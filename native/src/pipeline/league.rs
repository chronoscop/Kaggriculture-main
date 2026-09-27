//! Frozen opponents and validation gates. Evaluation never contributes training rows.
use super::behavior::Profile;
use super::matchmaking::{self, OpponentResults};
use super::rollout::{self, Collection, Opponent};
use crate::learning::policy::{Policy, Rng};
use kagg_engine::json::Json;
use std::collections::BTreeMap;

pub const CAPACITY: usize = 8;
pub const SNAPSHOT_EVERY: u64 = 10;
pub struct Snapshot {
    pub iteration: u64,
    pub weights: Json,
    pub profile: Option<Profile>,
}
pub struct League {
    pub snapshots: Vec<Snapshot>,
    pub champion_iteration: u64,
    pub outcomes: BTreeMap<u64, OpponentResults>,
}
#[derive(Clone, Debug)]
pub struct Score {
    pub games: usize,
    pub cash: f64,
    pub margin: f64,
    pub win_rate: f64,
    pub draw_rate: f64,
    pub work: f64,
    pub harvest: f64,
    pub idle_fraction: f64,
    pub inactive_games: usize,
}
impl Score {
    pub fn from_collection(c: &Collection) -> Self {
        let n = c.games.len() as f64;
        let mut score = Self {
            games: c.games.len(),
            cash: 0.,
            margin: 0.,
            win_rate: 0.,
            draw_rate: 0.,
            work: 0.,
            harvest: 0.,
            idle_fraction: 0.,
            inactive_games: 0,
        };
        for g in &c.games {
            let seat = g.learner;
            let cash = g.state.farms[seat].money;
            let margin = cash - g.state.farms[1 - seat].money;
            let s = &g.agents[seat].stats;
            score.cash += cash / n;
            score.margin += margin / n;
            score.win_rate += f64::from(margin > 0.) / n;
            score.draw_rate += f64::from(margin == 0.) / n;
            score.work += s.work as f64 / n;
            score.harvest += s.harvested_units as f64 / n;
            let total = s.work + s.walking + s.idle;
            score.idle_fraction += s.idle as f64 / total.max(1) as f64 / n;
            score.inactive_games += usize::from(s.work == 0 || s.harvested_units == 0);
        }
        score
    }
    pub fn json(&self) -> Json {
        Json::Obj(vec![
            ("games".into(), Json::Num(self.games as f64)),
            ("mean_cash".into(), Json::Num(self.cash)),
            ("mean_margin".into(), Json::Num(self.margin)),
            ("win_rate".into(), Json::Num(self.win_rate)),
            ("draw_rate".into(), Json::Num(self.draw_rate)),
            ("mean_work".into(), Json::Num(self.work)),
            ("mean_harvested_units".into(), Json::Num(self.harvest)),
            ("idle_fraction".into(), Json::Num(self.idle_fraction)),
            (
                "inactive_games".into(),
                Json::Num(self.inactive_games as f64),
            ),
        ])
    }
}
/// The heuristic is a basic production guard, not a record to beat on every promotion.
pub fn anchor_guard(candidate: &Score, champion: &Score) -> bool {
    candidate.inactive_games == 0 && candidate.cash > 3000. && candidate.cash >= 0.9 * champion.cash
}
/// Pool admission is separate from best-model promotion.
pub fn qualifies_for_pool(candidate: &Score, duel: &Score) -> bool {
    candidate.inactive_games == 0
        && candidate.cash > 3000.
        && duel.inactive_games == 0
        && duel.cash > 3000.
        && duel.margin > 0.
        && duel.win_rate > 0.5
}
impl League {
    pub fn new(policy: &Policy) -> Result<Self, String> {
        Ok(Self {
            champion_iteration: 0,
            outcomes: BTreeMap::new(),
            snapshots: vec![Snapshot {
                iteration: 0,
                weights: policy.weights_json()?,
                profile: None,
            }],
        })
    }
    pub fn restore(j: &Json) -> Result<Self, String> {
        if !j.is_arr() || j.arr().is_empty() || j.arr().len() > CAPACITY {
            return Err("checkpoint lacks valid opponent pool".into());
        }
        let mut snapshots = Vec::new();
        let mut outcomes = BTreeMap::new();
        for item in j.arr() {
            let iteration = item
                .get("iteration")
                .str()
                .parse::<u64>()
                .map_err(|_| "invalid pool iteration")?;
            if !item.get("weights").is_obj()
                || snapshots
                    .last()
                    .is_some_and(|s: &Snapshot| s.iteration >= iteration)
            {
                return Err("invalid pool weights or order".into());
            }
            outcomes.insert(
                iteration,
                OpponentResults::restore(item.get("training_outcomes"))?,
            );
            snapshots.push(Snapshot {
                iteration,
                weights: item.get("weights").clone(),
                profile: Profile::parse(item.get("behavior_profile"))?,
            });
        }
        let marked: Vec<_> = j
            .arr()
            .iter()
            .filter(|s| s.get("champion").bool())
            .collect();
        if marked.len() > 1
            || (marked.is_empty() && j.arr().iter().any(|s| !s.get("champion").is_null()))
        {
            return Err("invalid champion marker".into());
        }
        // Legacy v7 pools kept the champion last.
        let champion_iteration = marked
            .first()
            .map(|s| s.get("iteration").str().parse::<u64>().unwrap())
            .unwrap_or_else(|| snapshots.last().unwrap().iteration);
        Ok(Self {
            snapshots,
            champion_iteration,
            outcomes,
        })
    }
    pub fn json(&self) -> Json {
        Json::Arr(
            self.snapshots
                .iter()
                .map(|s| {
                    Json::Obj(vec![
                        ("iteration".into(), Json::Str(s.iteration.to_string())),
                        (
                            "champion".into(),
                            Json::Bool(s.iteration == self.champion_iteration),
                        ),
                        ("weights".into(), s.weights.clone()),
                        (
                            "training_outcomes".into(),
                            self.outcomes
                                .get(&s.iteration)
                                .cloned()
                                .unwrap_or_default()
                                .json(),
                        ),
                        (
                            "behavior_profile".into(),
                            s.profile.as_ref().map(Profile::json).unwrap_or(Json::Null),
                        ),
                    ])
                })
                .collect(),
        )
    }
    pub fn policies(&self, device: i32) -> Result<Vec<Policy>, String> {
        self.snapshots
            .iter()
            .map(|s| {
                let mut p = Policy::mixed_routes(device, 0, 1e-4)?;
                p.load_weights(&s.weights)?;
                Ok(p)
            })
            .collect()
    }
    pub fn champion_index(&self) -> usize {
        self.snapshots
            .iter()
            .position(|s| s.iteration == self.champion_iteration)
            .expect("champion must remain in training pool")
    }

    pub fn ensure_profiles(
        &mut self,
        pool: &[Policy],
        seeds: &[i64],
        workers: usize,
        eval_seed: u64,
    ) -> Result<(), String> {
        for (i, snapshot) in self.snapshots.iter_mut().enumerate() {
            if snapshot.profile.is_none() {
                println!(
                    "{{\"stage\":\"profile_opponent\",\"iteration\":{}}}",
                    snapshot.iteration
                );
                let c = rollout::collect(
                    &pool[i],
                    seeds,
                    workers,
                    Opponent::Heuristic,
                    true,
                    &mut Rng(eval_seed),
                    false,
                )?;
                snapshot.profile = Some(Profile::from_collection(&c));
            }
        }
        Ok(())
    }
    fn distant_indices(&self, selected: &[usize], count: usize) -> Vec<usize> {
        let mut chosen = selected.to_vec();
        let mut added = Vec::new();
        for _ in 0..count {
            let next = (0..self.snapshots.len())
                .filter(|i| !chosen.contains(i))
                .max_by(|&a, &b| {
                    let distance = |i: usize| {
                        self.snapshots[i]
                            .profile
                            .as_ref()
                            .map(|p| {
                                chosen
                                    .iter()
                                    .filter_map(|&j| {
                                        self.snapshots[j].profile.as_ref().map(|q| p.distance(q))
                                    })
                                    .reduce(f64::min)
                                    .unwrap_or(0.)
                            })
                            .unwrap_or(0.)
                    };
                    distance(a).total_cmp(&distance(b)).then(
                        self.snapshots[a]
                            .iteration
                            .cmp(&self.snapshots[b].iteration),
                    )
                });
            if let Some(i) = next {
                chosen.push(i);
                added.push(i)
            } else {
                break;
            }
        }
        added
    }
    fn recent_indices(&self) -> Vec<usize> {
        (0..self.snapshots.len()).rev().take(2).collect()
    }
    fn difficulty(&self, i: usize, iteration: u64) -> f64 {
        let r = self
            .outcomes
            .get(&self.snapshots[i].iteration)
            .cloned()
            .unwrap_or_default();
        // Deployment-mode difficulty; missing/stale probe evidence shrinks to 0.5.
        1. - r.greedy.score(iteration)
    }

    pub fn roster(&self, iteration: u64) -> Roster {
        let recent = self.recent_indices();
        let mut historical: Vec<_> = (0..self.snapshots.len())
            .filter(|i| !recent.contains(i))
            .collect();
        if historical.is_empty() {
            historical.push(self.champion_index());
        }
        let scores: Vec<_> = historical
            .iter()
            .map(|&i| {
                self.outcomes
                    .get(&self.snapshots[i].iteration)
                    .cloned()
                    .unwrap_or_default()
                    .greedy
                    .score(iteration)
            })
            .collect();
        Roster {
            champion: self.champion_index(),
            recent,
            historical,
            probabilities: matchmaking::probabilities(&scores),
            phase: iteration as usize,
        }
    }
    /// Called BEFORE PPO and any pool mutation; slot indices still refer to the actual opponents.
    pub fn observe_training(&mut self, c: &Collection, iteration: u64) {
        for g in &c.games {
            if let Opponent::Frozen(i) = g.opponent {
                if g.state.step != 719 {
                    continue;
                }
                let version = self.snapshots[i].iteration;
                let outcomes = self.outcomes.entry(version).or_default();
                let r = if g.greedy_probe {
                    &mut outcomes.greedy
                } else {
                    &mut outcomes.sampled
                };
                let cash = g.state.farms[g.learner].money;
                let rival_cash = g.state.farms[1 - g.learner].money;
                r.observe(iteration, cash, rival_cash);
                match g.exploration_regime.as_str() {
                    "focused" => outcomes.focused.observe(iteration, cash, rival_cash),
                    "broad" => outcomes.broad.observe(iteration, cash, rival_cash),
                    _ => {}
                }
            }
        }
    }
    pub fn matchmaking_report(&self, iteration: u64) -> Json {
        let roster = self.roster(iteration);
        Json::Arr(
            self.snapshots
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let r = self.outcomes.get(&s.iteration).cloned().unwrap_or_default();
                    let probability = roster
                        .historical
                        .iter()
                        .position(|&j| j == i)
                        .map(|j| roster.probabilities[j])
                        .unwrap_or(0.);
                    Json::Obj(vec![
                        ("iteration".into(), Json::Str(s.iteration.to_string())),
                        ("recent".into(), Json::Bool(roster.recent.contains(&i))),
                        ("champion".into(), Json::Bool(i == roster.champion)),
                        ("pfsp_probability".into(), Json::Num(probability)),
                        (
                            "difficulty_source".into(),
                            Json::Str("greedy_training_probes".into()),
                        ),
                        (
                            "difficulty_score".into(),
                            Json::Num(r.greedy.score(iteration)),
                        ),
                        ("focused".into(), r.focused.report(iteration)),
                        ("broad".into(), r.broad.report(iteration)),
                        ("sampled".into(), r.sampled.report(iteration)),
                        ("greedy".into(), r.greedy.report(iteration)),
                    ])
                })
                .collect(),
        )
    }
    /// Periodic snapshots advance the curriculum independently of promotion/validation.
    pub fn freeze_recent(
        &mut self,
        policy: &Policy,
        iteration: u64,
        force: bool,
    ) -> Result<bool, String> {
        if self.snapshots.iter().any(|s| s.iteration == iteration)
            || (!force && iteration % SNAPSHOT_EVERY != 0)
        {
            return Ok(false);
        }
        self.admit(
            Snapshot {
                iteration,
                weights: policy.weights_json()?,
                profile: None,
            },
            false,
        );
        Ok(true)
    }
    pub(crate) fn admit(&mut self, snapshot: Snapshot, promoted: bool) {
        let iteration = snapshot.iteration;
        if promoted {
            self.champion_iteration = iteration;
        }
        if let Some(existing) = self.snapshots.iter_mut().find(|s| s.iteration == iteration) {
            *existing = snapshot;
            self.outcomes.remove(&iteration);
        } else {
            self.snapshots.push(snapshot);
        }
        self.snapshots.sort_by_key(|s| s.iteration);
        if self.snapshots.len() > CAPACITY {
            let mut keep = vec![self.champion_index()];
            for i in self.recent_indices() {
                if !keep.contains(&i) {
                    keep.push(i);
                }
            }
            // A long-term reference survives churn even when currently easy.
            if let Some(i) = (0..self.snapshots.len()).find(|i| !keep.contains(i)) {
                keep.push(i);
            }
            let mut difficult: Vec<_> = (0..self.snapshots.len())
                .filter(|i| !keep.contains(i))
                .collect();
            difficult.sort_by(|&a, &b| {
                self.difficulty(b, iteration)
                    .total_cmp(&self.difficulty(a, iteration))
                    .then(
                        self.snapshots[a]
                            .iteration
                            .cmp(&self.snapshots[b].iteration),
                    )
            });
            for i in difficult
                .into_iter()
                .filter(|&i| self.difficulty(i, iteration) > 0.5)
                .take(2)
            {
                keep.push(i);
            }
            // Spread remaining references over training history; no behavior-profile ranking.
            while keep.len() < CAPACITY {
                let next = (0..self.snapshots.len())
                    .filter(|i| !keep.contains(i))
                    .max_by_key(|&i| {
                        keep.iter()
                            .map(|&j| {
                                self.snapshots[i]
                                    .iteration
                                    .abs_diff(self.snapshots[j].iteration)
                            })
                            .min()
                            .unwrap_or(0)
                    })
                    .unwrap();
                keep.push(next);
            }
            self.snapshots = std::mem::take(&mut self.snapshots)
                .into_iter()
                .enumerate()
                .filter(|(i, _)| keep.contains(i))
                .map(|(_, s)| s)
                .collect();
        }
        self.outcomes
            .retain(|iteration, _| self.snapshots.iter().any(|s| s.iteration == *iteration));
    }
    pub fn evaluate_and_promote(
        &mut self,
        candidate: &Policy,
        iteration: u64,
        seeds: &[i64],
        workers: usize,
        eval_seed: u64,
    ) -> Result<(bool, Json), String> {
        let started = std::time::Instant::now();
        let pool = self.policies(candidate.device)?;
        let profile_games = 0;
        let ci = self.champion_index();
        let champion_iteration = self.champion_iteration;
        let anchor = |p: &Policy, det: bool| {
            rollout::collect(
                p,
                seeds,
                workers,
                Opponent::Heuristic,
                det,
                &mut Rng(eval_seed),
                false,
            )
        };
        let fixed = anchor(candidate, true)?;
        let greedy = Score::from_collection(&fixed);
        let profile = Profile::from_collection(&fixed);
        let stochastic = Score::from_collection(&anchor(candidate, false)?);
        let champion = Score::from_collection(&anchor(&pool[ci], true)?);
        let duel = Score::from_collection(&rollout::collect_with_pool(
            candidate,
            seeds,
            workers,
            Opponent::Frozen(ci),
            &pool,
            true,
            &mut Rng(eval_seed),
            false,
        )?);
        let roster = self.roster(iteration);
        let mut panel = vec![ci];
        if let Some(&i) = roster.recent.iter().find(|&&i| i != ci) {
            if !panel.contains(&i) {
                panel.push(i)
            }
        }
        for i in self.distant_indices(&panel, 1) {
            panel.push(i)
        }
        let (screen_seeds, confirm_seeds) = gate_seeds(eval_seed, iteration, seeds.len())?;
        let screen = evaluate_panel(
            candidate,
            &pool,
            ci,
            &panel,
            &screen_seeds,
            workers,
            eval_seed,
        )?;
        let basic = anchor_guard(&greedy, &champion);
        let confirmation = if basic && screen.qualifies() {
            Some(evaluate_panel(
                candidate,
                &pool,
                ci,
                &panel,
                &confirm_seeds,
                workers,
                eval_seed,
            )?)
        } else {
            None
        };
        let promoted = confirmation.as_ref().is_some_and(PanelResult::qualifies);
        let novelty = self
            .snapshots
            .iter()
            .filter_map(|s| s.profile.as_ref().map(|q| profile.distance(q)))
            .reduce(f64::min)
            .unwrap_or(0.);
        let competitive = screen.competitive && greedy.inactive_games == 0 && greedy.cash > 3000.;
        let admitted = promoted || competitive;
        let reason = if promoted {
            "promoted"
        } else if competitive {
            "competitive"
        } else {
            "none"
        };
        let panel_iterations = Json::Arr(
            panel
                .iter()
                .map(|&i| Json::Num(self.snapshots[i].iteration as f64))
                .collect(),
        );
        if admitted {
            self.admit(
                Snapshot {
                    iteration,
                    weights: candidate.weights_json()?,
                    profile: Some(profile.clone()),
                },
                promoted,
            )
        }
        Ok((
            promoted,
            Json::Obj(vec![
                ("iteration".into(), Json::Num(iteration as f64)),
                (
                    "seeds".into(),
                    Json::Arr(seeds.iter().map(|&s| Json::Num(s as f64)).collect()),
                ),
                (
                    "champion_iteration_before".into(),
                    Json::Num(champion_iteration as f64),
                ),
                ("promoted".into(), Json::Bool(promoted)),
                ("pool_admitted".into(), Json::Bool(admitted)),
                ("admission_reason".into(), Json::Str(reason.into())),
                ("behavior_novelty".into(), Json::Num(novelty)),
                ("behavior_profile".into(), profile.json()),
                ("pool_size".into(), Json::Num(self.snapshots.len() as f64)),
                ("panel_iterations".into(), panel_iterations),
                (
                    "evaluation_games_executed".into(),
                    Json::Num(
                        (profile_games
                            + seeds.len() * 8
                            + (2 * panel.len() - 1) * screen_seeds.len() * 2
                            + if confirmation.is_some() {
                                (2 * panel.len() - 1) * confirm_seeds.len() * 2
                            } else {
                                0
                            }) as f64,
                    ),
                ),
                ("screen".into(), screen.json(&screen_seeds)),
                (
                    "confirmation".into(),
                    confirmation
                        .as_ref()
                        .map(|c| c.json(&confirm_seeds))
                        .unwrap_or(Json::Null),
                ),
                ("greedy_vs_heuristic".into(), greedy.json()),
                ("sampled_vs_heuristic".into(), stochastic.json()),
                ("champion_vs_heuristic".into(), champion.json()),
                ("greedy_vs_champion".into(), duel.json()),
                ("seconds".into(), Json::Num(started.elapsed().as_secs_f64())),
            ]),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct Roster {
    pub champion: usize,
    pub recent: Vec<usize>,
    pub historical: Vec<usize>,
    pub probabilities: Vec<f64>,
    pub phase: usize,
}
impl Roster {
    pub fn schedule(&self, count: usize, rng: &mut Rng) -> Vec<(Opponent, &'static str, bool)> {
        let mut rows = Vec::new();
        let choose_pfsp = |rng: &mut Rng| {
            let mut u = rng.uniform() as f64;
            for (&i, &p) in self.historical.iter().zip(&self.probabilities) {
                u -= p;
                if u <= 0. {
                    return i;
                }
            }
            self.historical.last().copied().unwrap_or(self.champion)
        };
        for index in 0..count {
            let i = index % 16;
            // 12 recent / 12 PFSP / 4 coverage / 2 self / 2 heuristic games per 32.
            let (op, role) = match i {
                0..=5 => (
                    Opponent::Frozen(self.recent[(index + self.phase) % self.recent.len()]),
                    "recent",
                ),
                6..=11 => (Opponent::Frozen(choose_pfsp(rng)), "pfsp"),
                12..=13 => (
                    Opponent::Frozen(
                        self.historical
                            [(self.phase * 2 + index / 16 * 2 + i - 12) % self.historical.len()],
                    ),
                    "coverage",
                ),
                14 => (Opponent::SelfPlay, "current"),
                _ => (Opponent::Heuristic, "heuristic"),
            };
            rows.push((op, role, false));
        }
        // Select one pair per quartet, independently of opponent/version and role.
        for start in (0..count / 4).map(|i| i * 4) {
            rows[start + (rng.uniform() * 4.) as usize].2 = true;
        }
        rows
    }
}
/// Reserve a separate, deterministic block for every iteration, including resumed runs.
pub fn gate_seeds(base: u64, iteration: u64, n: usize) -> Result<(Vec<i64>, Vec<i64>), String> {
    let start = base as u128 + 1_000_000 + (iteration as u128) * (4 * n as u128);
    if n == 0 || start + 3 * n as u128 > i64::MAX as u128 {
        return Err("gate seed overflow".into());
    }
    Ok((
        (0..n).map(|i| (start + i as u128) as i64).collect(),
        (n..3 * n).map(|i| (start + i as u128) as i64).collect(),
    ))
}
struct PanelResult {
    gain: f64,
    win_gain: f64,
    productive: bool,
    competitive: bool,
    rows: Vec<Json>,
}
impl PanelResult {
    fn qualifies(&self) -> bool {
        self.productive && self.gain > 0. && self.win_gain > 0.
    }
    fn json(&self, seeds: &[i64]) -> Json {
        Json::Obj(vec![
            (
                "seeds".into(),
                Json::Arr(seeds.iter().map(|&s| Json::Num(s as f64)).collect()),
            ),
            ("mean_relative_margin_gain".into(), Json::Num(self.gain)),
            ("mean_win_rate_gain".into(), Json::Num(self.win_gain)),
            ("productive".into(), Json::Bool(self.productive)),
            ("qualifies".into(), Json::Bool(self.qualifies())),
            ("matchups".into(), Json::Arr(self.rows.clone())),
        ])
    }
}
fn competitive_improvement(c: &Score, baseline: Option<&Score>) -> bool {
    let relative = |s: &Score| s.margin / (2. * s.cash - s.margin).max(1.);
    c.inactive_games == 0
        && c.cash > 3000.
        && c.margin > 0.
        && c.win_rate > 0.5
        && baseline.is_none_or(|b| {
            relative(c) > relative(b) + 1e-6
                && c.win_rate + 0.5 * c.draw_rate >= b.win_rate + 0.5 * b.draw_rate
        })
}
fn evaluate_panel(
    candidate: &Policy,
    pool: &[Policy],
    ci: usize,
    panel: &[usize],
    seeds: &[i64],
    workers: usize,
    rng_seed: u64,
) -> Result<PanelResult, String> {
    let play = |p: &Policy, i| -> Result<Score, String> {
        Ok(Score::from_collection(&rollout::collect_with_pool(
            p,
            seeds,
            workers,
            Opponent::Frozen(i),
            pool,
            true,
            &mut Rng(rng_seed),
            false,
        )?))
    };
    let relative = |s: &Score| s.margin / (2. * s.cash - s.margin).max(1.);
    let mut result = PanelResult {
        gain: 0.,
        win_gain: 0.,
        productive: true,
        competitive: false,
        rows: Vec::new(),
    };
    for &i in panel {
        let c = play(candidate, i)?;
        let baseline = if i == ci {
            None
        } else {
            Some(play(&pool[ci], i)?)
        };
        result.gain += relative(&c) - baseline.as_ref().map(&relative).unwrap_or(0.);
        result.win_gain += c.win_rate + 0.5 * c.draw_rate
            - baseline
                .as_ref()
                .map(|s| s.win_rate + 0.5 * s.draw_rate)
                .unwrap_or(0.5);
        result.productive &= c.inactive_games == 0 && c.cash > 3000.;
        result.competitive |= competitive_improvement(&c, baseline.as_ref());
        result.rows.push(Json::Obj(vec![
            ("opponent_slot".into(), Json::Num(i as f64)),
            ("candidate".into(), c.json()),
            (
                "champion".into(),
                baseline.map(|b| b.json()).unwrap_or(Json::Null),
            ),
        ]));
    }
    result.gain /= panel.len() as f64;
    result.win_gain /= panel.len() as f64;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotating_screen_and_confirmation_never_reuse_seeds() {
        let (a, b) = gate_seeds(1_000_000_000, 1886, 4).unwrap();
        let (c, d) = gate_seeds(1_000_000_000, 1887, 4).unwrap();
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 8);
        let all: std::collections::BTreeSet<_> =
            a.iter().chain(&b).chain(&c).chain(&d).copied().collect();
        assert_eq!(all.len(), 24);
        assert!(all.iter().all(|s| *s >= 1_001_000_000));
        assert!(gate_seeds(u64::MAX, u64::MAX, 4).is_err());
    }
    #[test]
    fn panel_requires_productivity_and_both_return_and_win_improvement() {
        let mut r = PanelResult {
            gain: 0.01,
            win_gain: 0.1,
            productive: true,
            competitive: true,
            rows: vec![],
        };
        assert!(r.qualifies());
        r.gain = -0.01;
        assert!(!r.qualifies());
        r.gain = 0.01;
        r.win_gain = 0.;
        assert!(!r.qualifies());
        r.win_gain = 0.1;
        r.productive = false;
        assert!(!r.qualifies());
    }
    #[test]
    fn unchanged_specialist_does_not_reenter_pool_by_beating_a_weak_opponent() {
        let c = Score {
            games: 8,
            cash: 60000.,
            margin: 50000.,
            win_rate: 1.,
            draw_rate: 0.,
            work: 500.,
            harvest: 100.,
            idle_fraction: 0.1,
            inactive_games: 0,
        };
        assert!(!competitive_improvement(&c, Some(&c)));
        let mut improved = c.clone();
        improved.cash += 5000.;
        improved.margin += 5000.;
        assert!(competitive_improvement(&improved, Some(&c)));
    }
}
