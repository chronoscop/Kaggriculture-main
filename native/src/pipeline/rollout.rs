//! CPU environment workers feed one batched GPU coordinator. Policies see Observation only.
use super::{
    encoding,
    executor::*,
    planner::{self, Problem},
};
use crate::learning::{
    experience::Experience,
    policy::{cash_returns, Decision, Policy, Rng, Sample},
};
use kagg_engine::{
    engine::{self, PlayerAction},
    json::Json,
    state::State,
};
use std::time::Instant;
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Opponent {
    SelfPlay,
    Heuristic,
    League,
    Frozen(usize),
}
impl Opponent {
    pub fn name(self) -> &'static str {
        match self {
            Self::SelfPlay => "selfplay",
            Self::Heuristic => "heuristic",
            Self::League => "league",
            Self::Frozen(_) => "historical",
        }
    }
}
struct Pending {
    seat: usize,
    problem: Problem,
    obs: Observation,
    sample: Sample,
}
pub struct Game {
    pub state: State,
    pub agents: [Executor; 2],
    pub rows: [Vec<Sample>; 2],
    pub seed: i64,
    pub learner: usize,
    pub opponent: Opponent,
    seat: usize,
    actor: usize,
    market_done: bool,
    trading: Option<super::trading::Trading>,
    markets: [Vec<Vec<String>>; 2],
    actions: [PlayerAction; 2],
    pending: Option<Pending>,
    pub trace: Vec<Json>,
    pub record: bool,
    pub exploration: f32,
    pub greedy_probe: bool,
    pub experiences: Vec<Experience>,
    pub trade_stats: [super::behavior::TradeStats; 2],
    pub opponent_role: String,
    pub exploration_regime: String,
    pub learned_decisions: [usize; 2],
    pub wait_probability: [f64; 2],
    pub selected_groups: [[usize; super::GROUPS]; 2],
}
impl Game {
    pub fn new(seed: i64, learner: usize, opponent: Opponent, record: bool) -> Self {
        let mut agents = [Executor::new(), Executor::new()];
        if opponent == Opponent::Heuristic {
            agents[1 - learner].market_mode = super::trading::MarketMode::Rule;
        }
        Self {
            state: State::new(seed),
            agents,
            rows: [vec![], vec![]],
            seed,
            learner,
            opponent,
            seat: 0,
            actor: 0,
            market_done: false,
            trading: None,
            markets: [vec![], vec![]],
            actions: Default::default(),
            pending: None,
            trace: vec![],
            record,
            exploration: 0.,
            greedy_probe: false,
            experiences: vec![],
            trade_stats: Default::default(),
            opponent_role: opponent.name().into(),
            exploration_regime: "standard".into(),
            learned_decisions: [0; 2],
            wait_probability: [0.; 2],
            selected_groups: [[0; super::GROUPS]; 2],
        }
    }
    fn queue_orders(&mut self, orders: Vec<Vec<String>>) {
        if let Some(t) = &mut self.trading {
            t.apply(orders);
        } else {
            self.markets[self.seat].extend(orders);
        }
    }
    fn request(&mut self, p: Problem, o: Observation) -> Option<Sample> {
        if p.choices.len() == 1 {
            let orders = p
                .select(0, &mut self.agents[self.seat], &o)
                .expect("singleton candidate");
            self.queue_orders(orders);
            return None;
        }
        if self.opponent == Opponent::Heuristic && self.seat != self.learner {
            let orders = p
                .select(p.heuristic(), &mut self.agents[self.seat], &o)
                .expect("generated index");
            self.queue_orders(orders);
            return None;
        }
        let (context, features) = encoding::encode(&o, &self.agents[self.seat], &p);
        let sample = Sample {
            context,
            features,
            action: 0,
            logp: 0.,
            value: 0.,
            reward: 0.,
            exploration: if !self.greedy_probe
                && (self.seat == self.learner || self.opponent == Opponent::SelfPlay)
            {
                self.exploration
            } else {
                0.
            },
            step: o.step,
            // Hypothetical order proceeds in the trade observation are not realized rewards.
            cash: (self.state.farms[self.seat].money / 10000.) as f32,
            ..Sample::default()
        };
        self.pending = Some(Pending {
            seat: self.seat,
            problem: p,
            obs: o,
            sample: sample.clone(),
        });
        Some(sample)
    }
    pub fn prepare(&mut self) -> Option<Sample> {
        assert!(self.pending.is_none());
        loop {
            if self.state.step >= 719 {
                return None;
            }
            let seat = self.seat;
            let o = Observation::from_state(&self.state, seat);
            self.agents[seat].observe(&o);
            if !self.market_done {
                self.market_done = true;
                if self.agents[seat].market_due(&o) {
                    self.agents[seat].last_market = o.step;
                    self.agents[seat].last_cash = o.farm.money;
                    let p = planner::investment_problem(&o, &self.agents[seat]);
                    if let Some(sample) = self.request(p, o.clone()) {
                        return Some(sample);
                    }
                }
            }
            while self.actor < o.private.inventories.len() {
                let actor = self.actor;
                self.actor += 1;
                if self.agents[seat]
                    .routes
                    .get(actor)
                    .and_then(Option::as_ref)
                    .is_some_and(|r| !r.steps.is_empty())
                {
                    continue;
                }
                let p = planner::route_problem(&o, &self.agents[seat], actor);
                if let Some(sample) = self.request(p, o.clone()) {
                    return Some(sample);
                }
            }
            if self.trading.is_none() {
                let (mut action, projected) =
                    self.agents[seat].project_action(&o, std::mem::take(&mut self.markets[seat]));
                self.trading = super::trading::Trading::begin_if_due(
                    &o,
                    projected,
                    &mut action.market,
                    &mut self.agents[seat],
                );
                self.actions[seat] = action;
            }
            if let Some(t) = &mut self.trading {
                if let Some(p) = t.next(&mut self.agents[seat]) {
                    let trade_obs = t.obs.clone();
                    if let Some(sample) = self.request(p, trade_obs) {
                        return Some(sample);
                    }
                    continue;
                }
                self.actions[seat].market = self.trading.take().unwrap().finish();
            }
            self.actor = 0;
            self.market_done = false;
            if seat == 0 {
                self.seat = 1;
                continue;
            }
            if self.record {
                self.trace.push(Json::Obj(vec![
                    ("step".into(), Json::Num(self.state.step as f64)),
                    (
                        "actions".into(),
                        Json::Arr(self.actions.iter().map(action_json).collect()),
                    ),
                ]));
            }
            super::behavior::observe_market(&self.state, &self.actions, &mut self.trade_stats);
            engine::step(&mut self.state, &self.actions);
            self.seat = 0;
        }
    }
    pub fn accept(&mut self, d: Decision) -> Result<(), String> {
        let mut pending = self.pending.take().ok_or("no pending decision")?;
        let orders =
            pending
                .problem
                .select(d.action, &mut self.agents[pending.seat], &pending.obs)?;
        self.queue_orders(orders);
        self.learned_decisions[pending.seat] += 1;
        self.wait_probability[pending.seat] += d.wait_probability as f64;
        self.selected_groups[pending.seat][pending.sample.features[d.action][31] as usize] += 1;
        if self.record {
            self.trace.push(Json::Obj(vec![
                ("step".into(), Json::Num(self.state.step as f64)),
                ("seat".into(), Json::Num(pending.seat as f64)),
                ("decision".into(), Json::Num(d.action as f64)),
                (
                    "policy_wait_probability".into(),
                    Json::Num(d.wait_probability as f64),
                ),
                ("behavior_logp".into(), Json::Num(d.logp as f64)),
                (
                    "candidate_count".into(),
                    Json::Num(pending.sample.features.len() as f64),
                ),
                (
                    "selected".into(),
                    Json::Str(format!("{:?}", pending.problem.choices[d.action])),
                ),
            ]));
        }
        pending.sample.action = d.action;
        pending.sample.logp = d.logp;
        pending.sample.value = d.value;
        // Frozen opponents may supply self-generated imitation experience only.
        // terminal_rows keeps their deterministic decisions out of PPO.
        self.rows[pending.seat].push(pending.sample);
        Ok(())
    }
    pub fn terminal_rows(&mut self) -> Vec<Sample> {
        assert_eq!(self.state.step, 719);
        let mut all = Vec::new();
        for seat in 0..2 {
            let mut rows = std::mem::take(&mut self.rows[seat]);
            let cash = self.state.farms[seat].money;
            cash_returns(&mut rows, 719, (cash / 10000.) as f32, 0.997);
            if let Some(mut e) = Experience::from_episode(
                self.seed,
                seat,
                cash - 3000.,
                self.agents[seat].stats.harvested_units,
                &rows,
            ) {
                e.learner_seat = Some(self.learner);
                e.opponent = if seat != self.learner || self.opponent == Opponent::SelfPlay {
                    "current"
                } else if self.opponent == Opponent::Heuristic {
                    "heuristic"
                } else {
                    "historical"
                }
                .into();
                self.experiences.push(e);
            }
            if !self.greedy_probe && (seat == self.learner || self.opponent == Opponent::SelfPlay) {
                all.extend(rows);
            }
        }
        all
    }
    pub fn report(&self) -> Json {
        Json::Obj(vec![
            ("seed".into(), Json::Num(self.seed as f64)),
            ("greedy_probe".into(), Json::Bool(self.greedy_probe)),
            (
                "exploration_regime".into(),
                Json::Str(self.exploration_regime.clone()),
            ),
            ("exploration".into(), Json::Num(self.exploration as f64)),
            ("opponent".into(), Json::Str(self.opponent.name().into())),
            (
                "opponent_role".into(),
                Json::Str(self.opponent_role.clone()),
            ),
            (
                "opponent_slot".into(),
                match self.opponent {
                    Opponent::Frozen(i) => Json::Num(i as f64),
                    _ => Json::Null,
                },
            ),
            (
                "policy_wait_probability".into(),
                Json::Arr(
                    (0..2)
                        .map(|seat| {
                            Json::Num(
                                self.wait_probability[seat]
                                    / self.learned_decisions[seat].max(1) as f64,
                            )
                        })
                        .collect(),
                ),
            ),
            (
                "selected_groups".into(),
                Json::Arr(
                    self.selected_groups
                        .iter()
                        .map(|gs| Json::Arr(gs.iter().map(|&n| Json::Num(n as f64)).collect()))
                        .collect(),
                ),
            ),
            (
                "trading".into(),
                Json::Arr(
                    (0..2)
                        .map(|seat| {
                            let t = &self.trade_stats[seat];
                            Json::Obj(vec![
                                (
                                    "mode".into(),
                                    Json::Str(self.agents[seat].market_mode.name().into()),
                                ),
                                ("sales_revenue".into(), Json::Num(t.revenue.iter().sum())),
                                ("total_spending".into(), Json::Num(t.spending)),
                                (
                                    "ending_stock".into(),
                                    Json::Num(self.state.private[seat].shed.sum() as f64),
                                ),
                                (
                                    "products".into(),
                                    Json::Obj(
                                        kagg_engine::state::PRODUCTS
                                            .iter()
                                            .enumerate()
                                            .map(|(j, name)| {
                                                (
                                                    (*name).into(),
                                                    Json::Obj(vec![
                                                        (
                                                            "sold_units".into(),
                                                            Json::Num(t.units[j] as f64),
                                                        ),
                                                        (
                                                            "sales_revenue".into(),
                                                            Json::Num(t.revenue[j]),
                                                        ),
                                                        (
                                                            "bought_units".into(),
                                                            Json::Num(t.bought_units[j] as f64),
                                                        ),
                                                        (
                                                            "purchase_cost".into(),
                                                            Json::Num(t.purchase_cost[j]),
                                                        ),
                                                        (
                                                            "ending_stock".into(),
                                                            Json::Num(
                                                                self.state.private[seat]
                                                                    .shed
                                                                    .get(name)
                                                                    as f64,
                                                            ),
                                                        ),
                                                    ]),
                                                )
                                            })
                                            .collect(),
                                    ),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("learner_seat".into(), Json::Num(self.learner as f64)),
            ("steps".into(), Json::Num(self.state.step as f64)),
            (
                "cash".into(),
                Json::Arr(
                    self.state
                        .farms
                        .iter()
                        .map(|f| Json::Num(f.money))
                        .collect(),
                ),
            ),
            (
                "stats".into(),
                Json::Arr(self.agents.iter().map(|a| stats_json(&a.stats)).collect()),
            ),
        ])
    }
}
pub fn stats_json(s: &Stats) -> Json {
    Json::Obj(vec![
        ("trade_sessions".into(), Json::Num(s.trade_sessions as f64)),
        (
            "trade_events".into(),
            Json::Obj(
                super::trading::EVENT_NAMES
                    .iter()
                    .zip(s.trade_events)
                    .map(|(name, count)| ((*name).into(), Json::Num(count as f64)))
                    .collect(),
            ),
        ),
        (
            "trade_decisions".into(),
            Json::Num(s.trade_decisions as f64),
        ),
        ("trade_holds".into(), Json::Num(s.trade_holds as f64)),
        ("sell_orders".into(), Json::Num(s.sell_orders as f64)),
        ("buy_orders".into(), Json::Num(s.buy_orders as f64)),
        ("routes".into(), Json::Num(s.routes as f64)),
        ("mixed_routes".into(), Json::Num(s.mixed_routes as f64)),
        (
            "material_routes".into(),
            Json::Num(s.material_routes as f64),
        ),
        ("walking".into(), Json::Num(s.walking as f64)),
        ("work".into(), Json::Num(s.work as f64)),
        ("idle".into(), Json::Num(s.idle as f64)),
        ("invalidated".into(), Json::Num(s.invalidated as f64)),
        (
            "receipt_failures".into(),
            Json::Num(s.receipt_failures as f64),
        ),
        ("investments".into(), Json::Num(s.investments as f64)),
        (
            "projects_requested".into(),
            Json::Num(s.projects_requested as f64),
        ),
        (
            "projects_started".into(),
            Json::Num(s.projects_started as f64),
        ),
        (
            "harvested_units".into(),
            Json::Num(s.harvested_units as f64),
        ),
        (
            "fertilizer_used".into(),
            Json::Num(s.fertilizer_used as f64),
        ),
        (
            "expired_projects".into(),
            Json::Num(s.expired_projects as f64),
        ),
    ])
}
pub struct Collection {
    pub games: Vec<Game>,
    pub samples: Vec<Sample>,
    pub experiences: Vec<Experience>,
    pub seconds: f64,
    pub inference_seconds: f64,
    pub inference_calls: usize,
    pub mean_batch: f64,
}
/// Descriptive cohort outcomes; opponent/seed mixes may differ, not a paired A/B.
pub fn exploration_summary(c: &Collection) -> Json {
    Json::Obj(
        ["greedy", "focused", "broad", "standard"]
            .iter()
            .map(|&mode| {
                let games: Vec<_> = c
                    .games
                    .iter()
                    .filter(|g| g.exploration_regime == mode)
                    .collect();
                let n = games.len() as f64;
                let cash = games
                    .iter()
                    .map(|g| g.state.farms[g.learner].money)
                    .sum::<f64>();
                let margin = games
                    .iter()
                    .map(|g| g.state.farms[g.learner].money - g.state.farms[1 - g.learner].money)
                    .sum::<f64>();
                let points = games
                    .iter()
                    .map(|g| {
                        let d = g.state.farms[g.learner].money - g.state.farms[1 - g.learner].money;
                        if d > 0. {
                            1.
                        } else if d == 0. {
                            0.5
                        } else {
                            0.
                        }
                    })
                    .sum::<f64>();
                let mean = |value| {
                    if n > 0. {
                        Json::Num(value / n)
                    } else {
                        Json::Null
                    }
                };
                (
                    mode.into(),
                    Json::Obj(vec![
                        ("games".into(), Json::Num(n)),
                        ("mean_cash".into(), mean(cash)),
                        ("mean_margin".into(), mean(margin)),
                        ("score_rate".into(), mean(points)),
                    ]),
                )
            })
            .collect(),
    )
}

pub fn collect(
    policy: &Policy,
    seeds: &[i64],
    workers: usize,
    opponent: Opponent,
    deterministic: bool,
    rng: &mut Rng,
    record: bool,
) -> Result<Collection, String> {
    collect_with_pool(
        policy,
        seeds,
        workers,
        opponent,
        &[],
        deterministic,
        rng,
        record,
    )
}

/// Pool indices are stable during a collection. Only the learner's rows are trained.
pub fn collect_with_pool(
    policy: &Policy,
    seeds: &[i64],
    workers: usize,
    opponent: Opponent,
    pool: &[Policy],
    deterministic: bool,
    rng: &mut Rng,
    record: bool,
) -> Result<Collection, String> {
    collect_impl(
        policy,
        seeds,
        workers,
        opponent,
        pool,
        deterministic,
        rng,
        record,
        0.,
        false,
        None,
    )
}

pub fn collect_exploring(
    policy: &Policy,
    seeds: &[i64],
    workers: usize,
    opponent: Opponent,
    pool: &[Policy],
    deterministic: bool,
    rng: &mut Rng,
    record: bool,
    exploration: f32,
) -> Result<Collection, String> {
    collect_impl(
        policy,
        seeds,
        workers,
        opponent,
        pool,
        deterministic,
        rng,
        record,
        exploration,
        true,
        None,
    )
}
pub fn collect_roster(
    policy: &Policy,
    seeds: &[i64],
    workers: usize,
    pool: &[Policy],
    rng: &mut Rng,
    record: bool,
    exploration: f32,
    roster: &super::league::Roster,
) -> Result<Collection, String> {
    collect_impl(
        policy,
        seeds,
        workers,
        Opponent::League,
        pool,
        false,
        rng,
        record,
        exploration,
        true,
        Some(roster),
    )
}
/// One probe per full block of four seed pairs. Across four blocks, each
/// opponent-schedule residue gets exactly one probe (2/2/4 games at batch=32).
pub(crate) fn probe_schedule(count: usize, rng: &mut Rng) -> Vec<bool> {
    let mut mask = vec![false; count];
    let mut residues = [0, 1, 2, 3];
    for block in 0..count / 4 {
        if block % 4 == 0 {
            rng.shuffle(&mut residues);
        }
        mask[block * 4 + residues[block % 4]] = true;
    }
    mask
}

/// Reserve one quarter of stochastic game pairs for broad exploration.
/// The rest still sample the learned policy, with no extra uniform proposal.
/// A pair's exploration mode stays fixed for the full season.
pub const FOCUSED_EXPLORATION: f32 = 0.;
pub(crate) fn league_exploration_plan(
    probes: &[bool],
    maximum: f32,
    rng: &mut Rng,
) -> Vec<(&'static str, f32)> {
    let mut plan: Vec<_> = probes
        .iter()
        .map(|&probe| {
            if probe {
                ("greedy", 0.)
            } else {
                ("focused", FOCUSED_EXPLORATION)
            }
        })
        .collect();
    let mut candidates: Vec<_> = (0..probes.len()).filter(|&i| !probes[i]).collect();
    for j in 0..candidates.len() / 4 {
        let selected = j + (rng.uniform() * (candidates.len() - j) as f64) as usize;
        candidates.swap(j, selected);
        plan[candidates[j]] = ("broad", maximum);
    }
    plan
}

fn collect_impl(
    policy: &Policy,
    seeds: &[i64],
    workers: usize,
    opponent: Opponent,
    pool: &[Policy],
    deterministic: bool,
    rng: &mut Rng,
    record: bool,
    exploration: f32,
    probes: bool,
    roster: Option<&super::league::Roster>,
) -> Result<Collection, String> {
    if !exploration.is_finite() || !(0. ..=1.).contains(&exploration) {
        return Err("exploration must be between 0 and 1".into());
    }
    if seeds.is_empty() || workers == 0 {
        return Err("seeds and workers must be nonempty".into());
    }
    if let Opponent::Frozen(i) = opponent {
        if i >= pool.len() {
            return Err("missing frozen opponent".into());
        }
    }
    if let Some(r) = roster {
        if r.recent.is_empty()
            || r.historical.is_empty()
            || r.probabilities.len() != r.historical.len()
            || r.probabilities.iter().any(|p| !p.is_finite() || *p < 0.)
            || (r.probabilities.iter().sum::<f64>() - 1.).abs() > 1e-8
        {
            return Err("invalid roster probabilities".into());
        }
        if std::iter::once(&r.champion)
            .chain(r.recent.iter())
            .chain(r.historical.iter())
            .any(|&i| i >= pool.len())
        {
            return Err("missing roster opponent".into());
        }
    }
    let schedule = roster.map(|r| r.schedule(seeds.len(), rng));
    let mut games = Vec::new();
    let offset = (rng.uniform() * 4.) as usize;
    let probe_mask = if probes && !deterministic {
        probe_schedule(seeds.len(), rng)
    } else {
        vec![false; seeds.len()]
    };
    let exploration_plan = schedule.as_ref().map(|schedule| {
        league_exploration_plan(
            &schedule.iter().map(|s| s.2).collect::<Vec<_>>(),
            exploration,
            rng,
        )
    });
    for (index, &seed) in seeds.iter().enumerate() {
        let selected = if let Some(schedule) = &schedule {
            schedule[index].0
        } else if opponent == Opponent::League {
            match (index + offset) % 4 {
                0 => Opponent::Heuristic,
                1 => Opponent::SelfPlay,
                _ if !pool.is_empty() => Opponent::Frozen(
                    ((rng.uniform() * pool.len() as f64) as usize).min(pool.len() - 1),
                ),
                _ => Opponent::SelfPlay,
            }
        } else {
            opponent
        };
        for seat in 0..2 {
            let mut game = Game::new(seed, seat, selected, record);
            game.agents[seat].market_mode = policy.market_mode;
            game.agents[1 - seat].market_mode = match selected {
                Opponent::Heuristic => super::trading::MarketMode::Rule,
                Opponent::Frozen(i) => pool[i].market_mode,
                _ => policy.market_mode,
            };
            game.greedy_probe = probes
                && !deterministic
                && schedule
                    .as_ref()
                    .map(|s| s[index].2)
                    .unwrap_or(probe_mask[index]);
            if let Some(s) = &schedule {
                game.opponent_role = s[index].1.into();
            }
            if let Some(plan) = &exploration_plan {
                game.exploration = plan[index].1;
                game.exploration_regime = plan[index].0.into();
            } else {
                game.exploration = if deterministic || game.greedy_probe {
                    0.
                } else {
                    exploration
                };
                game.exploration_regime = if deterministic || game.greedy_probe {
                    "greedy"
                } else {
                    "standard"
                }
                .into();
            }
            games.push(game);
        }
    }
    let started = Instant::now();
    let mut inference_seconds = 0.;
    let mut calls = 0;
    let mut requests = 0;
    loop {
        let width = games.len().div_ceil(workers.min(games.len()));
        let batches = std::thread::scope(|scope| {
            let handles: Vec<_> = games
                .chunks_mut(width)
                .enumerate()
                .map(|(chunk, games)| {
                    scope.spawn(move || {
                        games
                            .iter_mut()
                            .enumerate()
                            .filter_map(|(j, g)| g.prepare().map(|s| (chunk * width + j, s)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().map_err(|_| "rollout worker panicked".to_string()))
                .collect::<Result<Vec<_>, _>>()
        })?;
        let pending: Vec<_> = batches.into_iter().flatten().collect();
        if pending.is_empty() {
            break;
        }
        // Keep GPU batches together by policy; never mix historical logits into PPO.
        let mut groups: Vec<Vec<(usize, Sample)>> =
            (0..pool.len() + 2).map(|_| Vec::new()).collect();
        for (index, sample) in pending {
            let g = &games[index];
            let group = if g.pending.as_ref().unwrap().seat != g.learner {
                match g.opponent {
                    Opponent::Frozen(i) => i + 1,
                    _ => 0,
                }
            } else {
                0
            };
            let group = if group == 0 && g.greedy_probe {
                pool.len() + 1
            } else {
                group
            };
            groups[group].push((index, sample));
        }
        for (group, pending) in groups.into_iter().enumerate() {
            if pending.is_empty() {
                continue;
            }
            let inputs: Vec<_> = pending.iter().map(|(_, s)| s.clone()).collect();
            let model = if group == 0 || group == pool.len() + 1 {
                policy
            } else {
                &pool[group - 1]
            };
            let t = Instant::now();
            let decisions = model.infer(&inputs, deterministic || group > 0, rng)?;
            inference_seconds += t.elapsed().as_secs_f64();
            calls += 1;
            requests += inputs.len();
            for ((index, _), decision) in pending.into_iter().zip(decisions) {
                games[index].accept(decision)?;
            }
        }
    }
    let samples = games.iter_mut().flat_map(Game::terminal_rows).collect();
    let experiences = games
        .iter_mut()
        .flat_map(|g| std::mem::take(&mut g.experiences))
        .collect();
    Ok(Collection {
        games,
        samples,
        experiences,
        seconds: started.elapsed().as_secs_f64(),
        inference_seconds,
        inference_calls: calls,
        mean_batch: if calls > 0 {
            requests as f64 / calls as f64
        } else {
            0.
        },
    })
}
