//! Frozen opponents and validation gates. Evaluation never contributes training rows.
use super::rollout::{self, Collection, Opponent};
use crate::learning::policy::{Policy, Rng};
use kagg_engine::json::Json;

pub const CAPACITY: usize = 4;
pub struct Snapshot {
    pub iteration: u64,
    pub weights: Json,
}
pub struct League {
    pub snapshots: Vec<Snapshot>,
}
#[derive(Clone, Debug)]
pub struct Score {
    pub games: usize,
    pub cash: f64,
    pub margin: f64,
    pub win_rate: f64,
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
/// Require real production and improvement against both an anchor and the champion.
/// This is a small validation gate, not a statistical claim about unseen opponents.
pub fn qualifies(candidate: &Score, champion: &Score, head_to_head: &Score) -> bool {
    candidate.inactive_games == 0
        && candidate.cash > 3000.
        && candidate.margin > champion.margin
        && head_to_head.margin > 0.
        && head_to_head.win_rate > 0.5
}
impl League {
    pub fn new(policy: &Policy) -> Result<Self, String> {
        Ok(Self {
            snapshots: vec![Snapshot {
                iteration: 0,
                weights: policy.weights_json()?,
            }],
        })
    }
    pub fn restore(j: &Json) -> Result<Self, String> {
        if !j.is_arr() || j.arr().is_empty() || j.arr().len() > CAPACITY {
            return Err("checkpoint lacks valid opponent pool".into());
        }
        let mut snapshots = Vec::new();
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
            snapshots.push(Snapshot {
                iteration,
                weights: item.get("weights").clone(),
            });
        }
        Ok(Self { snapshots })
    }
    pub fn json(&self) -> Json {
        Json::Arr(
            self.snapshots
                .iter()
                .map(|s| {
                    Json::Obj(vec![
                        ("iteration".into(), Json::Str(s.iteration.to_string())),
                        ("weights".into(), s.weights.clone()),
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
        let champion_index = pool.len() - 1;
        let champion_iteration = self.snapshots[champion_index].iteration;
        let anchor = |p: &Policy, deterministic: bool| -> Result<Score, String> {
            Ok(Score::from_collection(&rollout::collect(
                p,
                seeds,
                workers,
                Opponent::Heuristic,
                deterministic,
                &mut Rng(eval_seed),
                false,
            )?))
        };
        let greedy = anchor(candidate, true)?;
        let stochastic = anchor(candidate, false)?;
        let champion = anchor(&pool[champion_index], true)?;
        let duel = Score::from_collection(&rollout::collect_with_pool(
            candidate,
            seeds,
            workers,
            Opponent::Frozen(champion_index),
            &pool,
            true,
            &mut Rng(eval_seed),
            false,
        )?);
        let promoted = qualifies(&greedy, &champion, &duel);
        if promoted {
            self.snapshots.push(Snapshot {
                iteration,
                weights: candidate.weights_json()?,
            });
            if self.snapshots.len() > CAPACITY {
                self.snapshots.remove(0);
            }
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
                ("pool_size".into(), Json::Num(self.snapshots.len() as f64)),
                ("greedy_vs_heuristic".into(), greedy.json()),
                ("sampled_vs_heuristic".into(), stochastic.json()),
                ("champion_vs_heuristic".into(), champion.json()),
                ("greedy_vs_champion".into(), duel.json()),
                ("seconds".into(), Json::Num(started.elapsed().as_secs_f64())),
            ]),
        ))
    }
}
