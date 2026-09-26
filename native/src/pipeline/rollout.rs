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
    markets: [Vec<Vec<String>>; 2],
    actions: [PlayerAction; 2],
    pending: Option<Pending>,
    pub trace: Vec<Json>,
    pub record: bool,
    pub exploration: f32,
    pub greedy_probe: bool,
    pub experiences: Vec<Experience>,
    pub learned_decisions: [usize; 2],
    pub wait_probability: [f64; 2],
    pub selected_groups: [[usize; 16]; 2],
}
impl Game {
    pub fn new(seed: i64, learner: usize, opponent: Opponent, record: bool) -> Self {
        Self {
            state: State::new(seed),
            agents: [Executor::new(), Executor::new()],
            rows: [vec![], vec![]],
            seed,
            learner,
            opponent,
            seat: 0,
            actor: 0,
            market_done: false,
            markets: [vec![], vec![]],
            actions: Default::default(),
            pending: None,
            trace: vec![],
            record,
            exploration: 0.,
            greedy_probe: false,
            experiences: vec![],
            learned_decisions: [0; 2],
            wait_probability: [0.; 2],
            selected_groups: [[0; 16]; 2],
        }
    }
    fn request(&mut self, p: Problem, o: Observation) -> Option<Sample> {
        if p.choices.len() == 1 {
            let orders = p
                .select(0, &mut self.agents[self.seat], &o)
                .expect("singleton candidate");
            self.markets[self.seat].extend(orders);
            return None;
        }
        if self.opponent == Opponent::Heuristic && self.seat != self.learner {
            let orders = p
                .select(p.heuristic(), &mut self.agents[self.seat], &o)
                .expect("generated index");
            self.markets[self.seat].extend(orders);
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
            cash: (o.farm.money / 10000.) as f32,
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
            self.actions[seat] =
                self.agents[seat].action(&o, std::mem::take(&mut self.markets[seat]));
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
        self.markets[pending.seat].extend(orders);
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
            if let Some(e) = Experience::from_episode(
                self.seed,
                seat,
                cash - 3000.,
                self.agents[seat].stats.harvested_units,
                &rows,
            ) {
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
            ("opponent".into(), Json::Str(self.opponent.name().into())),
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
    )
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
    let mut games = Vec::new();
    let offset = (rng.uniform() * 4.) as usize;
    for (index, &seed) in seeds.iter().enumerate() {
        let selected = if opponent == Opponent::League {
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
            game.greedy_probe = probes && !deterministic && index % 4 == 3;
            game.exploration = if deterministic || game.greedy_probe {
                0.
            } else {
                exploration
            };
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
