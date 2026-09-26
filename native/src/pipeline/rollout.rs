//! CPU environment workers feed one batched GPU coordinator. Policies see Observation only.
use super::{
    encoding,
    executor::*,
    planner::{self, Problem},
};
use crate::learning::policy::{Decision, Policy, Rng, Sample};
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
}
impl Opponent {
    pub fn name(self) -> &'static str {
        match self {
            Self::SelfPlay => "selfplay",
            Self::Heuristic => "heuristic",
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
        }
    }
    fn request(&mut self, p: Problem, o: Observation) -> Option<Sample> {
        if p.choices.len() < 2 {
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
        if self.record {
            self.trace.push(Json::Obj(vec![
                ("step".into(), Json::Num(self.state.step as f64)),
                ("seat".into(), Json::Num(pending.seat as f64)),
                ("decision".into(), Json::Num(d.action as f64)),
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
        self.rows[pending.seat].push(pending.sample);
        Ok(())
    }
    pub fn terminal_rows(&mut self) -> Vec<Sample> {
        assert_eq!(self.state.step, 719);
        let mut all = Vec::new();
        for seat in 0..2 {
            let reward =
                ((self.state.farms[seat].money - self.state.farms[1 - seat].money) / 10000.) as f32;
            for mut row in std::mem::take(&mut self.rows[seat]) {
                row.reward = reward;
                all.push(row);
            }
        }
        all
    }
    pub fn report(&self) -> Json {
        Json::Obj(vec![
            ("seed".into(), Json::Num(self.seed as f64)),
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
    if seeds.is_empty() || workers == 0 {
        return Err("seeds and workers must be nonempty".into());
    }
    let mut games: Vec<_> = seeds
        .iter()
        .flat_map(|&seed| (0..2).map(move |seat| Game::new(seed, seat, opponent, record)))
        .collect();
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
        let inputs: Vec<_> = pending.iter().map(|(_, s)| s.clone()).collect();
        let t = Instant::now();
        let decisions = policy.infer(&inputs, deterministic, rng)?;
        inference_seconds += t.elapsed().as_secs_f64();
        calls += 1;
        requests += inputs.len();
        for ((index, _), decision) in pending.into_iter().zip(decisions) {
            games[index].accept(decision)?;
        }
    }
    let samples = games.iter_mut().flat_map(Game::terminal_rows).collect();
    Ok(Collection {
        games,
        samples,
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
