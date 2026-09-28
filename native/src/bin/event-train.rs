//! Bounded policy improvement: accepted portfolio -> one scoped change -> full-game gate.
#[cfg(feature = "train")]
mod app {
    use kagg_engine::{
        engine::{self, PlayerAction},
        json::{self, Json},
        state::State as GameState,
    };
    use route_rl_native::{
        learning::{
            plan_compare::{self, Bank, Pair},
            policy::{Policy, Rng, Sample},
            tensor,
        },
        pipeline::{
            event_portfolio::{Deployed, Portfolio, Runtime, SLOTS},
            executor::Observation,
            plan_events::Choice,
            plan_prototype::{Agent, Config},
        },
    };
    use std::{
        io::Write,
        path::{Path, PathBuf},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::Instant,
    };
    fn n(x: impl Into<f64>) -> Json {
        Json::Num(x.into())
    }
    fn read(p: &str) -> Result<Json, String> {
        json::parse(&std::fs::read_to_string(p).map_err(|e| e.to_string())?)
    }
    fn write(p: &Path, j: &Json) -> Result<(), String> {
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, j.dump()).map_err(|e| e.to_string())?;
        std::fs::rename(tmp, p).map_err(|e| e.to_string())
    }
    fn append(p: &Path, j: &Json) -> Result<(), String> {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .map_err(|e| e.to_string())?;
        writeln!(f, "{}", j.dump()).map_err(|e| e.to_string())
    }
    fn uint(j: &Json, k: &str) -> Result<u64, String> {
        j.get(k)
            .str()
            .parse()
            .map_err(|_| format!("invalid checkpoint {k}"))
    }
    struct Options {
        out: PathBuf,
        resume: Option<String>,
        foundation: Option<String>,
        config: String,
        iterations: usize,
        games: usize,
        workers: usize,
        points: usize,
        alternatives: usize,
        epochs: usize,
        batch: usize,
        eval_every: usize,
        eval_games: usize,
        confirm_games: usize,
        seed: u64,
        eval_seed: u64,
        device: i32,
        lr: f64,
    }
    impl Options {
        fn parse() -> Result<Self, String> {
            let mut o = Self {
                out: "runs/event_plan_trial".into(),
                resume: None,
                foundation: None,
                config: "native/configs/plan_prototype_v1.json".into(),
                iterations: 100,
                games: 16,
                workers: 7,
                points: 2,
                alternatives: 2,
                epochs: 8,
                batch: 64,
                eval_every: 5,
                eval_games: 8,
                confirm_games: 16,
                seed: 1200,
                eval_seed: 1100000000,
                device: 0,
                lr: 0.0003,
            };
            let args: Vec<_> = std::env::args().skip(1).collect();
            if args.iter().any(|s| s == "--help") {
                println!("event-train --out DIR [--resume latest.json | --base-checkpoint v3_best.json] --iterations 100 --games-per-update 16 --branch-points 2 --alternatives 2 --workers 7 --device cuda|cpu --epochs 8 --batch-size 64 --learning-rate 0.0003 --eval-every 5 --eval-games 8 --confirm-games 16 --seed 1200 --eval-seed 1100000000 --config native/configs/plan_prototype_v1.json\nNew event-plan-improvement-v4 checkpoints only. One scoped proposal is gated before deployment; learner never directly controls collection. iterations = additional comparison/NN updates; games-per-update = full base trajectories; each branch runs its remaining season to termination.");
                std::process::exit(0);
            }
            if args.len() % 2 != 0 {
                return Err("each flag needs a value".into());
            }
            for a in args.chunks(2) {
                let v = &a[1];
                let u = || v.parse::<usize>().map_err(|_| format!("invalid {}", a[0]));
                match a[0].as_str() {
                    "--out" => o.out = v.into(),
                    "--resume" => o.resume = Some(v.clone()),
                    "--base-checkpoint" => o.foundation = Some(v.clone()),
                    "--config" => o.config = v.clone(),
                    "--iterations" => o.iterations = u()?,
                    "--games-per-update" => o.games = u()?,
                    "--workers" => o.workers = u()?,
                    "--branch-points" => o.points = u()?,
                    "--alternatives" => o.alternatives = u()?,
                    "--epochs" => o.epochs = u()?,
                    "--batch-size" => o.batch = u()?,
                    "--eval-every" => o.eval_every = u()?,
                    "--eval-games" => o.eval_games = u()?,
                    "--confirm-games" => o.confirm_games = u()?,
                    "--seed" => o.seed = u()? as u64,
                    "--eval-seed" => o.eval_seed = u()? as u64,
                    "--learning-rate" => o.lr = v.parse().map_err(|_| "invalid learning rate")?,
                    "--device" => {
                        o.device = match v.as_str() {
                            "cuda" => 0,
                            "cpu" => -1,
                            _ => return Err("device: cpu or cuda".into()),
                        }
                    }
                    _ => {
                        return Err(format!(
                        "unknown option {}; old PPO flags are not valid for comparison training",
                        a[0]
                    ))
                    }
                }
            }
            if [
                o.iterations,
                o.games,
                o.workers,
                o.points,
                o.alternatives,
                o.epochs,
                o.batch,
                o.eval_every,
                o.eval_games,
            ]
            .contains(&0)
                || o.games % 2 != 0
                || o.eval_games % 2 != 0
                || o.confirm_games % 2 != 0
                || o.confirm_games < o.eval_games
                || o.eval_games > 1024
                || o.confirm_games > 1024
                || o.eval_seed > 2_000_000_000
                || o.points > 4
                || o.alternatives != 2
                || (o.foundation.is_some() && o.resume.is_some())
                || o.batch < 2
                || !o.lr.is_finite()
                || o.lr <= 0.
                || o.seed >= 100000000
                || o.eval_seed < 1000000000
            {
                return Err("positive sizes/lr, even games, branch-points 1..4, alternatives =2, batch >=2, disjoint training/eval seeds required".into());
            }
            Ok(o)
        }
    }
    const SCHEMA: &str = "event-plan-improvement-v4";
    const CONTRACT: &str = route_rl_native::pipeline::event_portfolio::CONTRACT;
    #[derive(Clone)]
    struct Job {
        seed: i64,
        seat: usize,
        opponent: usize,
        rng: u64,
        slots: Vec<usize>,
    }
    #[derive(Clone)]
    struct World {
        game: GameState,
        own: Deployed,
        rival: Deployed,
        fixed: Agent,
    }
    impl World {
        fn new(seed: i64, c: &Config) -> Self {
            Self {
                game: GameState::new(seed),
                own: Deployed::new(c.clone()),
                rival: Deployed::new(c.clone()),
                fixed: Agent::new(c.clone()),
            }
        }
        fn advance(
            &mut self,
            job: &Job,
            own: PlayerAction,
            opponents: &[Runtime],
        ) -> Result<(), String> {
            let mut actions: [PlayerAction; 2] = Default::default();
            actions[job.seat] = own;
            let obs = Observation::from_state(&self.game, 1 - job.seat);
            actions[1 - job.seat] = if job.opponent == 0 {
                self.fixed.action(&obs)
            } else {
                self.rival.action(&obs, &opponents[job.opponent - 1])?
            };
            engine::step(&mut self.game, &actions);
            Ok(())
        }
        fn finish(
            &mut self,
            job: &Job,
            accepted: &Runtime,
            opponents: &[Runtime],
        ) -> Result<usize, String> {
            let start = self.game.step;
            while self.game.step < 719 {
                let obs = Observation::from_state(&self.game, job.seat);
                let a = self.own.action(&obs, accepted)?;
                self.advance(job, a, opponents)?;
            }
            self.own
                .controller
                .observe(&Observation::from_state(&self.game, job.seat));
            Ok((self.game.step - start) as usize)
        }
        fn cash(&self, seat: usize) -> [f64; 2] {
            [self.game.farms[seat].money, self.game.farms[1 - seat].money]
        }
        fn result(&self, job: &Job) -> Json {
            let cash = self.cash(job.seat);
            let mut j = self.own.controller.report();
            for (k, v) in [
                ("seed", n(job.seed as f64)),
                ("seat", n(job.seat as f64)),
                ("opponent", n(job.opponent as f64)),
                ("cash", n(cash[0])),
                ("opponent_cash", n(cash[1])),
                ("margin", n(cash[0] - cash[1])),
                ("relative_margin", n(relative_margin(cash))),
                ("score", n(score(cash))),
            ] {
                j.set_path(k, v);
            }
            j
        }
    }
    fn score(c: [f64; 2]) -> f64 {
        if c[0] > c[1] {
            1.
        } else if c[0] < c[1] {
            0.
        } else {
            0.5
        }
    }
    fn relative_margin(c: [f64; 2]) -> f64 {
        (c[0] - c[1]) / (c[0].abs() + c[1].abs()).max(1.)
    }
    struct Fork {
        world: World,
        row: Sample,
        reference: Choice,
        alternatives: Vec<(usize, Choice)>,
        start_progress: usize,
        reference_index: usize,
        slot: usize,
        learner_changes: bool,
    }
    /// Retain a mandatory rotating coverage point, then alternate between correcting
    /// the current learner and covering resource-event kinds/ordinals. Only real
    /// observed opportunities are eligible; missing targets never waste a branch.
    fn branch_points(observed: &[(usize, bool)], desired: &[usize], rotation: usize) -> Vec<usize> {
        if observed.is_empty() || desired.is_empty() {
            return vec![];
        }
        let mut order: Vec<_> = (0..observed.len()).collect();
        order.sort_by_key(|i| observed[*i].0);
        let coverage = order
            .iter()
            .copied()
            .find(|i| observed[*i].0 == desired[0])
            .unwrap_or(order[rotation % order.len()]);
        let mut chosen = vec![coverage];
        for round in 0..desired.len().saturating_sub(1) {
            let turn = rotation + round;
            let prefer_disagreement = turn % 2 == 0;
            let resource_kind = (turn / 2) % 3;
            let resource_ordinal = (turn / 6) % 4;
            let next = order
                .iter()
                .copied()
                .filter(|i| !chosen.contains(i))
                .min_by_key(|i| {
                    let (slot, disagreement) = observed[*i];
                    let resource = slot >= 4;
                    if prefer_disagreement {
                        (
                            usize::from(!disagreement),
                            (slot + SLOTS - turn % SLOTS) % SLOTS,
                            0,
                        )
                    } else {
                        (
                            usize::from(!resource),
                            if resource {
                                (slot / 4 - 1 + 3 - resource_kind) % 3
                            } else {
                                3
                            },
                            (slot % 4 + 4 - resource_ordinal) % 4,
                        )
                    }
                });
            match next {
                Some(i) => chosen.push(i),
                None => break,
            }
        }
        chosen
    }
    struct Game {
        report: Json,
        pairs: Vec<Pair>,
        comparisons: Vec<Json>,
        steps: usize,
        branches: usize,
    }
    fn play(
        job: &Job,
        accepted: &Runtime,
        learner: Option<&Policy>,
        opponents: &[Runtime],
        config: &Config,
        iteration: u64,
        revision: u64,
        alternatives: usize,
    ) -> Result<Game, String> {
        let mut world = World::new(job.seed, config);
        let mut rng = Rng(job.rng);
        let mut forks = vec![];
        while world.game.step < 719 {
            let obs = Observation::from_state(&world.game, job.seat);
            // Shared with event-agent: observe and claim the opportunity exactly once.
            let action = if let Some(mut d) = world.own.prepare(&obs, accepted)? {
                if let Some(slot) = d.slot.filter(|_| !job.slots.is_empty()) {
                    if alternatives > 0 && d.choices.len() > 1 {
                        let mut pool: Vec<_> =
                            (0..d.choices.len()).filter(|i| *i != d.selected).collect();
                        rng.shuffle(&mut pool);
                        // Always measure the CURRENT neural choice before broad exploration.
                        let mut learner_changes = false;
                        if let Some(p) = learner {
                            let proposed = p.infer(&[d.row.clone()], true, &mut Rng(0))?[0].action;
                            learner_changes = proposed != d.selected;
                            if let Some(at) = pool.iter().position(|i| *i == proposed) {
                                pool.swap(0, at);
                            }
                        }
                        if pool.len() > 2 {
                            let first = &d.choices[pool[0]].next;
                            if let Some(at) =
                                (1..pool.len()).find(|at| d.choices[pool[*at]].next != *first)
                            {
                                pool.swap(1, at);
                            }
                        }
                        pool.truncate(alternatives);
                        forks.push(Fork {
                            world: world.clone(),
                            row: d.row.clone(),
                            reference: d.choices[d.selected].clone(),
                            alternatives: pool
                                .iter()
                                .map(|i| (*i, d.choices[*i].clone()))
                                .collect(),
                            start_progress: world.own.controller.progress.len(),
                            reference_index: d.selected,
                            slot,
                            learner_changes,
                        });
                    }
                }
                world
                    .own
                    .execute_choice(&obs, d.choices.swap_remove(d.selected), accepted)?
            } else {
                world.own.continue_action(&obs, accepted)?
            };
            world.advance(job, action, opponents)?;
        }
        world
            .own
            .controller
            .observe(&Observation::from_state(&world.game, job.seat));
        let reference_cash = world.cash(job.seat);
        let mut result = Game {
            report: world.result(job),
            pairs: vec![],
            comparisons: vec![],
            steps: 719,
            branches: 0,
        };
        let observed: Vec<_> = forks.iter().map(|f| (f.slot, f.learner_changes)).collect();
        // Both seats share rotation; the state still determines which events exist.
        let rotation = job.seed as usize + iteration as usize;
        let chosen = branch_points(&observed, &job.slots, rotation);
        let mut forks: Vec<_> = forks.into_iter().map(Some).collect();
        for index in chosen {
            let fork = forks[index].take().expect("distinct branch points");
            for (alternative_index, alternative) in fork.alternatives {
                let mut branch = fork.world.clone();
                let obs = Observation::from_state(&branch.game, job.seat);
                let new_start = branch.own.controller.progress.len();
                let new_end = new_start
                    + if !alternative.keep && alternative.next.is_some() {
                        alternative.sites.len()
                    } else {
                        0
                    };
                let a = branch
                    .own
                    .execute_choice(&obs, alternative.clone(), accepted)?;
                branch.advance(job, a, opponents)?;
                result.steps += 1 + branch.finish(job, accepted, opponents)?;
                result.branches += 1;
                let cash = branch.cash(job.seat);
                let pref = plan_compare::preference(reference_cash, cash);
                let progress = |w: &World, start: usize, end: usize| {
                    Json::Arr(
                        w.own.controller.progress[start..end.min(w.own.controller.progress.len())]
                            .iter()
                            .map(|p| p.json())
                            .collect(),
                    )
                };
                let evidence = Json::Obj(vec![
                    (
                        "event".into(),
                        fork.world
                            .own
                            .last_event
                            .as_ref()
                            .map(|e| e.json())
                            .unwrap_or(Json::Null),
                    ),
                    ("iteration".into(), n(iteration as f64)),
                    ("incumbent_revision".into(), Json::Str(revision.to_string())),
                    ("slot_id".into(), n(fork.slot as f64)),
                    ("learner_changes".into(), Json::Bool(fork.learner_changes)),
                    ("seed".into(), n(job.seed as f64)),
                    ("seat".into(), n(job.seat as f64)),
                    ("opponent".into(), n(job.opponent as f64)),
                    ("step".into(), n(obs.step as f64)),
                    ("reference_index".into(), n(fork.reference_index as f64)),
                    ("alternative_index".into(), n(alternative_index as f64)),
                    ("full_row".into(), fork.row.json()),
                    ("reference_plan".into(), fork.reference.json()),
                    ("alternative_plan".into(), alternative.json()),
                    (
                        "reference_cash".into(),
                        Json::Arr(reference_cash.map(n).to_vec()),
                    ),
                    ("alternative_cash".into(), Json::Arr(cash.map(n).to_vec())),
                    (
                        "reference_execution".into(),
                        progress(
                            &world,
                            fork.start_progress,
                            fork.start_progress
                                + if !fork.reference.keep && fork.reference.next.is_some() {
                                    fork.reference.sites.len()
                                } else {
                                    0
                                },
                        ),
                    ),
                    (
                        "alternative_execution".into(),
                        progress(&branch, new_start, new_end),
                    ),
                    ("decisive".into(), Json::Bool(pref.is_some())),
                    (
                        "alternative_better".into(),
                        Json::Bool(pref.as_ref().is_some_and(|(t, _)| t[1] > t[0])),
                    ),
                ]);
                {
                    let (target, gain) = pref.unwrap_or_else(|| (vec![0.5, 0.5], 0.));
                    let winner = if target[1] > target[0] {
                        &alternative
                    } else {
                        &fork.reference
                    };
                    let kind = winner.next.as_ref().map(|p| p.name()).unwrap_or("");
                    let kinds = [
                        "WHEAT",
                        "CARROT",
                        "TOMATO",
                        "STRAWBERRY",
                        "MELON",
                        "GOOSE",
                        "COW",
                        "SHEEP",
                        "",
                    ];
                    let bucket = job.opponent * 27
                        + (obs.day() as usize / 10).min(2) * 9
                        + kinds.iter().position(|x| *x == kind).unwrap();
                    result.pairs.push(Pair {
                        row: Sample {
                            features: vec![
                                fork.reference.features.clone(),
                                alternative.features.clone(),
                            ],
                            ..fork.row.clone()
                        },
                        target,
                        gain,
                        iteration,
                        seed: job.seed,
                        seat: job.seat,
                        opponent: job.opponent,
                        bucket,
                        evidence: evidence.clone(),
                    });
                }
                result.comparisons.push(evidence);
            }
        }
        Ok(result)
    }
    fn collect(
        jobs: Vec<Job>,
        accepted: &Portfolio,
        learner: Option<&Json>,
        opponents: &[Portfolio],
        config: &Config,
        workers: usize,
        iteration: u64,
        alternatives: usize,
    ) -> Result<Vec<Game>, String> {
        let jobs = Arc::new(jobs);
        let cursor = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];
        for _ in 0..workers
            .min(jobs.len())
            .min(route_rl_native::resources::available_workers())
        {
            let js = jobs.clone();
            let ix = cursor.clone();
            let portfolio = accepted.clone();
            let lw = learner.cloned();
            let os = opponents.to_vec();
            let c = config.clone();
            handles.push(std::thread::spawn(
                move || -> Result<Vec<(usize, Game)>, String> {
                    tensor::worker_threads();
                    let runtime = Runtime::load(&portfolio, -1)?;
                    let mut lp = None;
                    if let Some(w) = lw {
                        let mut p = Policy::plans(-1, 0, 0.0003)?;
                        p.load_weights(&w)?;
                        lp = Some(p);
                    }
                    let models = os
                        .iter()
                        .map(|s| Runtime::load(s, -1))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut games = vec![];
                    loop {
                        let i = ix.fetch_add(1, Ordering::Relaxed);
                        if i >= js.len() {
                            break;
                        }
                        games.push((
                            i,
                            play(
                                &js[i],
                                &runtime,
                                lp.as_ref(),
                                &models,
                                &c,
                                iteration,
                                portfolio.revision,
                                alternatives,
                            )?,
                        ));
                    }
                    Ok(games)
                },
            ));
        }
        let mut rows = vec![];
        let mut error = None;
        for h in handles {
            match h.join() {
                Ok(Ok(v)) => rows.extend(v),
                Ok(Err(e)) => error = Some(e),
                Err(_) => error = Some("scoped rollout worker panicked".into()),
            }
        }
        if let Some(e) = error {
            return Err(e);
        }
        rows.sort_by_key(|(i, _)| *i);
        Ok(rows.into_iter().map(|(_, g)| g).collect())
    }
    fn roster(accepted: &Portfolio, previous: &Portfolio) -> Vec<usize> {
        let mut slots = vec![0];
        if accepted.slots.iter().any(Option::is_some)
            || accepted.foundation.slots.iter().any(Option::is_some)
        {
            slots.push(1);
        }
        if previous.slots.iter().any(Option::is_some) && previous.revision != accepted.revision {
            slots.push(2);
        }
        slots
    }
    fn jobs(
        base: u64,
        count: usize,
        roster: &[usize],
        points: usize,
        iteration: u64,
        rng: &mut Rng,
    ) -> Vec<Job> {
        (0..count)
            .map(|i| Job {
                seed: (base + i as u64 / 2) as i64,
                seat: i % 2,
                opponent: roster[(i / 2) % roster.len()],
                rng: rng.next(),
                slots: (0..points)
                    .map(|k| ((i / 2) + k + iteration.saturating_sub(1) as usize) % SLOTS)
                    .collect(),
            })
            .collect()
    }
    fn eval_jobs(base: u64, count: usize, roster: &[usize]) -> Vec<Job> {
        (0..count)
            .flat_map(|i| {
                roster.iter().map(move |opponent| Job {
                    seed: (base + i as u64 / 2) as i64,
                    seat: i % 2,
                    opponent: *opponent,
                    rng: 0,
                    slots: vec![],
                })
            })
            .collect()
    }
    fn summary(gs: &[Game]) -> Json {
        let mean = |k: &str| {
            gs.iter().map(|g| g.report.get(k).f64()).sum::<f64>() / gs.len().max(1) as f64
        };
        Json::Obj(vec![
            ("games".into(), n(gs.len() as f64)),
            ("mean_cash".into(), n(mean("cash"))),
            ("mean_margin".into(), n(mean("margin"))),
            ("score_rate".into(), n(mean("score"))),
            ("mean_links_requested".into(), n(mean("links_requested"))),
            (
                "mean_links_yielded".into(),
                n(mean("links_successor_yielded")),
            ),
            (
                "by_opponent".into(),
                Json::Obj(
                    (0..3)
                        .filter_map(|slot| {
                            let rows: Vec<_> = gs
                                .iter()
                                .filter(|g| g.report.get("opponent").i64() == slot)
                                .collect();
                            if rows.is_empty() {
                                return None;
                            }
                            let avg = |k: &str| {
                                rows.iter().map(|g| g.report.get(k).f64()).sum::<f64>()
                                    / rows.len() as f64
                            };
                            Some((
                                slot.to_string(),
                                Json::Obj(vec![
                                    ("games".into(), n(rows.len() as f64)),
                                    ("mean_cash".into(), n(avg("cash"))),
                                    ("mean_margin".into(), n(avg("margin"))),
                                    ("score_rate".into(), n(avg("score"))),
                                ]),
                            ))
                        })
                        .collect(),
                ),
            ),
            (
                "games_detail".into(),
                Json::Arr(gs.iter().map(|g| g.report.clone()).collect()),
            ),
        ])
    }
    /// Matched seeds/seats/opponents. Never gate a candidate against unrelated runs.
    fn gate(candidate: &[Game], incumbent: &[Game]) -> Result<Json, String> {
        if candidate.len() != incumbent.len() || candidate.is_empty() {
            return Err("gate needs matching nonempty paired games".into());
        }
        let mut score_gain = 0.;
        let mut margin_gain = 0.;
        let mut by_opponent = std::collections::BTreeMap::<i64, (f64, f64, usize)>::new();
        let mut by_seed = std::collections::BTreeMap::<i64, (f64, usize)>::new();
        let mut inactive_regression = false;
        for (c, b) in candidate.iter().zip(incumbent) {
            for key in ["seed", "seat", "opponent"] {
                if c.report.get(key) != b.report.get(key) {
                    return Err("unpaired evaluation records".into());
                }
            }
            let ds = c.report.get("score").f64() - b.report.get("score").f64();
            let dm = c.report.get("relative_margin").f64() - b.report.get("relative_margin").f64();
            score_gain += ds;
            margin_gain += dm;
            let e = by_opponent
                .entry(c.report.get("opponent").i64())
                .or_default();
            e.0 += ds;
            e.1 += dm;
            e.2 += 1;
            let e = by_seed.entry(c.report.get("seed").i64()).or_default();
            e.0 += dm;
            e.1 += 1;
            inactive_regression |= b.report.get("harvested_units").f64() > 0.
                && c.report.get("harvested_units").f64() == 0.;
        }
        score_gain /= candidate.len() as f64;
        margin_gain /= candidate.len() as f64;
        let positive = by_seed
            .values()
            .filter(|(d, n)| *d / *n as f64 > 0.0001)
            .count();
        let negative = by_seed
            .values()
            .filter(|(d, n)| *d / (*n as f64) < -0.0001)
            .count();
        let no_matchup_regression = by_opponent
            .values()
            .all(|(s, m, n)| *s >= -1e-9 && *m / *n as f64 >= -0.005);
        let better = score_gain > 1e-9 || (score_gain >= -1e-9 && margin_gain > 0.002);
        let qualifies =
            better && positive > negative && no_matchup_regression && !inactive_regression;
        Ok(Json::Obj(vec![
            ("qualifies".into(), Json::Bool(qualifies)),
            ("mean_score_gain".into(), n(score_gain)),
            ("mean_relative_margin_gain".into(), n(margin_gain)),
            ("positive_seeds".into(), n(positive as f64)),
            ("negative_seeds".into(), n(negative as f64)),
            (
                "no_matchup_regression".into(),
                Json::Bool(no_matchup_regression),
            ),
            (
                "inactive_regression".into(),
                Json::Bool(inactive_regression),
            ),
        ]))
    }
    struct State {
        iteration: u64,
        rng: Rng,
        config: Config,
        accepted: Portfolio,
        previous: Portfolio,
        bank: Bank,
        seed: u64,
        next_seed: u64,
        eval_seed: u64,
        steps: u64,
        base_games: u64,
        branches: u64,
        attempts: [u64; SLOTS],
        monitor: Json,
    }
    impl State {
        fn checkpoint(&self, p: &Policy) -> Result<Json, String> {
            Ok(Json::Obj(vec![
                ("schema".into(), Json::Str(SCHEMA.into())),
                ("policy_contract".into(), Json::Str(CONTRACT.into())),
                ("iteration".into(), Json::Str(self.iteration.to_string())),
                ("model".into(), p.checkpoint(self.iteration, &self.rng)?),
                ("config".into(), self.config.json()),
                ("deployment".into(), self.accepted.json()),
                ("previous_accepted".into(), self.previous.json()),
                ("comparison_bank".into(), self.bank.json()),
                ("seed".into(), Json::Str(self.seed.to_string())),
                ("next_seed".into(), Json::Str(self.next_seed.to_string())),
                ("eval_seed".into(), Json::Str(self.eval_seed.to_string())),
                ("training_steps".into(), Json::Str(self.steps.to_string())),
                ("base_games".into(), Json::Str(self.base_games.to_string())),
                (
                    "branch_rollouts".into(),
                    Json::Str(self.branches.to_string()),
                ),
                (
                    "gate_attempts".into(),
                    Json::Arr(
                        self.attempts
                            .iter()
                            .map(|x| Json::Str(x.to_string()))
                            .collect(),
                    ),
                ),
                ("deployed_vs_rule".into(), self.monitor.clone()),
            ]))
        }
        fn restore(j: &Json, p: &mut Policy) -> Result<Self, String> {
            if j.get("schema").str() != SCHEMA || j.get("policy_contract").str() != CONTRACT {
                return Err("requires event-plan-improvement-v4 full checkpoint; rejected v2 learner cannot become an accepted strategy".into());
            }
            let (iteration, rng) = p.restore(j.get("model"))?;
            if !p.plan_residual || iteration != uint(j, "iteration")? {
                return Err("model/outer iteration mismatch".into());
            }
            let accepted = Portfolio::parse(j.get("deployment"))?;
            let previous = Portfolio::parse(j.get("previous_accepted"))?;
            let bank = Bank::parse(j.get("comparison_bank"))?;
            if previous.revision > accepted.revision {
                return Err("previous strategy revision exceeds accepted revision".into());
            }
            for row in bank.elite.iter().chain(&bank.recent) {
                if row.incumbent_revision()? != accepted.revision {
                    return Err("stale comparison labels in active checkpoint".into());
                }
            }
            let attempts: Vec<u64> = j
                .get("gate_attempts")
                .arr()
                .iter()
                .map(|x| x.str().parse().map_err(|_| "invalid gate attempt count"))
                .collect::<Result<_, _>>()?;
            Ok(Self {
                iteration,
                rng,
                config: Config::parse(j.get("config"))?,
                accepted,
                previous,
                bank,
                seed: uint(j, "seed")?,
                next_seed: uint(j, "next_seed")?,
                eval_seed: uint(j, "eval_seed")?,
                steps: uint(j, "training_steps")?,
                base_games: uint(j, "base_games")?,
                branches: uint(j, "branch_rollouts")?,
                attempts: attempts
                    .try_into()
                    .map_err(|_| "invalid scope counter length")?,
                monitor: j.get("deployed_vs_rule").clone(),
            })
        }
        fn promote(&mut self, candidate: Portfolio) -> Result<(), String> {
            if candidate.revision != self.accepted.revision + 1
                || candidate.foundation != self.accepted.foundation
                || candidate.slots.len() != SLOTS
                || candidate
                    .slots
                    .iter()
                    .zip(&self.accepted.slots)
                    .filter(|(a, b)| a != b)
                    .count()
                    != 1
            {
                return Err("candidate must replace exactly one scope at the next revision".into());
            }
            self.previous = self.accepted.clone();
            self.accepted = candidate;
            // Outcome labels depend on the continuation policy. Archive is in comparisons.jsonl;
            // frozen accepted slot models retain actual behavior across revisions.
            self.bank = Bank::default();
            self.attempts = [0; SLOTS];
            Ok(())
        }
    }
    fn monitor(o: &Options, s: &mut State) -> Result<(), String> {
        let gs = collect(
            eval_jobs(s.eval_seed, o.eval_games, &[0]),
            &s.accepted,
            None,
            &[],
            &s.config,
            o.workers,
            s.iteration,
            0,
        )?;
        s.monitor = summary(&gs);
        s.monitor
            .set_path("accepted_revision", n(s.accepted.revision as f64));
        Ok(())
    }
    fn evaluate(o: &Options, s: &mut State, p: &Policy) -> Result<(), String> {
        let rows: Vec<_> = s.bank.elite.iter().chain(&s.bank.recent).cloned().collect();
        let support = plan_compare::support_report(p, &rows, s.accepted.revision)?;
        // Data selects which ONE scope deserves a full candidate gate. It never promotes it.
        let slot = support
            .get("slots")
            .arr()
            .iter()
            .filter(|r| {
                r.get("supported_seeds").i64() >= 2 && r.get("total_observed_delta").f64() > 0.
            })
            .max_by(|a, b| {
                let rank = |r: &Json| {
                    r.get("supported_changes").f64()
                        / (1. + s.attempts[r.get("slot_id").i64() as usize] as f64)
                };
                rank(a)
                    .total_cmp(&rank(b))
                    .then_with(|| b.get("slot_id").i64().cmp(&a.get("slot_id").i64()))
            })
            .map(|r| r.get("slot_id").i64() as usize);
        let before = s.accepted.revision;
        let mut report = Json::Obj(vec![
            ("iteration".into(), n(s.iteration as f64)),
            ("accepted_revision_before".into(), n(before as f64)),
            ("support".into(), support),
            ("promoted".into(), Json::Bool(false)),
            ("deployed_vs_rule".into(), s.monitor.clone()),
        ]);
        if let Some(slot) = slot {
            s.attempts[slot] += 1;
            let proposal = s.accepted.propose(slot, s.iteration, p.weights_json()?)?;
            let opponents = vec![s.accepted.clone(), s.previous.clone()];
            let roster = roster(&s.accepted, &s.previous);
            let base = s
                .eval_seed
                .checked_add(
                    1_000_000 + s.iteration.checked_mul(10000).ok_or("eval seed overflow")?,
                )
                .ok_or("eval seed overflow")?;
            let js = eval_jobs(base, o.eval_games, &roster);
            let candidate = collect(
                js.clone(),
                &proposal,
                None,
                &opponents,
                &s.config,
                o.workers,
                s.iteration,
                0,
            )?;
            let incumbent = collect(
                js,
                &s.accepted,
                None,
                &opponents,
                &s.config,
                o.workers,
                s.iteration,
                0,
            )?;
            let screen = gate(&candidate, &incumbent)?;
            report.set_path("candidate_slot", n(slot as f64));
            report.set_path("screen_seeds_start", n(base as f64));
            report.set_path("candidate", summary(&candidate));
            report.set_path("incumbent", summary(&incumbent));
            report.set_path("screen", screen.clone());
            if matches!(screen.get("qualifies"), Json::Bool(true)) {
                let js = eval_jobs(base + 1000, o.confirm_games, &roster);
                let candidate = collect(
                    js.clone(),
                    &proposal,
                    None,
                    &opponents,
                    &s.config,
                    o.workers,
                    s.iteration,
                    0,
                )?;
                let incumbent = collect(
                    js,
                    &s.accepted,
                    None,
                    &opponents,
                    &s.config,
                    o.workers,
                    s.iteration,
                    0,
                )?;
                let confirmation = gate(&candidate, &incumbent)?;
                report.set_path("confirmation_seeds_start", n((base + 1000) as f64));
                report.set_path("confirmation_candidate", summary(&candidate));
                report.set_path("confirmation_incumbent", summary(&incumbent));
                report.set_path("confirmation", confirmation.clone());
                if matches!(confirmation.get("qualifies"), Json::Bool(true)) {
                    s.promote(proposal)?;
                    monitor(o, s)?;
                    report.set_path("promoted", Json::Bool(true));
                    report.set_path("deployed_vs_rule", s.monitor.clone());
                    write(&o.out.join("best.json"), &s.checkpoint(p)?)?;
                }
            }
        } else {
            report.set_path(
                "reason",
                Json::Str("no scope has positive aggregate tested gain and support from two distinct training seeds".into()),
            );
        }
        report.set_path("accepted_revision_after", n(s.accepted.revision as f64));
        append(&o.out.join("evaluations.jsonl"), &report)?;
        println!("evaluation iteration={} proposed_slot={:?} promoted={} accepted_revision={} deployed_cash={}",s.iteration,slot,s.accepted.revision!=before,s.accepted.revision,s.monitor.get("mean_cash").f64());
        Ok(())
    }
    pub fn run() -> Result<(), String> {
        let o = Options::parse()?;
        tensor::threads(1);
        tensor::worker_threads();
        std::fs::create_dir_all(&o.out).map_err(|e| e.to_string())?;
        if o.out.join("latest.json").exists() || o.out.join("manifest.json").exists() {
            return Err("use a new output directory".into());
        }
        let mut p = Policy::plans(o.device, o.seed, o.lr)?;
        let mut foundation = Portfolio::empty();
        let mut config = Config::parse(&read(&o.config)?)?;
        if let Some(path) = &o.foundation {
            let j = read(path)?;
            if j.get("schema").str() != "plan-improvement-v3"
                || j.get("policy_contract").str() != "plan-chain-320x32-scoped-v3"
            {
                return Err(
                    "base-checkpoint requires a v3 accepted portfolio, not its raw learner".into(),
                );
            }
            foundation = Portfolio::from_foundation(
                route_rl_native::pipeline::plan_portfolio::Portfolio::parse(j.get("deployment"))?,
            );
            config = Config::parse(j.get("config"))?;
        }
        let mut s = if let Some(path) = &o.resume {
            State::restore(&read(path)?, &mut p)?
        } else {
            State {
                iteration: 0,
                rng: Rng(o.seed ^ 0x1acf789),
                config,
                accepted: foundation.clone(),
                previous: foundation.clone(),
                bank: Bank::default(),
                seed: o.seed,
                next_seed: o.seed,
                eval_seed: o.eval_seed,
                steps: 0,
                base_games: 0,
                branches: 0,
                attempts: [0; SLOTS],
                monitor: Json::Null,
            }
        };
        write(
            &o.out.join("manifest.json"),
            &Json::Obj(vec![
                ("schema".into(), Json::Str(SCHEMA.into())),
                ("policy_contract".into(), Json::Str(CONTRACT.into())),
                ("iterations_additional".into(), n(o.iterations as f64)),
                ("base_games_per_update".into(), n(o.games as f64)),
                ("branch_points".into(), n(o.points as f64)),
                ("alternatives".into(), n(o.alternatives as f64)),
                ("branch_sampling".into(), Json::Str("one mandatory rotating event scope; remaining points alternate learner disagreement and rotating resource kinds/ordinals".into())),
                ("epochs".into(), n(o.epochs as f64)),
                ("batch_size".into(), n(o.batch as f64)),
                ("eval_every".into(), n(o.eval_every as f64)),
                ("eval_games".into(), n(o.eval_games as f64)),
                ("confirm_games".into(), n(o.confirm_games as f64)),
                (
                    "device".into(),
                    Json::Str(if o.device < 0 { "cpu" } else { "cuda" }.into()),
                ),
                (
                    "resume".into(),
                    o.resume.clone().map(Json::Str).unwrap_or(Json::Null),
                ),
                ("base_checkpoint".into(),o.foundation.clone().map(Json::Str).unwrap_or(Json::Null)),
                ("learning_rate".into(), n(p.lr)),
                (
                    "workers".into(),
                    n(o.workers
                        .min(route_rl_native::resources::available_workers())
                        as f64),
                ),
                (
                    "event_scopes".into(),
                    Json::Str("harvest/material/funding/review: first four events each, shared consumption in branch and deployment".into()),
                ),
            ]),
        )?;
        if o.resume.is_none() {
            monitor(&o, &mut s)?;
        }
        write(&o.out.join("initial.json"), &s.checkpoint(&p)?)?;
        write(&o.out.join("best.json"), &s.checkpoint(&p)?)?;
        write(&o.out.join("latest.json"), &s.checkpoint(&p)?)?;
        append(
            &o.out.join("evaluations.jsonl"),
            &Json::Obj(vec![
                ("iteration".into(), n(s.iteration as f64)),
                ("initial".into(), Json::Bool(true)),
                (
                    "accepted_revision_after".into(),
                    n(s.accepted.revision as f64),
                ),
                ("deployed_vs_rule".into(), s.monitor.clone()),
            ]),
        )?;
        println!("event-plan-improvement-v4: accepted_revision={} baseline_collection=true one_scope_per_candidate_gate=true workers={}",s.accepted.revision,o.workers.min(route_rl_native::resources::available_workers()));
        for _ in 0..o.iterations {
            let started = Instant::now();
            let iteration = s.iteration + 1;
            if s.next_seed + o.games as u64 / 2 >= 1_000_000_000 {
                return Err("training seed range exhausted".into());
            }
            let roster = roster(&s.accepted, &s.previous);
            let js = jobs(
                s.next_seed,
                o.games,
                &roster,
                o.points,
                iteration,
                &mut s.rng,
            );
            s.next_seed += o.games as u64 / 2;
            let gs = collect(
                js,
                &s.accepted,
                Some(&p.weights_json()?),
                &[s.accepted.clone(), s.previous.clone()],
                &s.config,
                o.workers,
                iteration,
                o.alternatives,
            )?;
            let collect_seconds = started.elapsed().as_secs_f64();
            let pairs: Vec<_> = gs.iter().flat_map(|g| g.pairs.iter().cloned()).collect();
            for g in &gs {
                for c in &g.comparisons {
                    append(&o.out.join("comparisons.jsonl"), c)?;
                }
            }
            let before = p.weights_json()?;
            let t = Instant::now();
            let update = plan_compare::update_improvement(
                &mut p,
                &pairs,
                &s.bank,
                s.accepted.revision,
                o.epochs,
                o.batch,
                &mut s.rng,
            )?;
            s.bank.admit(&pairs);
            s.iteration = iteration;
            let steps = gs.iter().map(|g| g.steps as u64).sum::<u64>();
            let branches = gs.iter().map(|g| g.branches as u64).sum::<u64>();
            s.steps += steps;
            s.branches += branches;
            s.base_games += gs.len() as u64;
            let mut coverage = [0; SLOTS];
            for g in &gs {
                for c in &g.comparisons {
                    coverage[c.get("slot_id").i64() as usize] += 1;
                }
            }
            let mut metric = summary(&gs);
            for (k, v) in [
                ("iteration", n(iteration as f64)),
                ("accepted_revision", n(s.accepted.revision as f64)),
                (
                    "accepted_scopes",
                    n(s.accepted.slots.iter().filter(|x| x.is_some()).count() as f64),
                ),
                ("rollout_policy", Json::Str("accepted_portfolio".into())),
                ("comparison_pairs", n(pairs.len() as f64)),
                (
                    "decisive_pairs",
                    n(gs.iter()
                        .flat_map(|g| &g.comparisons)
                        .filter(|j| matches!(j.get("decisive"), Json::Bool(true)))
                        .count() as f64),
                ),
                (
                    "better_alternatives",
                    n(gs.iter()
                        .flat_map(|g| &g.comparisons)
                        .filter(|j| matches!(j.get("alternative_better"), Json::Bool(true)))
                        .count() as f64),
                ),
                (
                    "slot_coverage",
                    Json::Arr(coverage.iter().map(|n| Json::Num(*n as f64)).collect()),
                ),
                ("branch_rollouts", n(branches as f64)),
                ("simulation_steps", n(steps as f64)),
                ("equivalent_full_games", n(steps as f64 / 719.)),
                ("total_base_games", n(s.base_games as f64)),
                ("total_branch_rollouts", n(s.branches as f64)),
                ("collect_seconds", n(collect_seconds)),
                ("update_seconds", n(t.elapsed().as_secs_f64())),
                ("weights_changed", Json::Bool(before != p.weights_json()?)),
                ("update", update),
                ("elite_pairs", n(s.bank.elite.len() as f64)),
                ("recent_pairs", n(s.bank.recent.len() as f64)),
            ] {
                metric.set_path(k, v);
            }
            append(&o.out.join("metrics.jsonl"), &metric)?;
            println!("{}", metric.dump());
            if iteration % o.eval_every as u64 == 0 {
                evaluate(&o, &mut s, &p)?;
                write(
                    &o.out.join(format!("checkpoint_{iteration:06}.json")),
                    &s.checkpoint(&p)?,
                )?;
            }
            write(&o.out.join("latest.json"), &s.checkpoint(&p)?)?;
        }
        Ok(())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        fn fake(seed: i64, opponent: i64, score: f64, margin: f64) -> Game {
            Game {
                report: Json::Obj(vec![
                    ("seed".into(), n(seed as f64)),
                    ("seat".into(), n(0.)),
                    ("opponent".into(), n(opponent as f64)),
                    ("score".into(), n(score)),
                    ("relative_margin".into(), n(margin)),
                    ("harvested_units".into(), n(100.)),
                ]),
                pairs: vec![],
                comparisons: vec![],
                steps: 0,
                branches: 0,
            }
        }
        #[test]
        fn gate_rejects_unchanged_and_regressive_candidates() {
            let b = vec![fake(1, 0, 0.5, 0.), fake(2, 0, 0.5, 0.)];
            let same = vec![fake(1, 0, 0.5, 0.), fake(2, 0, 0.5, 0.)];
            assert_eq!(
                gate(&same, &b).unwrap().get("qualifies"),
                &Json::Bool(false)
            );
            let good = vec![fake(1, 0, 0.5, 0.01), fake(2, 0, 0.5, 0.02)];
            assert_eq!(gate(&good, &b).unwrap().get("qualifies"), &Json::Bool(true));
            let bad = vec![fake(1, 0, 0., 0.02), fake(2, 0, 0., 0.02)];
            assert_eq!(gate(&bad, &b).unwrap().get("qualifies"), &Json::Bool(false));
            let bad_key = vec![fake(3, 0, 1., 0.2), fake(2, 0, 1., 0.2)];
            assert!(gate(&bad_key, &b).is_err());
        }
        #[test]
        fn scheduled_collection_covers_all_event_scopes_with_paired_seats() {
            let js = jobs(100, 32, &[0, 1, 2], 2, 1, &mut Rng(1));
            let mut seen = std::collections::BTreeSet::new();
            for pair in js.chunks(2) {
                assert_eq!(pair[0].slots, pair[1].slots);
                assert_eq!(pair[0].seed, pair[1].seed);
                assert_eq!(pair[0].opponent, pair[1].opponent);
                seen.insert(pair[0].slots[0]);
            }
            assert_eq!(seen.len(), SLOTS);
        }
        #[test]
        fn branch_sampling_reserves_coverage_even_with_many_resource_events() {
            let observed: Vec<_> = (0..SLOTS).map(|slot| (slot, false)).collect();
            for rotation in 0..24 {
                let chosen = branch_points(&observed, &[3, 0], rotation);
                assert_eq!(observed[chosen[0]].0, 3);
                assert_eq!(chosen.len(), 2);
                assert_ne!(chosen[0], chosen[1]);
            }
        }
        #[test]
        fn branch_sampling_rotates_resource_kinds_and_ordinals() {
            let observed: Vec<_> = (0..SLOTS).map(|slot| (slot, false)).collect();
            let mut covered = std::collections::BTreeSet::new();
            for rotation in (1..24).step_by(2) {
                let chosen = branch_points(&observed, &[0, 1], rotation);
                assert_eq!(observed[chosen[0]].0, 0);
                covered.insert(observed[chosen[1]].0);
            }
            assert_eq!(covered, (4..SLOTS).collect());
        }
        #[test]
        fn branch_sampling_targets_disagreement_and_fills_absent_scopes() {
            let observed = vec![(0, false), (4, false), (8, false), (15, true)];
            let chosen = branch_points(&observed, &[0, 1], 0);
            assert_eq!(chosen, vec![0, 3]);
            let sparse = vec![(0, false), (2, true), (3, false)];
            let mut covered = std::collections::BTreeSet::new();
            for rotation in 0..sparse.len() {
                let chosen = branch_points(&sparse, &[12, 13, 14, 15], rotation);
                covered.insert(sparse[chosen[0]].0);
                assert_eq!(chosen.len(), sparse.len());
                assert_eq!(
                    chosen
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len(),
                    sparse.len()
                );
            }
            assert_eq!(covered, [0, 2, 3].into_iter().collect());
            assert!(branch_points(&[], &[0], 0).is_empty());
            assert!(branch_points(&sparse, &[], 0).is_empty());
        }
        fn state(p: &Policy) -> State {
            let _ = p;
            State {
                iteration: 7,
                rng: Rng(2),
                config: Config::default(),
                accepted: Portfolio::empty(),
                previous: Portfolio::empty(),
                bank: Bank::default(),
                seed: 1,
                next_seed: 99,
                eval_seed: 1100000000,
                steps: 7190,
                base_games: 4,
                branches: 9,
                attempts: [0; SLOTS],
                monitor: Json::Null,
            }
        }
        #[test]
        fn checkpoint_restores_learner_and_deployed_composition_separately() {
            tensor::threads(1);
            let p = Policy::plans(-1, 2, 0.001).unwrap();
            let mut s = state(&p);
            s.promote(s.accepted.propose(2, 3, p.weights_json().unwrap()).unwrap())
                .unwrap();
            let j = s.checkpoint(&p).unwrap();
            let mut q = Policy::plans(-1, 8, 0.01).unwrap();
            let restored = State::restore(&j, &mut q).unwrap();
            assert_eq!(j, restored.checkpoint(&q).unwrap());
            assert!(restored.accepted.slots[2].is_some());
            assert!(restored.previous.slots.iter().all(Option::is_none));
            let mut old = j;
            old.set_path("schema", Json::Str("plan-comparison-v2".into()));
            assert!(State::restore(&old, &mut q).is_err());
        }
        #[test]
        fn exploration_does_not_change_accepted_base_trajectory() {
            tensor::worker_threads();
            let accepted = Runtime::load(&Portfolio::empty(), -1).unwrap();
            let learner = Policy::plans(-1, 72, 0.0003).unwrap();
            let job = Job {
                seed: 29,
                seat: 0,
                opponent: 0,
                rng: 44,
                slots: vec![0],
            };
            let a = play(&job, &accepted, None, &[], &Config::default(), 1, 0, 0).unwrap();
            let b = play(
                &job,
                &accepted,
                Some(&learner),
                &[],
                &Config::default(),
                1,
                0,
                1,
            )
            .unwrap();
            assert_eq!(a.report, b.report);
            assert_eq!(b.branches, 1);
            assert_eq!(b.comparisons[0].get("slot_id").i64(), 0);
            assert_eq!(
                b.comparisons[0].get("reference_cash").arr()[0].f64(),
                a.report.get("cash").f64()
            );
        }
        #[test]
        fn rejected_proposal_and_multi_scope_commit_preserve_incumbent() {
            tensor::worker_threads();
            let p = Policy::plans(-1, 72, 0.0003).unwrap();
            let mut s = state(&p);
            let snapshot = s.checkpoint(&p).unwrap();
            let candidate = s.accepted.propose(2, 8, p.weights_json().unwrap()).unwrap();
            let scores = vec![fake(1, 0, 0.5, 0.), fake(2, 0, 0.5, 0.)];
            if *gate(&scores, &scores).unwrap().get("qualifies") == Json::Bool(true) {
                s.promote(candidate.clone()).unwrap();
            }
            assert_eq!(snapshot, s.checkpoint(&p).unwrap());
            let mut invalid = candidate.clone();
            invalid.slots[3] = invalid.slots[2].clone();
            assert!(s.promote(invalid).is_err());
            assert_eq!(snapshot, s.checkpoint(&p).unwrap());
            s.promote(candidate).unwrap();
            assert_eq!(s.accepted.revision, 1);
            assert_eq!(s.previous.revision, 0);
        }
        #[test]
        fn same_forced_choice_preserves_claimed_scope_and_terminal_result() {
            tensor::threads(1);
            let p = Runtime::load(&Portfolio::empty(), -1).unwrap();
            let job = Job {
                seed: 2,
                seat: 0,
                opponent: 0,
                rng: 1,
                slots: vec![],
            };
            let mut w = World::new(2, &Config::default());
            loop {
                let o = Observation::from_state(&w.game, 0);
                let mut probe = w.clone();
                if probe.own.prepare(&o, &p).unwrap().is_some() {
                    break;
                }
                let a = w.own.action(&o, &p).unwrap();
                w.advance(&job, a, &[]).unwrap();
                assert!(w.game.step < 600);
            }
            let o = Observation::from_state(&w.game, 0);
            let mut b = w.clone();
            let d = b.own.prepare(&o, &p).unwrap().unwrap();
            let a = w.own.action(&o, &p).unwrap();
            w.advance(&job, a, &[]).unwrap();
            w.finish(&job, &p, &[]).unwrap();
            let a = b
                .own
                .execute_choice(&o, d.choices[d.selected].clone(), &p)
                .unwrap();
            b.advance(&job, a, &[]).unwrap();
            b.finish(&job, &p, &[]).unwrap();
            assert_eq!(w.game.digest(), b.game.digest());
        }
    }
}
#[cfg(feature = "train")]
fn main() {
    if let Err(e) = app::run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("build with --features train");
    std::process::exit(1);
}
