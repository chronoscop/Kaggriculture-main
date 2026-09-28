//! Stable complete policy iteration within a declared production-event scope.
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
            score_confirmation, tensor,
        },
        pipeline::{
            event_policy::{Version, CONTRACT, SCHEMA},
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
        evaluate_checkpoint: Option<String>,
        diagnose_revisions: Option<String>,
        init: Option<String>,
        foundation: Option<String>,
        config: String,
        iterations: usize,
        games: usize,
        workers: usize,
        points: usize,
        alternatives: usize,
        branch_steps: usize,
        max_branches: usize,
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
                out: "runs/event_policy_trial".into(),
                resume: None,
                evaluate_checkpoint: None,
                diagnose_revisions: None,
                init: None,
                foundation: None,
                config: "native/configs/plan_prototype_v1.json".into(),
                iterations: 100,
                games: 16,
                workers: 7,
                points: 2,
                alternatives: 2,
                branch_steps: 1440,
                max_branches: 4,
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
                println!("event-train --out NEW_DIR [--init-from accepted_event.json | --base-checkpoint v3_best.json | --resume v7_latest.json] --iterations 100 --games-per-update 16 --workers 7 --device cuda|cpu --branch-points 2 --alternatives 2 --branch-steps-per-game 1440 --max-branches-per-game 4 --epochs 8 --batch-size 64 --learning-rate 0.0003 --eval-every 5 --eval-games 8 --confirm-games 16\nOne shared network, fixed accepted continuation until promotion. Default scope unchanged: four seasonal harvest arrangements and one same-batch revision each. Outcome-independent replay. Independent confirmation progresses at the next scheduled evaluations, preserving the exact frozen candidate. iterations are additional; init-from preserves accepted deployment ONLY; proposal, labels and Adam start fresh with normalized inputs.");
                println!("Read-only diagnostic: --evaluate-checkpoint v7_latest.json --out NEW_DIR --eval-games 64 --eval-seed NEW_SEED --workers 7 --device cpu. Evaluates the latest learner, ignores pending candidates, performs no training or promotion; eval-games counts both seats per opponent (64 = 32 seeds). ");
                println!("Mechanism diagnostic: --diagnose-revisions v7_latest.json --out NEW_DIR --eval-games 8 --eval-seed SEED --workers 7 --device cpu. Compares accepted, new arrangements only, new revisions only, and both; tests changed revisions individually under accepted continuation. Maximum 16 games per opponent (8 seeds). Writes traces and comparisons only; never trains or promotes.");
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
                    "--evaluate-checkpoint" => o.evaluate_checkpoint = Some(v.clone()),
                    "--diagnose-revisions" => o.diagnose_revisions = Some(v.clone()),
                    "--init-from" => o.init = Some(v.clone()),
                    "--base-checkpoint" => o.foundation = Some(v.clone()),
                    "--config" => o.config = v.clone(),
                    "--iterations" => o.iterations = u()?,
                    "--games-per-update" => o.games = u()?,
                    "--workers" => o.workers = u()?,
                    "--branch-points" => o.points = u()?,
                    "--alternatives" => o.alternatives = u()?,
                    "--branch-steps-per-game" => o.branch_steps = u()?,
                    "--max-branches-per-game" => o.max_branches = u()?,
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
                    _ => return Err(format!("unknown option {}", a[0])),
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
                o.confirm_games,
            ]
            .contains(&0)
                || o.games % 2 != 0
                || o.eval_games % 2 != 0
                || o.confirm_games % 2 != 0
                || (o.evaluate_checkpoint.is_none()
                    && o.diagnose_revisions.is_none()
                    && o.confirm_games < o.eval_games)
                || o.confirm_games > 1024
                || o.points > 4
                || o.alternatives > 2
                || o.branch_steps < 719
                || o.max_branches == 0
                || o.max_branches > 8
                || o.batch < 2
                || !o.lr.is_finite()
                || o.lr <= 0.
                || o.seed >= 100_000_000
                || !(1_000_000_000..=2_000_000_000).contains(&o.eval_seed)
                || [
                    o.resume.is_some(),
                    o.init.is_some(),
                    o.foundation.is_some(),
                    o.evaluate_checkpoint.is_some(),
                    o.diagnose_revisions.is_some(),
                ]
                .iter()
                .filter(|x| **x)
                .count()
                    > 1
            {
                return Err("invalid sizes/seed partition: even games, points 1..4, alternatives 1..2, branch budget >=719, max branches 1..8, and one initialization source".into());
            }
            Ok(o)
        }
    }
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
    struct Game {
        report: Json,
        pairs: Vec<Pair>,
        comparisons: Vec<Json>,
        steps: usize,
        branches: usize,
    }
    /// Only actual terminal outcomes become labels. Prefix identifies the conditional
    /// second state, which must never be mistaken for an incumbent trajectory state.
    fn record_pair(
        result: &mut Game,
        job: &Job,
        iteration: u64,
        revision: u64,
        slot: usize,
        row: &Sample,
        choices: &[Choice],
        reference: usize,
        alternative: usize,
        reference_cash: [f64; 2],
        alternative_cash: [f64; 2],
        mut evidence: Json,
    ) -> Result<(), String> {
        let pref = plan_compare::match_score_preference(reference_cash, alternative_cash)?;
        for (k, v) in [
            (
                "objective",
                Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into()),
            ),
            ("policy_contract", Json::Str(CONTRACT.into())),
            (
                "learner_input_encoding",
                Json::Str(route_rl_native::learning::policy::EVENT_INPUT_ENCODING.into()),
            ),
            ("iteration", n(iteration as f64)),
            ("incumbent_revision", Json::Str(revision.to_string())),
            ("slot_id", n(slot as f64)),
            ("seed", n(job.seed as f64)),
            ("seat", n(job.seat as f64)),
            ("opponent", n(job.opponent as f64)),
            ("step", n(row.step as f64)),
            ("reference_index", n(reference as f64)),
            ("alternative_index", n(alternative as f64)),
            ("full_row", row.json()),
            ("reference_plan", choices[reference].json()),
            ("alternative_plan", choices[alternative].json()),
            ("reference_cash", Json::Arr(reference_cash.map(n).to_vec())),
            (
                "alternative_cash",
                Json::Arr(alternative_cash.map(n).to_vec()),
            ),
            ("decisive", Json::Bool(pref.is_some())),
            (
                "alternative_better",
                Json::Bool(pref.as_ref().is_some_and(|(t, _)| t[1] > t[0])),
            ),
        ] {
            evidence.set_path(k, v);
        }
        // Only distinct executed choices form a ranking target.
        if reference != alternative {
            let (target, gain) = pref.unwrap_or_else(|| (vec![0.5, 0.5], 0.));
            // Diagnostic category identifies the tested alternative BEFORE its
            // outcome. It never controls retention or sampling.
            let kind = choices[alternative]
                .next
                .as_ref()
                .map(|p| p.name())
                .unwrap_or("");
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
            result.pairs.push(Pair {
                row: Sample {
                    features: vec![
                        choices[reference].features.clone(),
                        choices[alternative].features.clone(),
                    ],
                    ..row.clone()
                },
                target,
                gain,
                iteration,
                seed: job.seed,
                seat: job.seat,
                opponent: job.opponent,
                bucket: job.opponent * 27
                    + (row.step.max(0) as usize / 240).min(2) * 9
                    + kinds.iter().position(|x| *x == kind).unwrap(),
                evidence: evidence.clone(),
            });
        }
        result.comparisons.push(evidence);
        Ok(())
    }
    #[derive(Clone)]
    struct Fork {
        world: World,
        row: Sample,
        choices: Vec<Choice>,
        reference: usize,
        slot: usize,
        followup: bool,
        proposed: usize,
        prefix: String,
    }
    #[derive(Default)]
    struct Budget {
        steps: usize,
        branches: usize,
    }
    impl Budget {
        fn available(&self, step: i64, step_limit: usize, count_limit: usize) -> bool {
            self.branches < count_limit && self.steps + (719 - step) as usize <= step_limit
        }
    }
    fn snapshot(
        world: &World,
        d: &route_rl_native::pipeline::event_portfolio::Decision,
        learner: &Policy,
        prefix: String,
    ) -> Result<Fork, String> {
        Ok(Fork {
            world: world.clone(),
            row: d.row.clone(),
            choices: d.choices.clone(),
            reference: d.selected,
            slot: d.slot.ok_or("missing scope")?,
            followup: d.followup,
            proposed: learner.infer(&[d.row.clone()], true, &mut Rng(0))?[0].action,
            prefix,
        })
    }
    /// Coverage over the declared scope, then disagreements. Conditional revision
    /// states are eligible, never given a separate network or a different continuation.
    fn select_forks(forks: &[Fork], desired: &[usize], rotation: usize) -> Vec<usize> {
        if forks.is_empty() || desired.is_empty() {
            return vec![];
        }
        let first = forks
            .iter()
            .position(|f| f.slot == desired[0])
            .unwrap_or(rotation % forks.len());
        let mut ids = vec![first];
        while ids.len() < desired.len().min(forks.len()) {
            let next = (0..forks.len())
                .filter(|i| !ids.contains(i))
                .min_by_key(|i| {
                    let f = &forks[*i];
                    (
                        usize::from(!f.followup),
                        usize::from(f.proposed == f.reference),
                        (f.slot + SLOTS - rotation % SLOTS) % SLOTS,
                    )
                })
                .unwrap();
            ids.push(next);
        }
        ids
    }
    fn alternatives(f: &Fork, limit: usize, rng: &mut Rng) -> Vec<usize> {
        let mut ids: Vec<_> = (0..f.choices.len()).filter(|i| *i != f.reference).collect();
        rng.shuffle(&mut ids);
        if let Some(at) = ids.iter().position(|i| *i == f.proposed) {
            ids.swap(0, at);
        }
        if ids.len() > 2 {
            if let Some(at) =
                (1..ids.len()).find(|i| f.choices[ids[*i]].next != f.choices[ids[0]].next)
            {
                ids.swap(1, at);
            }
        }
        ids.truncate(limit);
        ids
    }
    /// ONE intervention; every later choice is made by the frozen accepted policy.
    /// Optionally save a conditional state encountered on this real trajectory.
    fn rollout_edit(
        f: &Fork,
        alternative: usize,
        job: &Job,
        accepted: &Runtime,
        opponents: &[Runtime],
        learner: Option<&Policy>,
        scope: &[usize],
    ) -> Result<(World, Option<Fork>), String> {
        let mut world = f.world.clone();
        let obs = Observation::from_state(&world.game, job.seat);
        let new_batch = world.own.controller.batches.len();
        let creates =
            !f.followup && !f.choices[alternative].keep && f.choices[alternative].next.is_some();
        let action = world
            .own
            .execute_choice(&obs, f.choices[alternative].clone(), accepted)?;
        world.advance(job, action, opponents)?;
        let mut conditional = None;
        while world.game.step < 719 {
            let obs = Observation::from_state(&world.game, job.seat);
            let action = if let Some(mut d) = world.own.prepare(&obs, accepted)? {
                if creates
                    && conditional.is_none()
                    && d.followup
                    && d.slot == Some(f.slot)
                    && world.own.last_event.as_ref().and_then(|e| e.batch) == Some(new_batch)
                    && d.slot.is_some_and(|s| scope.contains(&s))
                {
                    if let Some(p) = learner {
                        conditional = Some(snapshot(
                            &world,
                            &d,
                            p,
                            format!(
                                "{}:{}:{}:{}:{}",
                                job.seed, job.seat, f.row.step, f.slot, alternative
                            ),
                        )?);
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
        Ok((world, conditional))
    }
    fn evidence(f: &Fork, kind: &str, version: u64, actual_choice: bool) -> Json {
        Json::Obj(vec![
            (
                "stage".into(),
                Json::Str(
                    if f.followup {
                        "revision"
                    } else {
                        "arrangement"
                    }
                    .into(),
                ),
            ),
            ("source".into(), Json::Str(kind.into())),
            ("prefix_id".into(), Json::Str(f.prefix.clone())),
            (
                "continuation_revision".into(),
                Json::Str(version.to_string()),
            ),
            ("current_greedy_tested".into(), Json::Bool(actual_choice)),
            (
                "event".into(),
                f.world
                    .own
                    .last_event
                    .as_ref()
                    .map(|e| e.json())
                    .unwrap_or(Json::Null),
            ),
        ])
    }
    fn play(
        job: &Job,
        accepted: &Runtime,
        learner: Option<&Policy>,
        opponents: &[Runtime],
        config: &Config,
        iteration: u64,
        revision: u64,
        scope: &[usize],
        alt_count: usize,
        step_limit: usize,
        count_limit: usize,
    ) -> Result<Game, String> {
        let mut world = World::new(job.seed, config);
        let mut forks = vec![];
        let mut events = 0;
        let mut edits = 0;
        let mut revisions = 0;
        while world.game.step < 719 {
            let obs = Observation::from_state(&world.game, job.seat);
            let action = if let Some(mut d) = world.own.prepare(&obs, accepted)? {
                if d.slot.is_some_and(|s| scope.contains(&s)) {
                    events += 1;
                    edits += usize::from(d.selected != 0);
                    revisions += usize::from(d.followup);
                    if let Some(p) = learner.filter(|_| d.choices.len() > 1) {
                        forks.push(snapshot(&world, &d, p, String::new())?);
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
        let mut report = world.result(job);
        report.set_path("scope_events", n(events as f64));
        report.set_path("scope_edits", n(edits as f64));
        report.set_path("revision_events", n(revisions as f64));
        let reference_cash = world.cash(job.seat);
        let mut result = Game {
            report,
            pairs: vec![],
            comparisons: vec![],
            steps: 719,
            branches: 0,
        };
        let mut rng = Rng(job.rng);
        let mut budget = Budget::default();
        let mut conditional_used = false;
        for i in select_forks(&forks, &job.slots, job.seed as usize + iteration as usize) {
            let f = &forks[i];
            for alternative in alternatives(f, alt_count, &mut rng) {
                if !budget.available(f.row.step, step_limit, count_limit) {
                    continue;
                }
                let (branch, conditional) = rollout_edit(
                    f,
                    alternative,
                    job,
                    accepted,
                    opponents,
                    if !conditional_used { learner } else { None },
                    scope,
                )?;
                budget.steps += (719 - f.row.step) as usize;
                budget.branches += 1;
                let cash = branch.cash(job.seat);
                let mut ev = evidence(
                    f,
                    "accepted_trajectory",
                    revision,
                    alternative == f.proposed,
                );
                ev.set_path("execution", branch.own.controller.report());
                record_pair(
                    &mut result,
                    job,
                    iteration,
                    revision,
                    f.slot,
                    &f.row,
                    &f.choices,
                    f.reference,
                    alternative,
                    reference_cash,
                    cash,
                    ev,
                )?;
                // At most ONE secondary comparison per base game. It spends the
                // SAME budget and cannot recursively branch. The first branch's
                // completed outcome is the exact reference, not an extra rollout.
                if let Some(second) = conditional.filter(|s| {
                    !conditional_used && budget.available(s.row.step, step_limit, count_limit)
                }) {
                    if let Some(next) = alternatives(&second, 1, &mut rng).first().copied() {
                        let (revised, _) =
                            rollout_edit(&second, next, job, accepted, opponents, None, scope)?;
                        budget.steps += (719 - second.row.step) as usize;
                        budget.branches += 1;
                        conditional_used = true;
                        record_pair(
                            &mut result,
                            job,
                            iteration,
                            revision,
                            second.slot,
                            &second.row,
                            &second.choices,
                            second.reference,
                            next,
                            cash,
                            revised.cash(job.seat),
                            evidence(
                                &second,
                                "conditional_exploration",
                                revision,
                                next == second.proposed,
                            ),
                        )?;
                    }
                }
            }
        }
        result.steps += budget.steps;
        result.branches = budget.branches;
        Ok(result)
    }
    fn collect(
        jobs: Vec<Job>,
        accepted: &Version,
        learner: Option<&Json>,
        opponents: &[Version],
        config: &Config,
        workers: usize,
        iteration: u64,
        alt_count: usize,
        step_limit: usize,
        count_limit: usize,
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
            let version = accepted.clone();
            let lw = learner.cloned();
            let os = opponents.to_vec();
            let c = config.clone();
            handles.push(std::thread::spawn(
                move || -> Result<Vec<(usize, Game)>, String> {
                    tensor::worker_threads();
                    let runtime = version.runtime(-1)?;
                    let lp = if let Some(w) = lw {
                        let mut p = Policy::event_plans(-1, 0, 0.0003)?;
                        p.load_weights(&w)?;
                        Some(p)
                    } else {
                        None
                    };
                    let opponents = os
                        .iter()
                        .map(|v| v.runtime(-1))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut result = vec![];
                    loop {
                        let i = ix.fetch_add(1, Ordering::Relaxed);
                        if i >= js.len() {
                            break;
                        }
                        result.push((
                            i,
                            play(
                                &js[i],
                                &runtime,
                                lp.as_ref(),
                                &opponents,
                                &c,
                                iteration,
                                version.revision,
                                &version.scope,
                                alt_count,
                                step_limit,
                                count_limit,
                            )?,
                        ));
                    }
                    Ok(result)
                },
            ));
        }
        let mut rows = vec![];
        let mut error = None;
        for h in handles {
            match h.join() {
                Ok(Ok(v)) => rows.extend(v),
                Ok(Err(e)) => error = Some(e),
                Err(_) => error = Some("rollout worker panicked".into()),
            }
        }
        if let Some(e) = error {
            return Err(e);
        }
        rows.sort_by_key(|(i, _)| *i);
        Ok(rows.into_iter().map(|(_, g)| g).collect())
    }
    fn roster(accepted: &Version, previous: &Version) -> Vec<usize> {
        let mut r = vec![0, 1];
        if previous.revision != accepted.revision {
            r.push(2);
        }
        r
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
            e.0 += ds;
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
        // These remain diagnostics. Cash margins, sign counts, production volume and
        // individual matchup fluctuations cannot veto an aggregate match-score gain.
        let no_matchup_regression = by_opponent.values().all(|(s, _, _)| *s >= -1e-9);
        let qualifies = score_gain > 1e-9;
        Ok(Json::Obj(vec![
            (
                "objective".into(),
                Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into()),
            ),
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
    fn seed_score_deltas(candidate: &[Game], incumbent: &[Game]) -> Result<Vec<f64>, String> {
        gate(candidate, incumbent)?; // validate matching seat/opponent keys
        let mut seeds = std::collections::BTreeMap::<i64, (f64, usize)>::new();
        for (c, b) in candidate.iter().zip(incumbent) {
            let e = seeds.entry(c.report.get("seed").i64()).or_default();
            e.0 += c.report.get("score").f64() - b.report.get("score").f64();
            e.1 += 1;
        }
        Ok(seeds.values().map(|(sum, n)| sum / (*n as f64)).collect())
    }
    /// Uniform historical reservoir + FIFO recent evidence. Inclusion depends on
    /// arrival order and a saved RNG ONLY, never the winner, sign, magnitude or crop.
    /// We preserve rare successes through the same probability as rare failures;
    /// accepted deployment, not outcome-biased replay, preserves proven ability.
    struct EvidenceBank {
        history: Vec<Pair>,
        recent: Vec<Pair>,
        seen: u64,
        rng: Rng,
    }
    impl Default for EvidenceBank {
        fn default() -> Self {
            Self {
                history: vec![],
                recent: vec![],
                seen: 0,
                rng: Rng(0x9312_a6e5_4410_873b),
            }
        }
    }
    impl EvidenceBank {
        const CAPACITY: usize = 2048;
        fn admit(&mut self, pairs: &[Pair]) {
            for pair in pairs {
                self.seen += 1;
                if self.history.len() < Self::CAPACITY {
                    self.history.push(pair.clone());
                } else {
                    // Exact unbiased uniform integer, independent of outcome.
                    let threshold = self.seen.wrapping_neg() % self.seen;
                    let at = loop {
                        let x = self.rng.next();
                        if x >= threshold {
                            break (x % self.seen) as usize;
                        }
                    };
                    if at < Self::CAPACITY {
                        self.history[at] = pair.clone();
                    }
                }
                self.recent.push(pair.clone());
            }
            if self.recent.len() > 512 {
                self.recent.drain(..self.recent.len() - 512);
            }
        }
        fn training(&self, _rng: &mut Rng) -> Bank {
            // Existing optimizer draws 50% fresh / 25% history / 25% recent.
            // "elite" is only the legacy optimizer's field name, NOT filtering.
            Bank {
                elite: self.history.clone(),
                recent: self.recent.clone(),
            }
        }
        fn json(&self) -> Json {
            Json::Obj(vec![
                (
                    "sampling".into(),
                    Json::Str("outcome_independent_reservoir_v1".into()),
                ),
                ("seen".into(), Json::Str(self.seen.to_string())),
                ("rng".into(), Json::Str(self.rng.0.to_string())),
                (
                    "history".into(),
                    Json::Arr(self.history.iter().map(Pair::json).collect()),
                ),
                (
                    "recent".into(),
                    Json::Arr(self.recent.iter().map(Pair::json).collect()),
                ),
            ])
        }
        fn parse(j: &Json, version: u64) -> Result<Self, String> {
            if j.get("sampling").str() != "outcome_independent_reservoir_v1" {
                return Err("winner-selected experience cannot resume as uniform evidence".into());
            }
            let out = Self {
                history: j
                    .get("history")
                    .arr()
                    .iter()
                    .map(Pair::parse)
                    .collect::<Result<_, _>>()?,
                recent: j
                    .get("recent")
                    .arr()
                    .iter()
                    .map(Pair::parse)
                    .collect::<Result<_, _>>()?,
                seen: uint(j, "seen")?,
                rng: Rng(uint(j, "rng")?),
            };
            if out.history.len() != (out.seen.min(Self::CAPACITY as u64) as usize)
                || out.recent.len() != (out.seen.min(512) as usize)
            {
                return Err("invalid evidence reservoir size".into());
            }
            for p in out.history.iter().chain(&out.recent) {
                if p.incumbent_revision()? != version
                    || p.evidence.get("continuation_revision").str() != version.to_string()
                    || p.evidence.get("policy_contract").str() != CONTRACT
                    || p.evidence.get("learner_input_encoding").str()
                        != route_rl_native::learning::policy::EVENT_INPUT_ENCODING
                    || p.evidence.get("objective").str() != plan_compare::MATCH_SCORE_OBJECTIVE
                    || !matches!(p.evidence.get("stage").str(), "arrangement" | "revision")
                    || p.seed < 0
                    || p.seed >= 1_000_000_000
                {
                    return Err("stale/incompatible/validation evidence in replay".into());
                }
            }
            Ok(out)
        }
    }
    fn outcome_summary(pairs: &[Pair]) -> Result<Json, String> {
        let mut result = Json::Obj(vec![]);
        for stage in ["arrangement", "revision"] {
            let deltas = pairs
                .iter()
                .filter(|p| p.evidence.get("stage").str() == stage)
                .map(Pair::improvement_target)
                .collect::<Result<Vec<_>, _>>()?;
            result.set_path(
                stage,
                Json::Obj(vec![
                    ("rows".into(), n(deltas.len() as f64)),
                    (
                        "better".into(),
                        n(deltas.iter().filter(|d| **d > 0.).count() as f64),
                    ),
                    (
                        "worse".into(),
                        n(deltas.iter().filter(|d| **d < 0.).count() as f64),
                    ),
                    (
                        "tie".into(),
                        n(deltas.iter().filter(|d| **d == 0.).count() as f64),
                    ),
                    (
                        "mean_target".into(),
                        n(deltas.iter().map(|d| *d as f64).sum::<f64>()
                            / deltas.len().max(1) as f64),
                    ),
                ]),
            );
        }
        Ok(result)
    }
    /// Unique measured states, evaluated AFTER the update with the full legal
    /// candidate list. A new untested argmax is UNKNOWN, never counted a success.
    fn selection_report(p: &Policy, pairs: &[Pair]) -> Result<Json, String> {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<String, Vec<&Pair>> = BTreeMap::new();
        for pair in pairs {
            let key = format!(
                "{}:{}:{}:{}:{}",
                pair.seed,
                pair.seat,
                pair.opponent,
                pair.row.step,
                pair.evidence.get("prefix_id").str()
            );
            groups.entry(key).or_default().push(pair);
        }
        let groups: Vec<_> = groups.into_values().collect();
        if groups.is_empty() {
            return Ok(Json::Null);
        }
        let rows = groups
            .iter()
            .map(|g| Sample::parse(g[0].evidence.get("full_row")))
            .collect::<Result<Vec<_>, _>>()?;
        let decisions = p.infer(&rows, true, &mut Rng(0))?;
        let mut out = Json::Obj(vec![]);
        for stage in ["arrangement", "revision"] {
            let mut known = 0;
            let mut unknown = 0;
            let mut keep = 0;
            let mut gain = 0.;
            let mut good = 0;
            let mut bad = 0;
            for (group, d) in groups.iter().zip(&decisions) {
                if group[0].evidence.get("stage").str() != stage {
                    continue;
                }
                let reference = group[0].evidence.get("reference_index").i64() as usize;
                let measured = if d.action == reference {
                    Some(0.)
                } else {
                    group
                        .iter()
                        .find(|r| r.evidence.get("alternative_index").i64() as usize == d.action)
                        .map(|r| r.improvement_target())
                        .transpose()?
                };
                keep += usize::from(d.action == 0);
                if let Some(delta) = measured {
                    known += 1;
                    gain += delta as f64;
                    good += usize::from(delta > 0.);
                    bad += usize::from(delta < 0.);
                } else {
                    unknown += 1;
                }
            }
            out.set_path(
                stage,
                Json::Obj(vec![
                    ("states".into(), n((known + unknown) as f64)),
                    ("keep".into(), n(keep as f64)),
                    ("tested".into(), n(known as f64)),
                    ("untested".into(), n(unknown as f64)),
                    ("better".into(), n(good as f64)),
                    ("worse".into(), n(bad as f64)),
                    (
                        "mean_tested_score_delta".into(),
                        n(gain / known.max(1) as f64),
                    ),
                ]),
            );
        }
        Ok(out)
    }
    /// A SINGLE immutable proposal survives between predetermined confirmation
    /// looks. New learner updates cannot replace it or reset its evidence budget.
    struct Pending {
        candidate: Version,
        reference_revision: u64,
        look: usize,
        confirm_games: usize,
        candidate_games: Vec<Game>,
        incumbent_games: Vec<Game>,
    }
    fn saved_games(gs: &[Game]) -> Json {
        Json::Arr(gs.iter().map(|g| g.report.clone()).collect())
    }
    fn restored_games(j: &Json) -> Vec<Game> {
        j.arr()
            .iter()
            .map(|r| Game {
                report: r.clone(),
                pairs: vec![],
                comparisons: vec![],
                steps: 719,
                branches: 0,
            })
            .collect()
    }
    impl Pending {
        fn json(&self) -> Json {
            Json::Obj(vec![
                ("candidate".into(), self.candidate.json()),
                (
                    "reference_revision".into(),
                    Json::Str(self.reference_revision.to_string()),
                ),
                ("look".into(), n(self.look as f64)),
                ("confirm_games".into(), n(self.confirm_games as f64)),
                ("candidate_games".into(), saved_games(&self.candidate_games)),
                ("incumbent_games".into(), saved_games(&self.incumbent_games)),
            ])
        }
        fn parse(j: &Json, accepted: &Version, previous: &Version) -> Result<Self, String> {
            let out = Self {
                candidate: Version::parse(j.get("candidate"))?,
                reference_revision: uint(j, "reference_revision")?,
                look: j.get("look").i64() as usize,
                confirm_games: j.get("confirm_games").i64() as usize,
                candidate_games: restored_games(j.get("candidate_games")),
                incumbent_games: restored_games(j.get("incumbent_games")),
            };
            accepted.validate_successor(&out.candidate)?;
            let r = roster(accepted, previous);
            if out.reference_revision != accepted.revision
                || out.look > 2
                || out.confirm_games == 0
                || out.confirm_games % 2 != 0
                || out.candidate_games.len() != out.incumbent_games.len()
            {
                return Err("invalid frozen confirmation state".into());
            }
            let expected = if out.look == 0 {
                0
            } else {
                out.confirm_games * (1 << (out.look - 1)) * r.len()
            };
            if out.candidate_games.len() != expected {
                return Err("confirmation sample count mismatch".into());
            }
            if expected > 0 {
                gate(&out.candidate_games, &out.incumbent_games)?;
            }
            Ok(out)
        }
    }
    struct State {
        iteration: u64,
        rng: Rng,
        config: Config,
        accepted: Version,
        previous: Version,
        bank: EvidenceBank,
        next_seed: u64,
        eval_seed: u64,
        next_eval_seed: u64,
        steps: u64,
        base_games: u64,
        branches: u64,
        monitor: Json,
        pending: Option<Pending>,
        evaluation_games: u64,
    }
    impl State {
        fn checkpoint(&self, p: &Policy) -> Result<Json, String> {
            Ok(Json::Obj(vec![
                ("schema".into(), Json::Str(SCHEMA.into())),
                (
                    "learner_input_encoding".into(),
                    Json::Str(route_rl_native::learning::policy::EVENT_INPUT_ENCODING.into()),
                ),
                ("policy_contract".into(), Json::Str(CONTRACT.into())),
                (
                    "objective".into(),
                    Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into()),
                ),
                ("iteration".into(), Json::Str(self.iteration.to_string())),
                ("model".into(), p.checkpoint(self.iteration, &self.rng)?),
                ("config".into(), self.config.json()),
                ("deployment".into(), self.accepted.json()),
                ("previous_accepted".into(), self.previous.json()),
                ("comparison_bank".into(), self.bank.json()),
                (
                    "pending_candidate".into(),
                    self.pending
                        .as_ref()
                        .map(|x| x.json())
                        .unwrap_or(Json::Null),
                ),
                ("next_seed".into(), Json::Str(self.next_seed.to_string())),
                ("eval_seed".into(), Json::Str(self.eval_seed.to_string())),
                (
                    "next_eval_seed".into(),
                    Json::Str(self.next_eval_seed.to_string()),
                ),
                ("training_steps".into(), Json::Str(self.steps.to_string())),
                ("base_games".into(), Json::Str(self.base_games.to_string())),
                (
                    "branch_rollouts".into(),
                    Json::Str(self.branches.to_string()),
                ),
                (
                    "evaluation_games".into(),
                    Json::Str(self.evaluation_games.to_string()),
                ),
                ("deployed_vs_rule".into(), self.monitor.clone()),
            ]))
        }
        fn restore(j: &Json, p: &mut Policy) -> Result<Self, String> {
            if j.get("schema").str() != SCHEMA
                || j.get("policy_contract").str() != CONTRACT
                || j.get("objective").str() != plan_compare::MATCH_SCORE_OBJECTIVE
            {
                return Err("--resume requires full event-policy-iteration-v7 checkpoint; use --init-from for an older accepted deployment and a fresh normalized learner".into());
            }
            let (iteration, rng) = p.restore(j.get("model"))?;
            if iteration != uint(j, "iteration")? || !p.plan_residual || !p.event_input_scaling {
                return Err("model iteration/architecture mismatch".into());
            }
            let accepted = Version::parse(j.get("deployment"))?;
            let previous = Version::parse(j.get("previous_accepted"))?;
            if j.get("deployment").get("contract").str() != CONTRACT
                || previous.revision > accepted.revision
                || previous.scope != accepted.scope
                || previous.foundation != accepted.foundation
            {
                return Err("invalid previous accepted strategy".into());
            }
            let bank = EvidenceBank::parse(j.get("comparison_bank"), accepted.revision)?;
            let pending = if j.get("pending_candidate").is_obj() {
                Some(Pending::parse(
                    j.get("pending_candidate"),
                    &accepted,
                    &previous,
                )?)
            } else {
                None
            };
            let out = Self {
                iteration,
                rng,
                config: Config::parse(j.get("config"))?,
                accepted,
                previous,
                bank,
                pending,
                next_seed: uint(j, "next_seed")?,
                eval_seed: uint(j, "eval_seed")?,
                next_eval_seed: uint(j, "next_eval_seed")?,
                steps: uint(j, "training_steps")?,
                base_games: uint(j, "base_games")?,
                branches: uint(j, "branch_rollouts")?,
                evaluation_games: uint(j, "evaluation_games")?,
                monitor: j.get("deployed_vs_rule").clone(),
            };
            if out.next_seed >= 1_000_000_000
                || out.eval_seed < 1_000_000_000
                || out.next_eval_seed < out.eval_seed + 1_000_000
            {
                return Err("training/evaluation seed domains overlap".into());
            }
            Ok(out)
        }
        fn promote(&mut self, candidate: Version) -> Result<(), String> {
            self.accepted.validate_successor(&candidate)?;
            self.previous = self.accepted.clone();
            self.accepted = candidate;
            self.bank = EvidenceBank::default();
            self.pending = None;
            Ok(())
        }
        fn eval_seeds(&mut self, count: usize) -> Result<u64, String> {
            let start = self.next_eval_seed;
            self.next_eval_seed = start
                .checked_add(count as u64 / 2)
                .ok_or("eval seeds exhausted")?;
            if self.next_eval_seed > i64::MAX as u64 {
                return Err("eval seeds exhausted".into());
            }
            Ok(start)
        }
    }
    fn full_games(
        js: Vec<Job>,
        version: &Version,
        s: &State,
        o: &Options,
    ) -> Result<Vec<Game>, String> {
        collect(
            js,
            version,
            None,
            &[s.accepted.clone(), s.previous.clone()],
            &s.config,
            o.workers,
            s.iteration,
            0,
            0,
            0,
        )
    }
    fn monitor(o: &Options, s: &mut State) -> Result<(), String> {
        let gs = full_games(
            eval_jobs(s.eval_seed, o.eval_games, &[0]),
            &s.accepted,
            s,
            o,
        )?;
        s.evaluation_games += gs.len() as u64;
        s.monitor = summary(&gs);
        s.monitor
            .set_path("accepted_revision", n(s.accepted.revision as f64));
        Ok(())
    }
    fn evaluate(o: &Options, s: &mut State, p: &mut Policy) -> Result<(), String> {
        let before = s.accepted.revision;
        let start = Instant::now();
        let mut report = Json::Obj(vec![
            ("iteration".into(), n(s.iteration as f64)),
            ("accepted_revision_before".into(), n(before as f64)),
            ("promoted".into(), Json::Bool(false)),
        ]);
        let roster = roster(&s.accepted, &s.previous);
        if s.pending.is_none() {
            let candidate = s.accepted.propose(s.iteration, p.weights_json()?)?;
            let seed = s.eval_seeds(o.eval_games)?;
            let js = eval_jobs(seed, o.eval_games, &roster);
            let cs = full_games(js.clone(), &candidate, s, o)?;
            let bs = full_games(js, &s.accepted, s, o)?;
            s.evaluation_games += (cs.len() + bs.len()) as u64;
            let screen = gate(&cs, &bs)?;
            report.set_path("screen_seeds_start", n(seed as f64));
            report.set_path("screen", screen.clone());
            report.set_path("candidate", summary(&cs));
            report.set_path("incumbent", summary(&bs));
            report.set_path("candidate_iteration", n(candidate.iteration as f64));
            if matches!(screen.get("qualifies"), Json::Bool(true)) {
                s.pending = Some(Pending {
                    candidate,
                    reference_revision: before,
                    look: 0,
                    confirm_games: o.confirm_games,
                    candidate_games: vec![],
                    incumbent_games: vec![],
                });
                // Save the candidate BEFORE expensive independent confirmation.
                write(&o.out.join("latest.json"), &s.checkpoint(p)?)?;
            }
        }
        if let Some(mut pending) = s.pending.take() {
            if pending.reference_revision != s.accepted.revision {
                return Err("confirmation reference changed".into());
            }
            pending.look += 1;
            let target = pending.confirm_games * (1 << (pending.look - 1));
            let done = pending.candidate_games.len() / roster.len();
            let seed = s.eval_seeds(target - done)?;
            let js = eval_jobs(seed, target - done, &roster);
            let cs = full_games(js.clone(), &pending.candidate, s, o)?;
            let bs = full_games(js, &s.accepted, s, o)?;
            s.evaluation_games += (cs.len() + bs.len()) as u64;
            pending.candidate_games.extend(cs);
            pending.incumbent_games.extend(bs);
            let evidence = score_confirmation::assess(
                &seed_score_deltas(&pending.candidate_games, &pending.incumbent_games)?,
                pending.look,
                3,
            )?;
            let mut confirmation = gate(&pending.candidate_games, &pending.incumbent_games)?;
            confirmation.set_path("evidence", evidence.clone());
            confirmation.set_path("qualifies", evidence.get("qualifies").clone());
            report.set_path("candidate_iteration", n(pending.candidate.iteration as f64));
            report.set_path("confirmation_seeds_start", n(seed as f64));
            report.set_path("confirmation", confirmation);
            report.set_path("confirmation_candidate", summary(&pending.candidate_games));
            report.set_path("confirmation_incumbent", summary(&pending.incumbent_games));
            match evidence.get("decision").str() {
                "promote" => {
                    // Start the next cycle at the ACTUALLY confirmed weights,
                    // not the learner that may have moved while this was pending.
                    let mut next = Policy::event_plans(o.device, s.iteration, p.lr)?;
                    next.load_weights(
                        pending
                            .candidate
                            .weights
                            .as_ref()
                            .ok_or("candidate has no model")?,
                    )?;
                    s.promote(pending.candidate)?;
                    *p = next;
                    monitor(o, s)?;
                    report.set_path("promoted", Json::Bool(true));
                    write(&o.out.join("best.json"), &s.checkpoint(p)?)?;
                }
                "extend" => s.pending = Some(pending),
                _ => {
                    // Preserved for audit; never restarted to fish for significance.
                    write(
                        &o.out.join(format!(
                            "candidate_{:06}_{}.json",
                            pending.candidate.iteration,
                            evidence.get("decision").str()
                        )),
                        &pending.json(),
                    )?;
                }
            }
        }
        report.set_path("accepted_revision_after", n(s.accepted.revision as f64));
        report.set_path(
            "pending_candidate_iteration",
            s.pending
                .as_ref()
                .map(|p| n(p.candidate.iteration as f64))
                .unwrap_or(Json::Null),
        );
        report.set_path("deployed_vs_rule", s.monitor.clone());
        report.set_path("seconds", n(start.elapsed().as_secs_f64()));
        report.set_path("total_evaluation_games", n(s.evaluation_games as f64));
        append(&o.out.join("evaluations.jsonl"), &report)?;
        println!("evaluation {}", report.dump());
        Ok(())
    }
    /// Preserve accepted behavior and its historical opponent; ignore proposal/Adam.
    fn import_reference(j: &Json) -> Result<(Version, Version), String> {
        if !matches!(
            j.get("schema").str(),
            "event-policy-iteration-v5" | "event-policy-iteration-v6" | SCHEMA
        ) || j.get("policy_contract").str() != CONTRACT
        {
            return Err("init-from requires an accepted shared event policy checkpoint".into());
        }
        let accepted = Version::parse(j.get("deployment"))?;
        let previous = Version::parse(j.get("previous_accepted"))?;
        if previous.revision > accepted.revision
            || previous.scope != accepted.scope
            || previous.foundation != accepted.foundation
        {
            return Err("incompatible imported historical opponent".into());
        }
        Ok((accepted, previous))
    }
    /// Diagnostic intervention only: executing models and action candidates stay unchanged.
    fn diagnostic_trajectory(
        job: &Job,
        accepted: &Runtime,
        learner: &Policy,
        opponents: &[Runtime],
        config: &Config,
        scope: &[usize],
        mode: usize,
        trace: Option<&Path>,
    ) -> Result<(Game, Vec<Fork>), String> {
        let mut w = World::new(job.seed, config);
        let mut forks = vec![];
        let mut selections = vec![];
        while w.game.step < 719 {
            let obs = Observation::from_state(&w.game, job.seat);
            let mut decision = Json::Null;
            let action = if let Some(mut d) = w.own.prepare(&obs, accepted)? {
                if d.slot.is_some_and(|s| scope.contains(&s)) {
                    let proposed = learner.infer(&[d.row.clone()], true, &mut Rng(0))?[0].action;
                    if mode == 0 && d.followup {
                        forks.push(snapshot(&w, &d, learner, String::new())?);
                    }
                    let selected =
                        if mode == 3 || (mode == 1 && !d.followup) || (mode == 2 && d.followup) {
                            proposed
                        } else {
                            d.selected
                        };
                    decision = Json::Obj(vec![
                        ("step".into(), n(obs.step as f64)),
                        ("followup".into(), Json::Bool(d.followup)),
                        ("event".into(), w.own.last_event.as_ref().unwrap().json()),
                        ("reference".into(), d.choices[d.selected].json()),
                        ("selected".into(), d.choices[selected].json()),
                        ("differs".into(), Json::Bool(selected != d.selected)),
                        ("candidate_count".into(), n(d.choices.len() as f64)),
                    ]);
                    selections.push(decision.clone());
                    d.selected = selected;
                }
                w.own
                    .execute_choice(&obs, d.choices.swap_remove(d.selected), accepted)?
            } else {
                w.own.continue_action(&obs, accepted)?
            };
            if let Some(path) = trace {
                append(
                    path,
                    &Json::Obj(vec![
                        (
                            "observation".into(),
                            json::parse(&kagg_engine::obsjson::seat_obs_json(&w.game, job.seat))?,
                        ),
                        (
                            "action".into(),
                            route_rl_native::pipeline::executor::action_json(&action),
                        ),
                        ("decision".into(), decision),
                    ]),
                )?;
            }
            w.advance(job, action, opponents)?;
        }
        w.own
            .controller
            .observe(&Observation::from_state(&w.game, job.seat));
        if let Some(path) = trace {
            append(
                path,
                &Json::Obj(vec![
                    (
                        "observation".into(),
                        json::parse(&kagg_engine::obsjson::seat_obs_json(&w.game, job.seat))?,
                    ),
                    ("action".into(), Json::Null),
                ]),
            )?;
        }
        let mut report = w.result(job);
        report.set_path("decisions", Json::Arr(selections));
        Ok((
            Game {
                report,
                pairs: vec![],
                comparisons: vec![],
                steps: 719,
                branches: 0,
            },
            forks,
        ))
    }
    fn diagnose_revisions(o: &Options, path: &str) -> Result<(), String> {
        if o.out.exists()
            && std::fs::read_dir(&o.out)
                .map_err(|e| e.to_string())?
                .next()
                .is_some()
        {
            return Err("diagnostic requires a new/empty directory".into());
        }
        if o.eval_games > 16 {
            return Err("mechanism diagnostic limited to 8 seeds".into());
        }
        let source = read(path)?;
        let mut policy = Policy::event_plans(-1, 0, o.lr)?;
        let s = State::restore(&source, &mut policy)?;
        let js = Arc::new(eval_jobs(
            o.eval_seed,
            o.eval_games,
            &roster(&s.accepted, &s.previous),
        ));
        std::fs::create_dir_all(&o.out).map_err(|e| e.to_string())?;
        write(
            &o.out.join("manifest.json"),
            &Json::Obj(vec![
                ("source".into(), Json::Str(path.into())),
                ("iteration".into(), n(s.iteration as f64)),
                ("seed_start".into(), n(o.eval_seed as f64)),
                ("seed_count".into(), n((o.eval_games / 2) as f64)),
                ("diagnostic_only".into(), Json::Bool(true)),
            ]),
        )?;
        let cursor = Arc::new(AtomicUsize::new(0));
        let mut threads = vec![];
        for _ in 0..o
            .workers
            .min(route_rl_native::resources::available_workers())
            .min(js.len())
        {
            let jobs = js.clone();
            let cursor = cursor.clone();
            let config = s.config.clone();
            let version = s.accepted.clone();
            let previous = s.previous.clone();
            let weights = policy.weights_json()?;
            let out = o.out.clone();
            let iteration = s.iteration;
            threads.push(std::thread::spawn(
                move || -> Result<Vec<(usize, Vec<Game>, Vec<Json>)>, String> {
                    tensor::worker_threads();
                    let accepted = version.runtime(-1)?;
                    let opponents = vec![version.runtime(-1)?, previous.runtime(-1)?];
                    let mut learner = Policy::event_plans(-1, 0, 0.0003)?;
                    learner.load_weights(&weights)?;
                    let mut results = vec![];
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() {
                            break;
                        }
                        let job = &jobs[i];
                        let mut variants = vec![];
                        let mut single = vec![];
                        for mode in 0..4 {
                            let trace = if job.seed == jobs[0].seed
                                && job.seat == 0
                                && job.opponent == 1
                                && (mode == 0 || mode == 3)
                            {
                                Some(out.join(format!("trace_mode{mode}.jsonl")))
                            } else {
                                None
                            };
                            let (game, forks) = diagnostic_trajectory(
                                job,
                                &accepted,
                                &learner,
                                &opponents,
                                &config,
                                &version.scope,
                                mode,
                                trace.as_deref(),
                            )?;
                            let cash = [
                                game.report.get("cash").f64(),
                                game.report.get("opponent_cash").f64(),
                            ];
                            for f in forks {
                                if f.proposed == f.reference {
                                    continue;
                                }
                                let (branch, _) = rollout_edit(
                                    &f,
                                    f.proposed,
                                    job,
                                    &accepted,
                                    &opponents,
                                    None,
                                    &version.scope,
                                )?;
                                let mut temp = Game {
                                    report: Json::Null,
                                    pairs: vec![],
                                    comparisons: vec![],
                                    steps: 0,
                                    branches: 0,
                                };
                                record_pair(
                                    &mut temp,
                                    job,
                                    iteration,
                                    version.revision,
                                    f.slot,
                                    &f.row,
                                    &f.choices,
                                    f.reference,
                                    f.proposed,
                                    cash,
                                    branch.cash(job.seat),
                                    evidence(
                                        &f,
                                        "diagnostic_single_revision",
                                        version.revision,
                                        true,
                                    ),
                                )?;
                                single.extend(temp.comparisons);
                            }
                            variants.push(game);
                        }
                        results.push((i, variants, single));
                    }
                    Ok(results)
                },
            ));
        }
        let mut results = vec![];
        for thread in threads {
            results.extend(thread.join().map_err(|_| "diagnostic worker panic")??);
        }
        results.sort_by_key(|r| r.0);
        let mut modes: Vec<Vec<Game>> = (0..4).map(|_| vec![]).collect();
        for (_, games, single) in results {
            for (m, game) in games.into_iter().enumerate() {
                modes[m].push(game);
            }
            for row in single {
                append(&o.out.join("single_revisions.jsonl"), &row)?;
            }
        }
        let report = Json::Obj(vec![
            ("baseline".into(), summary(&modes[0])),
            ("new_arrangements_only".into(), summary(&modes[1])),
            ("new_revisions_only".into(), summary(&modes[2])),
            ("new_both".into(), summary(&modes[3])),
            ("arrangements_gain".into(), gate(&modes[1], &modes[0])?),
            ("revisions_gain".into(), gate(&modes[2], &modes[0])?),
            ("both_gain".into(), gate(&modes[3], &modes[0])?),
        ]);
        write(&o.out.join("diagnosis.json"), &report)?;
        println!("mechanism diagnostic complete; no training or promotion");
        Ok(())
    }
    fn diagnostic_candidate(s: &State, p: &Policy) -> Result<Version, String> {
        // Read the current learner explicitly, never a pending or accepted model.
        s.accepted.propose(s.iteration, p.weights_json()?)
    }
    fn evaluate_checkpoint(o: &Options, path: &str) -> Result<(), String> {
        if o.out.exists()
            && std::fs::read_dir(&o.out)
                .map_err(|e| e.to_string())?
                .next()
                .is_some()
        {
            return Err("diagnostic requires a new/empty output directory".into());
        }
        let mut p = Policy::event_plans(-1, 0, o.lr)?;
        let s = State::restore(&read(path)?, &mut p)?;
        if o.eval_seed < s.next_eval_seed {
            return Err(
                "diagnostic seed must be beyond the checkpoint's used evaluation range".into(),
            );
        }
        let candidate = diagnostic_candidate(&s, &p)?;
        let opponent_slots = roster(&s.accepted, &s.previous);
        let js = eval_jobs(o.eval_seed, o.eval_games, &opponent_slots);
        let manifest = Json::Obj(vec![
            (
                "schema".into(),
                Json::Str("event-policy-diagnostic-v1".into()),
            ),
            ("source".into(), Json::Str(path.into())),
            ("candidate_iteration".into(), n(s.iteration as f64)),
            ("accepted_revision".into(), n(s.accepted.revision as f64)),
            ("seed_start".into(), n(o.eval_seed as f64)),
            ("seed_count".into(), n((o.eval_games / 2) as f64)),
            ("games_per_policy".into(), n(js.len() as f64)),
            ("diagnostic_only".into(), Json::Bool(true)),
            ("looks".into(), n(1.)),
            (
                "opponents".into(),
                Json::Arr(opponent_slots.iter().map(|x| n(*x as f64)).collect()),
            ),
            (
                "candidate_weights".into(),
                candidate.weights.clone().unwrap(),
            ),
        ]);
        std::fs::create_dir_all(&o.out).map_err(|e| e.to_string())?;
        write(&o.out.join("manifest.json"), &manifest)?;
        println!("diagnostic learner iteration {}: {} fresh seeds, {} games per policy; no training/promotion", s.iteration, o.eval_games/2, js.len());
        let start = Instant::now();
        let cs = full_games(js.clone(), &candidate, &s, o)?;
        println!("candidate finished: {} games", cs.len());
        let bs = full_games(js, &s.accepted, &s, o)?;
        if cs
            .iter()
            .chain(&bs)
            .any(|g| g.branches != 0 || !g.pairs.is_empty())
        {
            return Err("diagnostic unexpectedly generated learning branches".into());
        }
        let result = Json::Obj(vec![
            ("iteration".into(), n(s.iteration as f64)),
            ("diagnostic_only".into(), Json::Bool(true)),
            ("promoted".into(), Json::Bool(false)),
            ("candidate".into(), summary(&cs)),
            ("incumbent".into(), summary(&bs)),
            ("comparison".into(), gate(&cs, &bs)?),
            (
                "evidence".into(),
                score_confirmation::assess(&seed_score_deltas(&cs, &bs)?, 1, 1)?,
            ),
            ("seconds".into(), n(start.elapsed().as_secs_f64())),
        ]);
        write(&o.out.join("evaluation.json"), &result)?;
        println!("diagnostic completed: {}", result.get("comparison").dump());
        Ok(())
    }
    pub fn run() -> Result<(), String> {
        let o = Options::parse()?;
        tensor::threads(1);
        tensor::worker_threads();
        if let Some(path) = &o.diagnose_revisions {
            return diagnose_revisions(&o, path);
        }
        if let Some(path) = &o.evaluate_checkpoint {
            return evaluate_checkpoint(&o, path);
        }
        std::fs::create_dir_all(&o.out).map_err(|e| e.to_string())?;
        if o.out.join("manifest.json").exists() || o.out.join("latest.json").exists() {
            return Err("use a NEW output directory".into());
        }
        let mut p = Policy::event_plans(o.device, o.seed, o.lr)?;
        let mut config = Config::parse(&read(&o.config)?)?;
        let mut base = Portfolio::empty();
        let mut next_seed = o.seed;
        if let Some(path) = &o.foundation {
            let j = read(path)?;
            if j.get("schema").str() != "plan-improvement-v3"
                || j.get("policy_contract").str() != "plan-chain-320x32-scoped-v3"
            {
                return Err("base-checkpoint requires accepted plan-improvement-v3".into());
            }
            base = Portfolio::from_foundation(
                route_rl_native::pipeline::plan_portfolio::Portfolio::parse(j.get("deployment"))?,
            );
            config = Config::parse(j.get("config"))?;
        }
        let mut imported = None;
        if let Some(path) = &o.init {
            let j = read(path)?;
            imported = Some(import_reference(&j)?);
            config = Config::parse(j.get("config"))?;
            next_seed = uint(&j, "next_seed")?.max(o.seed);
            eprintln!("initialization: accepted deployment retains its original input encoding; fresh normalized proposal, Adam and evidence; source unchanged");
        }
        let (initial, initial_previous) = match imported {
            Some(versions) => versions,
            None => {
                let v = Version::initial(base, vec![0, 1, 2, 3])?;
                (v.clone(), v)
            }
        };
        let mut s = if let Some(path) = &o.resume {
            State::restore(&read(path)?, &mut p)?
        } else {
            State {
                iteration: 0,
                rng: Rng(o.seed ^ 0x1acf789),
                config,
                accepted: initial.clone(),
                previous: initial_previous,
                bank: EvidenceBank::default(),
                next_seed,
                eval_seed: o.eval_seed,
                next_eval_seed: o.eval_seed + 1_000_000,
                steps: 0,
                base_games: 0,
                branches: 0,
                evaluation_games: 0,
                monitor: Json::Null,
                pending: None,
            }
        };
        write(
            &o.out.join("manifest.json"),
            &Json::Obj(vec![
                ("schema".into(), Json::Str(SCHEMA.into())),
                (
                    "learner_input_encoding".into(),
                    Json::Str(route_rl_native::learning::policy::EVENT_INPUT_ENCODING.into()),
                ),
                ("policy_contract".into(), Json::Str(CONTRACT.into())),
                (
                    "objective".into(),
                    Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into()),
                ),
                ("iterations_additional".into(), n(o.iterations as f64)),
                ("base_games_per_update".into(), n(o.games as f64)),
                ("branch_points".into(), n(o.points as f64)),
                ("alternatives".into(), n(o.alternatives as f64)),
                ("branch_steps_per_game".into(), n(o.branch_steps as f64)),
                ("max_branches_per_game".into(), n(o.max_branches as f64)),
                ("epochs".into(), n(o.epochs as f64)),
                ("batch_size".into(), n(o.batch as f64)),
                ("learning_rate".into(), n(p.lr)),
                ("eval_every".into(), n(o.eval_every as f64)),
                ("eval_games".into(), n(o.eval_games as f64)),
                ("confirm_games".into(), n(o.confirm_games as f64)),
                ("confirmation_looks".into(), n(3.)),
                (
                    "scope".into(),
                    Json::Arr(s.accepted.scope.iter().map(|x| n(*x as f64)).collect()),
                ),
                (
                    "continuation_refresh".into(),
                    Json::Str("ONLY after whole-policy promotion".into()),
                ),
                ("conditional_comparisons_per_game_max".into(), n(1.)),
                (
                    "replay_sampling".into(),
                    Json::Str("outcome_independent_reservoir_v1".into()),
                ),
                (
                    "workers".into(),
                    n(o.workers
                        .min(route_rl_native::resources::available_workers())
                        as f64),
                ),
                (
                    "device".into(),
                    Json::Str(if o.device < 0 { "cpu" } else { "cuda" }.into()),
                ),
                (
                    "init_from".into(),
                    o.init.clone().map(Json::Str).unwrap_or(Json::Null),
                ),
                (
                    "base_checkpoint".into(),
                    o.foundation.clone().map(Json::Str).unwrap_or(Json::Null),
                ),
                (
                    "resume".into(),
                    o.resume.clone().map(Json::Str).unwrap_or(Json::Null),
                ),
            ]),
        )?;
        if o.resume.is_none() || s.monitor.get("games").i64() != o.eval_games as i64 {
            monitor(&o, &mut s)?;
        }
        for name in ["initial.json", "best.json", "latest.json"] {
            write(&o.out.join(name), &s.checkpoint(&p)?)?;
        }
        append(
            &o.out.join("evaluations.jsonl"),
            &Json::Obj(vec![
                ("initial".into(), Json::Bool(true)),
                ("iteration".into(), n(s.iteration as f64)),
                (
                    "accepted_revision_after".into(),
                    n(s.accepted.revision as f64),
                ),
                ("deployed_vs_rule".into(), s.monitor.clone()),
            ]),
        )?;
        println!("event-policy-iteration-v7 accepted_revision={} fixed_continuation=true shared_network=true scope={:?} budget_per_game={}steps/{}branches",s.accepted.revision,s.accepted.scope,o.branch_steps,o.max_branches);
        for _ in 0..o.iterations {
            let started = Instant::now();
            let iteration = s.iteration + 1;
            if s.next_seed + o.games as u64 / 2 >= 1_000_000_000 {
                return Err("training seed range exhausted".into());
            }
            let mut js = jobs(
                s.next_seed,
                o.games,
                &roster(&s.accepted, &s.previous),
                o.points,
                iteration,
                &mut s.rng,
            );
            for job in &mut js {
                for slot in &mut job.slots {
                    *slot = s.accepted.scope[*slot % s.accepted.scope.len()];
                }
            }
            s.next_seed += o.games as u64 / 2;
            let before = p.weights_json()?;
            let gs = collect(
                js,
                &s.accepted,
                Some(&before),
                &[s.accepted.clone(), s.previous.clone()],
                &s.config,
                o.workers,
                iteration,
                o.alternatives,
                o.branch_steps,
                o.max_branches,
            )?;
            let collect_seconds = started.elapsed().as_secs_f64();
            let pairs: Vec<_> = gs.iter().flat_map(|g| g.pairs.iter().cloned()).collect();
            for g in &gs {
                for c in &g.comparisons {
                    append(&o.out.join("comparisons.jsonl"), c)?;
                }
            }
            let timer = Instant::now();
            let replay = s.bank.training(&mut s.rng);
            let update = plan_compare::update_improvement(
                &mut p,
                &pairs,
                &replay,
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
            if steps > o.games as u64 * (719 + o.branch_steps) as u64
                || branches > o.games as u64 * o.max_branches as u64
            {
                return Err("simulation budget exceeded".into());
            }
            let mut metric = summary(&gs);
            let mut coverage = [0usize; SLOTS];
            for pair in &pairs {
                coverage[pair.evidence.get("slot_id").i64() as usize] += 1;
            }
            for (k, v) in [
                ("iteration", n(iteration as f64)),
                ("accepted_revision", n(s.accepted.revision as f64)),
                ("continuation_revision", n(s.accepted.revision as f64)),
                (
                    "rollout_policy",
                    Json::Str("fixed_accepted_complete_policy".into()),
                ),
                ("comparison_pairs", n(pairs.len() as f64)),
                (
                    "decisive_pairs",
                    n(pairs.iter().filter(|p| p.gain > 0.).count() as f64),
                ),
                (
                    "better_alternatives",
                    n(pairs.iter().filter(|p| p.target[1] > p.target[0]).count() as f64),
                ),
                (
                    "arrangement_pairs",
                    n(pairs
                        .iter()
                        .filter(|p| p.evidence.get("stage").str() == "arrangement")
                        .count() as f64),
                ),
                (
                    "revision_pairs",
                    n(pairs
                        .iter()
                        .filter(|p| p.evidence.get("stage").str() == "revision")
                        .count() as f64),
                ),
                (
                    "current_greedy_tested",
                    n(pairs
                        .iter()
                        .filter(|p| {
                            matches!(p.evidence.get("current_greedy_tested"), Json::Bool(true))
                        })
                        .count() as f64),
                ),
                (
                    "slot_coverage",
                    Json::Arr(coverage.iter().map(|x| n(*x as f64)).collect()),
                ),
                ("branch_rollouts", n(branches as f64)),
                ("simulation_steps", n(steps as f64)),
                ("equivalent_full_games", n(steps as f64 / 719.)),
                ("total_base_games", n(s.base_games as f64)),
                ("total_branch_rollouts", n(s.branches as f64)),
                ("collect_seconds", n(collect_seconds)),
                ("update_seconds", n(timer.elapsed().as_secs_f64())),
                ("weights_changed", Json::Bool(before != p.weights_json()?)),
                ("update", update),
                ("history_pairs", n(s.bank.history.len() as f64)),
                ("recent_pairs", n(s.bank.recent.len() as f64)),
                ("evidence_seen", n(s.bank.seen as f64)),
                ("fresh_outcomes", outcome_summary(&pairs)?),
                ("post_update_choices", selection_report(&p, &pairs)?),
                (
                    "input_health",
                    p.input_diagnostics(&pairs.iter().map(|r| r.row.clone()).collect::<Vec<_>>())?,
                ),
                ("history_outcomes", outcome_summary(&s.bank.history)?),
                ("recent_outcomes", outcome_summary(&s.bank.recent)?),
            ] {
                metric.set_path(k, v);
            }
            append(&o.out.join("metrics.jsonl"), &metric)?;
            println!("{}", metric.dump());
            if iteration % o.eval_every as u64 == 0 {
                evaluate(&o, &mut s, &mut p)?;
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
        fn state() -> State {
            let accepted = Version::initial(Portfolio::empty(), vec![0, 1, 2, 3]).unwrap();
            State {
                iteration: 0,
                rng: Rng(19),
                config: Config::default(),
                accepted: accepted.clone(),
                previous: accepted,
                bank: EvidenceBank::default(),
                next_seed: 1200,
                eval_seed: 1_100_000_000,
                next_eval_seed: 1_101_000_000,
                steps: 0,
                base_games: 0,
                branches: 0,
                evaluation_games: 0,
                monitor: Json::Null,
                pending: None,
            }
        }
        fn pair(stage: &str, seed: i64) -> Pair {
            let mut g = Game {
                report: Json::Null,
                pairs: vec![],
                comparisons: vec![],
                steps: 0,
                branches: 0,
            };
            let choices: Vec<_> = (0..2)
                .map(|i| Choice {
                    sites: vec![],
                    next: None,
                    cycles: 1,
                    lead: 24,
                    floor: 0.,
                    keep: i == 0,
                    features: vec![i as f32; 32],
                })
                .collect();
            let row = Sample {
                context: vec![0.; 320],
                features: choices.iter().map(|c| c.features.clone()).collect(),
                ..Default::default()
            };
            record_pair(
                &mut g,
                &Job {
                    seed,
                    seat: 0,
                    opponent: 0,
                    rng: 0,
                    slots: vec![0],
                },
                1,
                0,
                0,
                &row,
                &choices,
                0,
                1,
                [1., 2.],
                [2., 1.],
                Json::Obj(vec![
                    ("stage".into(), Json::Str(stage.into())),
                    ("continuation_revision".into(), Json::Str("0".into())),
                ]),
            )
            .unwrap();
            g.pairs.remove(0)
        }
        #[test]
        fn diagnostic_uses_latest_learner_without_replacing_pending_or_accepted() {
            tensor::worker_threads();
            let mut s = state();
            let p = Policy::event_plans(-1, 9, 0.0003).unwrap();
            let older = Policy::event_plans(-1, 8, 0.0003).unwrap();
            s.iteration = 20;
            s.pending = Some(Pending {
                candidate: s
                    .accepted
                    .propose(10, older.weights_json().unwrap())
                    .unwrap(),
                reference_revision: s.accepted.revision,
                look: 1,
                confirm_games: 16,
                candidate_games: vec![],
                incumbent_games: vec![],
            });
            let before = s.checkpoint(&p).unwrap();
            let c = diagnostic_candidate(&s, &p).unwrap();
            assert_eq!(c.iteration, 20);
            assert_eq!(c.weights.as_ref().unwrap(), &p.weights_json().unwrap());
            assert_ne!(c.weights, s.pending.as_ref().unwrap().candidate.weights);
            assert_eq!(before, s.checkpoint(&p).unwrap());
        }
        #[test]
        fn normalized_init_preserves_legacy_accepted_and_historical_opponent() {
            tensor::worker_threads();
            let old = Policy::plans(-1, 73, 0.0003).unwrap();
            let initial = state().accepted;
            let accepted = initial.propose(30, old.weights_json().unwrap()).unwrap();
            let j = Json::Obj(vec![
                (
                    "schema".into(),
                    Json::Str("event-policy-iteration-v6".into()),
                ),
                ("policy_contract".into(), Json::Str(CONTRACT.into())),
                ("deployment".into(), accepted.json()),
                ("previous_accepted".into(), initial.json()),
                (
                    "model".into(),
                    Json::Str("unaccepted proposal must be ignored".into()),
                ),
            ]);
            let (back, previous) = import_reference(&j).unwrap();
            assert_eq!(back, accepted);
            assert_eq!(previous, initial);
            assert!(matches!(
                back.weights.as_ref().unwrap().get("_event_input_encoding"),
                Json::Null
            ));
            let learner = Policy::event_plans(-1, 73, 0.0003).unwrap();
            assert!(learner.event_input_scaling);
        }
        #[test]
        fn reference_and_both_label_roles_survive_updates_and_resume() {
            tensor::worker_threads();
            let p = Policy::event_plans(-1, 1, 0.0003).unwrap();
            let mut s = state();
            for i in 1..=10 {
                s.iteration = i;
                s.bank.admit(&[
                    pair("arrangement", i as i64),
                    pair("revision", 100 + i as i64),
                ]);
            }
            let mut restored = Policy::event_plans(-1, 2, 0.0003).unwrap();
            let back = State::restore(&s.checkpoint(&p).unwrap(), &mut restored).unwrap();
            assert_eq!(back.accepted, s.accepted);
            assert_eq!(back.bank.json(), s.bank.json());
            assert_eq!(back.bank.recent.len(), 20);
            assert_eq!(back.bank.history.len(), 20);
            let candidate = s.accepted.propose(10, p.weights_json().unwrap()).unwrap();
            let mut invalid = candidate.clone();
            invalid.scope.push(4);
            assert!(s.promote(invalid).is_err());
            assert!(!s.bank.recent.is_empty());
            s.promote(candidate).unwrap();
            assert!(s.bank.recent.is_empty());
            assert!(s.bank.history.is_empty());
        }
        #[test]
        fn replay_membership_is_identical_when_all_outcomes_are_flipped() {
            let mut positive = EvidenceBank::default();
            let mut negative = EvidenceBank::default();
            for i in 0..3000 {
                let p = pair(
                    if i % 3 == 0 {
                        "revision"
                    } else {
                        "arrangement"
                    },
                    i,
                );
                let mut q = p.clone();
                q.target.reverse();
                q.evidence
                    .set_path("reference_cash", Json::Arr(vec![n(2.), n(1.)]));
                q.evidence
                    .set_path("alternative_cash", Json::Arr(vec![n(1.), n(2.)]));
                positive.admit(&[p]);
                negative.admit(&[q]);
            }
            assert_eq!(
                positive.history.iter().map(|p| p.seed).collect::<Vec<_>>(),
                negative.history.iter().map(|p| p.seed).collect::<Vec<_>>()
            );
            assert_eq!(
                positive.recent.iter().map(|p| p.seed).collect::<Vec<_>>(),
                negative.recent.iter().map(|p| p.seed).collect::<Vec<_>>()
            );
            assert_eq!(positive.history.len(), EvidenceBank::CAPACITY);
            let mut restored = EvidenceBank::parse(&positive.json(), 0).unwrap();
            for i in 3000..3100 {
                let p = pair("arrangement", i);
                positive.admit(&[p.clone()]);
                restored.admit(&[p]);
            }
            assert_eq!(positive.json(), restored.json());
        }
        #[test]
        fn successes_failures_and_ties_all_enter_replay_without_outcome_quota() {
            let mut bank = EvidenceBank::default();
            for i in 0..100 {
                let mut p = pair("arrangement", i);
                let delta = if i < 10 {
                    1.
                } else if i < 40 {
                    -1.
                } else {
                    0.
                };
                p.evidence
                    .set_path("reference_cash", Json::Arr(vec![n(1.), n(1.)]));
                p.evidence
                    .set_path("alternative_cash", Json::Arr(vec![n(1. + delta), n(1.)]));
                p.relabel_match_score().unwrap();
                bank.admit(&[p]);
            }
            let counts = outcome_summary(&bank.history).unwrap();
            let a = counts.get("arrangement");
            assert_eq!(a.get("better").i64(), 10);
            assert_eq!(a.get("worse").i64(), 30);
            assert_eq!(a.get("tie").i64(), 60);
        }
        #[test]
        fn current_argmax_is_unknown_until_that_choice_was_actually_tested() {
            tensor::worker_threads();
            let p = Policy::event_plans(-1, 1, 0.0003).unwrap();
            let mut measured = pair("arrangement", 7);
            let known = selection_report(&p, &[measured.clone()]).unwrap();
            assert_eq!(known.get("arrangement").get("tested").i64(), 1);
            assert_eq!(known.get("arrangement").get("better").i64(), 1);
            let mut row = Sample::parse(measured.evidence.get("full_row")).unwrap();
            let mut untested = vec![0.; 32];
            untested[30] = 3.;
            untested[31] = 1.;
            row.features.push(untested);
            measured.evidence.set_path("full_row", row.json());
            let unknown = selection_report(&p, &[measured]).unwrap();
            assert_eq!(unknown.get("arrangement").get("untested").i64(), 1);
            assert_eq!(unknown.get("arrangement").get("better").i64(), 0);
        }
        #[test]
        fn sampling_only_upgrade_does_not_silently_restore_old_winner_replay() {
            tensor::worker_threads();
            let p = Policy::event_plans(-1, 1, 0.0003).unwrap();
            let mut j = state().checkpoint(&p).unwrap();
            j.set_path("schema", Json::Str("event-policy-iteration-v6".into()));
            let mut q = Policy::event_plans(-1, 2, 0.0003).unwrap();
            assert!(State::restore(&j, &mut q).is_err());
        }
        #[test]
        fn stale_labels_and_validation_data_are_rejected() {
            let mut b = EvidenceBank::default();
            b.admit(&[pair("arrangement", 1)]);
            assert!(EvidenceBank::parse(&b.json(), 1).is_err());
            let mut b = EvidenceBank::default();
            b.admit(&[pair("revision", 1_100_000_000)]);
            assert!(EvidenceBank::parse(&b.json(), 0).is_err());
        }
        #[test]
        fn candidate_identity_and_independent_confirmation_survive_checkpoint() {
            tensor::worker_threads();
            let mut p = Policy::event_plans(-1, 1, 0.0003).unwrap();
            let mut s = state();
            let candidate = s.accepted.propose(2, p.weights_json().unwrap()).unwrap();
            s.pending = Some(Pending {
                candidate: candidate.clone(),
                reference_revision: 0,
                look: 0,
                confirm_games: 16,
                candidate_games: vec![],
                incumbent_games: vec![],
            });
            // The learner can change while the proposal is pending.
            p.load_weights(
                &Policy::event_plans(-1, 99, 0.0003)
                    .unwrap()
                    .weights_json()
                    .unwrap(),
            )
            .unwrap();
            let mut q = Policy::event_plans(-1, 3, 0.0003).unwrap();
            let back = State::restore(&s.checkpoint(&p).unwrap(), &mut q).unwrap();
            assert_eq!(back.pending.as_ref().unwrap().candidate, candidate);
            assert_ne!(
                back.pending
                    .as_ref()
                    .unwrap()
                    .candidate
                    .weights
                    .as_ref()
                    .unwrap(),
                &q.weights_json().unwrap()
            );
            assert_eq!(back.pending.unwrap().look, 0);
            let paired = eval_jobs(s.eval_seed + 1000, 16, &[0, 1]);
            let make = |gain: f64| {
                paired
                    .iter()
                    .map(|job| Game {
                        report: Json::Obj(vec![
                            ("seed".into(), n(job.seed as f64)),
                            ("seat".into(), n(job.seat as f64)),
                            ("opponent".into(), n(job.opponent as f64)),
                            ("score".into(), n(gain)),
                            ("relative_margin".into(), n(0.)),
                        ]),
                        pairs: vec![],
                        comparisons: vec![],
                        steps: 719,
                        branches: 0,
                    })
                    .collect()
            };
            let pending = s.pending.as_mut().unwrap();
            pending.look = 1;
            pending.candidate_games = make(1.);
            pending.incumbent_games = make(0.5);
            let after_look = State::restore(&s.checkpoint(&p).unwrap(), &mut q).unwrap();
            assert_eq!(after_look.pending.as_ref().unwrap().look, 1);
            assert_eq!(
                after_look.pending.as_ref().unwrap().candidate_games.len(),
                32
            );
            assert_eq!(after_look.pending.as_ref().unwrap().candidate, candidate);

            let a = s.eval_seeds(8).unwrap();
            let b = s.eval_seeds(16).unwrap();
            assert_eq!(b, a + 4);
        }
        #[test]
        fn budget_never_starts_a_branch_that_cannot_finish() {
            let b = Budget {
                steps: 1000,
                branches: 2,
            };
            assert!(!b.available(200, 1440, 4));
            assert!(b.available(300, 1440, 4));
            assert!(!b.available(718, 1440, 2));
        }
        #[test]
        fn sparse_exploration_preserves_base_game_and_uses_one_continuation() {
            tensor::worker_threads();
            let s = state();
            let runtime = s.accepted.runtime(-1).unwrap();
            let p = Policy::event_plans(-1, 2, 0.0003).unwrap();
            let job = Job {
                seed: 1201,
                seat: 0,
                opponent: 0,
                rng: 7,
                slots: vec![0, 1],
            };
            let base = play(
                &job,
                &runtime,
                None,
                &[],
                &s.config,
                1,
                0,
                &s.accepted.scope,
                0,
                0,
                0,
            )
            .unwrap();
            let explored = play(
                &job,
                &runtime,
                Some(&p),
                &[],
                &s.config,
                1,
                0,
                &s.accepted.scope,
                2,
                1440,
                4,
            )
            .unwrap();
            assert_eq!(base.report, explored.report);
            assert!(!explored.pairs.is_empty());
            assert!(explored.steps <= 719 + 1440);
            assert!(explored.branches <= 4);
            assert!(
                explored
                    .pairs
                    .iter()
                    .filter(|p| p.evidence.get("source").str() == "conditional_exploration")
                    .count()
                    <= 1
            );
            for pair in &explored.pairs {
                assert_eq!(pair.evidence.get("continuation_revision").str(), "0");
                assert!(s
                    .accepted
                    .scope
                    .contains(&(pair.evidence.get("slot_id").i64() as usize)));
                assert_eq!(
                    pair.evidence.get("objective").str(),
                    plan_compare::MATCH_SCORE_OBJECTIVE
                );
            }
        }
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
            let good = vec![fake(1, 0, 1., -0.1), fake(2, 0, 0.5, -0.2)];
            assert_eq!(gate(&good, &b).unwrap().get("qualifies"), &Json::Bool(true));
            let bad = vec![fake(1, 0, 0., 0.02), fake(2, 0, 0., 0.02)];
            assert_eq!(gate(&bad, &b).unwrap().get("qualifies"), &Json::Bool(false));
            let bad_key = vec![fake(3, 0, 1., 0.2), fake(2, 0, 1., 0.2)];
            assert!(gate(&bad_key, &b).is_err());
        }
        #[test]
        fn gate_uses_only_match_score_with_cash_and_seed_signs_as_diagnostics() {
            let b = vec![
                fake(1, 0, 0., 0.),
                fake(1, 1, 0., 0.),
                fake(2, 0, 1., 0.9),
                fake(3, 0, 1., 0.9),
            ];
            let c = vec![
                fake(1, 0, 1., 0.001),
                fake(1, 1, 1., 0.001),
                fake(2, 0, 0.5, 0.),
                fake(3, 0, 0.5, 0.),
            ];
            let g = gate(&c, &b).unwrap();
            assert_eq!(g.get("positive_seeds").i64(), 1);
            assert_eq!(g.get("negative_seeds").i64(), 2);
            assert!(g.get("mean_relative_margin_gain").f64() < 0.);
            assert_eq!(g.get("qualifies"), &Json::Bool(true));
            let cash_only = vec![
                fake(1, 0, 0., 0.1),
                fake(1, 1, 0., 0.1),
                fake(2, 0, 1., 0.99),
                fake(3, 0, 1., 0.99),
            ];
            assert_eq!(
                gate(&cash_only, &b).unwrap().get("qualifies"),
                &Json::Bool(false)
            );
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
