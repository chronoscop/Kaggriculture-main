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
            event_policy::{
                Version, BATCH_CONTRACT, COLLECTION_CONTRACT, CONTRACT, MENU_BATCH_CONTRACT,
                MENU_CONTRACT, NORMALIZED_SCHEMA, PREFIX_SCHEMA, SCHEMA,
            },
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
        warm_start: Option<String>,
        batch_lifetime: Option<bool>,
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
                warm_start: None,
                batch_lifetime: None,
                foundation: None,
                config: "native/configs/plan_prototype_v1.json".into(),
                iterations: 100,
                games: 16,
                workers: 7,
                points: 2,
                alternatives: 2,
                branch_steps: 4320,
                max_branches: 8,
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
                println!("event-train --out NEW_DIR [--init-from accepted_event.json | --base-checkpoint v3_best.json | --resume v9_latest.json] --iterations 100 --games-per-update 16 --workers 7 --device cuda|cpu --branch-points 2 --alternatives 2 --branch-steps-per-game 4320 --max-branches-per-game 8 --epochs 8 --batch-size 64 --learning-rate 0.0003 --eval-every 5 --eval-games 8 --confirm-games 16\nOne shared network, fixed accepted continuation until promotion. Default scope unchanged: four seasonal harvest arrangements and one same-batch revision each. Outcome-independent replay. Independent confirmation progresses at the next scheduled evaluations, preserving the exact frozen candidate. iterations are additional; --warm-start v9_latest.json preserves normalized proposal weights but resets Adam, labels and confirmation. --init-from preserves deployment (and accepted normalized weights after v8 confirmation). --followup-scope single|batch: batch requires prior v8 independent acceptance; resume cannot change scope. Two base trajectories per training condition; local and same-prefix segment arms share accepted continuation; full candidate sequence outcomes never label individual choices.");
                println!("v9 complete menus: 2–4 deterministic executable choices, all terminal arms compared atomically. Replay/batches use complete sets. Legacy init-from preserves accepted actions and starts a fresh menu learner. --alternatives is unused for complete menus; branch budgets still apply. Scope expansion is disabled. observations.jsonl reports the latest learner against fixed opponents/seeds; independent promotion never uses this panel.");
                println!("Read-only diagnostic: --evaluate-checkpoint v8_latest.json --out NEW_DIR --eval-games 64 --eval-seed NEW_SEED --workers 7 --device cpu. Evaluates the latest learner, ignores pending candidates, performs no training or promotion; eval-games counts both seats per opponent (64 = 32 seeds). ");
                println!("Mechanism diagnostic: --diagnose-revisions v8_latest.json --out NEW_DIR --eval-games 8 --eval-seed SEED --workers 7 --device cpu. Compares accepted, new arrangements only, new revisions only, and both; tests changed revisions individually under accepted continuation. Maximum 16 games per opponent (8 seeds). Writes traces and comparisons only; never trains or promotes.");
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
                    "--warm-start" => o.warm_start = Some(v.clone()),
                    "--followup-scope" => {
                        o.batch_lifetime = Some(match v.as_str() {
                            "single" => false,
                            "batch" => true,
                            _ => return Err("followup-scope: single or batch".into()),
                        })
                    }
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
                    o.warm_start.is_some(),
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
        sequence: Option<Json>,
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
            ("collection_contract", Json::Str(COLLECTION_CONTRACT.into())),
            (
                "continuation_contract",
                if evidence.get("continuation_contract").str().is_empty() {
                    Json::Str(CONTRACT.into())
                } else {
                    evidence.get("continuation_contract").clone()
                },
            ),
            (
                "source_snapshot",
                Json::Str(format!("collection_{iteration:06}")),
            ),
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
            sequence: None,
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
    /// Candidate state distribution is collected with the exact deployment runtime.
    /// Local targets instead share the stable accepted suffix policy. In particular,
    /// the candidate's terminal result is NEVER reused as a local reference label.
    fn candidate_trajectory(
        job: &Job,
        candidate: &Runtime,
        learner: &Policy,
        opponents: &[Runtime],
        config: &Config,
        scope: &[usize],
        source_id: &str,
    ) -> Result<(World, Vec<Fork>, Vec<Json>), String> {
        let mut world = World::new(job.seed, config);
        let mut forks = vec![];
        let mut trace = vec![];
        while world.game.step < 719 {
            let obs = Observation::from_state(&world.game, job.seat);
            let action = if let Some(mut d) = world.own.prepare(&obs, candidate)? {
                if d.slot.is_some_and(|s| scope.contains(&s)) {
                    let prefix = format!(
                        "{source_id}:{}:{}:{}:{}",
                        job.seed,
                        job.seat,
                        job.opponent,
                        trace.len()
                    );
                    forks.push(snapshot(&world, &d, learner, prefix.clone())?);
                    trace.push(Json::Obj(vec![
                        ("prefix_id".into(), Json::Str(prefix)),
                        ("step".into(), n(obs.step as f64)),
                        ("slot".into(), n(d.slot.unwrap() as f64)),
                        ("followup".into(), Json::Bool(d.followup)),
                        ("legal_candidates".into(), n(d.choices.len() as f64)),
                        (
                            "production_candidates".into(),
                            n(d.choices.iter().filter(|c| c.next.is_some()).count() as f64),
                        ),
                        ("selected".into(), n(d.selected as f64)),
                        ("plan".into(), d.choices[d.selected].json()),
                        (
                            "event".into(),
                            world.own.last_event.as_ref().unwrap().json(),
                        ),
                    ]));
                }
                world
                    .own
                    .execute_choice(&obs, d.choices.swap_remove(d.selected), candidate)?
            } else {
                world.own.continue_action(&obs, candidate)?
            };
            world.advance(job, action, opponents)?;
        }
        world
            .own
            .controller
            .observe(&Observation::from_state(&world.game, job.seat));
        Ok((world, forks, trace))
    }
    /// Change a bounded same-batch segment. Other scopes and the suffix after
    /// its last editable event use the SAME accepted runtime as both local arms.
    fn rollout_segment(
        f: &Fork,
        job: &Job,
        accepted: &Runtime,
        candidate: &Runtime,
        learner: &Policy,
        opponents: &[Runtime],
    ) -> Result<(World, Vec<Json>), String> {
        let mut world = f.world.clone();
        let mut edits = vec![];
        let obs = Observation::from_state(&world.game, job.seat);
        let action = world
            .own
            .execute_choice(&obs, f.choices[f.proposed].clone(), candidate)?;
        world.advance(job, action, opponents)?;
        while world.game.step < 719 {
            let obs = Observation::from_state(&world.game, job.seat);
            let action = if let Some(mut d) = world.own.prepare(&obs, accepted)? {
                let same_batch = d.followup && d.slot == Some(f.slot);
                let runtime = if same_batch { candidate } else { accepted };
                if same_batch {
                    d.selected = learner.infer(&[d.row.clone()], true, &mut Rng(0))?[0].action;
                    edits.push(Json::Obj(vec![
                        ("step".into(), n(obs.step as f64)),
                        (
                            "event".into(),
                            world.own.last_event.as_ref().unwrap().json(),
                        ),
                        ("selected".into(), d.choices[d.selected].json()),
                        ("legal_candidates".into(), n(d.choices.len() as f64)),
                    ]));
                }
                world
                    .own
                    .execute_choice(&obs, d.choices.swap_remove(d.selected), runtime)?
            } else {
                world.own.continue_action(&obs, accepted)?
            };
            world.advance(job, action, opponents)?;
        }
        world
            .own
            .controller
            .observe(&Observation::from_state(&world.game, job.seat));
        Ok((world, edits))
    }
    /// Execute the actual incumbent/candidate prefix. Both use the versioned
    /// menu; before the first promotion its first choice is the exact old policy.
    fn menu_trajectory(
        job: &Job,
        accepted: &Runtime,
        candidate: &Runtime,
        learner: &Policy,
        opponents: &[Runtime],
        config: &Config,
        scope: &[usize],
        source: &str,
        use_candidate: bool,
    ) -> Result<(World, Vec<Fork>), String> {
        let mut world = World::new(job.seed, config);
        let mut forks = vec![];
        let runtime = if use_candidate { candidate } else { accepted };
        while world.game.step < 719 {
            let obs = Observation::from_state(&world.game, job.seat);
            let action = if let Some(mut d) = world.own.prepare(&obs, candidate)? {
                if d.slot.is_some_and(|s| scope.contains(&s)) {
                    if !use_candidate {
                        d.selected = if accepted.menu_reference.is_some() {
                            accepted.select(d.slot, &d.row)?
                        } else {
                            0
                        };
                    }
                    if d.choices.len() > 1 {
                        forks.push(snapshot(
                            &world,
                            &d,
                            learner,
                            format!(
                                "{source}:{}:{}:{}:{}",
                                job.seed, job.seat, job.opponent, obs.step
                            ),
                        )?);
                    }
                }
                world
                    .own
                    .execute_choice(&obs, d.choices.swap_remove(d.selected), runtime)?
            } else {
                world.own.continue_action(&obs, runtime)?
            };
            world.advance(job, action, opponents)?;
        }
        world
            .own
            .controller
            .observe(&Observation::from_state(&world.game, job.seat));
        Ok((world, forks))
    }
    fn play_menu_training(
        job: &Job,
        accepted: &Runtime,
        candidate: &Runtime,
        learner: &Policy,
        opponents: &[Runtime],
        config: &Config,
        iteration: u64,
        revision: u64,
        scope: &[usize],
        step_limit: usize,
        count_limit: usize,
    ) -> Result<Game, String> {
        // Lifetime expansion requires its own accepted scope; no legacy-to-menu
        // migration is allowed to quietly expand the data collection scope.
        if accepted.batch_lifetime != candidate.batch_lifetime {
            return Err("complete-set training requires the accepted follow-up scope; do not expand scope during migration".into());
        }
        let (aw, af) = menu_trajectory(
            job,
            accepted,
            candidate,
            learner,
            opponents,
            config,
            scope,
            &format!("collection_{iteration:06}:accepted"),
            false,
        )?;
        let (cw, cf) = menu_trajectory(
            job,
            accepted,
            candidate,
            learner,
            opponents,
            config,
            scope,
            &format!("collection_{iteration:06}:candidate"),
            true,
        )?;
        let mut result = Game {
            report: aw.result(job),
            pairs: vec![],
            comparisons: vec![],
            sequence: None,
            steps: 1438,
            branches: 0,
        };
        let mut budget = Budget::default();
        let mut sets = vec![];
        let rotation = (job.seed as usize).wrapping_add(iteration as usize);
        let accepted_ids = select_forks(&af, &job.slots, rotation);
        let mut candidate_ids = select_forks(&cf, &job.slots, rotation + 1);
        // With two sets, cover an accepted arrangement and a real candidate
        // follow-up, rather than sampling the first arrangement twice.
        if let Some(at) = candidate_ids.iter().position(|i| cf[*i].followup) {
            candidate_ids.swap(0, at);
        }

        let mut queue = vec![];
        for i in 0..accepted_ids.len().max(candidate_ids.len()) {
            let a = accepted_ids
                .get(i)
                .map(|j| (&af[*j], "accepted_trajectory"));
            let c = candidate_ids.get(i).map(|j| (&cf[*j], "candidate_prefix"));
            for item in if rotation % 2 == 0 { [a, c] } else { [c, a] } {
                if let Some(v) = item {
                    queue.push(v);
                }
            }
        }
        for (f, source) in queue {
            if sets.len() >= job.slots.len() {
                break;
            }
            let arms = f.choices.len();
            let cost = (719 - f.row.step) as usize;
            // Atomic reservation: never train an incompletely compared menu.
            if budget.branches + arms > count_limit || budget.steps + arms * cost > step_limit {
                continue;
            }
            let mut outcomes = vec![];
            for j in 0..arms {
                let (end, _) = rollout_edit(f, j, job, accepted, opponents, None, scope)?;
                outcomes.push(end.cash(job.seat));
            }
            budget.branches += arms;
            budget.steps += arms * cost;
            let mut ev = evidence(f, source, revision, true);
            for (k, v) in [
                (
                    "continuation_contract",
                    Json::Str(
                        if accepted.menu_reference.is_some() {
                            if accepted.batch_lifetime {
                                MENU_BATCH_CONTRACT
                            } else {
                                MENU_CONTRACT
                            }
                        } else if accepted.batch_lifetime {
                            BATCH_CONTRACT
                        } else {
                            CONTRACT
                        }
                        .into(),
                    ),
                ),
                ("candidate_set_id", Json::Str(f.prefix.clone())),
                (
                    "candidate_encoding",
                    Json::Str(route_rl_native::pipeline::plan_menu::ENCODING.into()),
                ),
                (
                    "all_terminal_cash",
                    Json::Arr(
                        outcomes
                            .iter()
                            .map(|c| Json::Arr(c.map(n).to_vec()))
                            .collect(),
                    ),
                ),
                (
                    "candidate_plans",
                    Json::Arr(f.choices.iter().map(Choice::json).collect()),
                ),
                ("terminal_step", n(719.)),
                ("rollin_selected", n(f.reference as f64)),
                ("learner_selected", n(f.proposed as f64)),
                (
                    "scope_contract",
                    Json::Str(
                        if candidate.batch_lifetime {
                            MENU_BATCH_CONTRACT
                        } else {
                            MENU_CONTRACT
                        }
                        .into(),
                    ),
                ),
            ] {
                ev.set_path(k, v);
            }
            for j in 1..arms {
                record_pair(
                    &mut result,
                    job,
                    iteration,
                    revision,
                    f.slot,
                    &f.row,
                    &f.choices,
                    0,
                    j,
                    outcomes[0],
                    outcomes[j],
                    ev.clone(),
                )?;
            }
            sets.push(ev);
        }
        result.steps += budget.steps;
        result.branches = budget.branches;
        result.sequence = Some(Json::Obj(vec![
            ("iteration".into(), n(iteration as f64)),
            ("seed".into(), n(job.seed as f64)),
            ("seat".into(), n(job.seat as f64)),
            ("opponent".into(), n(job.opponent as f64)),
            ("accepted_outcome".into(), aw.result(job)),
            ("candidate_outcome".into(), cw.result(job)),
            ("complete_sets".into(), Json::Arr(sets)),
            ("accepted_events".into(), n(af.len() as f64)),
            ("candidate_events".into(), n(cf.len() as f64)),
            (
                "collection_contract".into(),
                Json::Str(COLLECTION_CONTRACT.into()),
            ),
        ]));
        Ok(result)
    }
    fn play_training(
        job: &Job,
        accepted: &Runtime,
        candidate: &Runtime,
        learner: &Policy,
        opponents: &[Runtime],
        config: &Config,
        iteration: u64,
        revision: u64,
        scope: &[usize],
        alt_count: usize,
        step_limit: usize,
        count_limit: usize,
    ) -> Result<Game, String> {
        if candidate.menu_reference.is_some() {
            return play_menu_training(
                job,
                accepted,
                candidate,
                learner,
                opponents,
                config,
                iteration,
                revision,
                scope,
                step_limit,
                count_limit,
            );
        }
        let source_id = format!("collection_{iteration:06}");
        let (world, forks, trace) = candidate_trajectory(
            job, candidate, learner, opponents, config, scope, &source_id,
        )?;
        // Reserve an entire two-suffix comparison before starting either arm.
        // Source priority rotates by seed (both seats use the same priority),
        // independent of outcomes. Limited budgets cannot erase champion evidence.
        let candidate_first = (job.seed as u64 + iteration) % 2 == 0;
        let accepted_steps = if candidate_first { 0 } else { step_limit / 2 };
        let accepted_count = if candidate_first { 0 } else { count_limit / 2 };
        let mut result = if candidate_first {
            Game {
                report: Json::Null,
                pairs: vec![],
                comparisons: vec![],
                sequence: None,
                steps: 719,
                branches: 0,
            }
        } else {
            play(
                job,
                accepted,
                Some(learner),
                opponents,
                config,
                iteration,
                revision,
                scope,
                alt_count,
                accepted_steps,
                accepted_count,
            )?
        };
        let mut budget = Budget {
            steps: result.steps - 719,
            branches: result.branches,
        };
        let mut ids: Vec<_> = (0..forks.len()).collect();
        // Rotate among actual follow-ups, including later events of an owned
        // batch. No selection depends on the branch's eventual win/loss.
        if !ids.is_empty() {
            let rotate = (job.seed as usize + iteration as usize) % ids.len();
            ids.rotate_left(rotate);
            ids.sort_by_key(|i| (!forks[*i].followup, forks[*i].proposed == 0));
            // One revision and one arrangement when both exist. This keeps a
            // first-arrangement + follow-up segment measurable in the single stage.
            if job.slots.len() > 1 && forks[ids[0]].followup {
                if let Some(at) = ids.iter().position(|i| !forks[*i].followup) {
                    let arrangement = ids.remove(at);
                    ids.insert(1, arrangement);
                }
            }
        }
        let mut segments = vec![];
        let mut rng = Rng(job.rng ^ 0x645cdf12);
        let mut tested = 0;
        for i in ids {
            if tested >= job.slots.len() {
                break;
            }
            let mut f = forks[i].clone();
            f.reference = 0; // Explicit Keep, not the candidate's selected revision.
            let cost = (719 - f.row.step) as usize;
            let needs_segment = (f.followup && candidate.batch_lifetime)
                || (!f.followup && f.proposed != 0 && f.choices[f.proposed].next.is_some());
            let arms = if needs_segment { 3 } else { 2 };
            if budget.branches + arms > count_limit || budget.steps + arms * cost > step_limit {
                continue;
            }
            let Some(alternative) = alternatives(&f, 1, &mut rng).first().copied() else {
                continue;
            };
            let (keep, _) = rollout_edit(&f, 0, job, accepted, opponents, None, scope)?;
            let (changed, _) =
                rollout_edit(&f, alternative, job, accepted, opponents, None, scope)?;
            let (segment_cash, segment_events) = if needs_segment {
                let (segment, events) =
                    rollout_segment(&f, job, accepted, candidate, learner, opponents)?;
                (segment.cash(job.seat), events)
            } else {
                // No later owned edit can occur. Reuse an identical completed arm.
                (
                    if f.proposed == 0 {
                        keep.cash(job.seat)
                    } else {
                        changed.cash(job.seat)
                    },
                    vec![],
                )
            };
            budget.steps += arms * cost;
            budget.branches += arms;
            tested += 1;
            segments.push(Json::Obj(vec![
                ("prefix_id".into(), Json::Str(f.prefix.clone())),
                ("slot".into(), n(f.slot as f64)),
                ("step".into(), n(f.row.step as f64)),
                (
                    "event".into(),
                    f.world.own.last_event.as_ref().unwrap().json(),
                ),
                ("candidate_first_index".into(), n(f.proposed as f64)),
                (
                    "reference_cash".into(),
                    Json::Arr(keep.cash(job.seat).map(n).to_vec()),
                ),
                (
                    "single_edit_cash".into(),
                    Json::Arr(changed.cash(job.seat).map(n).to_vec()),
                ),
                ("single_edit_index".into(), n(alternative as f64)),
                (
                    "segment_cash".into(),
                    Json::Arr(segment_cash.map(n).to_vec()),
                ),
                (
                    "segment_score_gain".into(),
                    n(score(segment_cash) - score(keep.cash(job.seat))),
                ),
                ("later_segment_events".into(), Json::Arr(segment_events)),
                (
                    "continuation_revision".into(),
                    Json::Str(revision.to_string()),
                ),
                (
                    "continuation_contract".into(),
                    Json::Str(
                        if accepted.batch_lifetime {
                            BATCH_CONTRACT
                        } else {
                            CONTRACT
                        }
                        .into(),
                    ),
                ),
                ("terminal_step".into(), n(719.)),
                ("reused_local_arm".into(), Json::Bool(!needs_segment)),
                ("local_target".into(), Json::Bool(false)),
            ]));
            let mut ev = evidence(
                &f,
                "candidate_prefix",
                revision,
                f.proposed == 0 || alternative == f.proposed,
            );
            ev.set_path("reference_execution", keep.own.controller.report());
            ev.set_path("execution", changed.own.controller.report());
            ev.set_path("candidate_selected_index", n(f.proposed as f64));
            ev.set_path("legal_candidate_count", n(f.choices.len() as f64));
            ev.set_path(
                "production_candidate_count",
                n(f.choices.iter().filter(|c| c.next.is_some()).count() as f64),
            );
            ev.set_path("reference_terminal_step", n(keep.game.step as f64));
            ev.set_path("alternative_terminal_step", n(changed.game.step as f64));
            record_pair(
                &mut result,
                job,
                iteration,
                revision,
                f.slot,
                &f.row,
                &f.choices,
                0,
                alternative,
                keep.cash(job.seat),
                changed.cash(job.seat),
                ev,
            )?;
        }
        if candidate_first {
            // Spend the remaining budget on champion states. Each source runs
            // exactly one full base trajectory, in addition to counted suffixes.
            let extra = play(
                job,
                accepted,
                Some(learner),
                opponents,
                config,
                iteration,
                revision,
                scope,
                alt_count,
                step_limit - budget.steps,
                count_limit - budget.branches,
            )?;
            budget.steps += extra.steps - 719;
            budget.branches += extra.branches;
            result.report = extra.report;
            result.pairs.extend(extra.pairs);
            result.comparisons.extend(extra.comparisons);
        }
        let sequence = Json::Obj(vec![
            (
                "collection_contract".into(),
                Json::Str(COLLECTION_CONTRACT.into()),
            ),
            ("source_snapshot".into(), Json::Str(source_id.clone())),
            ("iteration".into(), n(iteration as f64)),
            ("seed".into(), n(job.seed as f64)),
            ("seat".into(), n(job.seat as f64)),
            ("opponent".into(), n(job.opponent as f64)),
            (
                "candidate_contract".into(),
                Json::Str(
                    if candidate.batch_lifetime {
                        BATCH_CONTRACT
                    } else {
                        CONTRACT
                    }
                    .into(),
                ),
            ),
            ("candidate_events".into(), Json::Arr(trace)),
            ("same_prefix_segments".into(), Json::Arr(segments)),
            ("candidate_outcome".into(), world.result(job)),
            ("accepted_outcome".into(), result.report.clone()),
            ("terminal_step".into(), n(world.game.step as f64)),
            (
                "score_gain".into(),
                n(score(world.cash(job.seat)) - result.report.get("score").f64()),
            ),
            ("local_target".into(), Json::Bool(false)),
            ("promotion_evidence".into(), Json::Bool(false)),
            ("local_candidate_pairs".into(), n(tested as f64)),
        ]);
        result.sequence = Some(sequence);
        for p in &mut result.pairs {
            p.evidence
                .set_path("source_snapshot", Json::Str(source_id.clone()));
            p.evidence.set_path(
                "continuation_contract",
                Json::Str(
                    if accepted.batch_lifetime {
                        BATCH_CONTRACT
                    } else {
                        CONTRACT
                    }
                    .into(),
                ),
            );
        }
        for e in &mut result.comparisons {
            e.set_path("source_snapshot", Json::Str(source_id.clone()));
            e.set_path(
                "continuation_contract",
                Json::Str(
                    if accepted.batch_lifetime {
                        BATCH_CONTRACT
                    } else {
                        CONTRACT
                    }
                    .into(),
                ),
            );
        }
        result.steps = 1438 + budget.steps;
        result.branches = budget.branches;
        Ok(result)
    }
    fn collect(
        jobs: Vec<Job>,
        accepted: &Version,
        learner: Option<&Version>,
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
                    let candidate = lw.as_ref().map(|v| v.runtime(-1)).transpose()?;
                    let lp = if let Some(v) = &lw {
                        let w = v.weights.as_ref().ok_or("candidate missing weights")?;
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
                        let game = if let (Some(candidate), Some(learner)) = (&candidate, &lp) {
                            play_training(
                                &js[i],
                                &runtime,
                                candidate,
                                learner,
                                &opponents,
                                &c,
                                iteration,
                                version.revision,
                                &version.scope,
                                alt_count,
                                step_limit,
                                count_limit,
                            )?
                        } else {
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
                            )?
                        };
                        result.push((i, game));
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
            let mut set_ids = std::collections::BTreeSet::new();
            for pair in pairs {
                let id = pair.evidence.get("candidate_set_id").str();
                if !id.is_empty() && !set_ids.insert(id.to_string()) {
                    continue;
                }
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
                if !p.evidence.get("candidate_set_id").str().is_empty() {
                    route_rl_native::learning::event_sets::validate(p)?;
                }
                if p.incumbent_revision()? != version
                    || p.evidence.get("continuation_revision").str() != version.to_string()
                    || p.evidence.get("policy_contract").str() != CONTRACT
                    || p.evidence.get("collection_contract").str() != COLLECTION_CONTRACT
                    || !matches!(
                        p.evidence.get("continuation_contract").str(),
                        CONTRACT | BATCH_CONTRACT | MENU_CONTRACT | MENU_BATCH_CONTRACT
                    )
                    || p.evidence.get("source_snapshot").str().is_empty()
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
    fn candidate_coverage(gs: &[Game]) -> Json {
        let local: Vec<_> = gs
            .iter()
            .flat_map(|g| &g.pairs)
            .filter(|p| {
                p.evidence.get("source").str() == "candidate_prefix"
                    && p.evidence.get("stage").str() == "revision"
            })
            .collect();
        let sizes: Vec<_> = local
            .iter()
            .map(|p| p.evidence.get("legal_candidate_count").i64())
            .collect();
        let sequences: Vec<_> = gs.iter().filter_map(|g| g.sequence.as_ref()).collect();
        Json::Obj(vec![
            ("revision_states_tested".into(), n(local.len() as f64)),
            (
                "legal_candidates_min".into(),
                sizes
                    .iter()
                    .min()
                    .map(|x| n(*x as f64))
                    .unwrap_or(Json::Null),
            ),
            (
                "legal_candidates_max".into(),
                sizes
                    .iter()
                    .max()
                    .map(|x| n(*x as f64))
                    .unwrap_or(Json::Null),
            ),
            (
                "keep_cancel_only".into(),
                n(local
                    .iter()
                    .filter(|p| p.evidence.get("production_candidate_count").i64() == 0)
                    .count() as f64),
            ),
            (
                "sequences_without_revision".into(),
                n(sequences
                    .iter()
                    .filter(|s| {
                        !s.get("candidate_events")
                            .arr()
                            .iter()
                            .any(|e| matches!(e.get("followup"), Json::Bool(true)))
                    })
                    .count() as f64),
            ),
            (
                "segments_with_later_events".into(),
                n(sequences
                    .iter()
                    .flat_map(|s| s.get("same_prefix_segments").arr())
                    .filter(|s| !s.get("later_segment_events").arr().is_empty())
                    .count() as f64),
            ),
            (
                "training_sequence_score_gain".into(),
                n(sequences
                    .iter()
                    .map(|s| s.get("score_gain").f64())
                    .sum::<f64>()
                    / sequences.len().max(1) as f64),
            ),
        ])
    }
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
                sequence: None,
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
        target_batch_lifetime: bool,
        menu_learning: bool,
        scope_confirmed: bool,
    }
    impl State {
        fn proposal(&self, iteration: u64, weights: Json) -> Result<Version, String> {
            let mut candidate = self.accepted.propose(iteration, weights)?;
            candidate.batch_lifetime = self.target_batch_lifetime;
            if self.menu_learning && candidate.menu_anchor.is_none() {
                candidate.menu_anchor = Some(Box::new(self.accepted.clone()));
            }
            self.accepted.validate_successor(&candidate)?;
            Ok(candidate)
        }
        fn checkpoint(&self, p: &Policy) -> Result<Json, String> {
            Ok(Json::Obj(vec![
                ("schema".into(), Json::Str(SCHEMA.into())),
                ("menu_learning".into(), Json::Bool(self.menu_learning)),
                (
                    "candidate_encoding".into(),
                    Json::Str(route_rl_native::pipeline::plan_menu::ENCODING.into()),
                ),
                (
                    "collection_contract".into(),
                    Json::Str(COLLECTION_CONTRACT.into()),
                ),
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
                (
                    "target_batch_lifetime".into(),
                    Json::Bool(self.target_batch_lifetime),
                ),
                ("scope_confirmed".into(), Json::Bool(self.scope_confirmed)),
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
                || j.get("collection_contract").str() != COLLECTION_CONTRACT
                || j.get("policy_contract").str() != CONTRACT
                || j.get("objective").str() != plan_compare::MATCH_SCORE_OBJECTIVE
            {
                return Err("--resume requires full event-policy-iteration-v9 checkpoint; use --init-from to preserve accepted deployment and invalidate old labels/Adam".into());
            }
            if matches!(j.get("menu_learning"), Json::Bool(true))
                && j.get("candidate_encoding").str()
                    != route_rl_native::pipeline::plan_menu::ENCODING
            {
                return Err("incompatible candidate encoding".into());
            }
            let (iteration, rng) = p.restore(j.get("model"))?;
            if iteration != uint(j, "iteration")? || !p.plan_residual || !p.event_input_scaling {
                return Err("model iteration/architecture mismatch".into());
            }
            let accepted = Version::parse(j.get("deployment"))?;
            let previous = Version::parse(j.get("previous_accepted"))?;
            if previous.revision > accepted.revision
                || previous.scope != accepted.scope
                || previous.foundation != accepted.foundation
            {
                return Err("invalid previous accepted strategy".into());
            }
            let bank = EvidenceBank::parse(j.get("comparison_bank"), accepted.revision)?;
            if bank
                .history
                .iter()
                .chain(&bank.recent)
                .any(|p| p.evidence.get("continuation_contract").str() != accepted.contract())
            {
                return Err("cached labels use a different continuation execution contract".into());
            }
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
                menu_learning: matches!(j.get("menu_learning"), Json::Bool(true)),
                target_batch_lifetime: matches!(j.get("target_batch_lifetime"), Json::Bool(true)),
                scope_confirmed: matches!(j.get("scope_confirmed"), Json::Bool(true)),
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
            if out.menu_learning
                && out
                    .bank
                    .history
                    .iter()
                    .chain(&out.bank.recent)
                    .any(|p| p.evidence.get("candidate_set_id").str().is_empty())
            {
                return Err("menu replay requires complete sets".into());
            }
            if out.accepted.batch_lifetime && !out.target_batch_lifetime
                || out
                    .pending
                    .as_ref()
                    .is_some_and(|p| p.candidate.batch_lifetime != out.target_batch_lifetime)
                || out.next_seed >= 1_000_000_000
                || out.eval_seed < 1_000_000_000
                || out.next_eval_seed < out.eval_seed + 1_000_000
            {
                return Err("training/evaluation seed domains overlap".into());
            }
            Ok(out)
        }
        fn promote(&mut self, candidate: Version) -> Result<(), String> {
            self.accepted.validate_successor(&candidate)?;
            if candidate.batch_lifetime != self.target_batch_lifetime {
                return Err("promotion must use the collected and evaluated responsibility".into());
            }
            self.previous = self.accepted.clone();
            self.accepted = candidate;
            self.scope_confirmed = true;
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
    fn observe_latest(o: &Options, s: &mut State, p: &Policy) -> Result<(), String> {
        if !s.menu_learning {
            return Ok(());
        }
        let candidate = s.proposal(s.iteration, p.weights_json()?)?;
        let anchor = candidate
            .menu_anchor
            .as_deref()
            .ok_or("missing menu anchor")?;
        let js = eval_jobs(s.eval_seed, o.eval_games, &[0, 1]);
        let cache = o.out.join("observation_reference.json");
        let key = Json::Obj(vec![
            ("anchor".into(), anchor.json()),
            ("config".into(), s.config.json()),
            ("seed".into(), n(s.eval_seed as f64)),
            ("games".into(), n(o.eval_games as f64)),
        ]);
        let saved = if cache.exists() {
            read(cache.to_str().unwrap())?
        } else {
            Json::Null
        };
        let reference = if saved.get("key") == &key {
            saved.get("summary").clone()
        } else {
            let gs = collect(
                js.clone(),
                anchor,
                None,
                &[anchor.clone()],
                &s.config,
                o.workers,
                s.iteration,
                0,
                0,
                0,
            )?;
            s.evaluation_games += gs.len() as u64;
            let summary = summary(&gs);
            write(
                &cache,
                &Json::Obj(vec![
                    ("key".into(), key),
                    ("summary".into(), summary.clone()),
                ]),
            )?;
            summary
        };
        let gs = collect(
            js,
            &candidate,
            None,
            &[anchor.clone()],
            &s.config,
            o.workers,
            s.iteration,
            0,
            0,
            0,
        )?;
        s.evaluation_games += gs.len() as u64;
        let current = summary(&gs);
        let row = Json::Obj(vec![
            ("iteration".into(), n(s.iteration as f64)),
            ("latest_candidate".into(), current.clone()),
            ("fixed_reference".into(), reference.clone()),
            (
                "mean_score_gain".into(),
                n(current.get("score_rate").f64() - reference.get("score_rate").f64()),
            ),
            ("used_for_promotion".into(), Json::Bool(false)),
        ]);
        append(&o.out.join("observations.jsonl"), &row)?;
        println!("observation {}", row.dump());
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
            let candidate = s.proposal(s.iteration, p.weights_json()?)?;
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
            "event-policy-iteration-v5"
                | "event-policy-iteration-v6"
                | NORMALIZED_SCHEMA
                | PREFIX_SCHEMA
                | SCHEMA
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
                sequence: None,
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
        if s.menu_learning {
            return Err("legacy intervention diagnostic does not support complete menus; use evaluate-checkpoint".into());
        }
        let js = Arc::new(eval_jobs(
            o.eval_seed,
            o.eval_games,
            &roster(&s.accepted, &s.previous),
        ));
        std::fs::create_dir_all(&o.out).map_err(|e| e.to_string())?;
        if o.batch_lifetime
            .is_some_and(|mode| mode != s.target_batch_lifetime)
        {
            return Err(
                "resume cannot change responsibility; use init-from after independent acceptance"
                    .into(),
            );
        }
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
                                    sequence: None,
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
    // Read-only compatibility does not convert old labels into active v8 replay.
    fn diagnostic_state(j: &Json, p: &mut Policy) -> Result<State, String> {
        if j.get("schema").str() == SCHEMA {
            return State::restore(j, p);
        }
        if !matches!(j.get("schema").str(), NORMALIZED_SCHEMA | PREFIX_SCHEMA)
            || j.get("policy_contract").str() != CONTRACT
            || j.get("objective").str() != plan_compare::MATCH_SCORE_OBJECTIVE
        {
            return Err("diagnostic requires v7/v8 normalized checkpoint".into());
        }
        let (iteration, rng) = p.restore(j.get("model"))?;
        if iteration != uint(j, "iteration")? || !p.event_input_scaling {
            return Err("diagnostic model/iteration mismatch".into());
        }
        let (accepted, previous) = import_reference(j)?;
        Ok(State {
            iteration,
            rng,
            config: Config::parse(j.get("config"))?,
            menu_learning: accepted.menu_anchor.is_some(),
            target_batch_lifetime: accepted.batch_lifetime,
            scope_confirmed: false,
            accepted,
            previous,
            bank: EvidenceBank::default(),
            pending: None,
            next_seed: uint(j, "next_seed")?,
            eval_seed: uint(j, "eval_seed")?,
            next_eval_seed: uint(j, "next_eval_seed")?,
            steps: uint(j, "training_steps")?,
            base_games: uint(j, "base_games")?,
            branches: uint(j, "branch_rollouts")?,
            evaluation_games: uint(j, "evaluation_games")?,
            monitor: j.get("deployed_vs_rule").clone(),
        })
    }
    fn diagnostic_candidate(s: &State, p: &Policy) -> Result<Version, String> {
        // Read the current learner explicitly, never a pending or accepted model.
        s.proposal(s.iteration, p.weights_json()?)
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
        let s = diagnostic_state(&read(path)?, &mut p)?;
        if s.target_batch_lifetime {
            return Err("revision ablation covers single scope only; use --evaluate-checkpoint for batch scope".into());
        }
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
        let mut expansion_authorized = false;
        let mut imported_eval_cursor = 0;
        if let Some(path) = o.init.as_ref().or(o.warm_start.as_ref()) {
            let j = read(path)?;
            imported = Some(import_reference(&j)?);
            expansion_authorized = j.get("schema").str() == SCHEMA
                && matches!(j.get("scope_confirmed"), Json::Bool(true));
            imported_eval_cursor = uint(&j, "next_eval_seed")?;
            if o.warm_start.is_some() {
                if j.get("schema").str() != SCHEMA
                    || !matches!(j.get("menu_learning"), Json::Bool(true))
                {
                    return Err(
                        "warm-start requires v9 menu weights; use init-from for older deployments"
                            .into(),
                    );
                }
                p.load_weights(j.get("model").get("weights"))?;
                if !p.event_input_scaling {
                    return Err("warm-start requires normalized weights".into());
                }
            } else if imported
                .as_ref()
                .is_some_and(|(v, _)| v.menu_anchor.is_some())
            {
                // Scope expansion starts from demonstrated choices, preserving the
                // accepted normalized learner instead of randomizing its abilities.
                let weights = j.get("deployment").get("weights");
                p.load_weights(weights)?;
                if !p.event_input_scaling {
                    return Err("scope expansion needs normalized accepted weights".into());
                }
            }
            config = Config::parse(j.get("config"))?;
            next_seed = uint(&j, "next_seed")?.max(o.seed);
            eprintln!("initialization: accepted deployment unchanged; proposal source explicit; fresh Adam, labels and confirmation; source unchanged");
        }
        let (initial, initial_previous) = match imported {
            Some(versions) => versions,
            None => {
                let v = Version::initial(base, vec![0, 1, 2, 3])?;
                (v.clone(), v)
            }
        };
        let target_batch_lifetime = o.batch_lifetime.unwrap_or(initial.batch_lifetime);
        if o.resume.is_none()
            && target_batch_lifetime
            && !initial.batch_lifetime
            && !(expansion_authorized && o.init.is_some())
        {
            return Err("batch scope requires --init-from a v8 checkpoint whose deployed policy passed independent confirmation; first run the single stage".into());
        }
        if o.resume.is_none() && initial.batch_lifetime != target_batch_lifetime {
            return Err("complete menus preserve accepted scope; scope expansion is not supported by this experiment".into());
        }
        if initial.batch_lifetime && !target_batch_lifetime {
            return Err("cannot shrink accepted batch responsibility".into());
        }
        let mut s = if let Some(path) = &o.resume {
            State::restore(&read(path)?, &mut p)?
        } else {
            State {
                iteration: 0,
                target_batch_lifetime,
                menu_learning: true,
                scope_confirmed: false,
                rng: Rng(o.seed ^ 0x1acf789),
                config,
                accepted: initial.clone(),
                previous: initial_previous,
                bank: EvidenceBank::default(),
                next_seed,
                eval_seed: o.eval_seed,
                next_eval_seed: (o.eval_seed + 1_000_000).max(imported_eval_cursor),
                steps: 0,
                base_games: 0,
                branches: 0,
                evaluation_games: 0,
                monitor: Json::Null,
                pending: None,
            }
        };
        if o.batch_lifetime
            .is_some_and(|mode| mode != s.target_batch_lifetime)
        {
            return Err(
                "resume cannot change responsibility; use init-from after independent acceptance"
                    .into(),
            );
        }
        write(
            &o.out.join("manifest.json"),
            &Json::Obj(vec![
                ("schema".into(), Json::Str(SCHEMA.into())),
                (
                    "collection_contract".into(),
                    Json::Str(COLLECTION_CONTRACT.into()),
                ),
                (
                    "learner_input_encoding".into(),
                    Json::Str(route_rl_native::learning::policy::EVENT_INPUT_ENCODING.into()),
                ),
                ("policy_contract".into(), Json::Str(CONTRACT.into())),
                (
                    "objective".into(),
                    Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into()),
                ),
                (
                    "candidate_encoding".into(),
                    Json::Str(route_rl_native::pipeline::plan_menu::ENCODING.into()),
                ),
                (
                    "learning_unit".into(),
                    Json::Str("complete_candidate_set".into()),
                ),
                ("max_menu_choices".into(), n(4.)),
                ("iterations_additional".into(), n(o.iterations as f64)),
                ("base_games_per_update".into(), n(o.games as f64)),
                ("candidate_games_per_update".into(), n(o.games as f64)),
                (
                    "target_batch_lifetime".into(),
                    Json::Bool(s.target_batch_lifetime),
                ),
                (
                    "warm_start".into(),
                    o.warm_start.clone().map(Json::Str).unwrap_or(Json::Null),
                ),
                ("branch_points".into(), n(o.points as f64)),
                (
                    "legacy_alternatives_unused".into(),
                    n(o.alternatives as f64),
                ),
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
                (
                    "candidate_prefix_comparisons_per_game_max".into(),
                    n(o.points as f64),
                ),
                (
                    "sequence_targets_used_for_local_learning".into(),
                    Json::Bool(false),
                ),
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
        println!("event-policy-iteration-v9 accepted_revision={} fixed_continuation=true shared_network=true scope={:?} budget_per_game={}steps/{}branches",s.accepted.revision,s.accepted.scope,o.branch_steps,o.max_branches);
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
            let candidate = s.proposal(iteration, before.clone())?;
            let snapshot_dir = o.out.join("collection_sources");
            std::fs::create_dir_all(&snapshot_dir).map_err(|e| e.to_string())?;
            write(
                &snapshot_dir.join(format!("collection_{iteration:06}.json")),
                &Json::Obj(vec![
                    (
                        "collection_contract".into(),
                        Json::Str(COLLECTION_CONTRACT.into()),
                    ),
                    ("iteration".into(), n(iteration as f64)),
                    ("candidate_prefix_policy".into(), candidate.json()),
                    ("local_continuation_policy".into(), s.accepted.json()),
                    ("accepted_opponent".into(), s.accepted.json()),
                    ("previous_opponent".into(), s.previous.json()),
                    ("config".into(), s.config.json()),
                ]),
            )?;
            let gs = collect(
                js,
                &s.accepted,
                Some(&candidate),
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
                if let Some(sequence) = &g.sequence {
                    append(&o.out.join("sequences.jsonl"), sequence)?;
                }
                for c in &g.comparisons {
                    append(&o.out.join("comparisons.jsonl"), c)?;
                }
            }
            let timer = Instant::now();
            let replay = s.bank.training(&mut s.rng);
            let update_fn = if s.menu_learning {
                route_rl_native::learning::event_sets::update
            } else {
                plan_compare::update_improvement
            };
            let update = update_fn(
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
            if steps > o.games as u64 * (1438 + o.branch_steps) as u64
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
                    Json::Str("accepted_and_frozen_candidate_prefixes".into()),
                ),
                ("comparison_pairs", n(pairs.len() as f64)),
                ("candidate_coverage", candidate_coverage(&gs)),
                (
                    "candidate_prefix_pairs",
                    n(pairs
                        .iter()
                        .filter(|p| p.evidence.get("source").str() == "candidate_prefix")
                        .count() as f64),
                ),
                (
                    "candidate_revision_pairs",
                    n(pairs
                        .iter()
                        .filter(|p| {
                            p.evidence.get("source").str() == "candidate_prefix"
                                && p.evidence.get("stage").str() == "revision"
                        })
                        .count() as f64),
                ),
                (
                    "candidate_sequence_games",
                    n(gs.iter().filter(|g| g.sequence.is_some()).count() as f64),
                ),
                ("target_batch_lifetime", Json::Bool(s.target_batch_lifetime)),
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
                    "complete_set_choices",
                    if s.menu_learning {
                        route_rl_native::learning::event_sets::metrics(&p, &pairs)?
                    } else {
                        Json::Null
                    },
                ),
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
                observe_latest(&o, &mut s, &p)?;
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
                target_batch_lifetime: false,
                menu_learning: false,
                scope_confirmed: false,
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
                sequence: None,
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
        fn complete_menu_collection_is_atomic_replayable_and_covers_actual_argmax() {
            tensor::worker_threads();
            let mut s = state();
            s.menu_learning = true;
            let p = Policy::event_plans(-1, 19, 0.0003).unwrap();
            let v = s.proposal(1, p.weights_json().unwrap()).unwrap();
            let accepted = s.accepted.runtime(-1).unwrap();
            let candidate = v.runtime(-1).unwrap();
            let job = Job {
                seed: 37,
                seat: 0,
                opponent: 0,
                rng: 0,
                slots: vec![0, 1],
            };
            let g = play_menu_training(
                &job,
                &accepted,
                &candidate,
                &p,
                &[],
                &s.config,
                1,
                0,
                &s.accepted.scope,
                4320,
                8,
            )
            .unwrap();
            assert!(!g.pairs.is_empty());
            assert!(g.steps <= 1438 + 4320 && g.branches <= 8);
            for pair in &g.pairs {
                route_rl_native::learning::event_sets::validate(pair).unwrap();
            }
            let metrics = route_rl_native::learning::event_sets::metrics(&p, &g.pairs).unwrap();
            assert_eq!(metrics.get("all").get("unsupported_argmax").i64(), 0);
            s.bank.admit(&g.pairs);
            assert_eq!(s.bank.seen, metrics.get("all").get("sets").i64() as u64);
            let mut restored = Policy::event_plans(-1, 0, 0.0003).unwrap();
            State::restore(&s.checkpoint(&p).unwrap(), &mut restored).unwrap();
            let short = play_menu_training(
                &job,
                &accepted,
                &candidate,
                &p,
                &[],
                &s.config,
                1,
                0,
                &s.accepted.scope,
                1,
                1,
            )
            .unwrap();
            assert!(short.pairs.is_empty());
            assert_eq!(short.branches, 0);
            assert_eq!(short.steps, 1438);
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
                        sequence: None,
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
        fn candidate_prefixes_share_terminal_suffix_without_segment_labels() {
            tensor::worker_threads();
            let s = state();
            let accepted = s.accepted.runtime(-1).unwrap();
            let mut p = Policy::event_plans(-1, 2, 0.0003).unwrap();
            // A test-only production preference exercises real follow-ups.
            // A fresh zero residual head ties every event choice and Keeps;
            // inventing a follow-up for that policy would test a false prefix.
            for parameter in &mut p.parameters[..10] {
                let mut data = vec![0.; parameter.shape.iter().product::<i64>() as usize];
                match parameter.name {
                    "candidate.0.weight" => {
                        data[2] = 1.;
                        data[1] = 0.1;
                    }
                    "score.0.weight" => data[0] = 1.,
                    "score.2.weight" => data[0] = 5.,
                    _ => {}
                }
                let value = tensor::Tensor::floats(&data, &parameter.shape, -1, false).unwrap();
                parameter.value.copy_from(&value).unwrap();
            }
            let candidate = s
                .proposal(1, p.weights_json().unwrap())
                .unwrap()
                .runtime(-1)
                .unwrap();
            let job = Job {
                seed: 1201,
                seat: 0,
                opponent: 0,
                rng: 7,
                slots: vec![0, 1],
            };
            let (world, forks, _) = candidate_trajectory(
                &job,
                &candidate,
                &p,
                &[],
                &s.config,
                &s.accepted.scope,
                "collection_000001",
            )
            .unwrap();
            let deployed = play(
                &job,
                &candidate,
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
            assert_eq!(world.cash(0)[0], deployed.report.get("cash").f64());
            assert_eq!(world.cash(0)[1], deployed.report.get("opponent_cash").f64());
            let collected = play_training(
                &job,
                &accepted,
                &candidate,
                &p,
                &[],
                &s.config,
                1,
                0,
                &s.accepted.scope,
                2,
                4320,
                8,
            )
            .unwrap();
            assert!(collected.steps <= 1438 + 4320);
            assert!(collected.branches <= 8);
            let sequence = collected.sequence.as_ref().unwrap();
            assert_eq!(sequence.get("local_target"), &Json::Bool(false));
            assert_eq!(sequence.get("promotion_evidence"), &Json::Bool(false));
            assert_eq!(
                sequence.get("candidate_outcome").get("cash"),
                deployed.report.get("cash")
            );
            assert!(collected
                .pairs
                .iter()
                .any(|p| p.evidence.get("source").str() == "accepted_trajectory"));
            let local = collected
                .pairs
                .iter()
                .find(|p| {
                    p.evidence.get("source").str() == "candidate_prefix"
                        && p.evidence.get("stage").str() == "revision"
                })
                .expect("actual candidate follow-up coverage");
            assert!(local.evidence.get("production_candidate_count").i64() > 0);
            assert_eq!(local.evidence.get("reference_index").i64(), 0);
            assert_eq!(local.evidence.get("reference_terminal_step").i64(), 719);
            assert_eq!(local.evidence.get("alternative_terminal_step").i64(), 719);
            let f = forks
                .iter()
                .find(|f| f.prefix == local.evidence.get("prefix_id").str())
                .unwrap();
            let alt = local.evidence.get("alternative_index").i64() as usize;
            let (reference, _) =
                rollout_edit(f, 0, &job, &accepted, &[], None, &s.accepted.scope).unwrap();
            let (alternative, _) =
                rollout_edit(f, alt, &job, &accepted, &[], None, &s.accepted.scope).unwrap();
            assert_eq!(
                local.evidence.get("reference_cash"),
                &Json::Arr(reference.cash(0).map(n).to_vec())
            );
            assert_eq!(
                local.evidence.get("alternative_cash"),
                &Json::Arr(alternative.cash(0).map(n).to_vec())
            );
            assert_eq!(
                local.improvement_target().unwrap() as f64,
                score(alternative.cash(0)) - score(reference.cash(0))
            );
            assert!(sequence
                .get("same_prefix_segments")
                .arr()
                .iter()
                .any(|s| !s.get("later_segment_events").arr().is_empty()));
            assert_eq!(collected.comparisons.len(), collected.pairs.len());
            let limited = play_training(
                &job,
                &accepted,
                &candidate,
                &p,
                &[],
                &s.config,
                1,
                0,
                &s.accepted.scope,
                1,
                719,
                1,
            )
            .unwrap();
            assert!(limited
                .pairs
                .iter()
                .all(|p| p.evidence.get("source").str() != "candidate_prefix"));
            assert!(limited.steps <= 1438 + 719);
            assert!(limited.branches <= 1);
        }
        #[test]
        fn old_diagnostics_remain_read_only_and_contract_changes_invalidate_labels() {
            tensor::worker_threads();
            let mut s = state();
            let p = Policy::event_plans(-1, 7, 0.0003).unwrap();
            let mut j = s.checkpoint(&p).unwrap();
            j.set_path("schema", Json::Str(NORMALIZED_SCHEMA.into()));
            j.set_path("collection_contract", Json::Null);
            j.set_path(
                "comparison_bank",
                Json::Str("diagnostic-only legacy data".into()),
            );
            let saved = j.clone();
            let mut q = Policy::event_plans(-1, 0, 0.0003).unwrap();
            assert!(State::restore(&j, &mut q).is_err());
            assert!(diagnostic_state(&j, &mut q).is_ok());
            assert_eq!(j, saved);
            let mut old = pair("revision", 1);
            old.evidence
                .set_path("continuation_contract", Json::Str(BATCH_CONTRACT.into()));
            s.bank.admit(&[old]);
            assert!(State::restore(&s.checkpoint(&p).unwrap(), &mut q).is_err());
            s.target_batch_lifetime = true;
            let proposal = s.proposal(1, p.weights_json().unwrap()).unwrap();
            assert!(proposal.batch_lifetime);
            assert!(!s.accepted.batch_lifetime);
            s.promote(proposal).unwrap();
            assert!(s.scope_confirmed && s.accepted.batch_lifetime);
            assert!(s.bank.history.is_empty());
            let back = State::restore(&s.checkpoint(&p).unwrap(), &mut q).unwrap();
            assert!(back.target_batch_lifetime && back.scope_confirmed);
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
                sequence: None,
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
