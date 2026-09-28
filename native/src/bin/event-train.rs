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
            score_confirmation, tensor,
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
                println!("event-train --out DIR [--resume latest.json | --base-checkpoint v3_best.json] --iterations 100 --games-per-update 16 --branch-points 2 --alternatives 2 --workers 7 --device cuda|cpu --epochs 8 --batch-size 64 --learning-rate 0.0003 --eval-every 5 --eval-games 8 --confirm-games 16 --seed 1200 --eval-seed 1100000000 --config native/configs/plan_prototype_v1.json\nSeason-paired event-plan-improvement-v4; older event checkpoints supported only with no accepted event patches. One scoped primary+followup package is gated before deployment; learner never directly controls collection. iterations = additional comparison/NN updates; games-per-update = full base trajectories; each branch runs its remaining season to termination; confirmation uses up to 4x confirm-games on new seeds.");
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
    /// the current learner and covering resource-event kinds/season phases. Only real
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
        // Same primary choice with a different continuation is package evidence,
        // not a contradictory training target on two identical action features.
        if reference != alternative {
            let (target, gain) = pref.unwrap_or_else(|| (vec![0.5, 0.5], 0.));
            let winner = if target[1] > target[0] {
                alternative
            } else {
                reference
            };
            let kind = choices[winner]
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
    fn play(
        job: &Job,
        accepted: &Runtime,
        learner: Option<&Policy>,
        opponents: &[Runtime],
        config: &Config,
        iteration: u64,
        revision: u64,
        alternatives: usize,
        continuation: Option<(&Portfolio, &Json, u64)>,
        audit: bool,
    ) -> Result<Game, String> {
        let mut world = World::new(job.seed, config);
        let mut rng = Rng(job.rng);
        let mut forks = vec![];
        let mut candidate_sets = vec![];
        while world.game.step < 719 {
            let obs = Observation::from_state(&world.game, job.seat);
            let action = if let Some(mut d) = world.own.prepare(&obs, accepted)? {
                if !d.followup && alternatives > 0 && !job.slots.is_empty() && d.choices.len() > 1 {
                    let slot = d.slot.ok_or("event scope missing")?;
                    let proposed = match learner {
                        Some(p) => p.infer(&[d.row.clone()], true, &mut Rng(0))?[0].action,
                        None => d.selected,
                    };
                    let mut pool: Vec<_> = if audit {
                        vec![proposed]
                    } else {
                        let mut ids: Vec<_> =
                            (0..d.choices.len()).filter(|i| *i != d.selected).collect();
                        rng.shuffle(&mut ids);
                        if let Some(at) = ids.iter().position(|i| *i == proposed) {
                            ids.swap(0, at);
                        }
                        if ids.len() > 2 {
                            let first = &d.choices[ids[0]].next;
                            if let Some(at) =
                                (1..ids.len()).find(|at| d.choices[ids[*at]].next != *first)
                            {
                                ids.swap(1, at);
                            }
                        }
                        ids
                    };
                    pool.truncate(alternatives);
                    forks.push(Fork {
                        world: world.clone(),
                        row: d.row.clone(),
                        reference: d.choices[d.selected].clone(),
                        alternatives: pool.iter().map(|i| (*i, d.choices[*i].clone())).collect(),
                        start_progress: world.own.controller.progress.len(),
                        reference_index: d.selected,
                        slot,
                        learner_changes: proposed != d.selected,
                    });
                    candidate_sets.push(d.choices.clone());
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
        let chosen = branch_points(
            &observed,
            &job.slots,
            job.seed as usize + iteration as usize,
        );
        let mut forks: Vec<_> = forks.into_iter().map(Some).collect();
        for index in chosen {
            let fork = forks[index].take().expect("distinct branch points");
            let mut paired_runtime = None;
            if let Some((portfolio, weights, _)) = continuation {
                let mut runtime = Runtime::load(portfolio, -1)?;
                runtime.override_followup(fork.slot, weights)?;
                paired_runtime = Some(runtime);
            }
            let branch_policy = paired_runtime.as_ref().unwrap_or(accepted);
            let mut alternatives = fork.alternatives.clone();
            let needs_reference =
                continuation.is_some() && !fork.reference.keep && fork.reference.next.is_some();
            if needs_reference {
                if let Some(at) = alternatives
                    .iter()
                    .position(|(i, _)| *i == fork.reference_index)
                {
                    alternatives.swap(0, at);
                } else {
                    alternatives.insert(0, (fork.reference_index, fork.reference.clone()));
                }
            }
            let mut common_reference_cash = reference_cash;
            for (alternative_index, alternative) in &alternatives {
                let auxiliary_reference = *alternative_index == fork.reference_index
                    && !fork
                        .alternatives
                        .iter()
                        .any(|(i, _)| i == alternative_index);
                let prefix = format!(
                    "{}:{}:{}:{}:{}:{}",
                    job.seed, job.seat, job.opponent, fork.slot, fork.row.step, alternative_index
                );
                let mut branch = fork.world.clone();
                let obs = Observation::from_state(&branch.game, job.seat);
                let new_start = branch.own.controller.progress.len();
                let new_batch = branch.own.controller.batches.len();
                let action = branch
                    .own
                    .execute_choice(&obs, alternative.clone(), branch_policy)?;
                if continuation.is_some() && !alternative.keep && alternative.next.is_some() {
                    if branch.own.controller.batches.len() != new_batch + 1 {
                        return Err("edit did not create expected batch".into());
                    }
                    branch.own.arm_followup(fork.slot, new_batch, obs.step);
                }
                branch.advance(job, action, opponents)?;
                result.steps += 1;
                let mut first_only_cash = None;
                let mut followup_event = Json::Null;
                // The same Runtime/Deployed path as event-agent. Intercept only to clone
                // the keep-vs-revise comparison at the one conditional followup event.
                while branch.game.step < 719 {
                    let obs = Observation::from_state(&branch.game, job.seat);
                    let action = if let Some(mut d) = branch.own.prepare(&obs, branch_policy)? {
                        if d.followup
                            && d.slot == Some(fork.slot)
                            && first_only_cash.is_none()
                            && continuation.is_some()
                        {
                            followup_event = branch
                                .own
                                .last_event
                                .as_ref()
                                .map(|e| e.json())
                                .unwrap_or(Json::Null);
                            let second_world = branch.clone();
                            let second_choices = d.choices.clone();
                            let second_row = d.row.clone();
                            let selected = d.selected;
                            // B: retain first arrangement at this review. C: frozen neural revision.
                            let mut hold = second_world.clone();
                            let a = hold.own.execute_choice(
                                &obs,
                                second_choices[0].clone(),
                                branch_policy,
                            )?;
                            hold.advance(job, a, opponents)?;
                            result.steps += 1 + hold.finish(job, branch_policy, opponents)?;
                            result.branches += 1;
                            let hold_cash = hold.cash(job.seat);
                            first_only_cash = Some(hold_cash);
                            let mut measured = vec![];
                            if selected != 0 {
                                measured.push(selected);
                            }
                            if !audit {
                                let mut ids: Vec<_> = (1..second_choices.len())
                                    .filter(|i| *i != selected)
                                    .collect();
                                rng.shuffle(&mut ids);
                                if let Some(i) = ids.first() {
                                    measured.push(*i);
                                }
                            }
                            let mut actual = if selected == 0 { Some(hold) } else { None };
                            for next in measured {
                                let mut revised = second_world.clone();
                                let a = revised.own.execute_choice(
                                    &obs,
                                    second_choices[next].clone(),
                                    branch_policy,
                                )?;
                                revised.advance(job, a, opponents)?;
                                result.steps +=
                                    1 + revised.finish(job, branch_policy, opponents)?;
                                result.branches += 1;
                                record_pair(
                                    &mut result,
                                    job,
                                    iteration,
                                    revision,
                                    fork.slot,
                                    &second_row,
                                    &second_choices,
                                    0,
                                    next,
                                    hold_cash,
                                    revised.cash(job.seat),
                                    Json::Obj(vec![
                                        ("stage".into(), Json::Str("followup".into())),
                                        ("prefix_id".into(), Json::Str(prefix.clone())),
                                        ("event".into(), followup_event.clone()),
                                        (
                                            "frozen_followup_version".into(),
                                            Json::Str(continuation.unwrap().2.to_string()),
                                        ),
                                        ("post_update_audit".into(), Json::Bool(audit)),
                                    ]),
                                )?;
                                if next == selected {
                                    actual = Some(revised);
                                }
                            }
                            branch = actual.ok_or("frozen followup choice was not executed")?;
                            break;
                        }
                        branch.own.execute_choice(
                            &obs,
                            d.choices.swap_remove(d.selected),
                            branch_policy,
                        )?
                    } else {
                        branch.own.continue_action(&obs, branch_policy)?
                    };
                    branch.advance(job, action, opponents)?;
                    result.steps += 1;
                }
                branch
                    .own
                    .controller
                    .observe(&Observation::from_state(&branch.game, job.seat));
                if first_only_cash.is_none() {
                    result.branches += 1;
                }
                let cash = branch.cash(job.seat);
                if *alternative_index == fork.reference_index {
                    common_reference_cash = cash;
                }
                let end = (new_start
                    + if !alternative.keep && alternative.next.is_some() {
                        alternative.sites.len()
                    } else {
                        0
                    })
                .min(branch.own.controller.progress.len());
                let evidence = Json::Obj(vec![
                    ("stage".into(), Json::Str("primary".into())),
                    (
                        "auxiliary_reference".into(),
                        Json::Bool(auxiliary_reference),
                    ),
                    (
                        "incumbent_cash".into(),
                        Json::Arr(reference_cash.map(n).to_vec()),
                    ),
                    (
                        "package_score_gain".into(),
                        n(score(cash) - score(reference_cash)),
                    ),
                    ("prefix_id".into(), Json::Str(prefix)),
                    (
                        "frozen_followup_version".into(),
                        Json::Str(continuation.map(|x| x.2).unwrap_or(0).to_string()),
                    ),
                    ("post_update_audit".into(), Json::Bool(audit)),
                    (
                        "event".into(),
                        fork.world
                            .own
                            .last_event
                            .as_ref()
                            .map(|e| e.json())
                            .unwrap_or(Json::Null),
                    ),
                    ("followup_event".into(), followup_event),
                    (
                        "keep_at_followup_cash".into(),
                        first_only_cash
                            .map(|c| Json::Arr(c.map(n).to_vec()))
                            .unwrap_or(Json::Null),
                    ),
                    ("learner_changes".into(), Json::Bool(fork.learner_changes)),
                    (
                        "incumbent_execution".into(),
                        Json::Arr(
                            world
                                .own
                                .controller
                                .progress
                                .iter()
                                .skip(fork.start_progress)
                                .take(if !fork.reference.keep && fork.reference.next.is_some() {
                                    fork.reference.sites.len()
                                } else {
                                    0
                                })
                                .map(|p| p.json())
                                .collect(),
                        ),
                    ),
                    (
                        "alternative_execution".into(),
                        Json::Arr(
                            branch.own.controller.progress[new_start..end]
                                .iter()
                                .map(|p| p.json())
                                .collect(),
                        ),
                    ),
                ]);
                record_pair(
                    &mut result,
                    job,
                    iteration,
                    revision,
                    fork.slot,
                    &fork.row,
                    &candidate_sets[index],
                    fork.reference_index,
                    *alternative_index,
                    common_reference_cash,
                    cash,
                    evidence,
                )?;
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
                    if let Some(w) = &lw {
                        let mut p = Policy::plans(-1, 0, 0.0003)?;
                        p.load_weights(if w.get("weights").is_obj() {
                            w.get("weights")
                        } else {
                            w
                        })?;
                        lp = Some(p);
                    }
                    let continuation =
                        lw.as_ref().filter(|w| w.get("followup").is_obj()).map(|w| {
                            (
                                &portfolio,
                                w.get("followup"),
                                w.get("followup_version").str().parse::<u64>().unwrap(),
                            )
                        });
                    let audit = lw
                        .as_ref()
                        .is_some_and(|w| matches!(w.get("audit"), Json::Bool(true)));
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
                                continuation,
                                audit,
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
        // Fixed for a block of primary updates. Targets depend on this continuation.
        followup_weights: Json,
        followup_version: u64,
        pending_followup_refresh: bool,
    }
    impl State {
        fn checkpoint(&self, p: &Policy) -> Result<Json, String> {
            Ok(Json::Obj(vec![
                ("schema".into(), Json::Str(SCHEMA.into())),
                ("policy_contract".into(), Json::Str(CONTRACT.into())),
                (
                    "objective".into(),
                    Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into()),
                ),
                ("iteration".into(), Json::Str(self.iteration.to_string())),
                ("model".into(), p.checkpoint(self.iteration, &self.rng)?),
                (
                    "frozen_followup_weights".into(),
                    self.followup_weights.clone(),
                ),
                (
                    "frozen_followup_version".into(),
                    Json::Str(self.followup_version.to_string()),
                ),
                ("config".into(), self.config.json()),
                (
                    "pending_followup_refresh".into(),
                    Json::Bool(self.pending_followup_refresh),
                ),
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
            let legacy_scope = matches!(
                j.get("policy_contract").str(),
                "event-batch-context320-actions32-v2"
                    | "event-batch-context320-actions32-season-pair-v3"
            );
            if j.get("schema").str() != SCHEMA
                || (!legacy_scope && j.get("policy_contract").str() != CONTRACT)
            {
                return Err("requires event-plan-improvement-v4 full checkpoint; rejected v2 learner cannot become an accepted strategy".into());
            }
            let (iteration, rng) = p.restore(j.get("model"))?;
            if !p.plan_residual || iteration != uint(j, "iteration")? {
                return Err("model/outer iteration mismatch".into());
            }
            let accepted = Portfolio::parse(j.get("deployment"))?;
            let previous = Portfolio::parse(j.get("previous_accepted"))?;
            let old_objective = j.get("objective").str();
            if !matches!(
                old_objective,
                "" | "terminal_utility_difference_regression" | plan_compare::MATCH_SCORE_OBJECTIVE
            ) {
                return Err("unsupported event training objective".into());
            }
            let mut bank = Bank::parse(j.get("comparison_bank"))?;
            if old_objective != plan_compare::MATCH_SCORE_OBJECTIVE {
                bank.relabel_match_score()?;
                eprintln!("objective upgrade: recomputed stored comparisons from terminal win/draw/loss; learner/Adam and accepted deployment preserved");
            } else if bank
                .elite
                .iter()
                .chain(&bank.recent)
                .any(|r| r.evidence.get("objective").str() != plan_compare::MATCH_SCORE_OBJECTIVE)
            {
                return Err("comparison bank objective does not match checkpoint".into());
            }
            if previous.revision > accepted.revision {
                return Err("previous strategy revision exceeds accepted revision".into());
            }
            for row in bank.elite.iter().chain(&bank.recent) {
                if row.incumbent_revision()? != accepted.revision {
                    return Err("stale comparison labels in active checkpoint".into());
                }
            }
            if legacy_scope {
                // Portfolio::parse rejects nonempty old event scopes. Event timing
                // and reservation fixes require fresh outcome labels. Source is untouched.
                bank = Bank::default();
                eprintln!("scope upgrade: preserved learner/Adam/frozen foundation; discarded comparisons from the old event execution contract; new season+paired contract requires new evidence");
            }
            let followup_weights = if legacy_scope {
                p.weights_json()?
            } else {
                j.get("frozen_followup_weights").clone()
            };
            if !followup_weights.is_obj() {
                return Err("missing frozen followup weights".into());
            }
            let mut verify = Policy::plans(-1, 0, 0.0003)?;
            verify.load_weights(&followup_weights)?;
            let followup_version = if legacy_scope {
                iteration + 1
            } else {
                uint(j, "frozen_followup_version")?
            };
            if bank.elite.iter().chain(&bank.recent).any(|r| {
                r.evidence.get("stage").str() == "primary"
                    && r.evidence.get("frozen_followup_version").str()
                        != followup_version.to_string()
            }) {
                return Err("stale frozen followup labels in checkpoint".into());
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
                attempts: if legacy_scope {
                    [0; SLOTS]
                } else {
                    attempts
                        .try_into()
                        .map_err(|_| "invalid scope counter length")?
                },
                monitor: j.get("deployed_vs_rule").clone(),
                followup_weights,
                followup_version,
                pending_followup_refresh: !legacy_scope
                    && matches!(j.get("pending_followup_refresh"), Json::Bool(true)),
            })
        }
        fn snapshot(&self, p: &Policy, audit: bool) -> Result<Json, String> {
            Ok(Json::Obj(vec![
                ("weights".into(), p.weights_json()?),
                ("followup".into(), self.followup_weights.clone()),
                (
                    "followup_version".into(),
                    Json::Str(self.followup_version.to_string()),
                ),
                ("audit".into(), Json::Bool(audit)),
            ]))
        }
        fn refresh_followup(&mut self, p: &Policy) -> Result<(), String> {
            self.followup_weights = p.weights_json()?;
            self.followup_version += 1;
            self.pending_followup_refresh = false;
            // Conditional second decisions terminate under the accepted portfolio;
            // primary labels additionally depend on the old frozen follower.
            self.bank
                .elite
                .retain(|r| r.evidence.get("stage").str() == "followup");
            self.bank
                .recent
                .retain(|r| r.evidence.get("stage").str() == "followup");
            Ok(())
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
        // Freeze BOTH networks before collecting current-choice evidence. There is
        // no gradient update between this audit, full-game gate and deployment.
        let frozen = s.snapshot(p, true)?;
        let audit_start = Instant::now();
        let audit_base = s.next_seed;
        if audit_base + o.games as u64 / 2 >= 1_000_000_000 {
            return Err("training audit seed range exhausted".into());
        }
        let audit_jobs = jobs(
            audit_base,
            o.games,
            &roster(&s.accepted, &s.previous),
            o.points,
            s.iteration,
            &mut s.rng,
        );
        s.next_seed += o.games as u64 / 2;
        let audit = collect(
            audit_jobs,
            &s.accepted,
            Some(&frozen),
            &[s.accepted.clone(), s.previous.clone()],
            &s.config,
            o.workers,
            s.iteration,
            1,
        )?;
        s.steps += audit.iter().map(|g| g.steps as u64).sum::<u64>();
        s.branches += audit.iter().map(|g| g.branches as u64).sum::<u64>();
        s.base_games += audit.len() as u64;
        let audit_pairs: Vec<_> = audit.iter().flat_map(|g| g.pairs.iter().cloned()).collect();
        s.bank.admit(&audit_pairs); // Used only on LATER updates, after this candidate is gated.
        let mut scopes = std::collections::BTreeMap::<
            usize,
            (f64, usize, std::collections::BTreeSet<i64>),
        >::new();
        for g in &audit {
            for c in &g.comparisons {
                append(&o.out.join("comparisons.jsonl"), c)?;
                if c.get("stage").str() != "primary"
                    || matches!(c.get("auxiliary_reference"), Json::Bool(true))
                {
                    continue;
                }
                let a = c.get("alternative_cash").arr();
                let b = c.get("incumbent_cash").arr();
                let delta = score([a[0].f64(), a[1].f64()]) - score([b[0].f64(), b[1].f64()]);
                let e = scopes.entry(c.get("slot_id").i64() as usize).or_default();
                e.0 += delta;
                e.1 += 1;
                if delta > 0. {
                    e.2.insert(c.get("seed").i64());
                }
            }
        }
        let support = Json::Obj(vec![
            (
                "source".into(),
                Json::Str("frozen_post_update_actual_package_rollouts".into()),
            ),
            ("training_seed_start".into(), n(audit_base as f64)),
            ("seconds".into(), n(audit_start.elapsed().as_secs_f64())),
            ("base_games".into(), n(audit.len() as f64)),
            (
                "branch_rollouts".into(),
                n(audit.iter().map(|g| g.branches).sum::<usize>() as f64),
            ),
            (
                "simulation_steps".into(),
                n(audit.iter().map(|g| g.steps).sum::<usize>() as f64),
            ),
            (
                "slots".into(),
                Json::Arr(
                    scopes
                        .iter()
                        .map(|(slot, (gain, count, seeds))| {
                            Json::Obj(vec![
                                ("slot_id".into(), n(*slot as f64)),
                                ("total_observed_delta".into(), n(*gain)),
                                ("tested_choices".into(), n(*count as f64)),
                                ("supported_seeds".into(), n(seeds.len() as f64)),
                                ("untested_choices".into(), n(0.)),
                            ])
                        })
                        .collect(),
                ),
            ),
        ]);
        // Small training support only allocates evaluation effort; independent
        // confirmation decides admission. Count seats as ONE seed of evidence.
        let slot = scopes
            .iter()
            .filter(|(_, (gain, _, seeds))| *gain > 0. && !seeds.is_empty())
            .max_by(|(a, x), (b, y)| {
                (x.0 / (1. + s.attempts[**a] as f64))
                    .total_cmp(&(y.0 / (1. + s.attempts[**b] as f64)))
                    .then_with(|| b.cmp(a))
            })
            .map(|(slot, _)| *slot);
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
            let proposal = s.accepted.propose_paired(
                slot,
                s.iteration,
                p.weights_json()?,
                s.followup_weights.clone(),
            )?;
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
                let mut candidate_all = vec![];
                let mut incumbent_all = vec![];
                let mut looks = vec![];
                let mut promoted = false;
                // One proposal, frozen weights/roster, cumulative NEW seeds. Split
                // the per-proposal test budget across three prespecified looks.
                for look in 1..=3 {
                    let target = o.confirm_games * (1 << (look - 1));
                    let done = candidate_all.len() / roster.len();
                    let js = eval_jobs(base + 1000 + done as u64 / 2, target - done, &roster);
                    candidate_all.extend(collect(
                        js.clone(),
                        &proposal,
                        None,
                        &opponents,
                        &s.config,
                        o.workers,
                        s.iteration,
                        0,
                    )?);
                    incumbent_all.extend(collect(
                        js,
                        &s.accepted,
                        None,
                        &opponents,
                        &s.config,
                        o.workers,
                        s.iteration,
                        0,
                    )?);
                    let mut confirmation = gate(&candidate_all, &incumbent_all)?;
                    let evidence = score_confirmation::assess(
                        &seed_score_deltas(&candidate_all, &incumbent_all)?,
                        look,
                        3,
                    )?;
                    promoted = matches!(evidence.get("qualifies"), Json::Bool(true));
                    confirmation.set_path("qualifies", Json::Bool(promoted));
                    confirmation.set_path("evidence", evidence.clone());
                    looks.push(confirmation.clone());
                    report.set_path("confirmation", confirmation);
                    if evidence.get("decision").str() != "extend" {
                        break;
                    }
                }
                report.set_path("confirmation_seeds_start", n((base + 1000) as f64));
                report.set_path("confirmation_looks", Json::Arr(looks));
                report.set_path("confirmation_candidate", summary(&candidate_all));
                report.set_path("confirmation_incumbent", summary(&incumbent_all));
                if promoted {
                    s.promote(proposal)?;
                    monitor(o, s)?;
                    report.set_path("promoted", Json::Bool(true));
                    report.set_path("deployed_vs_rule", s.monitor.clone());
                    s.pending_followup_refresh = true;
                    write(&o.out.join("best.json"), &s.checkpoint(p)?)?;
                }
            }
        } else {
            report.set_path(
                "reason",
                Json::Str("frozen current package choices have no positive tested aggregate gain on this training audit batch".into()),
            );
        }
        if frozen.get("weights") != &p.weights_json()?
            || frozen.get("followup") != &s.followup_weights
        {
            return Err("candidate changed between audit and gate".into());
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
                followup_weights: p.weights_json()?,
                followup_version: 1,
                pending_followup_refresh: false,
            }
        };
        write(
            &o.out.join("manifest.json"),
            &Json::Obj(vec![
                ("schema".into(), Json::Str(SCHEMA.into())),
                ("policy_contract".into(), Json::Str(CONTRACT.into())),
                ("objective".into(), Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into())),
                ("iterations_additional".into(), n(o.iterations as f64)),
                ("base_games_per_update".into(), n(o.games as f64)),
                ("branch_points".into(), n(o.points as f64)),
                ("alternatives".into(), n(o.alternatives as f64)),
                ("branch_sampling".into(), Json::Str("one mandatory rotating event scope; remaining points alternate learner disagreement and rotating resource kinds/season phases".into())),
                ("epochs".into(), n(o.epochs as f64)),
                ("batch_size".into(), n(o.batch as f64)),
                ("eval_every".into(), n(o.eval_every as f64)),
                ("eval_games".into(), n(o.eval_games as f64)),
                ("confirm_games".into(), n(o.confirm_games as f64)),
                ("confirmation_max_multiplier".into(),n(4.)),
                ("confirmation_looks".into(),n(3.)),
                ("follower_refresh".into(),Json::Str("after the next update following a frozen gate; learn audit labels before discarding stale primary continuation labels".into())),
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
                    Json::Str("harvest/material/funding/review x four 180-turn phases; one primary plus at most one same-batch later review per accepted scope".into()),
                ),
            ]),
        )?;
        if o.resume.is_none() || s.monitor.get("games").i64() != o.eval_games as i64 {
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
            if s.next_seed + o.games as u64 >= 1_000_000_000 {
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
                Some(&s.snapshot(&p, false)?),
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
            // Give last gate's freshly tested argmax labels an update under the
            // SAME follower before changing its continuation and clearing primary labels.
            if s.pending_followup_refresh {
                s.refresh_followup(&p)?;
            }
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
                (
                    "objective",
                    Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into()),
                ),
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
                s.pending_followup_refresh = true;
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
        fn primary_labels_use_common_follower_but_audit_uses_incumbent() {
            let job = Job {
                seed: 1,
                seat: 0,
                opponent: 0,
                rng: 0,
                slots: vec![0],
            };
            let mut result = fake(1, 0, 0., 0.);
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
                &mut result,
                &job,
                1,
                0,
                0,
                &row,
                &choices,
                0,
                1,
                [300., 200.],
                [250., 200.],
                Json::Obj(vec![
                    ("stage".into(), Json::Str("primary".into())),
                    ("incumbent_cash".into(), Json::Arr(vec![n(100.), n(200.)])),
                    ("package_score_gain".into(), n(1.)),
                ]),
            )
            .unwrap();
            assert_eq!(result.pairs[0].improvement_target().unwrap(), 0.);
            assert_eq!(result.comparisons[0].get("package_score_gain").f64(), 1.);
            // Same primary action with a changed continuation is gate evidence,
            // not an impossible nonzero target on identical features.
            record_pair(
                &mut result,
                &job,
                1,
                0,
                0,
                &row,
                &choices,
                0,
                0,
                [100., 200.],
                [300., 200.],
                Json::Obj(vec![]),
            )
            .unwrap();
            assert_eq!(result.pairs.len(), 1);
            assert_eq!(result.comparisons.len(), 2);
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
        #[test]
        fn legacy_event_checkpoint_relabels_evidence_without_resetting_deployment() {
            tensor::worker_threads();
            let p = Policy::plans(-1, 2, 0.001).unwrap();
            let mut s = state(&p);
            let row = Sample {
                context: vec![0.; 320],
                features: vec![vec![0.; 32]; 2],
                ..Default::default()
            };
            s.bank.admit(&[Pair {
                row,
                target: vec![0.1, 0.9],
                gain: 0.8,
                iteration: 7,
                seed: 1,
                seat: 0,
                opponent: 0,
                bucket: 0,
                evidence: Json::Obj(vec![
                    ("incumbent_revision".into(), Json::Str("0".into())),
                    ("reference_cash".into(), Json::Arr(vec![n(100.), n(90.)])),
                    ("alternative_cash".into(), Json::Arr(vec![n(1000.), n(90.)])),
                ]),
            }]);
            let mut j = s.checkpoint(&p).unwrap();
            j.set_path("objective", Json::Null);
            let mut q = Policy::plans(-1, 8, 0.01).unwrap();
            let restored = State::restore(&j, &mut q).unwrap();
            assert_eq!(restored.accepted, s.accepted);
            assert_eq!(restored.iteration, s.iteration);
            assert_eq!(restored.next_seed, s.next_seed);
            assert_eq!(
                p.checkpoint(s.iteration, &s.rng).unwrap(),
                q.checkpoint(restored.iteration, &restored.rng).unwrap()
            );
            assert!(restored
                .bank
                .elite
                .iter()
                .chain(&restored.bank.recent)
                .all(|r| r.improvement_target().unwrap() == 0. && r.gain == 0.));
            assert_eq!(
                restored.checkpoint(&q).unwrap().get("objective").str(),
                plan_compare::MATCH_SCORE_OBJECTIVE
            );
            j.set_path("objective", Json::Str("unknown".into()));
            assert!(State::restore(&j, &mut q).is_err());
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
                followup_weights: p.weights_json().unwrap(),
                followup_version: 1,
                pending_followup_refresh: false,
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
            let a = play(
                &job,
                &accepted,
                None,
                &[],
                &Config::default(),
                1,
                0,
                0,
                None,
                false,
            )
            .unwrap();
            let b = play(
                &job,
                &accepted,
                Some(&learner),
                &[],
                &Config::default(),
                1,
                0,
                1,
                None,
                false,
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
        fn retained_pair(stage: &str, seed: i64) -> Pair {
            let mut features = vec![vec![0.; 32]; 2];
            features[1][3] = 1.;
            Pair {
                row: Sample {
                    context: vec![0.; 320],
                    features,
                    step: 73,
                    ..Default::default()
                },
                target: vec![0.9, 0.1],
                gain: 1.,
                iteration: 7,
                seed,
                seat: 0,
                opponent: 0,
                bucket: 0,
                evidence: Json::Obj(vec![
                    (
                        "objective".into(),
                        Json::Str(plan_compare::MATCH_SCORE_OBJECTIVE.into()),
                    ),
                    ("incumbent_revision".into(), Json::Str("0".into())),
                    ("stage".into(), Json::Str(stage.into())),
                    ("frozen_followup_version".into(), Json::Str("1".into())),
                    ("reference_cash".into(), Json::Arr(vec![n(100.), n(90.)])),
                    ("alternative_cash".into(), Json::Arr(vec![n(90.), n(100.)])),
                ]),
            }
        }
        #[test]
        fn old_empty_scope_resume_preserves_optimizer_foundation_seeds_and_clears_scope_evidence() {
            tensor::worker_threads();
            let mut p = Policy::plans(-1, 501, 0.0007).unwrap();
            // Nonzero moments distinguish a full resume from a weights-only load.
            let param = &mut p.parameters[0];
            let size = param.shape.iter().product::<i64>() as usize;
            param.m = tensor::Tensor::floats(&vec![0.125; size], &param.shape, -1, false).unwrap();
            param.v = tensor::Tensor::floats(&vec![0.25; size], &param.shape, -1, false).unwrap();
            param.step = 9;
            let mut s = state(&p);
            let foundation = route_rl_native::pipeline::plan_portfolio::Portfolio::empty()
                .propose(2, 55, p.weights_json().unwrap())
                .unwrap();
            s.accepted = Portfolio::from_foundation(foundation.clone());
            s.previous = s.accepted.clone();
            s.attempts = [3; SLOTS];
            s.bank.admit(&[retained_pair("primary", 7)]);
            let mut old = s.checkpoint(&p).unwrap();
            old.set_path(
                "policy_contract",
                Json::Str("event-batch-context320-actions32-v2".into()),
            );
            for field in ["deployment", "previous_accepted"] {
                let mut d = old.get(field).clone();
                d.set_path(
                    "contract",
                    Json::Str("event-batch-context320-actions32-v2".into()),
                );
                old.set_path(field, d);
            }
            old.set_path("frozen_followup_weights", Json::Null);
            old.set_path("frozen_followup_version", Json::Null);
            let mut q = Policy::plans(-1, 999, 0.1).unwrap();
            let restored = State::restore(&old, &mut q).unwrap();
            assert_eq!(
                p.checkpoint(s.iteration, &s.rng).unwrap(),
                q.checkpoint(restored.iteration, &restored.rng).unwrap()
            );
            assert_eq!(restored.accepted.foundation, foundation);
            assert_eq!(restored.accepted, s.accepted);
            assert_eq!(restored.previous, s.previous);
            assert_eq!(
                (restored.seed, restored.next_seed, restored.eval_seed),
                (s.seed, s.next_seed, s.eval_seed)
            );
            assert_eq!(
                (restored.steps, restored.base_games, restored.branches),
                (s.steps, s.base_games, s.branches)
            );
            assert!(restored.bank.elite.is_empty() && restored.bank.recent.is_empty());
            assert_eq!(restored.attempts, [0; SLOTS]);
            assert_eq!(restored.followup_weights, q.weights_json().unwrap());
            assert_eq!(restored.followup_version, s.iteration + 1);
            assert_eq!(
                old.get("policy_contract").str(),
                "event-batch-context320-actions32-v2",
                "source checkpoint must remain untouched"
            );
        }
        #[test]
        fn paired_checkpoint_keeps_distinct_primary_follower_and_learning_snapshots() {
            tensor::worker_threads();
            let p = Policy::plans(-1, 601, 0.0003).unwrap();
            let primary = Policy::plans(-1, 602, 0.0003)
                .unwrap()
                .weights_json()
                .unwrap();
            let follower = Policy::plans(-1, 603, 0.0003)
                .unwrap()
                .weights_json()
                .unwrap();
            assert_ne!(primary, follower);
            let mut s = state(&p);
            s.promote(
                s.accepted
                    .propose_paired(4, 7, primary.clone(), follower.clone())
                    .unwrap(),
            )
            .unwrap();
            s.pending_followup_refresh = true;
            let snapshot = s.checkpoint(&p).unwrap();
            let mut q = Policy::plans(-1, 604, 0.01).unwrap();
            let restored = State::restore(&snapshot, &mut q).unwrap();
            assert_eq!(snapshot, restored.checkpoint(&q).unwrap());
            let slot = restored.accepted.slots[4].as_ref().unwrap();
            assert_eq!(slot.weights, primary);
            assert_eq!(slot.followup, Some(follower));
            assert_eq!(restored.followup_weights, p.weights_json().unwrap());
            assert!(restored.pending_followup_refresh);
            assert_ne!(restored.followup_weights, slot.weights);
        }
        #[test]
        fn follower_refresh_retains_only_actual_conditional_outcomes() {
            tensor::worker_threads();
            let p = Policy::plans(-1, 701, 0.0003).unwrap();
            let mut s = state(&p);
            let primary = retained_pair("primary", 7);
            let followup = retained_pair("followup", 8);
            s.bank.admit(&[primary, followup.clone()]);
            assert_eq!(s.bank.elite.len(), 2);
            let deployed = s.accepted.clone();
            let q = Policy::plans(-1, 702, 0.0003).unwrap();
            s.pending_followup_refresh = true;
            s.refresh_followup(&q).unwrap();
            assert!(!s.pending_followup_refresh);
            assert_eq!(s.followup_version, 2);
            assert_eq!(s.followup_weights, q.weights_json().unwrap());
            assert_eq!(s.accepted, deployed);
            for retained in [&s.bank.elite, &s.bank.recent] {
                assert_eq!(retained.len(), 1);
                assert_eq!(retained[0].json(), followup.json());
                assert_eq!(retained[0].improvement_target().unwrap(), -1.);
            }
        }
        fn choose_feature_weights(feature: usize) -> Json {
            let mut p = Policy::plans(-1, 802, 0.0003).unwrap();
            for (i, param) in p.parameters.iter_mut().enumerate() {
                let size = param.shape.iter().product::<i64>() as usize;
                let mut data = vec![0.; size];
                if i == 4 {
                    data[feature] = 1.;
                }
                if i == 6 || i == 8 {
                    data[0] = 1.;
                }
                param.value = tensor::Tensor::floats(&data, &param.shape, -1, true).unwrap();
            }
            p.weights_json().unwrap()
        }
        #[test]
        fn forced_primary_and_paired_deployment_share_physical_continuation() {
            tensor::worker_threads();
            let primary = choose_feature_weights(3); // CARROT
            let follower = choose_feature_weights(4); // TOMATO
            let accepted = Portfolio::empty();
            let mut branch_runtime = Runtime::load(&accepted, -1).unwrap();
            branch_runtime.override_followup(0, &follower).unwrap();
            let candidate = accepted.propose_paired(0, 1, primary, follower).unwrap();
            let deployed_runtime = Runtime::load(&candidate, -1).unwrap();
            let mut a = World::new(91, &Config::default());
            a.game.step = 73;
            a.game.farms[0].money = 0.;
            a.game.farms[0].farmer = (4, 4);
            for x in [2, 3] {
                a.game.farms[0].tiles[4][x] = kagg_engine::state::Cell::Plant {
                    crop: "WHEAT".into(),
                    planted_day: 0,
                    watered_today: true,
                    consecutive_unwatered: 0,
                    yield_units: 4,
                    max_lifespan_step: 144,
                    fertilized_until_day: -1,
                };
            }
            let mut b = a.clone();
            let job = Job {
                seed: 91,
                seat: 0,
                opponent: 0,
                rng: 1,
                slots: vec![],
            };
            let o = Observation::from_state(&a.game, 0);
            let da = a.own.prepare(&o, &branch_runtime).unwrap().unwrap();
            let db = b.own.prepare(&o, &deployed_runtime).unwrap().unwrap();
            assert_ne!(da.selected, db.selected);
            assert!(!db.choices[db.selected].keep);
            let choice = da.choices[db.selected].clone();
            let x = a.own.execute_choice(&o, choice, &branch_runtime).unwrap();
            a.own.arm_followup(0, 0, o.step);
            let y = b
                .own
                .execute_choice(&o, db.choices[db.selected].clone(), &deployed_runtime)
                .unwrap();
            a.advance(&job, x, &[]).unwrap();
            b.advance(&job, y, &[]).unwrap();
            assert_eq!(a.game.digest(), b.game.digest());
            let mut followers = 0;
            for _ in 0..48 {
                let oa = Observation::from_state(&a.game, 0);
                let ob = Observation::from_state(&b.game, 0);
                let da = a.own.prepare(&oa, &branch_runtime).unwrap();
                let db = b.own.prepare(&ob, &deployed_runtime).unwrap();
                assert_eq!(
                    da.as_ref().map(|d| (d.slot, d.followup, d.selected)),
                    db.as_ref().map(|d| (d.slot, d.followup, d.selected))
                );
                followers += usize::from(da.as_ref().is_some_and(|d| d.followup));
                let x = match da {
                    Some(d) => a
                        .own
                        .execute_choice(&oa, d.choices[d.selected].clone(), &branch_runtime)
                        .unwrap(),
                    None => a.own.continue_action(&oa, &branch_runtime).unwrap(),
                };
                let y = match db {
                    Some(d) => b
                        .own
                        .execute_choice(&ob, d.choices[d.selected].clone(), &deployed_runtime)
                        .unwrap(),
                    None => b.own.continue_action(&ob, &deployed_runtime).unwrap(),
                };
                a.advance(&job, x, &[]).unwrap();
                b.advance(&job, y, &[]).unwrap();
                assert_eq!(a.game.digest(), b.game.digest());
            }
            assert_eq!(
                followers, 1,
                "the paired choice must really execute once, not merely serialize"
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
