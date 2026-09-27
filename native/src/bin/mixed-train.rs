#[cfg(feature = "train")]
fn run(args: Vec<String>) -> Result<(), String> {
    use kagg_engine::json::{self, Json};
    use route_rl_native::{
        learning::{
            experience::ExperienceBank,
            policy::{Policy, Rng},
            tensor,
        },
        pipeline::{
            self,
            league::League,
            rollout::{self, Opponent},
        },
        resources,
    };
    use std::{
        fs::{self, OpenOptions},
        io::Write,
        path::{Path, PathBuf},
        time::Instant,
    };
    fn write(path: &Path, value: &Json) -> Result<(), String> {
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| e.to_string())?;
        f.write_all(value.dump().as_bytes())
            .map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        fs::rename(tmp, path).map_err(|e| e.to_string())
    }
    let help="mixed-train --out DIR [--iterations 2 --games-per-update 16 --workers 7 --device cuda --epochs 2 --batch-size 256 --seed 1200 --opponent league|heuristic|selfplay --opponent-checkpoint FILE(evaluate-only) --eval-every 5 --eval-games 8 --eval-seed 1000000000 --exploration 0.2 --imitation-weight 0.05 --market-mode learned|rule --resume CHECKPOINT --trace 0|1 --mode train|evaluate]";
    if args.iter().any(|s| s == "--help") {
        println!("{help}");
        return Ok(());
    }
    let allowed = [
        "--out",
        "--iterations",
        "--games-per-update",
        "--workers",
        "--device",
        "--epochs",
        "--batch-size",
        "--seed",
        "--opponent",
        "--opponent-checkpoint",
        "--resume",
        "--trace",
        "--mode",
        "--eval-every",
        "--eval-games",
        "--eval-seed",
        "--exploration",
        "--imitation-weight",
        "--market-mode",
    ];
    if args.len() % 2 != 0 || args.chunks(2).any(|p| !allowed.contains(&p[0].as_str())) {
        return Err(help.into());
    }
    let mut keys = std::collections::BTreeSet::new();
    if args.chunks(2).any(|p| !keys.insert(p[0].clone())) {
        return Err("duplicate option".into());
    }
    let get = |k: &str| args.windows(2).find(|p| p[0] == k).map(|p| p[1].as_str());
    let n = |k: &str, d: usize| {
        get(k)
            .map(|s| s.parse::<usize>().map_err(|_| format!("invalid {k}")))
            .unwrap_or(Ok(d))
    };
    let exploration = get("--exploration")
        .unwrap_or("0.2")
        .parse::<f32>()
        .map_err(|_| "invalid exploration")?;
    let imitation_weight = get("--imitation-weight")
        .unwrap_or("0.05")
        .parse::<f64>()
        .map_err(|_| "invalid imitation-weight")?;
    if !exploration.is_finite()
        || !(0. ..=1.).contains(&exploration)
        || !imitation_weight.is_finite()
        || !(0. ..=1.).contains(&imitation_weight)
    {
        return Err("exploration and imitation-weight must be in [0,1]".into());
    }
    let market_mode =
        pipeline::trading::MarketMode::parse(get("--market-mode").unwrap_or("learned"))?;
    let iterations = n("--iterations", 2)?;
    let games = n("--games-per-update", 16)?;
    let workers = n("--workers", resources::available_workers())?;
    let epochs = n("--epochs", 2)?;
    let batch = n("--batch-size", 256)?;
    let eval_every = n("--eval-every", 5)?;
    let eval_games = n("--eval-games", 8)?;
    let eval_seed = get("--eval-seed")
        .unwrap_or("1000000000")
        .parse::<u64>()
        .map_err(|_| "invalid eval-seed")?;
    if eval_every == 0 || eval_games < 2 || eval_games % 2 != 0 {
        return Err("eval-every must be positive; eval-games must be even and >= 2".into());
    }
    if iterations == 0 || games < 2 || games % 2 != 0 || workers == 0 || epochs == 0 || batch == 0 {
        return Err("positive counts and even games-per-update >= 2 required".into());
    }
    let seed = get("--seed")
        .unwrap_or("1200")
        .parse::<u64>()
        .map_err(|_| "invalid seed")?;
    if seed as u128 + iterations as u128 * games as u128 > i64::MAX as u128 {
        return Err("seed overflow".into());
    }
    let device = match get("--device").unwrap_or("cuda") {
        "cpu" => -1,
        "cuda" => 0,
        _ => return Err("invalid device".into()),
    };
    let mut opponent = match get("--opponent").unwrap_or("league") {
        "league" => Opponent::League,
        "heuristic" => Opponent::Heuristic,
        "selfplay" => Opponent::SelfPlay,
        _ => return Err("invalid opponent".into()),
    };
    let trace = match get("--trace").unwrap_or("0") {
        "0" => false,
        "1" => true,
        _ => return Err("trace must be 0 or 1".into()),
    };
    let evaluate = match get("--mode").unwrap_or("train") {
        "train" => false,
        "evaluate" => true,
        _ => return Err("invalid mode".into()),
    };
    if get("--opponent-checkpoint").is_some() {
        if !evaluate || get("--opponent").is_some() {
            return Err("opponent-checkpoint requires --mode evaluate and must not be combined with --opponent".into());
        }
        opponent = Opponent::Frozen(0);
    }
    if evaluate && get("--resume").is_none() {
        return Err(
            "evaluation requires --exploration 0.2 --imitation-weight 0.05 --resume CHECKPOINT"
                .into(),
        );
    }
    let train_end = seed as u128 + iterations as u128 * games as u128 / 2;
    let eval_end = eval_seed as u128 + eval_games as u128 / 2;
    if eval_end > i64::MAX as u128
        || (!evaluate && (eval_seed as u128) < train_end && eval_end > seed as u128)
    {
        return Err("validation seeds overlap training seeds or overflow".into());
    }
    let rotate_start = eval_seed as u128 + 1_000_000;
    let rotate_end = rotate_start + (iterations as u128 + 1) * (4 * (eval_games / 2)) as u128;
    if rotate_end > i64::MAX as u128
        || (!evaluate && (seed as u128) < rotate_end && train_end > rotate_start)
    {
        return Err("rotating validation seeds overlap training seeds or overflow".into());
    }
    let out = PathBuf::from(get("--out").ok_or(help)?);
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    if out.join("metrics.jsonl").exists() && get("--resume").is_none() {
        return Err("output already contains a run; use --resume or a new directory".into());
    }
    if evaluate && out.join("metrics.jsonl").exists() {
        return Err("use a fresh evaluation output directory".into());
    }
    let validate_evaluation_seeds = |checkpoint: &Json, iteration: u64| -> Result<(), String> {
        let config = checkpoint.get("run");
        let start = config
            .get("seed")
            .str()
            .parse::<u64>()
            .map_err(|_| "checkpoint lacks training seed provenance")?;
        let count = (iteration as u128) * (config.get("games_per_update").i64() as u128) / 2;
        let end = start as u128 + count;
        if (seed as u128) < end && (seed as u128 + games as u128 / 2) > start as u128 {
            return Err("evaluation seeds overlap training seeds".into());
        }
        let fixed = config
            .get("eval_seed")
            .str()
            .parse::<u128>()
            .map_err(|_| "invalid validation provenance")?;
        let n = config.get("eval_games").i64() as u128 / 2;
        let requested_end = seed as u128 + games as u128 / 2;
        let overlaps = |lo: u128, hi: u128| (seed as u128) < hi && requested_end > lo;
        if overlaps(fixed, fixed + n)
            || (config.get("evaluation_schedule").str() == "fixed-rotating-confirm-v1"
                && overlaps(
                    fixed + 1_000_000,
                    fixed + 1_000_000 + (iteration as u128 + 1) * 4 * n,
                ))
        {
            return Err("independent evaluation seeds overlap model-selection seeds".into());
        }
        Ok(())
    };
    tensor::threads(1);
    let mut policy = Policy::mixed_routes(device, seed, 1e-4)?;
    policy.market_mode = market_mode;
    let mut rng = Rng(seed);
    let mut first = 1;
    let mut league = League::new(&policy)?;
    let mut bank = ExperienceBank::default();
    if let Some(path) = get("--resume") {
        let checkpoint = json::parse(&fs::read_to_string(path).map_err(|e| e.to_string())?)?;
        let (iteration, saved_rng) = policy.restore(&checkpoint)?;
        if (!evaluate || get("--market-mode").is_some()) && policy.market_mode != market_mode {
            return Err("resume must preserve market-mode".into());
        }
        league = League::restore(checkpoint.get("league"))?;
        bank = ExperienceBank::restore(checkpoint.get("experience_bank"))?;
        if evaluate {
            validate_evaluation_seeds(&checkpoint, iteration)?;
        }

        if !evaluate {
            let config = checkpoint.get("run");
            if config.get("training_revision").str() != pipeline::TRAINING_REVISION {
                return Err("unsupported training revision; use a matching trainer".into());
            }
            if config.get("seed").str() != seed.to_string()
                || config.get("games_per_update").i64() != games as i64
                || config.get("opponent").str() != opponent.name()
                || config.get("epochs").i64() != epochs as i64
                || config.get("batch_size").i64() != batch as i64
                || config.get("eval_every").i64() != eval_every as i64
                || config.get("eval_games").i64() != eval_games as i64
                || config.get("eval_seed").str() != eval_seed.to_string()
                || config.get("exploration").f64() as f32 != exploration
                || config.get("imitation_weight").f64() != imitation_weight
            {
                return Err(
                    "resume must preserve seed, games, opponent, epochs, batch size, exploration, imitation and evaluation settings".into(),
                );
            }
            first = iteration as usize + 1;
            rng = saved_rng;
        }
    }
    if first > iterations && !evaluate {
        return Err("iterations is cumulative and must exceed checkpoint iteration".into());
    }
    let mut config=Json::Obj(vec![("schema".into(),Json::Str(pipeline::SCHEMA.into())),("seed".into(),Json::Str(seed.to_string())),("games_per_update".into(),Json::Num(games as f64)),("workers".into(),Json::Num(workers as f64)),("opponent".into(),Json::Str(opponent.name().into())),("epochs".into(),Json::Num(epochs as f64)),("batch_size".into(),Json::Num(batch as f64)),("eval_every".into(),Json::Num(eval_every as f64)),("eval_games".into(),Json::Num(eval_games as f64)),("eval_seed".into(),Json::Str(eval_seed.to_string())),("exploration".into(),Json::Num(exploration as f64)),("imitation_weight".into(),Json::Num(imitation_weight)),("greedy_probe_fraction".into(),Json::Num(0.25)),("gae_lambda_per_step".into(),Json::Num(0.997)),("reward".into(),Json::Str("actual own cash changes / 10000; gamma=1, time-aware GAE; terminal value=0".into())),("planner".into(),Json::Str("bounded mixed route insertion; committed production dispatch; learned project/route choices; market-mode controls sales and product procurement".into()))]);
    config.set_path(
        "opponent_checkpoint",
        get("--opponent-checkpoint")
            .map(|p| Json::Str(p.into()))
            .unwrap_or(Json::Null),
    );
    config.set_path("market_mode", Json::Str(policy.market_mode.name().into()));
    config.set_path(
        "market_schedule",
        Json::Str(
            if policy.market_mode == pipeline::trading::MarketMode::Learned {
                "periodic4-and-observed-events-including-rival-supply-final-day-v3"
            } else {
                "legacy-automatic-market"
            }
            .into(),
        ),
    );
    config.set_path("market_choices", Json::Str(if policy.market_mode == pipeline::trading::MarketMode::Learned { "joint-product-direction-size-global-stop; fractions10-25-50-75-100-plus-unit-and-needs; one-order-per-product" } else { "legacy-fixed-reserves-and-procurement" }.into()));
    config.set_path("public_supply_features", Json::Str("observed-rival-tiles; first-maturity-steps-and-24-72-step-cohorts; 4-24-step-count-yield-deltas; observed-24-step-additions-removals; no-sales-forecast".into()));
    config.set_path(
        "training_revision",
        Json::Str(pipeline::TRAINING_REVISION.into()),
    );
    config.set_path(
        "network_architecture",
        Json::Str("independent-actor-critic".into()),
    );
    config.set_path(
        "gradient_clipping",
        Json::Str("separate-actor-critic-norm-0.5".into()),
    );
    config.set_path("opponent_pool_capacity", Json::Num(8.));
    config.set_path(
        "pfsp_difficulty_source",
        Json::Str("greedy_training_probes".into()),
    );
    config.set_path(
        "exploration_schedule",
        Json::Str("policy18-broad6-greedy8-per32".into()),
    );
    config.set_path(
        "focused_exploration",
        Json::Num(rollout::FOCUSED_EXPLORATION as f64),
    );
    config.set_path(
        "opponent_schedule",
        Json::Str("recent12-pfsp12-coverage4-current2-heuristic2".into()),
    );
    config.set_path(
        "snapshot_every",
        Json::Num(pipeline::league::SNAPSHOT_EVERY as f64),
    );
    config.set_path(
        "pfsp_half_life_iterations",
        Json::Num(pipeline::matchmaking::HALF_LIFE),
    );
    config.set_path(
        "pfsp_prior_games",
        Json::Num(pipeline::matchmaking::PRIOR_GAMES),
    );
    config.set_path(
        "pfsp_uniform_mix",
        Json::Num(pipeline::matchmaking::UNIFORM_MIX),
    );
    config.set_path(
        "pfsp_max_share",
        Json::Num(pipeline::matchmaking::MAX_SHARE),
    );
    config.set_path("experience_capacity", Json::Num(192.));
    config.set_path(
        "evaluation_schedule",
        Json::Str("fixed-rotating-confirm-v1".into()),
    );
    config.set_path(
        "resume_from",
        get("--resume")
            .map(|s| Json::Str(s.into()))
            .unwrap_or(Json::Null),
    );
    write(&out.join("manifest.json"), &config)?;
    let mut metrics = OpenOptions::new()
        .append(true)
        .create(true)
        .open(out.join("metrics.jsonl"))
        .map_err(|e| e.to_string())?;
    let eval_seeds: Vec<_> = (0..eval_games / 2)
        .map(|i| (eval_seed + i as u64) as i64)
        .collect();
    // Frozen models stay on the coordinator device and are rebuilt after pool admission.
    if !evaluate && opponent == Opponent::League {
        league.freeze_recent(&policy, (first - 1) as u64, true)?;
    }
    let mut pool = league.policies_with_mode(device, policy.market_mode)?;
    if let Some(path) = get("--opponent-checkpoint") {
        let checkpoint = json::parse(&fs::read_to_string(path).map_err(|e| e.to_string())?)?;
        let mut rival = Policy::mixed_routes(device, 0, 1e-4)?;
        let (iteration, _) = rival.restore(&checkpoint)?;
        validate_evaluation_seeds(&checkpoint, iteration)?;
        league = League::new(&rival)?;
        league.snapshots[0].iteration = iteration;
        league.champion_iteration = iteration;
        pool = vec![rival];
    }
    if !evaluate && opponent == Opponent::League {
        league.ensure_profiles(&pool, &eval_seeds, workers, eval_seed)?;
    }
    for iteration in first..=if evaluate { first } else { iterations } {
        let seeds: Vec<_> = (0..games / 2)
            .map(|i| (seed + ((iteration - 1) * games / 2 + i) as u64) as i64)
            .collect();
        println!("{{\"stage\":\"collect\",\"iteration\":{iteration},\"games\":{games},\"workers\":{workers}}}");
        let pool_before = Json::Arr(
            league
                .snapshots
                .iter()
                .map(|s| Json::Num(s.iteration as f64))
                .collect(),
        );
        let matchmaking_before = league.matchmaking_report(iteration as u64);
        let result = if evaluate {
            rollout::collect_with_pool(
                &policy, &seeds, workers, opponent, &pool, true, &mut rng, trace,
            )?
        } else if opponent == Opponent::League {
            rollout::collect_roster(
                &policy,
                &seeds,
                workers,
                &pool,
                &mut rng,
                trace,
                exploration,
                &league.roster(iteration as u64),
            )?
        } else {
            rollout::collect_exploring(
                &policy,
                &seeds,
                workers,
                opponent,
                &pool,
                false,
                &mut rng,
                trace,
                exploration,
            )?
        };
        if !evaluate && opponent == Opponent::League {
            league.observe_training(&result, iteration as u64);
        }
        // Record against pre-admission slot identities, before pruning can shift indices.
        let opponent_outcomes = league.matchmaking_report(iteration as u64);
        let mut recent_snapshot_added = false;
        let started = Instant::now();
        let update = if evaluate {
            None
        } else {
            Some(policy.update(&result.samples, epochs, batch, &mut rng)?)
        };
        let ppo_seconds = started.elapsed().as_secs_f64();
        let mut imitation = (0, 0.);
        if !evaluate {
            for e in &result.experiences {
                let mut e = e.clone();
                e.collected_iteration = iteration as u64;
                bank.insert(e);
            }
            let replay = bank.sample(batch.min(256), &mut rng);
            imitation = policy.imitate(&replay, imitation_weight)?;
        }
        let update_seconds = started.elapsed().as_secs_f64();
        if !evaluate {
            if update.as_ref().unwrap().updates == 0 {
                return Err("no PPO update completed".into());
            }
            let mut promoted = false;
            let mut pool_changed = false;
            if iteration % eval_every == 0 || iteration == iterations {
                println!("{{\"stage\":\"validate\",\"iteration\":{iteration}}}");
                let (accepted, report) = league.evaluate_and_promote(
                    &policy,
                    iteration as u64,
                    &eval_seeds,
                    workers,
                    eval_seed,
                )?;
                promoted = accepted;
                let mut evaluations = OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(out.join("evaluations.jsonl"))
                    .map_err(|e| e.to_string())?;
                writeln!(evaluations, "{}", report.dump()).map_err(|e| e.to_string())?;
                println!("{}", report.dump());
                pool_changed = report.get("pool_admitted").bool();
            }
            if opponent == Opponent::League {
                recent_snapshot_added = league.freeze_recent(&policy, iteration as u64, false)?;
                pool_changed |= recent_snapshot_added;
            }
            if pool_changed {
                pool = league.policies_with_mode(device, policy.market_mode)?;
            }
            let mut checkpoint = policy.checkpoint(iteration as u64, &rng)?;
            if let Json::Obj(fields) = &mut checkpoint {
                fields.push(("run".into(), config.clone()));
                fields.push(("league".into(), league.json()));
                fields.push(("experience_bank".into(), bank.json()));
            }
            if promoted {
                write(&out.join("best.json"), &checkpoint)?;
            }
            write(&out.join("latest.json"), &checkpoint)?;
        }
        if trace {
            for g in &result.games {
                let text = g
                    .trace
                    .iter()
                    .map(Json::dump)
                    .collect::<Vec<_>>()
                    .join("\n");
                fs::write(
                    out.join(format!(
                        "trace_{}_{}_{}.jsonl",
                        iteration, g.seed, g.learner
                    )),
                    text,
                )
                .map_err(|e| e.to_string())?;
            }
        }
        let game_reports = Json::Arr(
            result
                .games
                .iter()
                .map(|g| {
                    let mut report = g.report();
                    report.set_path(
                        "opponent_iteration",
                        match g.opponent {
                            Opponent::Frozen(i) => pool_before.arr()[i].clone(),
                            _ => Json::Null,
                        },
                    );
                    report
                })
                .collect(),
        );
        let report = Json::Obj(vec![
            ("schema".into(), Json::Str(pipeline::SCHEMA.into())),
            ("iteration".into(), Json::Num(iteration as f64)),
            (
                "pool_iterations".into(),
                Json::Arr(
                    league
                        .snapshots
                        .iter()
                        .map(|s| Json::Num(s.iteration as f64))
                        .collect(),
                ),
            ),
            ("opponent_pool_before".into(), pool_before),
            ("matchmaking_before".into(), matchmaking_before),
            ("opponent_outcomes".into(), opponent_outcomes),
            (
                "exploration_summary".into(),
                rollout::exploration_summary(&result),
            ),
            (
                "recent_snapshot_added".into(),
                Json::Bool(recent_snapshot_added),
            ),
            (
                "training_revision".into(),
                Json::Str(pipeline::TRAINING_REVISION.into()),
            ),
            (
                "champion_iteration".into(),
                Json::Num(league.champion_iteration as f64),
            ),
            ("experience_composition".into(), bank.summary()),
            ("samples".into(), Json::Num(result.samples.len() as f64)),
            ("rollout_seconds".into(), Json::Num(result.seconds)),
            (
                "inference_seconds".into(),
                Json::Num(result.inference_seconds),
            ),
            (
                "inference_calls".into(),
                Json::Num(result.inference_calls as f64),
            ),
            ("mean_inference_batch".into(), Json::Num(result.mean_batch)),
            ("update_seconds".into(), Json::Num(update_seconds)),
            ("ppo_seconds".into(), Json::Num(ppo_seconds)),
            (
                "experience_episodes".into(),
                Json::Num(bank.episodes.len() as f64),
            ),
            (
                "new_profitable_episodes".into(),
                Json::Num(result.experiences.len() as f64),
            ),
            (
                "market_mode".into(),
                Json::Str(policy.market_mode.name().into()),
            ),
            (
                "trade_samples".into(),
                Json::Num(
                    result
                        .samples
                        .iter()
                        .filter(|r| r.features[r.action][31] >= 16.)
                        .count() as f64,
                ),
            ),
            ("imitation_samples".into(), Json::Num(imitation.0 as f64)),
            ("imitation_loss".into(), Json::Num(imitation.1)),
            (
                "greedy_probe_games".into(),
                Json::Num(result.games.iter().filter(|g| g.greedy_probe).count() as f64),
            ),
            (
                "updates".into(),
                Json::Num(update.as_ref().map(|u| u.updates).unwrap_or(0) as f64),
            ),
            (
                "loss".into(),
                update
                    .as_ref()
                    .map(|u| Json::Num(u.loss))
                    .unwrap_or(Json::Null),
            ),
            (
                "kl_stopped".into(),
                Json::Bool(update.as_ref().is_some_and(|u| u.kl_stopped)),
            ),
            (
                "policy_loss".into(),
                update
                    .as_ref()
                    .map(|u| Json::Num(u.policy_loss))
                    .unwrap_or(Json::Null),
            ),
            (
                "value_loss".into(),
                update
                    .as_ref()
                    .map(|u| Json::Num(u.value_loss))
                    .unwrap_or(Json::Null),
            ),
            (
                "policy_entropy".into(),
                update
                    .as_ref()
                    .map(|u| Json::Num(u.entropy))
                    .unwrap_or(Json::Null),
            ),
            (
                "mean_ppo_kl".into(),
                update
                    .as_ref()
                    .map(|u| Json::Num(u.mean_kl))
                    .unwrap_or(Json::Null),
            ),
            ("games".into(), game_reports),
        ]);
        writeln!(metrics, "{}", report.dump()).map_err(|e| e.to_string())?;
        metrics.flush().map_err(|e| e.to_string())?;
        println!("{}", report.dump());
    }
    Ok(())
}
#[cfg(feature = "train")]
fn main() {
    if let Err(e) = run(std::env::args().skip(1).collect()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("requires --features train");
    std::process::exit(1);
}

#[cfg(all(test, feature = "train"))]
mod tests {
    #[test]
    fn cli_checkpoint_resumes_complete_learning_state() {
        use kagg_engine::json;
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("mixed-v7-cli-{}-{unique}", std::process::id()));
        let base: Vec<String> = vec![
            "--out".into(),
            dir.to_string_lossy().into_owned(),
            "--device".into(),
            "cpu".into(),
            "--games-per-update".into(),
            "2".into(),
            "--workers".into(),
            "2".into(),
            "--epochs".into(),
            "1".into(),
            "--batch-size".into(),
            "64".into(),
            "--eval-games".into(),
            "2".into(),
            "--eval-every".into(),
            "1".into(),
            "--seed".into(),
            "76001".into(),
        ];
        let mut first = base.clone();
        first.extend(["--iterations".into(), "1".into()]);
        super::run(first).unwrap();
        let checkpoint = dir.join("latest.json");
        let ck = json::parse(&std::fs::read_to_string(&checkpoint).unwrap()).unwrap();
        assert_eq!(ck.get("schema").str(), "mixed-production-v8-ppo-v1");
        assert!(ck.get("experience_bank").is_arr());
        assert!(ck.get("league").is_arr());
        let mut resumed = base;
        resumed.extend([
            "--iterations".into(),
            "2".into(),
            "--resume".into(),
            checkpoint.to_string_lossy().into_owned(),
        ]);
        let mut wrong = resumed.clone();
        wrong.extend(["--exploration".into(), "0.1".into()]);
        assert!(super::run(wrong)
            .unwrap_err()
            .contains("resume must preserve"));
        let mut wrong_mode = resumed.clone();
        wrong_mode.extend(["--market-mode".into(), "rule".into()]);
        assert!(super::run(wrong_mode)
            .unwrap_err()
            .contains("preserve market-mode"));
        super::run(resumed).unwrap();
        let ck = json::parse(&std::fs::read_to_string(&checkpoint).unwrap()).unwrap();
        assert_eq!(ck.get("iteration").str(), "2");
        assert_eq!(
            std::fs::read_to_string(dir.join("metrics.jsonl"))
                .unwrap()
                .lines()
                .count(),
            2
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("evaluations.jsonl"))
                .unwrap()
                .lines()
                .count(),
            2
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
