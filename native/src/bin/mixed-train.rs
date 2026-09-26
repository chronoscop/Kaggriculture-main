#[cfg(feature = "train")]
fn run() -> Result<(), String> {
    use kagg_engine::json::{self, Json};
    use route_rl_native::{
        learning::{
            policy::{Policy, Rng},
            tensor,
        },
        pipeline::{
            self,
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
    let args: Vec<_> = std::env::args().skip(1).collect();
    let help="mixed-train --out DIR [--iterations 2 --games-per-update 16 --workers 7 --device cuda --epochs 2 --batch-size 256 --seed 1200 --opponent heuristic|selfplay --resume CHECKPOINT --trace 0|1 --mode train|evaluate]";
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
        "--resume",
        "--trace",
        "--mode",
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
    let iterations = n("--iterations", 2)?;
    let games = n("--games-per-update", 16)?;
    let workers = n("--workers", resources::available_workers())?;
    let epochs = n("--epochs", 2)?;
    let batch = n("--batch-size", 256)?;
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
    let opponent = match get("--opponent").unwrap_or("heuristic") {
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
    if evaluate && get("--resume").is_none() {
        return Err("evaluation requires --resume CHECKPOINT".into());
    }
    let out = PathBuf::from(get("--out").ok_or(help)?);
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    if out.join("metrics.jsonl").exists() && get("--resume").is_none() {
        return Err("output already contains a run; use --resume or a new directory".into());
    }
    if evaluate && out.join("metrics.jsonl").exists() {
        return Err("use a fresh evaluation output directory".into());
    }
    tensor::threads(1);
    let mut policy = Policy::mixed_routes(device, seed, 1e-4)?;
    let mut rng = Rng(seed);
    let mut first = 1;
    if let Some(path) = get("--resume") {
        let checkpoint = json::parse(&fs::read_to_string(path).map_err(|e| e.to_string())?)?;
        let (iteration, saved_rng) = policy.restore(&checkpoint)?;
        if evaluate {
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
        }

        if !evaluate {
            let config = checkpoint.get("run");
            if config.get("seed").str() != seed.to_string()
                || config.get("games_per_update").i64() != games as i64
                || config.get("opponent").str() != opponent.name()
                || config.get("epochs").i64() != epochs as i64
                || config.get("batch_size").i64() != batch as i64
            {
                return Err(
                    "resume must preserve seed, games, opponent, epochs and batch size".into(),
                );
            }
            first = iteration as usize + 1;
            rng = saved_rng;
        }
    }
    if first > iterations && !evaluate {
        return Err("iterations is cumulative and must exceed checkpoint iteration".into());
    }
    let config=Json::Obj(vec![("schema".into(),Json::Str(pipeline::SCHEMA.into())),("seed".into(),Json::Str(seed.to_string())),("games_per_update".into(),Json::Num(games as f64)),("workers".into(),Json::Num(workers as f64)),("opponent".into(),Json::Str(opponent.name().into())),("epochs".into(),Json::Num(epochs as f64)),("batch_size".into(),Json::Num(batch as f64)),("reward".into(),Json::Str("actual terminal own cash minus opponent cash, divided by 10000; no baseline reference".into())),("planner".into(),Json::Str("bounded mixed route insertion; rule sale settlement; learned project/route choices".into()))]);
    write(&out.join("manifest.json"), &config)?;
    let mut metrics = OpenOptions::new()
        .append(true)
        .create(true)
        .open(out.join("metrics.jsonl"))
        .map_err(|e| e.to_string())?;
    for iteration in first..=if evaluate { first } else { iterations } {
        let seeds: Vec<_> = (0..games / 2)
            .map(|i| (seed + ((iteration - 1) * games / 2 + i) as u64) as i64)
            .collect();
        println!("{{\"stage\":\"collect\",\"iteration\":{iteration},\"games\":{games},\"workers\":{workers}}}");
        let result = rollout::collect(
            &policy, &seeds, workers, opponent, evaluate, &mut rng, trace,
        )?;
        let started = Instant::now();
        let update = if evaluate {
            None
        } else {
            Some(policy.update(&result.samples, epochs, batch, &mut rng)?)
        };
        let update_seconds = started.elapsed().as_secs_f64();
        if !evaluate {
            if update.as_ref().unwrap().updates == 0 {
                return Err("no PPO update completed".into());
            }
            let mut checkpoint = policy.checkpoint(iteration as u64, &rng)?;
            if let Json::Obj(fields) = &mut checkpoint {
                fields.push(("run".into(), config.clone()));
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
        let report = Json::Obj(vec![
            ("schema".into(), Json::Str(pipeline::SCHEMA.into())),
            ("iteration".into(), Json::Num(iteration as f64)),
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
                "games".into(),
                Json::Arr(result.games.iter().map(|g| g.report()).collect()),
            ),
        ]);
        writeln!(metrics, "{}", report.dump()).map_err(|e| e.to_string())?;
        metrics.flush().map_err(|e| e.to_string())?;
        println!("{}", report.dump());
    }
    Ok(())
}
#[cfg(feature = "train")]
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("requires --features train");
    std::process::exit(1);
}
