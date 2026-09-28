//! Bounded prototype evaluation and full-season parameter search, entirely Rust.
#[cfg(feature = "train")]
mod app {
    use kagg_engine::{
        engine::{self, PlayerAction},
        json::{self, Json},
        obsstate,
        state::State,
    };
    use route_rl_native::{
        learning::{
            policy::{Policy, Rng, Sample},
            tensor,
        },
        pipeline::{
            encoding,
            executor::*,
            plan_prototype::{Agent, Config},
            planner,
            trading::MarketMode,
        },
    };
    use std::{
        io::{BufRead, Write},
        path::PathBuf,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::Instant,
    };
    fn number(j: &Json, k: &str) -> f64 {
        j.get(k).f64()
    }
    fn legacy_action(
        model: &Policy,
        e: &mut Executor,
        o: &Observation,
        rng: &mut Rng,
    ) -> Result<PlayerAction, String> {
        e.market_mode = MarketMode::Rule;
        e.observe(o);
        let choose = |p: planner::Problem,
                      e: &mut Executor,
                      rng: &mut Rng|
         -> Result<Vec<Vec<String>>, String> {
            let (context, features) = encoding::encode(o, e, &p);
            let d = model.infer(
                &[Sample {
                    context,
                    features,
                    ..Sample::default()
                }],
                true,
                rng,
            )?;
            p.select(d[0].action, e, o)
        };
        let mut orders = vec![];
        if e.market_due(o) {
            e.last_market = o.step;
            e.last_cash = o.farm.money;
            orders = choose(planner::investment_problem(o, e), e, rng)?;
        }
        for actor in 0..o.private.inventories.len() {
            if e.routes
                .get(actor)
                .and_then(Option::as_ref)
                .is_none_or(|r| r.steps.is_empty())
            {
                choose(planner::route_problem(o, e, actor), e, rng)?;
            }
        }
        Ok(e.action(o, orders))
    }
    #[derive(Clone)]
    struct Job {
        seed: i64,
        seat: usize,
        variant: String,
        config: Config,
        opponent: Option<Config>,
        opponent_id: usize,
    }
    fn play(job: &Job, model: &Policy) -> Result<Json, String> {
        let start = Instant::now();
        let mut state = State::new(job.seed);
        let mut old = [Executor::new(), Executor::new()];
        let mut proto = Agent::new(job.config.clone());
        let mut rng = Rng(0);
        let mut rival = job.opponent.clone().map(Agent::new);
        let mut opening = 0;
        let mut peak = 0;
        let mut workers = 0;
        let mut plantings = 0;
        let mut replants = 0;
        let mut last_harvest = std::collections::BTreeSet::new();
        while state.step < 719 {
            let mut actions: [PlayerAction; 2] = Default::default();
            for seat in 0..2 {
                let o = Observation::from_state(&state, seat);
                actions[seat] = if seat == job.seat && job.variant != "legacy" {
                    proto.action(&o)
                } else if seat != job.seat && rival.is_some() {
                    rival.as_mut().unwrap().action(&o)
                } else {
                    legacy_action(model, &mut old[seat], &o, &mut rng)?
                };
                if seat == job.seat && job.variant == "legacy" {
                    let n = o
                        .farm
                        .tiles
                        .iter()
                        .flatten()
                        .filter(|t| {
                            matches!(
                                t,
                                kagg_engine::state::Cell::Plant { .. }
                                    | kagg_engine::state::Cell::Structure {
                                        animal: Some(_),
                                        ..
                                    }
                            )
                        })
                        .count();
                    peak = peak.max(n);
                    workers = workers.max(o.farm.hands.len() + 1);
                    if state.step == 48 {
                        opening = n;
                    }
                }
            }
            let before = state.farms[job.seat].clone();
            let before_private = state.private[job.seat].clone();
            engine::step(&mut state, &actions);
            let cmds =
                std::iter::once(&actions[job.seat].farmer).chain(actions[job.seat].hands.iter());
            for (actor, cmd) in cmds.enumerate().take(before.hands.len() + 1) {
                let p = pos(&before, actor);
                if cmd.op == "HARVEST" {
                    if let kagg_engine::state::Cell::Plant { crop, .. } = tile(&before, p) {
                        if state.private[job.seat]
                            .inventories
                            .get(actor)
                            .is_some_and(|i| {
                                i.get(crop) > before_private.inventories[actor].get(crop)
                            })
                        {
                            last_harvest.insert(p);
                        }
                    }
                }
                if cmd.op == "PLACE" && kagg_engine::rules::animal(&cmd.item).is_some() {
                    last_harvest.remove(&p);
                }
                if cmd.op == "PLANT"
                    && *tile(&before, p) == kagg_engine::state::Cell::Empty
                    && matches!(tile(&state.farms[job.seat],p),kagg_engine::state::Cell::Plant{crop,..} if crop==&cmd.item)
                {
                    plantings += 1;
                    if last_harvest.remove(&p) {
                        replants += 1;
                    }
                }
            }
        }
        let own = state.farms[job.seat].money;
        let opp = state.farms[1 - job.seat].money;
        let mut result = if job.variant != "legacy" {
            proto.report()
        } else {
            Json::Obj(vec![
                ("peak_plots".into(), Json::Num(peak as f64)),
                ("opening_plots".into(), Json::Num(opening as f64)),
                ("peak_workers".into(), Json::Num(workers as f64)),
                ("work".into(), Json::Num(old[job.seat].stats.work as f64)),
                (
                    "harvested_units".into(),
                    Json::Num(old[job.seat].stats.harvested_units as f64),
                ),
                (
                    "expired_projects".into(),
                    Json::Num(old[job.seat].stats.expired_projects as f64),
                ),
                (
                    "invalidated".into(),
                    Json::Num(old[job.seat].stats.invalidated as f64),
                ),
            ])
        };
        result.set_path("opponent_slot", Json::Num(job.opponent_id as f64));
        result.set_path("successful_plantings", Json::Num(plantings as f64));
        result.set_path("successful_replants", Json::Num(replants as f64));
        for (k, v) in [
            ("seed", Json::Num(job.seed as f64)),
            ("seat", Json::Num(job.seat as f64)),
            ("variant", Json::Str(job.variant.clone())),
            ("cash", Json::Num(own)),
            ("opponent_cash", Json::Num(opp)),
            ("margin", Json::Num(own - opp)),
            (
                "win",
                Json::Num(if own > opp {
                    1.
                } else if own == opp {
                    0.5
                } else {
                    0.
                }),
            ),
            (
                "land",
                Json::Num(state.farms[job.seat].unlocked_quadrants.len() as f64),
            ),
            ("seconds", Json::Num(start.elapsed().as_secs_f64())),
        ] {
            result.set_path(k, v);
        }
        Ok(result)
    }
    fn evaluate(
        config: &Config,
        seeds: &[i64],
        workers: usize,
        weights: &Json,
        variants: &[&str],
        opponents: &[Config],
    ) -> Result<Json, String> {
        let mut jobs = vec![];
        for &v in variants {
            for &seed in seeds {
                for seat in 0..2 {
                    for slot in 0..=opponents.len() {
                        let mut config = config.clone();
                        if v == "legacy_routes" {
                            config.economic_routes = false;
                        }
                        jobs.push(Job {
                            seed,
                            seat,
                            variant: v.into(),
                            config,
                            opponent: slot.checked_sub(1).map(|i| opponents[i].clone()),
                            opponent_id: slot,
                        });
                    }
                }
            }
        }
        let jobs = Arc::new(jobs);
        let index = Arc::new(AtomicUsize::new(0));
        let weights = Arc::new(weights.clone());
        let mut handles = vec![];
        for _ in 0..workers.min(jobs.len()) {
            let jobs = Arc::clone(&jobs);
            let index = Arc::clone(&index);
            let weights = Arc::clone(&weights);
            handles.push(std::thread::spawn(
                move || -> Result<Vec<(usize, Json)>, String> {
                    let mut model = Policy::mixed_routes(-1, 0, 1e-4)?;
                    model.load_weights(&weights)?;
                    model.market_mode = MarketMode::Rule;
                    let mut results = vec![];
                    loop {
                        let i = index.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() {
                            break;
                        }
                        results.push((i, play(&jobs[i], &model)?));
                    }
                    Ok(results)
                },
            ));
        }
        let mut rows = vec![];
        for h in handles {
            rows.extend(h.join().map_err(|_| "evaluation worker panicked")??);
        }
        rows.sort_by_key(|(i, _)| *i);
        let games: Vec<_> = rows.into_iter().map(|(_, j)| j).collect();
        let mut report = Json::Obj(vec![("games".into(), Json::Arr(games.clone()))]);
        for v in variants {
            let r: Vec<_> = games
                .iter()
                .filter(|j| j.get("variant").str() == *v)
                .collect();
            let mut s = vec![("games".into(), Json::Num(r.len() as f64))];
            for k in [
                "cash",
                "margin",
                "win",
                "opening_plots",
                "peak_plots",
                "peak_workers",
                "harvested_units",
                "successful_plantings",
                "successful_replants",
                "expired_projects",
                "invalidated",
                "seconds",
            ] {
                s.push((
                    format!("mean_{k}"),
                    Json::Num(r.iter().map(|j| number(j, k)).sum::<f64>() / r.len() as f64),
                ));
            }
            report.set_path(v, Json::Obj(s));
        }
        Ok(report)
    }
    fn write(path: &std::path::Path, j: &Json) -> Result<(), String> {
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, j.dump() + "\n").map_err(|e| e.to_string())?;
        std::fs::rename(&temporary, path).map_err(|e| e.to_string())
    }
    fn mutate(c: &Config, rng: &mut Rng) -> Config {
        let mut n = c.clone();
        let noise = |r: &mut Rng, scale: f64| (r.uniform() * 2. - 1.) * scale;
        n.plots_per_worker = (n.plots_per_worker + noise(rng, 0.8)).clamp(2., 6.);
        n.cash_reserve = (n.cash_reserve + noise(rng, 160.)).clamp(0., 1500.);
        n.animal_share = (n.animal_share + noise(rng, 0.10)).clamp(0., 0.5);
        n.short_share = (n.short_share + noise(rng, 0.20)).clamp(0., 1.);
        n.expansion_fill = (n.expansion_fill + noise(rng, 0.12)).clamp(0.4, 1.);
        n.forecast_days = (n.forecast_days + noise(rng, 3.)).clamp(0., 16.);
        if rng.uniform() < 0.3 {
            n.renewal_lead_days =
                (n.renewal_lead_days + if rng.uniform() < 0.5 { -1 } else { 1 }).clamp(0, 2);
        }

        n
    }
    pub fn run() -> Result<(), String> {
        tensor::threads(1);
        let args: Vec<_> = std::env::args().skip(1).collect();
        let mode = args.first().map(String::as_str).unwrap_or("help");
        if mode == "help" || mode == "--help" {
            println!("plan-prototype eval|search|agent --checkpoint RULE_MODEL --config CONFIG --out DIRECTORY --seeds N... --confirm-seeds N... --workers 4 --rounds 3 --population 6 --search-seed 2026 --seed-stride 100 --opponent-config FILE --variants prototype legacy legacy_routes\nagent uses --config only; eval runs legacy, prototype, legacy_routes; search performs Rust policy-parameter search, not PPO. Fresh output directory required for search.");
            return Ok(());
        }
        let mut checkpoint = "runs/mixed_v8_market3_rule_trial/best.json".to_owned();
        let mut config = Config::default();
        let mut seeds = vec![810001, 810002];
        let mut confirm = vec![910001, 910002];
        let mut workers = 4usize;
        let mut rounds = 3usize;
        let mut population = 6usize;
        let mut rng = Rng(2026);
        let mut out = PathBuf::from("runs/plan_prototype_trial");
        let mut i = 1;
        let mut opponents = vec![];
        let mut variants = vec![
            "legacy".to_owned(),
            "prototype".to_owned(),
            "legacy_routes".to_owned(),
        ];
        let mut stride = 0i64;
        while i < args.len() {
            let flag = &args[i];
            i += 1;
            match flag.as_str() {
                "--variants" => {
                    variants.clear();
                    while i < args.len() && !args[i].starts_with("--") {
                        let v = &args[i];
                        if !matches!(v.as_str(), "legacy" | "prototype" | "legacy_routes") {
                            return Err("invalid evaluation variant".into());
                        }
                        variants.push(v.clone());
                        i += 1;
                    }
                    if variants.is_empty() {
                        return Err("empty variants".into());
                    }
                }
                "--seeds" | "--confirm-seeds" => {
                    let mut v = vec![];
                    while i < args.len() && !args[i].starts_with("--") {
                        v.push(args[i].parse::<i64>().map_err(|_| "invalid seed")?);
                        i += 1;
                    }
                    if v.is_empty() {
                        return Err("empty seed list".into());
                    }
                    if flag == "--seeds" {
                        seeds = v;
                    } else {
                        confirm = v;
                    }
                }
                _ => {
                    let v = args.get(i).ok_or("missing option value")?;
                    i += 1;
                    match flag.as_str() {
                        "--seed-stride" => {
                            stride = v.parse::<i64>().map_err(|_| "invalid seed stride")?
                        }
                        "--opponent-config" => opponents.push(Config::parse(&json::parse(
                            &std::fs::read_to_string(v).map_err(|e| e.to_string())?,
                        )?)?),
                        "--checkpoint" => checkpoint = v.clone(),
                        "--out" => out = PathBuf::from(v),
                        "--config" => {
                            config = Config::parse(&json::parse(
                                &std::fs::read_to_string(v).map_err(|e| e.to_string())?,
                            )?)?;
                        }
                        "--workers" => workers = v.parse().map_err(|_| "invalid workers")?,
                        "--rounds" => rounds = v.parse().map_err(|_| "invalid rounds")?,
                        "--population" => {
                            population = v.parse().map_err(|_| "invalid population")?
                        }
                        "--search-seed" => {
                            rng = Rng(v.parse().map_err(|_| "invalid search seed")?)
                        }
                        _ => return Err(format!("unknown option {flag}")),
                    }
                }
            }
        }
        if mode == "agent" {
            let mut agent = Agent::new(config);
            for l in std::io::stdin().lock().lines() {
                let j = json::parse(&l.map_err(|e| e.to_string())?)?;
                let obs = if j.get("observation").is_obj() {
                    j.get("observation")
                } else {
                    &j
                };
                let (s, seat) = obsstate::from_obs(obs).ok_or("invalid observation")?;
                if s.step == 0 {
                    agent = Agent::new(agent.config.clone());
                }
                let a = agent.action(&Observation::from_state(&s, seat));
                println!("{}", action_json(&a).dump());
                std::io::stdout().flush().map_err(|e| e.to_string())?;
            }
            return Ok(());
        }
        if !matches!(mode, "eval" | "search") || workers == 0 || rounds == 0 || population < 2 {
            return Err("invalid mode or non-positive experiment size".into());
        }
        for list in [&seeds, &confirm] {
            let unique: std::collections::BTreeSet<_> = list.iter().collect();
            if unique.len() != list.len() {
                return Err("duplicate seeds".into());
            }
        }
        if stride < 0 {
            return Err("seed stride cannot be negative".into());
        }
        let training_seeds: Vec<Vec<i64>> = (0..rounds)
            .map(|r| {
                seeds
                    .iter()
                    .map(|s| {
                        stride
                            .checked_mul(r as i64)
                            .and_then(|n| s.checked_add(n))
                            .ok_or("seed overflow".to_owned())
                    })
                    .collect()
            })
            .collect::<Result<_, String>>()?;
        if training_seeds.iter().flatten().any(|s| confirm.contains(s)) {
            return Err("search and confirmation seeds must be disjoint across all rounds".into());
        }
        std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
        if mode == "search" && out.join("search.jsonl").exists() {
            return Err(
                "search output exists; use a fresh directory and --config to warm start".into(),
            );
        }
        let ck = json::parse(&std::fs::read_to_string(&checkpoint).map_err(|e| e.to_string())?)?;
        if ck.get("policy_contract").str() != route_rl_native::pipeline::ENCODING
            || ck.get("market_mode").str() != "rule"
        {
            return Err("requires matching v8 market3 RULE checkpoint".into());
        }
        let weights = ck.get("weights").clone();
        drop(ck);
        write(&out.join("initial.json"), &config.json())?;
        let mut manifest = Json::Obj(vec![
            ("mode".into(), Json::Str(mode.into())),
            ("checkpoint".into(), Json::Str(checkpoint)),
            ("search_seed".into(), Json::Str(rng.0.to_string())),
            (
                "seeds".into(),
                Json::Arr(seeds.iter().map(|&x| Json::Num(x as f64)).collect()),
            ),
            (
                "confirmation_seeds".into(),
                Json::Arr(confirm.iter().map(|&x| Json::Num(x as f64)).collect()),
            ),
            ("workers".into(), Json::Num(workers as f64)),
            ("population".into(), Json::Num(population as f64)),
            ("rounds".into(), Json::Num(rounds as f64)),
        ]);
        manifest.set_path(
            "opponent_configs",
            Json::Arr(opponents.iter().map(Config::json).collect()),
        );
        manifest.set_path("seed_stride", Json::Num(stride as f64));
        manifest.set_path(
            "implementation",
            Json::Str("plan-prototype-v1-search3".into()),
        );
        write(&out.join("manifest.json"), &manifest)?;
        if mode == "eval" {
            let report = evaluate(
                &config,
                &seeds,
                workers,
                &weights,
                &variants.iter().map(String::as_str).collect::<Vec<_>>(),
                &opponents,
            )?;
            write(&out.join("evaluation.json"), &report)?;
            for v in &variants {
                println!("{v}: {}", report.get(v).dump());
            }
            return Ok(());
        }
        let mut log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("search.jsonl"))
            .map_err(|e| e.to_string())?;
        let baseline = evaluate(
            &config,
            &confirm,
            workers,
            &weights,
            &["prototype"],
            &opponents,
        )?;
        write(&out.join("initial_confirmation.json"), &baseline)?;
        let initial = config.clone();
        write(&out.join("best.json"), &initial.json())?;
        let mut elite = vec![config.clone()];
        for round in 1..=rounds {
            let mut candidates = elite.clone();
            while candidates.len() < population {
                let p = candidates.len() % elite.len();
                candidates.push(mutate(&elite[p], &mut rng));
            }
            let mut ranked = vec![];
            for (i, c) in candidates.into_iter().enumerate() {
                let r = evaluate(
                    &c,
                    &training_seeds[round - 1],
                    workers,
                    &weights,
                    &["prototype"],
                    &opponents,
                )?;
                let summary = r.get("prototype").clone();
                let mut row = Json::Obj(vec![
                    ("round".into(), Json::Num(round as f64)),
                    ("candidate".into(), Json::Num(i as f64)),
                    ("config".into(), c.json()),
                    ("summary".into(), summary.clone()),
                    ("games".into(), r.get("games").clone()),
                ]);
                row.set_path("rng", Json::Str(rng.0.to_string()));
                writeln!(log, "{}", row.dump()).map_err(|e| e.to_string())?;
                log.flush().map_err(|e| e.to_string())?;
                println!("round={round} candidate={i} {}", summary.dump());
                ranked.push((
                    number(&summary, "mean_win"),
                    number(&summary, "mean_margin"),
                    c,
                ));
            }
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.total_cmp(&a.1)));
            elite = ranked
                .into_iter()
                .take(2.min(population - 1))
                .map(|x| x.2)
                .collect();
            config = elite[0].clone();
            write(&out.join("latest.json"), &config.json())?;
            write(&out.join(&format!("round_{round:04}.json")), &config.json())?;
        }
        let mut report = evaluate(
            &config,
            &confirm,
            workers,
            &weights,
            &["prototype"],
            &opponents,
        )?;
        let a = baseline.get("prototype");
        let b = report.get("prototype");
        let accepted = number(b, "mean_win") >= number(a, "mean_win")
            && number(b, "mean_margin") > number(a, "mean_margin");
        if accepted {
            write(&out.join("best.json"), &config.json())?;
        }
        report.set_path("accepted_over_initial", Json::Bool(accepted));
        write(&out.join("confirmation.json"), &report)?;
        println!("accepted_over_initial={accepted}");
        println!("initial confirmation {}", baseline.get("prototype").dump());
        println!("final confirmation {}", report.get("prototype").dump());
        Ok(())
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
