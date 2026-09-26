//! Persistent JSONL agent boundary for evaluation only; training is entirely in-process Rust.
#[cfg(feature = "train")]
fn run() -> Result<(), String> {
    use kagg_engine::{
        json::{self, Json},
        obsstate,
    };
    use route_rl_native::{
        learning::{
            policy::{Policy, Rng, Sample},
            tensor,
        },
        pipeline::{encoding, executor::*, planner},
    };
    use std::io::{self, BufRead, Write};
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut policy = None;
    if args.len() != 2 || args[0] != "--checkpoint" {
        return Err("usage: mixed-agent --checkpoint PATH|heuristic".into());
    }
    tensor::threads(1);
    if args[1] != "heuristic" {
        let j = json::parse(&std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?)?;
        let mut p = Policy::mixed_routes(-1, 0, 1e-4)?;
        p.restore(&j)?;
        policy = Some(p);
    }
    let mut agents = [Executor::new(), Executor::new()];
    let mut last = [-1i64; 2];
    let mut rng = Rng(0);
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for line in io::stdin().lock().lines() {
        let j = json::parse(&line.map_err(|e| e.to_string())?)?;
        let obs = if j.get("observation").is_obj() {
            j.get("observation")
        } else {
            &j
        };
        let (s, seat) = obsstate::from_obs(obs).ok_or("invalid observation")?;
        let o = Observation::from_state(&s, seat);
        if o.step <= last[seat] {
            agents[seat] = Executor::new();
        }
        last[seat] = o.step;
        let e = &mut agents[seat];
        e.observe(&o);
        let mut orders = Vec::new();
        let mut choose =
            |p: planner::Problem, e: &mut Executor| -> Result<Vec<Vec<String>>, String> {
                let index = if let Some(model) = &policy {
                    let (c, f) = encoding::encode(&o, e, &p);
                    model.infer(
                        &[Sample {
                            context: c,
                            features: f,
                            action: 0,
                            logp: 0.,
                            value: 0.,
                            reward: 0.,
                            ..Sample::default()
                        }],
                        true,
                        &mut rng,
                    )?[0]
                        .action
                } else {
                    p.heuristic()
                };
                p.select(index, e, &o)
            };
        if e.market_due(&o) {
            e.last_market = o.step;
            e.last_cash = o.farm.money;
            orders = choose(planner::investment_problem(&o, e), e)?;
        }
        for actor in 0..o.private.inventories.len() {
            if e.routes
                .get(actor)
                .and_then(Option::as_ref)
                .is_none_or(|r| r.steps.is_empty())
            {
                choose(planner::route_problem(&o, e, actor), e)?;
            }
        }
        let action = e.action(&o, orders);
        let result: Json = action_json(&action);
        writeln!(output, "{}", result.dump()).map_err(|e| e.to_string())?;
        output.flush().map_err(|e| e.to_string())?;
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
