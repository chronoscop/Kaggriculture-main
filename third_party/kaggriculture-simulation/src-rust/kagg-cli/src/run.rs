//! Front-end subcommands backed by `kagg-sim`: tournament, self-play,
//! world catalogs and paired comparison.

use kagg_engine::json::{num, quote, Json};
use kagg_sim::seeding::idle_catalog;
use kagg_sim::selfplay::{self, SelfPlay};
use kagg_sim::tournament::{self, Tournament};
use kagg_sim::util::{apply_overrides, load_json, parse_seeds};

/// Split `<config> [--set k=v]... [--workers N] [--quiet]`.
pub fn config_args(args: &[String]) -> Result<Json, String> {
    let path = args.first().ok_or("missing config path")?;
    let mut cfg = load_json(path)?;
    let mut overrides = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--set" => {
                overrides.push(args.get(i + 1).ok_or("--set needs key=value")?.clone());
                i += 2;
            }
            "--workers" => {
                let w = args.get(i + 1).ok_or("--workers needs a number")?;
                overrides.push(format!("workers={w}"));
                i += 2;
            }
            "--quiet" => {
                overrides.push("progress=false".into());
                i += 1;
            }
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    apply_overrides(&mut cfg, &overrides)?;
    Ok(cfg)
}

pub fn tournament_cmd(args: &[String]) -> Result<(), String> {
    let t = Tournament::from_json(&config_args(args)?)?;
    let summary = t.execute()?;
    let parsed = kagg_engine::json::parse(&summary).map_err(|e| e.to_string())?;
    println!("{}", tournament::markdown(&parsed));
    eprintln!("results: {}", t.out_dir);
    Ok(())
}

pub fn selfplay_cmd(args: &[String]) -> Result<(), String> {
    let sp = SelfPlay::from_json(&config_args(args)?)?;
    println!("{}", sp.execute()?);
    eprintln!("results: {}", sp.out_dir);
    Ok(())
}

pub fn template(which: &str) -> Result<(), String> {
    match which {
        "tournament" => println!("{}", tournament::TEMPLATE),
        "selfplay" => println!("{}", selfplay::TEMPLATE),
        _ => return Err("template: tournament | selfplay".into()),
    }
    Ok(())
}

/// `kagg worlds <seeds> [key_depth] [threads]` -> JSON catalog (idle drive).
pub fn worlds_cmd(args: &[String]) -> Result<(), String> {
    let seeds = parse_seeds(args.first().ok_or("worlds <seeds> [k] [threads]")?)?;
    let k: usize = match args.get(1) {
        Some(x) => x.parse().map_err(|_| format!("bad key_depth {x:?}"))?,
        None => 2,
    };
    let k = k.clamp(1, kagg_sim::seeding::MAX_KEY_DEPTH);
    let threads: usize = match args.get(2) {
        Some(x) => x.parse().map_err(|_| format!("bad threads {x:?}"))?,
        None => 2,
    };
    let mut by_seed = Vec::new();
    let mut by_world: std::collections::BTreeMap<String, Vec<i64>> = Default::default();
    let mut sorted = seeds.clone();
    sorted.sort();
    sorted.dedup();
    // label exactly the listed seeds
    let cat = idle_catalog(&sorted, k, threads);
    for s in &sorted {
        let w = cat.get(s).cloned().flatten();
        by_seed.push(format!(
            "\"{s}\": {}",
            w.as_deref().map(quote).unwrap_or("null".into())
        ));
        by_world
            .entry(w.unwrap_or_else(|| "?".into()))
            .or_default()
            .push(*s);
    }
    let bw: Vec<String> = by_world
        .iter()
        .map(|(w, ss)| {
            let v: Vec<String> = ss.iter().map(|x| x.to_string()).collect();
            format!("{}: [{}]", quote(w), v.join(", "))
        })
        .collect();
    println!(
        "{{\"drive\": \"idle\", \"key_depth\": {k}, \"n\": {}, \"worlds\": {{{}}}, \"by_world\": {{{}}}}}",
        num(sorted.len() as f64),
        by_seed.join(", "),
        bw.join(", ")
    );
    Ok(())
}

pub fn compare_cmd(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("compare <results_a.jsonl> <results_b.jsonl> <agent> [agent_b]".into());
    }
    let b = args.get(3).unwrap_or(&args[2]);
    println!("{}", tournament::compare(&args[0], &args[1], &args[2], b));
    Ok(())
}
