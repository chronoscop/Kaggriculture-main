//! `kagg` -- command-line driver for the Kaggriculture engine port.
//!
//! ```text
//! kagg batch <jobs.tsv> [threads]   tape-pair rollouts -> final banks
//! kagg serve                        stdio step environment (docs/serve-protocol.md)
//! kagg episode <two-seat tape>      per-step state digests (differential tests)
//! kagg bench <two-seat tape> <n>    replay a tape n times, report steps/s
//!
//! parallel front ends (kagg-sim):
//! kagg tournament <config.json> [--set k=v]... [--workers N] [--quiet]
//! kagg selfplay <config.json> [--set k=v]... [--workers N] [--quiet]
//! kagg template tournament|selfplay  print a starter config
//! kagg worlds <seeds> [k] [threads]  idle-drive world catalog (JSON)
//! kagg compare <a.jsonl> <b.jsonl> <agent> [agent_b]   paired McNemar
//!
//! read-only diagnostics:
//! kagg rules                        pure rule tables as JSON
//! kagg prices <inv>                 all quoted prices at one inventory level
//! kagg price-sweep <item> <lo> <hi> <step>   raw (bits) + quoted prices
//! kagg rng-probe <seed> <day>       per-day RNG reference vectors
//! kagg weeds <seed> <day> <n>       the first n draws of one day's RNG
//! ```

mod batch;
mod diag;
mod run;
mod serve;

use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::state::{State, FINAL_STEP};
use kagg_engine::tape::{parse_action_line, parse_seed_line};
use std::env;
use std::io::{BufWriter, Write};

const USAGE: &str = "usage: kagg <command>
  batch <jobs.tsv> [threads]      tape-pair rollouts -> final banks
  serve                           stdio step environment
  episode <tape>                  per-step digests from a two-seat tape
  bench <tape> <reps>             throughput on a two-seat tape
  rules | prices <inv> | price-sweep <item> <lo> <hi> <step>
  rng-probe <seed> <day> | weeds <seed> <day> <n>
  tournament <config.json> [--set k=v]... [--workers N] [--quiet]
  selfplay <config.json> [--set k=v]... [--workers N] [--quiet]
  template tournament|selfplay
  worlds <seeds> [key_depth] [threads]
  compare <a.jsonl> <b.jsonl> <agent> [agent_b]
  version";

fn die(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

/// A two-seat tape: `SEED <n>` then seat-0 and seat-1 lines per step.
fn load_two_seat(path: &str) -> (i64, Vec<[PlayerAction; 2]>) {
    let raw =
        std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("cannot open {path}: {e}")));
    let mut lines = raw.lines();
    let seed = parse_seed_line(lines.next().unwrap_or_else(|| die("empty tape")))
        .unwrap_or_else(|e| die(&e));
    let body: Vec<&str> = lines.collect();
    let pairs = body
        .chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| [parse_action_line(c[0]), parse_action_line(c[1])])
        .collect();
    (seed, pairs)
}

/// Run a two-seat tape and print `<step> <digest>` after every step, then
/// `FINAL <bits0> <bits1>` (money as IEEE-754 bit patterns). Stops at the
/// official terminal step even if the tape is longer.
fn episode(path: &str) {
    let (seed, pairs) = load_two_seat(path);
    let mut st = State::new(seed);
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for actions in pairs.iter() {
        if st.step >= FINAL_STEP {
            break;
        }
        engine::step(&mut st, actions);
        writeln!(out, "{} {}", st.step, st.digest()).unwrap();
    }
    writeln!(
        out,
        "FINAL {} {}",
        st.farms[0].money.to_bits(),
        st.farms[1].money.to_bits()
    )
    .unwrap();
}

/// Replay one two-seat tape `reps` times; report steps/second as JSON.
fn bench(path: &str, reps: usize) {
    let (seed, pairs) = load_two_seat(path);
    let t0 = std::time::Instant::now();
    let mut total_steps = 0u64;
    let mut sink = 0.0f64;
    for _ in 0..reps {
        let mut st = State::new(seed);
        for actions in pairs.iter() {
            if st.step >= FINAL_STEP {
                break;
            }
            engine::step(&mut st, actions);
            total_steps += 1;
        }
        sink += st.farms[0].money;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{{\"steps\": {}, \"seconds\": {:.4}, \"steps_per_sec\": {:.0}, \
         \"episodes_per_sec\": {:.2}, \"sink\": {}}}",
        total_steps,
        dt,
        total_steps as f64 / dt,
        reps as f64 / dt,
        sink
    );
}

fn ok_or_die(r: Result<(), String>) {
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn arg<T: std::str::FromStr>(args: &[String], i: usize) -> T {
    args.get(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| die(&format!("bad or missing argument {i}\n{USAGE}")))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    match cmd {
        "batch" if args.len() >= 3 => {
            let threads = args.get(3).and_then(|t| t.parse().ok()).unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });
            batch::batch(&args[2], threads)
        }
        "serve" => serve::serve(),
        "episode" if args.len() == 3 => episode(&args[2]),
        "bench" if args.len() == 4 => bench(&args[2], arg(&args, 3)),
        "rules" => diag::rules_dump(),
        "prices" if args.len() == 3 => diag::prices(arg(&args, 2)),
        "price-sweep" if args.len() == 6 => {
            diag::price_sweep(&args[2], arg(&args, 3), arg(&args, 4), arg(&args, 5))
        }
        "rng-probe" if args.len() == 4 => diag::rng_probe(arg(&args, 2), arg(&args, 3)),
        "weeds" if args.len() == 5 => diag::weeds(arg(&args, 2), arg(&args, 3), arg(&args, 4)),
        "tournament" => ok_or_die(run::tournament_cmd(&args[2..])),
        "selfplay" => ok_or_die(run::selfplay_cmd(&args[2..])),
        "template" => ok_or_die(run::template(args.get(2).map(String::as_str).unwrap_or(""))),
        "worlds" => ok_or_die(run::worlds_cmd(&args[2..])),
        "compare" => ok_or_die(run::compare_cmd(&args[2..])),
        "version" => println!(
            "kagg {} (engine kaggle-environments {})",
            env!("CARGO_PKG_VERSION"),
            kagg_engine::market::ENGINE_VERSION
        ),
        _ => die(USAGE),
    }
}
