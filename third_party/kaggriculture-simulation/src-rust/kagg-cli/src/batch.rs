//! `kagg batch`: play many tape pairs across threads and report final banks.
//!
//! Jobs file: one job per line, `seed \t tapeA \t tapeB` (blank lines and
//! `#` comments ignored). seed `-` means "use tapeA's SEED header".
//! Output, in INPUT order: `idx \t seed \t bank0 \t bank1`, or
//! `idx \t ERR \t <msg>`.

use kagg_engine::tape::{load_tape, play_pair};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub fn parse_jobs(raw: &str) -> Vec<(String, String, String)> {
    raw.lines()
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut p = l.split('\t');
            (
                p.next().unwrap_or("-").to_string(),
                p.next().unwrap_or("").to_string(),
                p.next().unwrap_or("").to_string(),
            )
        })
        .collect()
}

pub fn run_job(seed_s: &str, pa: &str, pb: &str) -> Result<(i64, f64, f64), String> {
    let ta = load_tape(pa)?;
    let tb = load_tape(pb)?;
    let seed = if seed_s == "-" {
        ta.seed
    } else {
        seed_s
            .trim()
            .parse()
            .map_err(|_| format!("bad seed {seed_s:?}"))?
    };
    let (b0, b1) = play_pair(seed, &ta, &tb);
    Ok((seed, b0, b1))
}

pub fn batch(jobs_path: &str, threads: usize) {
    let raw = std::fs::read_to_string(jobs_path).unwrap_or_else(|e| {
        eprintln!("cannot read {jobs_path}: {e}");
        std::process::exit(2);
    });
    let jobs = parse_jobs(&raw);
    let results: Mutex<Vec<Option<String>>> = Mutex::new(vec![None; jobs.len()]);
    let next = AtomicUsize::new(0);
    let nthreads = threads.max(1).min(jobs.len().max(1));

    std::thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let (seed_s, pa, pb) = &jobs[i];
                let line = match run_job(seed_s, pa, pb) {
                    Ok((seed, b0, b1)) => format!("{i}\t{seed}\t{b0}\t{b1}"),
                    Err(e) => format!("{i}\tERR\t{e}"),
                };
                results.lock().unwrap()[i] = Some(line);
            });
        }
    });

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for line in results.into_inner().unwrap().into_iter().flatten() {
        if writeln!(out, "{line}").is_err() {
            break; // the reader went away
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_parse_skips_comments_and_blank_lines() {
        let j = parse_jobs("# header\n\n3\ta.tape\tb.tape\r\n-\tc\td\n");
        assert_eq!(j.len(), 2);
        assert_eq!(j[0], ("3".into(), "a.tape".into(), "b.tape".into()));
        assert_eq!(j[1].0, "-");
    }
}
