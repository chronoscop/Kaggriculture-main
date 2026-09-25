//! End-to-end tests of every `kagg` subcommand (runs the built binary).

use std::io::Write;
use std::process::{Command, Stdio};

fn kagg() -> Command {
    Command::new(env!("CARGO_BIN_EXE_kagg"))
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("kagg_cli_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[&str]) -> (i32, String, String) {
    let out = kagg().args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn write_tapes(dir: &std::path::Path) -> (String, String, String) {
    let a = dir.join("a.tape");
    let b = dir.join("b.tape");
    let ep = dir.join("ep.tape");
    let mut sa = String::from("SEED 7\n");
    let mut sb = String::from("SEED 7\n");
    let mut se = String::from("SEED 7\n");
    for i in 0..720 {
        let la = if i % 3 == 0 {
            "PLANT WHEAT\t\tBUY_SEED WHEAT 1"
        } else {
            "WATER\t\t"
        };
        let lb = if i % 5 == 0 {
            "SOUTH\t\t;HIRE"
        } else {
            "PASS\t\t"
        };
        sa.push_str(la);
        sa.push('\n');
        sb.push_str(lb);
        sb.push('\n');
        se.push_str(&format!("{la}\n{lb}\n"));
    }
    std::fs::write(&a, sa).unwrap();
    std::fs::write(&b, sb).unwrap();
    std::fs::write(&ep, se).unwrap();
    (
        a.to_string_lossy().to_string(),
        b.to_string_lossy().to_string(),
        ep.to_string_lossy().to_string(),
    )
}

#[test]
fn usage_and_version() {
    let (code, _, err) = run(&[]);
    assert_eq!(code, 2);
    assert!(err.contains("usage"));
    let (code, out, _) = run(&["version"]);
    assert_eq!(code, 0);
    assert!(out.contains("1.32.7"));
    assert_eq!(run(&["bench", "x"]).0, 2);
}

#[test]
fn episode_bench_and_batch_agree() {
    let d = tmp("ebb");
    let (a, b, ep) = write_tapes(&d);
    let (code, out, _) = run(&["episode", &ep]);
    assert_eq!(code, 0);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 720); // 719 digests + FINAL
    assert!(lines[718].starts_with("719 "));
    let fin: Vec<u64> = lines[719]
        .split(' ')
        .skip(1)
        .map(|x| x.parse().unwrap())
        .collect();
    let (code, out, _) = run(&["bench", &ep, "3"]);
    assert_eq!(code, 0);
    assert!(out.contains("\"steps\": 2157"));
    let jobs = d.join("jobs.tsv");
    std::fs::write(
        &jobs,
        format!("# comment\n-\t{a}\t{b}\n7\t{a}\t{b}\n9\t{a}\tmissing.tape\n"),
    )
    .unwrap();
    let (code, out, _) = run(&["batch", &jobs.to_string_lossy(), "2"]);
    assert_eq!(code, 0);
    let rows: Vec<Vec<&str>> = out.lines().map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 3);
    let b0: f64 = rows[0][2].parse().unwrap();
    let b1: f64 = rows[0][3].parse().unwrap();
    assert_eq!(b0.to_bits(), fin[0]);
    assert_eq!(b1.to_bits(), fin[1]);
    assert_eq!(rows[1][1], "7");
    assert_eq!(rows[2][1], "ERR");
}

#[test]
fn serve_protocol_round_trip() {
    let mut child = kagg()
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "RESET 3").unwrap();
        writeln!(stdin, "STEP2 PASS\t\tHIRE\x1ePASS\t\t").unwrap();
        writeln!(stdin, "GENGAME 3\x1e\x1e").unwrap();
        writeln!(stdin, "BOGUS").unwrap();
        writeln!(stdin, "QUIT").unwrap();
    }
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4);
    assert!(lines[0].contains("\"step\": 0"));
    assert!(lines[1].contains("\"step\": 1"));
    assert!(lines[2].starts_with("{\"days\""));
    assert!(lines[3].contains("error"));
}

#[test]
fn diagnostics() {
    for args in [
        vec!["rules"],
        vec!["prices", "9000"],
        vec!["price-sweep", "CARROT", "9000", "9100", "50"],
        vec!["rng-probe", "1", "2"],
        vec!["weeds", "1", "2", "5"],
    ] {
        let (code, out, _) = run(&args);
        assert_eq!(code, 0, "{args:?}");
        assert!(out.starts_with('{') || out.starts_with('['), "{args:?}");
    }
}

#[test]
fn tournament_selfplay_worlds_compare_template() {
    let d = tmp("front");
    let (a, _, _) = write_tapes(&d);
    let (code, tpl, _) = run(&["template", "tournament"]);
    assert_eq!(code, 0);
    assert!(tpl.contains("\"schedule\""));
    assert_eq!(run(&["template", "nope"]).0, 1);
    let cfg = d.join("t.json");
    std::fs::write(
        &cfg,
        format!(
            r#"{{"name": "cli", "candidate": {{"name": "tape", "type": "tape", "path": {a:?}}},
               "panel": [{{"name": "rnd", "type": "builtin", "kind": "random", "seed": 1}}],
               "worlds": {{"strategy": "range", "start": 0, "count": 2}},
               "output": {{"dir": {dir:?}, "resume": false}}}}"#,
            dir = d.to_string_lossy()
        ),
    )
    .unwrap();
    let (code, out, err) = run(&[
        "tournament",
        &cfg.to_string_lossy(),
        "--workers",
        "2",
        "--quiet",
    ]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("## Standings"));
    let res = d.join("cli").join("results.jsonl");
    assert_eq!(std::fs::read_to_string(&res).unwrap().lines().count(), 4);
    let r = res.to_string_lossy().to_string();
    let (code, out, _) = run(&["compare", &r, &r, "tape"]);
    assert_eq!(code, 0);
    assert!(out.contains("\"n_pairs\": 4"));
    assert_eq!(run(&["tournament", &cfg.to_string_lossy(), "--bogus"]).0, 1);
    assert_eq!(run(&["tournament", "missing.json"]).0, 1);

    let (code, sp, _) = run(&["template", "selfplay"]);
    assert_eq!(code, 0);
    let spc = d.join("sp.json");
    let sp = sp.replace(
        "\"dir\": \"selfplay\"",
        &format!("\"dir\": {:?}", d.to_string_lossy()),
    );
    std::fs::write(&spc, sp).unwrap();
    let (code, out, err) = run(&[
        "selfplay",
        &spc.to_string_lossy(),
        "--set",
        "worlds.pool=[0,60]",
        "--set",
        "samples.stride=360",
        "--quiet",
    ]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\"games\""));

    let (code, out, _) = run(&["worlds", "0-9", "1", "2"]);
    assert_eq!(code, 0);
    assert!(out.contains("\"n\": 10"));
    assert_eq!(run(&["compare", "a"]).0, 1);
}
