//! Configurable, parallel tournaments across worlds.
//!
//! ```json
//! {
//!   "name": "my-tournament",
//!   "candidate": {"name": "mine", "type": "python", "path": "main.py"},
//!   "panel": [
//!     {"name": "rnd", "type": "builtin", "kind": "random", "seed": 1},
//!     {"name": "opp", "type": "tape", "path": "opp.tape"},
//!     {"name": "other", "type": "python", "path": "other/main.py"}
//!   ],
//!   "schedule": "gauntlet",        // gauntlet | round_robin
//!   "seats": "both",               // both | seat0 | seat1 | alternate
//!   "worlds": {"strategy": "stratified", "pool": [0, 3000],
//!              "per_world": 4, "key_depth": 2},
//!   "key_depth": 2,                // world key used in summaries
//!   "world_weighting": "uniform",  // none | uniform | {"W": w, "*": w}
//!   "workers": 8,
//!   "on_error": "forfeit",         // forfeit | exclude
//!   "output": {"dir": "tournaments", "resume": true},
//!   "python": {"exe": "python", "path": [], "stderr": "inherit"},
//!   "samples": null,               // see samples.rs
//!   "sinks": []                    // see sink.rs
//! }
//! ```

use crate::agent::{AgentSpec, PythonCfg};
use crate::runner::{self, Game, OnError, Row, RunCfg};
use crate::samples::SampleCfg;
use crate::seeding;
use crate::sink::{SinkSpec, Sinks};
use crate::stats;
use crate::util::{bool_or, f64_or, i64_or, merge, str_or};
use kagg_engine::json::{self, num, quote, Json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

pub const TEMPLATE: &str = r#"{
 "name": "example",
 "candidate": {"name": "mine", "type": "builtin", "kind": "random", "seed": 1},
 "panel": [
  {"name": "chaos", "type": "builtin", "kind": "chaos", "seed": 2},
  {"name": "idle", "type": "builtin", "kind": "idle"}
 ],
 "schedule": "gauntlet",
 "seats": "both",
 "worlds": {"strategy": "stratified", "pool": [0, 2000], "per_world": 3, "key_depth": 2},
 "key_depth": 2,
 "world_weighting": "uniform",
 "workers": 4,
 "on_error": "forfeit",
 "output": {"dir": "tournaments", "resume": true},
 "python": {"exe": "python", "path": [], "stderr": "inherit"},
 "samples": null,
 "sinks": []
}"#;

const DEFAULTS: &str = r#"{
 "name": "tournament", "schedule": "gauntlet", "seats": "both",
 "worlds": {"strategy": "range", "start": 0, "count": 20},
 "key_depth": 2, "world_weighting": "none", "workers": 2,
 "on_error": "forfeit", "output": {"dir": "tournaments", "resume": true},
 "python": {}, "samples": null, "sinks": []
}"#;

pub struct Tournament {
    pub cfg: Json,
    pub name: String,
    pub candidate: Option<Arc<AgentSpec>>,
    pub panel: Vec<Arc<AgentSpec>>,
    pub schedule: String,
    pub seats: String,
    pub run: RunCfg,
    pub sinks: Vec<SinkSpec>,
    pub worlds: Json,
    pub weighting: Json,
    pub out_dir: String,
}

pub fn seat_orders(mode: &str, i: usize) -> Vec<usize> {
    match mode {
        "seat0" => vec![0],
        "seat1" => vec![1],
        "alternate" => vec![i % 2],
        _ => vec![0, 1],
    }
}

pub fn parse_common(
    cfg: &Json,
    default_dir: &str,
) -> Result<(String, RunCfg, Vec<SinkSpec>, String), String> {
    let name = str_or(cfg, "name", "run").to_string();
    if name.is_empty() || name.contains(['/', '\\']) {
        return Err(format!("bad run name {name:?}"));
    }
    let out = cfg.get("output");
    let out_dir = std::path::Path::new(str_or(out, "dir", default_dir))
        .join(&name)
        .to_string_lossy()
        .to_string();
    let on_error = match str_or(cfg, "on_error", "forfeit") {
        "forfeit" => OnError::Forfeit,
        "exclude" => OnError::Exclude,
        o => return Err(format!("on_error must be forfeit or exclude, got {o:?}")),
    };
    let samples = if cfg.get("samples").is_obj() {
        Some(SampleCfg::from_json(cfg.get("samples"))?)
    } else {
        None
    };
    let mut sinks = Vec::new();
    for s in cfg.get("sinks").arr() {
        sinks.push(SinkSpec::from_json(s)?);
    }
    let workers = i64_or(cfg, "workers", 2);
    if workers < 1 {
        return Err("workers must be >= 1".into());
    }
    let run = RunCfg {
        workers: workers as usize,
        key_depth: i64_or(cfg, "key_depth", 2).clamp(1, 8) as usize,
        on_error,
        results_path: std::path::Path::new(&out_dir)
            .join("results.jsonl")
            .to_string_lossy()
            .to_string(),
        resume: bool_or(out, "resume", true),
        python: PythonCfg::from_json(cfg.get("python")),
        samples,
        progress: bool_or(cfg, "progress", true),
        time_limits: runner::TimeLimits::from_json(cfg.get("time_limits"))?,
    };
    Ok((name, run, sinks, out_dir))
}

impl Tournament {
    pub fn from_json(user: &Json) -> Result<Self, String> {
        let cfg = merge(&json::parse(DEFAULTS).expect("defaults"), user);
        let (name, run, sinks, out_dir) = parse_common(&cfg, "tournaments")?;
        let schedule = str_or(&cfg, "schedule", "gauntlet").to_string();
        if schedule != "gauntlet" && schedule != "round_robin" {
            return Err("schedule must be gauntlet or round_robin".into());
        }
        let seats = str_or(&cfg, "seats", "both").to_string();
        if !["both", "seat0", "seat1", "alternate"].contains(&seats.as_str()) {
            return Err("seats must be both, seat0, seat1 or alternate".into());
        }
        let candidate = if cfg.get("candidate").is_obj() {
            Some(Arc::new(AgentSpec::from_json(
                cfg.get("candidate"),
                "candidate",
            )?))
        } else {
            None
        };
        if schedule == "gauntlet" && candidate.is_none() {
            return Err("a gauntlet needs a candidate".into());
        }
        let mut panel = Vec::new();
        for (i, p) in cfg.get("panel").arr().iter().enumerate() {
            panel.push(Arc::new(AgentSpec::from_json(p, &format!("panel{i}"))?));
        }
        if panel.is_empty() {
            return Err("the panel is empty".into());
        }
        let mut names: Vec<&str> = panel.iter().map(|p| p.name.as_str()).collect();
        if let Some(c) = &candidate {
            names.push(&c.name);
        }
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        if sorted.len() != names.len() {
            return Err(format!("agent names must be unique: {names:?}"));
        }
        Ok(Tournament {
            name,
            candidate,
            panel,
            schedule,
            seats,
            run,
            sinks,
            worlds: cfg.get("worlds").clone(),
            weighting: cfg.get("world_weighting").clone(),
            out_dir,
            cfg,
        })
    }

    pub fn games(&self, seeds: &seeding::SeedPlan) -> Vec<Game> {
        let mut pairs: Vec<(Arc<AgentSpec>, Arc<AgentSpec>, &str, &str)> = Vec::new();
        if self.schedule == "gauntlet" {
            let c = self.candidate.clone().expect("candidate");
            for p in &self.panel {
                pairs.push((c.clone(), p.clone(), "candidate", "panel"));
            }
        } else {
            let mut pool: Vec<(Arc<AgentSpec>, &str)> = Vec::new();
            if let Some(c) = &self.candidate {
                pool.push((c.clone(), "candidate"));
            }
            for p in &self.panel {
                pool.push((p.clone(), "panel"));
            }
            for i in 0..pool.len() {
                for j in i + 1..pool.len() {
                    pairs.push((pool[i].0.clone(), pool[j].0.clone(), pool[i].1, pool[j].1));
                }
            }
        }
        let mut out = Vec::new();
        for (a, b, ra, rb) in pairs {
            for (i, (seed, target)) in seeds.iter().enumerate() {
                for seat in seat_orders(&self.seats, i) {
                    let (x, y, rx, ry) = if seat == 0 {
                        (&a, &b, ra, rb)
                    } else {
                        (&b, &a, rb, ra)
                    };
                    out.push(Game {
                        id: Game::make_id(x, y, *seed, ""),
                        seed: *seed,
                        agents: [x.clone(), y.clone()],
                        roles: [rx.to_string(), ry.to_string()],
                        target_world: target.clone(),
                        tag: String::new(),
                    });
                }
            }
        }
        out
    }

    /// Run end to end; writes config.json, results.jsonl, summary.json,
    /// summary.md into the output directory. Returns the summary JSON text.
    pub fn execute(&self) -> Result<String, String> {
        let t0 = Instant::now();
        std::fs::create_dir_all(&self.out_dir).map_err(|e| format!("{}: {e}", self.out_dir))?;
        std::fs::write(
            std::path::Path::new(&self.out_dir).join("config.json"),
            self.cfg.dump(),
        )
        .map_err(|e| e.to_string())?;
        let seeds = seeding::select(&self.worlds, self.run.workers)?;
        let games = self.games(&seeds);
        let sinks = Sinks::start(&self.sinks, self.run.resume)?;
        let rows = runner::run(&games, &self.run, &sinks)?;
        sinks.finish()?;
        let secs = t0.elapsed().as_secs_f64();
        let focus = self.candidate.as_ref().map(|c| c.name.clone());
        let summary = summarize(&rows, focus.as_deref(), &self.weighting);
        let head = format!(
            "\"name\": {}, \"schedule\": {}, \"seeds\": {}, \"workers\": {}, \"seconds\": {}, \"games_per_sec\": {}, \"out_dir\": {}",
            quote(&self.name),
            quote(&self.schedule),
            seeds.len(),
            self.run.workers,
            num((secs * 100.0).round() / 100.0),
            num((rows.len() as f64 / secs.max(1e-9) * 100.0).round() / 100.0),
            quote(&self.out_dir)
        );
        let full = format!("{{{head}, {}", &summary[1..]);
        let p = std::path::Path::new(&self.out_dir);
        std::fs::write(p.join("summary.json"), &full).map_err(|e| e.to_string())?;
        let parsed = json::parse(&full).map_err(|e| e.to_string())?;
        std::fs::write(p.join("summary.md"), markdown(&parsed)).map_err(|e| e.to_string())?;
        Ok(full)
    }
}

#[derive(Default)]
struct Acc {
    s: Vec<f64>,
    m: Vec<f64>,
}

impl Acc {
    fn push(&mut self, s: f64, m: f64) {
        self.s.push(s);
        self.m.push(m);
    }

    fn block(&self) -> String {
        let n = self.s.len();
        if n == 0 {
            return "{\"games\": 0}".into();
        }
        let w = self.s.iter().filter(|&&x| x == 1.0).count();
        let d = self.s.iter().filter(|&&x| x == 0.5).count();
        let (lo, hi) = stats::ci95(&self.s);
        format!(
            "{{\"games\": {n}, \"wins\": {w}, \"draws\": {d}, \"losses\": {}, \"score\": {}, \"ci95\": [{}, {}], \"mean_margin\": {}}}",
            n - w - d,
            num(r4(stats::mean(&self.s))),
            num(r4(lo)),
            num(r4(hi)),
            num((stats::mean(&self.m) * 10.0).round() / 10.0)
        )
    }
}

fn r4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

/// Standings for every agent; breakdowns for `focus`. Returns JSON text.
pub fn summarize(rows: &[Row], focus: Option<&str>, weighting: &Json) -> String {
    let mut agents: BTreeMap<String, (Acc, usize, Vec<f64>, f64)> = BTreeMap::new();
    let mut matrix: BTreeMap<String, BTreeMap<String, Acc>> = BTreeMap::new();
    let mut by_opp: BTreeMap<String, Acc> = BTreeMap::new();
    let mut by_world: BTreeMap<String, Acc> = BTreeMap::new();
    let mut by_seat: BTreeMap<String, Acc> = BTreeMap::new();
    let mut overall = Acc::default();
    let mut errors = 0;
    for r in rows {
        if let Some((Some(seat), _)) = &r.error {
            agents.entry(r.agents[*seat].clone()).or_default().1 += 1;
        }
        if r.error.is_some() {
            errors += 1;
        }
        let Some(sc) = r.scores else { continue };
        let margin = r.banks.map(|b| b[0] - b[1]).unwrap_or(0.0);
        for seat in 0..2 {
            let me = &r.agents[seat];
            let m = if seat == 0 { margin } else { -margin };
            let e = agents.entry(me.clone()).or_default();
            e.0.push(sc[seat], m);
            if r.act_ms_mean[seat] > 0.0 {
                e.2.push(r.act_ms_mean[seat]);
                e.3 = e.3.max(r.act_ms_max[seat]);
            }
            matrix
                .entry(me.clone())
                .or_default()
                .entry(r.agents[1 - seat].clone())
                .or_default()
                .push(sc[seat], m);
            if Some(me.as_str()) == focus {
                overall.push(sc[seat], m);
                by_opp
                    .entry(r.agents[1 - seat].clone())
                    .or_default()
                    .push(sc[seat], m);
                by_world
                    .entry(r.world.clone().unwrap_or_else(|| "?".into()))
                    .or_default()
                    .push(sc[seat], m);
                by_seat
                    .entry(seat.to_string())
                    .or_default()
                    .push(sc[seat], m);
            }
        }
    }
    let mut standings: Vec<(f64, String, String)> = agents
        .iter()
        .map(|(name, (acc, errs, ms, mx))| {
            let b = acc.block();
            let score = if acc.s.is_empty() { -1.0 } else { stats::mean(&acc.s) };
            let ms_mean = if ms.is_empty() { "null".to_string() } else { num(r4(stats::mean(ms))) };
            (
                score,
                name.clone(),
                format!(
                    "{{\"agent\": {}, \"errors\": {errs}, \"act_ms_mean\": {ms_mean}, \"act_ms_max\": {}, {}",
                    quote(name),
                    num(r4(*mx)),
                    &b[1..]
                ),
            )
        })
        .collect();
    standings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let st: Vec<String> = standings.into_iter().map(|x| x.2).collect();
    let mat: Vec<String> = matrix
        .iter()
        .map(|(a, row)| {
            let cells: Vec<String> = row
                .iter()
                .map(|(b, acc)| format!("{}: {}", quote(b), num(r4(stats::mean(&acc.s)))))
                .collect();
            format!("{}: {{{}}}", quote(a), cells.join(", "))
        })
        .collect();
    let mut out = format!(
        "{{\"games\": {}, \"errors\": {errors}, \"standings\": [{}], \"matrix\": {{{}}}",
        rows.len(),
        st.join(", "),
        mat.join(", ")
    );
    if let Some(f) = focus {
        let blocks = |m: &BTreeMap<String, Acc>| {
            let v: Vec<String> = m
                .iter()
                .map(|(k, a)| format!("{}: {}", quote(k), a.block()))
                .collect();
            format!("{{{}}}", v.join(", "))
        };
        let weighted = world_weighted(&by_world, weighting);
        out.push_str(&format!(
            ", \"focus\": {{\"agent\": {}, \"overall\": {}, \"world_weighted_score\": {}, \"by_opponent\": {}, \"by_world\": {}, \"by_seat\": {}}}",
            quote(f),
            overall.block(),
            weighted.map(|x| num(r4(x))).unwrap_or_else(|| "null".into()),
            blocks(&by_opp),
            blocks(&by_world),
            blocks(&by_seat)
        ));
    }
    out.push('}');
    out
}

fn world_weighted(by_world: &BTreeMap<String, Acc>, weighting: &Json) -> Option<f64> {
    let worlds: Vec<(&String, f64)> = by_world
        .iter()
        .filter(|(_, a)| !a.s.is_empty())
        .map(|(w, a)| (w, stats::mean(&a.s)))
        .collect();
    if worlds.is_empty() {
        return None;
    }
    let weight = |w: &str| -> Option<f64> {
        match weighting {
            Json::Str(s) if s == "uniform" => Some(1.0),
            Json::Obj(_) => {
                let v = weighting.get(w);
                Some(if v.is_num() {
                    v.f64()
                } else {
                    f64_or(weighting, "*", 1.0)
                })
            }
            _ => None,
        }
    };
    let mut tot = 0.0;
    let mut acc = 0.0;
    for (w, s) in worlds {
        let x = weight(w)?;
        tot += x;
        acc += x * s;
    }
    if tot > 0.0 {
        Some(acc / tot)
    } else {
        None
    }
}

/// Markdown report from a summary JSON.
pub fn markdown(s: &Json) -> String {
    let f2 = |j: &Json| {
        if j.is_num() {
            format!("{:.3}", j.f64())
        } else {
            "-".into()
        }
    };
    let mut out = vec![
        format!("# Tournament: {}", s.get("name").str()),
        String::new(),
        format!(
            "{} game(s), {} with errors, {} s ({} games/s, {} workers).",
            s.get("games").i64(),
            s.get("errors").i64(),
            s.get("seconds").f64(),
            s.get("games_per_sec").f64(),
            s.get("workers").i64()
        ),
        String::new(),
        "## Standings".into(),
        String::new(),
        "| # | agent | games | W/D/L | score | 95% CI | mean margin | errors | ms/turn mean / max |".into(),
        "|---:|---|---:|---|---:|---|---:|---:|---|".into(),
    ];
    for (i, b) in s.get("standings").arr().iter().enumerate() {
        out.push(format!(
            "| {} | {} | {} | {}/{}/{} | {} | {} .. {} | {} | {} | {} / {} |",
            i + 1,
            b.get("agent").str(),
            b.get("games").i64(),
            b.get("wins").i64(),
            b.get("draws").i64(),
            b.get("losses").i64(),
            f2(b.get("score")),
            f2(b.get("ci95").idx(0)),
            f2(b.get("ci95").idx(1)),
            b.get("mean_margin").f64().round(),
            b.get("errors").i64(),
            f2(b.get("act_ms_mean")),
            f2(b.get("act_ms_max"))
        ));
    }
    let f = s.get("focus");
    if f.is_obj() {
        out.push(String::new());
        out.push(format!("## {}", f.get("agent").str()));
        out.push(String::new());
        out.push(format!(
            "Overall score {} over {} games; world-weighted score {}.",
            f2(f.get("overall").get("score")),
            f.get("overall").get("games").i64(),
            f2(f.get("world_weighted_score"))
        ));
        for (title, key) in [
            ("By opponent", "by_opponent"),
            ("By realized world", "by_world"),
            ("By seat", "by_seat"),
        ] {
            out.push(String::new());
            out.push(format!("### {title}"));
            out.push(String::new());
            out.push("| key | games | W/D/L | score | 95% CI |".into());
            out.push("|---|---:|---|---:|---|".into());
            for (k, b) in f.get(key).obj() {
                out.push(format!(
                    "| {k} | {} | {}/{}/{} | {} | {} .. {} |",
                    b.get("games").i64(),
                    b.get("wins").i64(),
                    b.get("draws").i64(),
                    b.get("losses").i64(),
                    f2(b.get("score")),
                    f2(b.get("ci95").idx(0)),
                    f2(b.get("ci95").idx(1))
                ));
            }
        }
    }
    out.push(String::new());
    out.join("\n")
}

/// Paired comparison of two result files for one agent (A vs B build),
/// pairing games by (opponent, seed, the agent's seat). Results files are
/// append-only across resumed runs, so the LATEST row per key wins.
pub fn compare(path_a: &str, path_b: &str, agent_a: &str, agent_b: &str) -> String {
    let keyed = |p: &str, name: &str| -> BTreeMap<(String, i64, usize), f64> {
        let mut m: BTreeMap<(String, i64, usize), Option<f64>> = BTreeMap::new();
        for r in runner::load_rows_ordered(p) {
            for seat in 0..2 {
                if r.agents[seat] == name {
                    let key = (r.agents[1 - seat].clone(), r.seed, seat);
                    m.insert(key, r.scores.map(|sc| sc[seat]));
                }
            }
        }
        m.into_iter()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect()
    };
    let a = keyed(path_a, agent_a);
    let b = keyed(path_b, agent_b);
    let keys: Vec<_> = a.keys().filter(|k| b.contains_key(*k)).cloned().collect();
    let sa: Vec<f64> = keys.iter().map(|k| a[k]).collect();
    let sb: Vec<f64> = keys.iter().map(|k| b[k]).collect();
    let p = stats::paired(&sa, &sb);
    let mut per: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for (k, (x, y)) in keys.iter().zip(sa.iter().zip(sb.iter())) {
        let e = per.entry(k.0.as_str()).or_default();
        if x > y {
            e.0 += 1;
        } else if y > x {
            e.1 += 1;
        }
    }
    let per_opp: Vec<String> = per
        .iter()
        .map(|(o, (x, y))| format!("{}: {{\"a_better\": {x}, \"b_better\": {y}}}", quote(o)))
        .collect();
    format!(
        "{{\"n_pairs\": {}, \"score_a\": {}, \"score_b\": {}, \"better_a\": {}, \"better_b\": {}, \
         \"discordant\": {}, \"p_value\": {}, \"score_diff\": {}, \"ci95\": [{}, {}], \
         \"significant\": {}, \"per_opponent\": {{{}}}}}",
        p.n,
        num(r4(stats::mean(&sa))),
        num(r4(stats::mean(&sb))),
        p.better_a,
        p.better_b,
        p.better_a + p.better_b,
        num(p.p_value),
        num(r4(p.diff)),
        num(r4(p.ci95.0)),
        num(r4(p.ci95.1)),
        p.p_value < 0.05,
        per_opp.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("kagg_tourn_{}_{name}", std::process::id()))
            .to_string_lossy()
            .to_string()
    }

    fn cfg(dir: &str, extra: &str) -> Json {
        let base = json::parse(TEMPLATE).unwrap();
        let mut over = json::parse(extra).unwrap();
        over.set_path("output.dir", Json::Str(dir.into()));
        over.set_path("progress", Json::Bool(false));
        merge(&base, &over)
    }

    #[test]
    fn validation() {
        for bad in [
            r#"{"schedule": "swiss"}"#,
            r#"{"seats": "x"}"#,
            r#"{"panel": []}"#,
            r#"{"on_error": "boom"}"#,
            r#"{"workers": 0}"#,
            r#"{"panel": [{"name": "mine", "type": "builtin", "kind": "idle"}]}"#,
            r#"{"name": "a/b"}"#,
        ] {
            assert!(Tournament::from_json(&cfg("x", bad)).is_err(), "{bad}");
        }
        let mut no_cand = cfg("x", "{}");
        no_cand.set_path("candidate", Json::Null);
        assert!(Tournament::from_json(&no_cand).is_err());
        no_cand.set_path("schedule", Json::Str("round_robin".into()));
        assert!(Tournament::from_json(&no_cand).is_ok());
    }

    #[test]
    fn schedules() {
        let t = Tournament::from_json(&cfg("x", "{}")).unwrap();
        let seeds = vec![(1, None), (2, None)];
        let g = t.games(&seeds);
        assert_eq!(g.len(), 2 * 2 * 2); // 2 opponents x 2 seeds x 2 seats
        assert!(g.iter().all(|x| x.roles.contains(&"candidate".to_string())));
        let t = Tournament::from_json(&cfg(
            "x",
            r#"{"schedule": "round_robin", "seats": "alternate"}"#,
        ))
        .unwrap();
        assert_eq!(t.games(&seeds).len(), 3 * 2); // 3 pairs x 2 seeds x 1
        assert_eq!(seat_orders("seat1", 0), vec![1]);
        assert_eq!(seat_orders("alternate", 3), vec![1]);
    }

    #[test]
    fn end_to_end_builtin_tournament() {
        let dir = tmpdir("e2e");
        let _ = std::fs::remove_dir_all(&dir);
        let t = Tournament::from_json(&cfg(
            &dir,
            r#"{"worlds": {"strategy": "range", "start": 0, "count": 3}, "workers": 2,
                "samples": {"stride": 360, "features": ["time"]},
                "sinks": [{"type": "jsonl", "path": "PLACEHOLDER", "records": ["sample"]}]}"#,
        ))
        .unwrap();
        // point the sink into the temp dir
        let mut t = t;
        let sp = std::path::Path::new(&dir)
            .join("samples.jsonl")
            .to_string_lossy()
            .to_string();
        t.sinks[0].kind = crate::sink::Kind::Jsonl(sp.clone());
        let s = json::parse(&t.execute().unwrap()).unwrap();
        assert_eq!(s.get("games").i64(), 12);
        assert_eq!(s.get("focus").get("overall").get("games").i64(), 12);
        let st = s.get("standings").arr();
        assert_eq!(st.len(), 3);
        let p = std::path::Path::new(&t.out_dir);
        assert!(p.join("summary.md").exists() && p.join("config.json").exists());
        let md = std::fs::read_to_string(p.join("summary.md")).unwrap();
        assert!(md.contains("## Standings"));
        // 12 games x 2 recorded steps (0, 360) x 2 seats
        assert_eq!(std::fs::read_to_string(&sp).unwrap().lines().count(), 48);
        // compare a run with itself: no discordant pairs
        let r = p.join("results.jsonl").to_string_lossy().to_string();
        let c = json::parse(&compare(&r, &r, "mine", "mine")).unwrap();
        assert_eq!(c.get("n_pairs").i64(), 12);
        assert_eq!(c.get("p_value").f64(), 1.0);
        assert!(c.get("per_opponent").is_obj());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn weighting() {
        let mut m = BTreeMap::new();
        let mut a = Acc::default();
        a.push(1.0, 0.0);
        let mut b = Acc::default();
        b.push(0.0, 0.0);
        b.push(0.0, 0.0);
        m.insert("A".to_string(), a);
        m.insert("B".to_string(), b);
        assert_eq!(world_weighted(&m, &Json::Str("uniform".into())), Some(0.5));
        let w = json::parse(r#"{"A": 3, "*": 1}"#).unwrap();
        assert_eq!(world_weighted(&m, &w), Some(0.75));
        assert_eq!(world_weighted(&m, &Json::Str("none".into())), None);
    }
}
