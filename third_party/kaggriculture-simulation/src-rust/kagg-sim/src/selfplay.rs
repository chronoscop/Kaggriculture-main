//! Self-play generation across worlds, streamed to output hooks.
//!
//! ```json
//! {
//!   "name": "sp",
//!   "agent": {"name": "me", "type": "python", "path": "main.py"},
//!   "opponents": {"mode": "mirror"},          // or:
//!   //  {"mode": "pool", "pool": [spec, ...], "p_mirror": 0.25, "rng_seed": 1}
//!   "worlds": {"strategy": "stratified", "pool": [0, 5000],
//!              "per_world": 20, "key_depth": 2},
//!   "seats": "both",
//!   "workers": 8,
//!   "samples": {"features": [], "labels": ["outcome", "return_to_go"],
//!               "gamma": 0.99, "seats": "all"},
//!   "sinks": [{"type": "jsonl", "path": "selfplay/sp/samples.jsonl"}],
//!   "output": {"dir": "selfplay", "resume": true}
//! }
//! ```
//!
//! In `mirror` mode both seats run the agent as independent instances: a
//! submission file is executed separately for each seat, so its module-level
//! state is per seat. (Both seats share one Python interpreter per worker,
//! so modules a submission IMPORTS are shared.) In `pool` mode each
//! game's opponent is drawn from the pool (or the agent itself with
//! probability `p_mirror`), deterministically from `rng_seed` and the game
//! index. Default sink: `<dir>/<name>/samples.jsonl` with every record.

use crate::agent::AgentSpec;
use crate::runner::{self, Game, Row};
use crate::seeding;
use crate::sink::{Kind, SinkSpec, Sinks};
use crate::stats;
use crate::tournament::{parse_common, seat_orders};
use crate::util::{f64_or, i64_or, merge, str_or};
use kagg_engine::json::{self, num, quote, Json};
use kagg_engine::policies::Rng;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

pub const TEMPLATE: &str = r#"{
 "name": "selfplay-example",
 "agent": {"name": "me", "type": "builtin", "kind": "random", "seed": "per_game"},
 "opponents": {"mode": "mirror"},
 "worlds": {"strategy": "stratified", "pool": [0, 2000], "per_world": 4, "key_depth": 2},
 "seats": "seat0",
 "workers": 4,
 "samples": {"features": [], "labels": ["outcome", "margin", "return_to_go"], "gamma": 1.0, "seats": "all", "stride": 1},
 "sinks": [],
 "output": {"dir": "selfplay", "resume": true},
 "python": {"exe": "python", "path": [], "stderr": "inherit"}
}"#;

const DEFAULTS: &str = r#"{
 "name": "selfplay", "opponents": {"mode": "mirror"},
 "worlds": {"strategy": "range", "start": 0, "count": 20},
 "seats": "seat0", "workers": 2, "key_depth": 2, "on_error": "forfeit",
 "samples": {}, "sinks": [], "output": {"dir": "selfplay", "resume": true},
 "python": {}
}"#;

pub struct SelfPlay {
    pub cfg: Json,
    pub name: String,
    pub agent: Arc<AgentSpec>,
    pub mirror: Arc<AgentSpec>,
    pub pool: Vec<Arc<AgentSpec>>,
    pub p_mirror: f64,
    pub rng_seed: u64,
    pub seats: String,
    pub run: runner::RunCfg,
    pub sinks: Vec<SinkSpec>,
    pub worlds: Json,
    pub out_dir: String,
}

impl SelfPlay {
    pub fn from_json(user: &Json) -> Result<Self, String> {
        let cfg = merge(&json::parse(DEFAULTS).expect("defaults"), user);
        let (name, run, mut sinks, out_dir) = parse_common(&cfg, "selfplay")?;
        if run.samples.is_none() {
            return Err("self-play needs a samples block (use {} for defaults)".into());
        }
        if sinks.is_empty() {
            sinks.push(SinkSpec {
                kind: Kind::Jsonl(
                    std::path::Path::new(&out_dir)
                        .join("samples.jsonl")
                        .to_string_lossy()
                        .to_string(),
                ),
                games: true,
                samples: true,
            });
        }
        if !cfg.get("agent").is_obj() {
            return Err("self-play needs an agent".into());
        }
        let agent = Arc::new(AgentSpec::from_json(cfg.get("agent"), "agent")?);
        // the mirror seat: same spec, distinct name so hosts/stats separate
        let mut mj = agent.json.clone();
        mj.set_path("name", Json::Str(format!("{}~mirror", agent.name)));
        let mirror = Arc::new(AgentSpec::from_json(&mj, "mirror")?);
        let opp = cfg.get("opponents");
        let mode = str_or(opp, "mode", "mirror");
        let mut pool = Vec::new();
        match mode {
            "mirror" => {}
            "pool" => {
                for (i, p) in opp.get("pool").arr().iter().enumerate() {
                    pool.push(Arc::new(AgentSpec::from_json(p, &format!("pool{i}"))?));
                }
                if pool.is_empty() {
                    return Err("opponents.pool is empty".into());
                }
            }
            m => return Err(format!("opponents.mode must be mirror or pool, got {m:?}")),
        }
        let seats = str_or(&cfg, "seats", "seat0").to_string();
        if !["both", "seat0", "seat1", "alternate"].contains(&seats.as_str()) {
            return Err("seats must be both, seat0, seat1 or alternate".into());
        }
        Ok(SelfPlay {
            name,
            agent,
            mirror,
            p_mirror: if mode == "mirror" {
                1.0
            } else {
                f64_or(opp, "p_mirror", 0.0)
            },
            rng_seed: i64_or(opp, "rng_seed", 0) as u64,
            pool,
            seats,
            run,
            sinks,
            worlds: cfg.get("worlds").clone(),
            out_dir,
            cfg,
        })
    }

    pub fn games(&self, seeds: &seeding::SeedPlan) -> Vec<Game> {
        let mut rng = Rng::new(self.rng_seed);
        let mut out = Vec::new();
        for (i, (seed, target)) in seeds.iter().enumerate() {
            let opp = if self.pool.is_empty() || rng.unit() < self.p_mirror {
                self.mirror.clone()
            } else {
                self.pool[rng.below(self.pool.len() as u64) as usize].clone()
            };
            for seat in seat_orders(&self.seats, i) {
                let (x, y, rx, ry) = if seat == 0 {
                    (&self.agent, &opp, "candidate", "opponent")
                } else {
                    (&opp, &self.agent, "opponent", "candidate")
                };
                out.push(Game {
                    id: Game::make_id(x, y, *seed, "selfplay"),
                    seed: *seed,
                    agents: [x.clone(), y.clone()],
                    roles: [rx.into(), ry.into()],
                    target_world: target.clone(),
                    tag: "selfplay".into(),
                });
            }
        }
        out
    }

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
        let written = sinks.finish()?;
        let summary = summary(
            &self.name,
            &rows,
            t0.elapsed().as_secs_f64(),
            &written,
            self.run.workers,
        );
        std::fs::write(
            std::path::Path::new(&self.out_dir).join("summary.json"),
            &summary,
        )
        .map_err(|e| e.to_string())?;
        Ok(summary)
    }
}

fn summary(name: &str, rows: &[Row], secs: f64, written: &[u64], workers: usize) -> String {
    let mut worlds: BTreeMap<String, usize> = BTreeMap::new();
    let mut cand: Vec<f64> = Vec::new();
    for r in rows {
        *worlds
            .entry(r.world.clone().unwrap_or_else(|| "?".into()))
            .or_default() += 1;
        if let Some(sc) = r.scores {
            if let Some(seat) = r.roles.iter().position(|x| x == "candidate") {
                cand.push(sc[seat]);
            }
        }
    }
    let w: Vec<String> = worlds
        .iter()
        .map(|(k, v)| format!("{}: {v}", quote(k)))
        .collect();
    let lines: Vec<String> = written.iter().map(|n| n.to_string()).collect();
    format!(
        "{{\"name\": {}, \"games\": {}, \"errors\": {}, \"seconds\": {}, \"games_per_sec\": {}, \"workers\": {workers}, \"candidate_score\": {}, \"records_written\": [{}], \"worlds\": {{{}}}}}",
        quote(name),
        rows.len(),
        rows.iter().filter(|r| r.error.is_some()).count(),
        num((secs * 100.0).round() / 100.0),
        num((rows.len() as f64 / secs.max(1e-9) * 100.0).round() / 100.0),
        if cand.is_empty() { "null".into() } else { num((stats::mean(&cand) * 10000.0).round() / 10000.0) },
        lines.join(", "),
        w.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("kagg_sp_{}_{name}", std::process::id()))
            .to_string_lossy()
            .to_string()
    }

    fn cfg(dir: &str, extra: &str) -> Json {
        let mut over = json::parse(extra).unwrap();
        over.set_path("output.dir", Json::Str(dir.into()));
        over.set_path("progress", Json::Bool(false));
        merge(&json::parse(TEMPLATE).unwrap(), &over)
    }

    #[test]
    fn validation() {
        for bad in [
            r#"{"opponents": {"mode": "zoo"}}"#,
            r#"{"opponents": {"mode": "pool", "pool": []}}"#,
            r#"{"seats": "q"}"#,
            r#"{"samples": null}"#,
        ] {
            assert!(SelfPlay::from_json(&cfg("x", bad)).is_err(), "{bad}");
        }
        let mut j = cfg("x", "{}");
        j.set_path("agent", Json::Null);
        assert!(SelfPlay::from_json(&j).is_err());
    }

    #[test]
    fn pool_mode_mixes_opponents() {
        let sp = SelfPlay::from_json(&cfg(
            "x",
            r#"{"opponents": {"mode": "pool", "p_mirror": 0.5, "rng_seed": 3,
                "pool": [{"name": "c", "type": "builtin", "kind": "chaos", "seed": 1}]},
               "seats": "both"}"#,
        ))
        .unwrap();
        let seeds: seeding::SeedPlan = (0..40).map(|s| (s, None)).collect();
        let g = sp.games(&seeds);
        assert_eq!(g.len(), 80);
        let names: std::collections::BTreeSet<String> = g
            .iter()
            .flat_map(|x| x.agents.iter().map(|a| a.name.clone()))
            .collect();
        assert!(names.contains("c") && names.contains("me~mirror") && names.contains("me"));
        assert!(g.iter().all(|x| x.tag == "selfplay"));
    }

    #[test]
    fn end_to_end_mirror_selfplay() {
        let dir = tmpdir("e2e");
        let _ = std::fs::remove_dir_all(&dir);
        let sp = SelfPlay::from_json(&cfg(
            &dir,
            r#"{"worlds": {"strategy": "stratified", "pool": [0, 200], "per_world": 1, "key_depth": 1},
                "samples": {"stride": 240, "features": ["money", "shops"], "labels": ["outcome", "return_to_go"]}}"#,
        ))
        .unwrap();
        let s = json::parse(&sp.execute().unwrap()).unwrap();
        let games = s.get("games").i64();
        assert!(games >= 5, "one game per first-shop world");
        let text = std::fs::read_to_string(std::path::Path::new(&sp.out_dir).join("samples.jsonl"))
            .unwrap();
        let recs: Vec<Json> = text.lines().map(|l| json::parse(l).unwrap()).collect();
        let samples: Vec<&Json> = recs
            .iter()
            .filter(|r| r.get("record").str() == "sample")
            .collect();
        // steps 0, 240, 480 x 2 seats per game
        assert_eq!(samples.len() as i64, games * 6);
        assert!(samples.iter().all(|r| r.get("target_world").is_str()));
        assert!(samples[0].get("features").get("money_me").is_num());
        assert!(samples[0].get("labels").get("return_to_go").is_num());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
