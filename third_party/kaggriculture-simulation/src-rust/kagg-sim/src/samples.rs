//! Per-step training samples: which steps, which seats, which features,
//! which labels.
//!
//! Config (`samples` block; all keys optional):
//!
//! ```json
//! {"features": ["time", "money", "market"],  // groups; [] = all, ["none"]
//!  "include_obs": false,      // the seat's full observation JSON
//!  "include_state": false,    // the full two-seat state JSON
//!  "include_action": true,    // the action as a tape line (+ JSON for
//!                             // Python agents when "action_json": true)
//!  "action_json": false,
//!  "labels": ["outcome", "margin", "return_to_go"],
//!  "gamma": 1.0, "margin_scale": 1.0,
//!  "seats": "all",            // "all" | "candidate" | [0, 1]
//!  "agents": [],              // only these agent names ([] = all)
//!  "stride": 1, "steps": [0, 719], "sample_rate": 1.0, "rng_seed": 0}
//! ```
//!
//! Labels: `outcome` (1/0.5/0), `won`, `margin` (x `margin_scale`),
//! `bank`, `reward` (bank change this step), `return_to_go`
//! (discounted by `gamma`), `steps_left`. Samples are buffered per game and
//! emitted when the game ends, because outcome labels are only known then.
//! Custom features / labels: stream samples (with `include_obs`) to a
//! `command` sink running your own processor (see `kaggsim.processor`).

use crate::util::{bool_or, f64_or, i64_or, strings};
use kagg_engine::features;
use kagg_engine::json::{num, quote, Json};
use kagg_engine::obsjson::{json_state, seat_obs_json};
use kagg_engine::policies::Rng;
use kagg_engine::state::{State, FINAL_STEP};

pub const LABELS: [&str; 7] = [
    "outcome",
    "won",
    "margin",
    "bank",
    "reward",
    "return_to_go",
    "steps_left",
];

#[derive(Clone, Debug)]
pub enum SeatSel {
    All,
    Candidate,
    List(Vec<usize>),
}

#[derive(Clone, Debug)]
pub struct SampleCfg {
    pub groups: Vec<String>,
    pub no_features: bool,
    pub include_obs: bool,
    pub include_state: bool,
    pub include_action: bool,
    pub action_json: bool,
    pub labels: Vec<String>,
    pub gamma: f64,
    pub margin_scale: f64,
    pub seats: SeatSel,
    pub agents: Vec<String>,
    pub stride: i64,
    pub steps: (i64, i64),
    pub sample_rate: f64,
    pub rng_seed: u64,
}

impl SampleCfg {
    pub fn from_json(j: &Json) -> Result<Self, String> {
        let mut groups = strings(j.get("features"));
        let no_features = groups.iter().any(|g| g == "none");
        if no_features {
            groups.clear();
        } else {
            features::check_groups(&groups)?;
        }
        let labels = if j.get("labels").is_arr() {
            strings(j.get("labels"))
        } else {
            vec!["outcome".into(), "margin".into(), "return_to_go".into()]
        };
        for l in &labels {
            if !LABELS.contains(&l.as_str()) {
                return Err(format!("unknown label {l:?}; choose from {LABELS:?}"));
            }
        }
        let seats = match j.get("seats") {
            Json::Str(s) if s == "candidate" => SeatSel::Candidate,
            Json::Str(s) if s == "all" => SeatSel::All,
            Json::Null => SeatSel::All,
            Json::Arr(a) => {
                let mut v = Vec::new();
                for x in a {
                    match x {
                        Json::Num(n) if *n == 0.0 || *n == 1.0 => v.push(*n as usize),
                        other => {
                            return Err(format!(
                                "samples.seats: seat must be 0 or 1, got {}",
                                other.dump()
                            ))
                        }
                    }
                }
                SeatSel::List(v)
            }
            other => return Err(format!("samples.seats: bad value {}", other.dump())),
        };
        let steps = if j.get("steps").is_arr() {
            (j.get("steps").idx(0).i64(), j.get("steps").idx(1).i64())
        } else {
            (0, FINAL_STEP)
        };
        let stride = i64_or(j, "stride", 1);
        if stride < 1 {
            return Err("samples.stride must be >= 1".into());
        }
        Ok(SampleCfg {
            groups,
            no_features,
            include_obs: bool_or(j, "include_obs", false),
            include_state: bool_or(j, "include_state", false),
            include_action: bool_or(j, "include_action", true),
            action_json: bool_or(j, "action_json", false),
            labels,
            gamma: f64_or(j, "gamma", 1.0),
            margin_scale: f64_or(j, "margin_scale", 1.0),
            seats,
            agents: strings(j.get("agents")),
            stride,
            steps,
            sample_rate: f64_or(j, "sample_rate", 1.0),
            rng_seed: i64_or(j, "rng_seed", 0) as u64,
        })
    }

    /// Seats recorded in a game with these roles / agent names.
    pub fn seats_for(&self, roles: &[String; 2], names: &[String; 2]) -> Vec<usize> {
        let mut s: Vec<usize> = match &self.seats {
            SeatSel::All => vec![0, 1],
            SeatSel::Candidate => (0..2).filter(|&i| roles[i] == "candidate").collect(),
            SeatSel::List(v) => v.clone(),
        };
        if !self.agents.is_empty() {
            s.retain(|&i| self.agents.contains(&names[i]));
        }
        s.sort_unstable();
        s.dedup();
        s
    }
}

struct Pending {
    seat: usize,
    step: i64,
    day: i64,
    idx: usize,
    body: String,
}

/// Per-game sample buffer.
pub struct Buffer<'a> {
    cfg: &'a SampleCfg,
    seats: Vec<usize>,
    rewards: [Vec<f64>; 2],
    pending: Vec<Pending>,
    rng: Rng,
}

impl<'a> Buffer<'a> {
    pub fn new(cfg: &'a SampleCfg, seats: Vec<usize>, game_hash: u64) -> Self {
        Buffer {
            cfg,
            seats,
            rewards: [Vec::new(), Vec::new()],
            pending: Vec::new(),
            rng: Rng::new(game_hash ^ cfg.rng_seed),
        }
    }

    pub fn active(&self) -> bool {
        !self.seats.is_empty()
    }

    /// Called with the PRE-step state and the chosen actions.
    pub fn record(&mut self, st: &State, lines: &[String; 2], action_json: &[Option<String>; 2]) {
        let step = st.step;
        let (lo, hi) = self.cfg.steps;
        if step < lo || step >= hi || (step - lo) % self.cfg.stride != 0 {
            return;
        }
        let seats = self.seats.clone();
        for seat in seats {
            if self.cfg.sample_rate < 1.0 && self.rng.unit() >= self.cfg.sample_rate {
                continue;
            }
            let mut parts: Vec<String> = Vec::new();
            if !self.cfg.no_features {
                let f = features::extract(st, seat, &self.cfg.groups).unwrap_or_default();
                parts.push(format!("\"features\": {}", features::to_json(&f)));
            }
            if self.cfg.include_obs {
                parts.push(format!("\"obs\": {}", seat_obs_json(st, seat)));
            }
            if self.cfg.include_state {
                parts.push(format!("\"state\": {}", json_state(st, false)));
            }
            if self.cfg.include_action {
                parts.push(format!("\"action_line\": {}", quote(&lines[seat])));
                if self.cfg.action_json {
                    if let Some(a) = &action_json[seat] {
                        parts.push(format!("\"action\": {a}"));
                    }
                }
            }
            self.pending.push(Pending {
                seat,
                step,
                day: st.day(),
                idx: self.rewards[seat].len(),
                body: parts.join(", "),
            });
        }
    }

    /// Called with the money change of every seat after each step.
    pub fn reward(&mut self, r: [f64; 2]) {
        for s in 0..2 {
            self.rewards[s].push(r[s]);
        }
    }

    /// Finish: one JSON line per sample (`meta` is a pre-rendered
    /// `"key": value, ...` prefix per seat).
    pub fn finish(self, banks: [f64; 2], meta: &[String; 2]) -> Vec<String> {
        let mut rtg: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        for s in 0..2 {
            let rs = &self.rewards[s];
            let mut out = vec![0.0; rs.len()];
            let mut acc = 0.0;
            for i in (0..rs.len()).rev() {
                acc = rs[i] + self.cfg.gamma * acc;
                out[i] = acc;
            }
            rtg[s] = out;
        }
        let mut lines = Vec::with_capacity(self.pending.len());
        for p in &self.pending {
            let (me, opp) = (p.seat, 1 - p.seat);
            let mut labs: Vec<String> = Vec::new();
            for l in &self.cfg.labels {
                let v = match l.as_str() {
                    "outcome" => num(crate::stats::score(banks[me], banks[opp])),
                    "won" => (banks[me] > banks[opp]).to_string(),
                    "margin" => num((banks[me] - banks[opp]) * self.cfg.margin_scale),
                    "bank" => num(banks[me]),
                    "reward" => num(*self.rewards[me].get(p.idx).unwrap_or(&0.0)),
                    "return_to_go" => num(*rtg[me].get(p.idx).unwrap_or(&0.0)),
                    "steps_left" => num((FINAL_STEP - p.step) as f64),
                    _ => "null".into(),
                };
                labs.push(format!("{}: {v}", quote(l)));
            }
            let body = if p.body.is_empty() {
                String::new()
            } else {
                format!("{}, ", p.body)
            };
            lines.push(format!(
                "{{\"record\": \"sample\", {}, \"step\": {}, \"day\": {}, {body}\"labels\": {{{}}}}}",
                meta[me],
                p.step,
                p.day,
                labs.join(", ")
            ));
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kagg_engine::json;

    fn cfg(src: &str) -> SampleCfg {
        SampleCfg::from_json(&json::parse(src).unwrap()).unwrap()
    }

    #[test]
    fn config_defaults_and_errors() {
        let c = cfg("{}");
        assert_eq!(c.labels, vec!["outcome", "margin", "return_to_go"]);
        assert_eq!(c.steps, (0, FINAL_STEP));
        for bad in [
            r#"{"labels": ["nope"]}"#,
            r#"{"features": ["nope"]}"#,
            r#"{"seats": 3}"#,
            r#"{"stride": 0}"#,
            r#"{"seats": [5]}"#,
        ] {
            assert!(
                SampleCfg::from_json(&json::parse(bad).unwrap()).is_err(),
                "{bad}"
            );
        }
        let c = cfg(r#"{"features": ["none"]}"#);
        assert!(c.no_features);
    }

    #[test]
    fn seat_selection() {
        let roles = ["candidate".to_string(), "panel".to_string()];
        let names = ["a".to_string(), "b".to_string()];
        assert_eq!(cfg("{}").seats_for(&roles, &names), vec![0, 1]);
        assert_eq!(
            cfg(r#"{"seats": "candidate"}"#).seats_for(&roles, &names),
            vec![0]
        );
        assert_eq!(cfg(r#"{"seats": [1]}"#).seats_for(&roles, &names), vec![1]);
        assert_eq!(
            cfg(r#"{"seats": [1, 0, 1]}"#).seats_for(&roles, &names),
            vec![0, 1]
        );
        assert_eq!(
            cfg(r#"{"agents": ["b"]}"#).seats_for(&roles, &names),
            vec![1]
        );
    }

    #[test]
    fn labels_and_return_to_go() {
        let c = cfg(
            r#"{"features": ["time"], "labels": ["outcome", "won", "margin", "bank", "reward", "return_to_go", "steps_left"], "gamma": 0.5, "stride": 2}"#,
        );
        let mut b = Buffer::new(&c, vec![0], 1);
        let mut st = State::new(1);
        let lines = ["PASS\t\t".to_string(), "PASS\t\t".to_string()];
        for t in 0..4 {
            st.step = t;
            b.record(&st, &lines, &[None, None]);
            b.reward([1.0, 0.0]);
        }
        let out = b.finish(
            [10.0, 4.0],
            &["\"seat\": 0".to_string(), "\"seat\": 1".to_string()],
        );
        assert_eq!(out.len(), 2); // steps 0 and 2
        let j = json::parse(&out[0]).unwrap();
        let l = j.get("labels");
        assert_eq!(l.get("outcome").f64(), 1.0);
        assert!(l.get("won").bool());
        assert_eq!(l.get("margin").f64(), 6.0);
        assert_eq!(l.get("bank").f64(), 10.0);
        assert_eq!(l.get("reward").f64(), 1.0);
        // 1 + .5 + .25 + .125
        assert_eq!(l.get("return_to_go").f64(), 1.875);
        assert_eq!(l.get("steps_left").f64(), 719.0);
        assert_eq!(j.get("features").get("step").i64(), 0);
        assert_eq!(j.get("action_line").str(), "PASS\t\t");
        let j2 = json::parse(&out[1]).unwrap();
        assert_eq!(j2.get("step").i64(), 2);
        assert_eq!(j2.get("labels").get("return_to_go").f64(), 1.5);
    }

    #[test]
    fn sampling_rate_and_obs() {
        let c = cfg(r#"{"sample_rate": 0.0, "include_obs": true}"#);
        let mut b = Buffer::new(&c, vec![0, 1], 7);
        let st = State::new(1);
        b.record(&st, &[String::new(), String::new()], &[None, None]);
        assert!(b
            .finish(
                [0.0, 0.0],
                &[String::from("\"a\": 1"), String::from("\"a\": 1")]
            )
            .is_empty());
        let c = cfg(
            r#"{"include_obs": true, "include_state": true, "features": ["none"], "action_json": true}"#,
        );
        let mut b = Buffer::new(&c, vec![1], 7);
        b.record(
            &st,
            &[String::new(), "WATER\t\t".into()],
            &[None, Some("{\"farmer\": [\"WATER\"]}".into())],
        );
        b.reward([0.0, 0.0]);
        let out = b.finish(
            [1.0, 1.0],
            &[String::from("\"a\": 1"), String::from("\"a\": 2")],
        );
        let j = json::parse(&out[0]).unwrap();
        assert_eq!(j.get("a").i64(), 2);
        assert_eq!(j.get("obs").get("player").i64(), 1);
        assert!(j.get("state").get("private").is_arr());
        assert!(j.get("features").is_null());
        assert_eq!(j.get("action").get("farmer").idx(0).str(), "WATER");
        assert_eq!(j.get("labels").get("outcome").f64(), 0.5);
    }
}
