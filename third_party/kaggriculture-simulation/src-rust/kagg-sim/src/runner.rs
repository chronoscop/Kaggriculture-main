//! The parallel game runner shared by tournaments and self-play.
//!
//! `workers` OS threads pull games from a shared queue. Each thread owns ONE
//! Python agent host (spawned lazily, reused across games; every game loads
//! fresh agents into slot 0 / slot 1, slot = seat) and its parsed tapes, so
//! threads share nothing but the queue and the result channel. A game's
//! engine state is a few kilobytes; tape and builtin seats cost no Python.
//!
//! Episode rules follow the official runner: actions for steps 0..718,
//! final banks read from the step-719 state, per-seat observations,
//! positional empty slots (actions travel as tape lines).
//!
//! Optional time limits mirror the official per-turn budget: each call may
//! take `act_s` seconds, and time beyond that is drawn from a per-game
//! `overage_s` bank; a seat that exhausts it loses the game (`on_error`).

use crate::agent::{AgentSpec, HostError, PyHost, PythonCfg};
use crate::samples::{Buffer, SampleCfg};
use crate::sink::Sinks;
use crate::stats;
use crate::util::fnv64;
use kagg_engine::engine;
use kagg_engine::json::{self, num, quote, Json};
use kagg_engine::obsjson::seat_obs_json;
use kagg_engine::policies::Policy;
use kagg_engine::state::{State, FINAL_STEP};
use kagg_engine::tape::{load_tape, parse_action_line};
use kagg_engine::world::world_key;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// Per-turn time budget (seconds), like the official `actTimeout` plus the
/// `remainingOverageTime` bank.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeLimits {
    pub act_s: f64,
    pub overage_s: f64,
}

impl TimeLimits {
    /// `{"act_s": 1.0, "overage_s": 60.0}`; `null` / absent = no limits.
    pub fn from_json(j: &Json) -> Result<Option<Self>, String> {
        if j.is_null() {
            return Ok(None);
        }
        let act_s = if j.get("act_s").is_num() {
            j.get("act_s").f64()
        } else {
            1.0
        };
        let overage_s = if j.get("overage_s").is_num() {
            j.get("overage_s").f64()
        } else {
            60.0
        };
        if act_s.is_nan() || act_s <= 0.0 || overage_s.is_nan() || overage_s < 0.0 {
            return Err("time_limits: act_s must be > 0 and overage_s >= 0".into());
        }
        Ok(Some(TimeLimits { act_s, overage_s }))
    }
}

#[derive(Clone, Debug)]
pub struct Game {
    pub id: String,
    pub seed: i64,
    pub agents: [Arc<AgentSpec>; 2],
    pub roles: [String; 2],
    pub target_world: Option<String>,
    pub tag: String,
}

impl Game {
    /// Deterministic id: both agents' content fingerprints, names, seed, tag.
    pub fn make_id(a: &AgentSpec, b: &AgentSpec, seed: i64, tag: &str) -> String {
        let key = format!(
            "{}|{}|{}|{}|{seed}|{tag}",
            a.fingerprint, b.fingerprint, a.name, b.name
        );
        format!("{:016x}", fnv64(key.as_bytes()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OnError {
    /// The erroring seat loses (what the official runner does).
    Forfeit,
    /// Keep the row, but give it no scores.
    Exclude,
}

#[derive(Clone, Debug)]
pub struct RunCfg {
    pub workers: usize,
    pub key_depth: usize,
    pub on_error: OnError,
    pub results_path: String,
    pub resume: bool,
    pub python: PythonCfg,
    pub samples: Option<SampleCfg>,
    pub progress: bool,
    /// Optional per-turn time limits (Python seats only).
    pub time_limits: Option<TimeLimits>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub id: String,
    pub seed: i64,
    pub tag: String,
    pub target_world: Option<String>,
    pub world: Option<String>,
    pub shops: Vec<String>,
    pub agents: [String; 2],
    pub roles: [String; 2],
    pub banks: Option<[f64; 2]>,
    pub scores: Option<[f64; 2]>,
    pub steps: i64,
    pub secs: f64,
    pub act_ms_mean: [f64; 2],
    pub act_ms_max: [f64; 2],
    /// (seat or None for an infrastructure failure, message)
    pub error: Option<(Option<usize>, String)>,
    pub forfeit: bool,
}

fn opt_str(s: &Option<String>) -> String {
    s.as_deref().map(quote).unwrap_or_else(|| "null".into())
}

impl Row {
    pub fn to_json(&self) -> String {
        self.to_json_record(None)
    }

    pub fn to_json_record(&self, record: Option<&str>) -> String {
        let pair = |a: &[f64; 2]| format!("[{}, {}]", num(a[0]), num(a[1]));
        let strs = |a: &[String]| {
            let v: Vec<String> = a.iter().map(|s| quote(s)).collect();
            format!("[{}]", v.join(", "))
        };
        let mut parts = Vec::new();
        if let Some(r) = record {
            parts.push(format!("\"record\": {}", quote(r)));
        }
        parts.push(format!("\"game_id\": {}", quote(&self.id)));
        parts.push(format!("\"seed\": {}", self.seed));
        parts.push(format!("\"tag\": {}", quote(&self.tag)));
        parts.push(format!("\"target_world\": {}", opt_str(&self.target_world)));
        parts.push(format!("\"world\": {}", opt_str(&self.world)));
        parts.push(format!("\"shops\": {}", strs(&self.shops)));
        parts.push(format!("\"agents\": {}", strs(&self.agents)));
        parts.push(format!("\"roles\": {}", strs(&self.roles)));
        parts.push(format!(
            "\"banks\": {}",
            self.banks
                .as_ref()
                .map(pair)
                .unwrap_or_else(|| "null".into())
        ));
        parts.push(format!(
            "\"scores\": {}",
            self.scores
                .as_ref()
                .map(pair)
                .unwrap_or_else(|| "null".into())
        ));
        let margin = self
            .banks
            .map(|b| num(b[0] - b[1]))
            .unwrap_or_else(|| "null".into());
        parts.push(format!("\"margin\": {margin}"));
        parts.push(format!("\"steps\": {}", self.steps));
        parts.push(format!(
            "\"secs\": {}",
            num((self.secs * 1000.0).round() / 1000.0)
        ));
        parts.push(format!("\"act_ms_mean\": {}", pair(&self.act_ms_mean)));
        parts.push(format!("\"act_ms_max\": {}", pair(&self.act_ms_max)));
        match &self.error {
            None => parts.push("\"error\": null".into()),
            Some((seat, msg)) => parts.push(format!(
                "\"error\": {{\"seat\": {}, \"message\": {}}}",
                seat.map(|s| s.to_string()).unwrap_or_else(|| "null".into()),
                quote(msg)
            )),
        }
        parts.push(format!("\"forfeit\": {}", self.forfeit));
        format!("{{{}}}", parts.join(", "))
    }

    pub fn from_json(j: &Json) -> Row {
        let pair = |v: &Json| -> Option<[f64; 2]> {
            if v.is_arr() {
                Some([v.idx(0).f64(), v.idx(1).f64()])
            } else {
                None
            }
        };
        let two = |v: &Json| [v.idx(0).str().to_string(), v.idx(1).str().to_string()];
        let os = |v: &Json| {
            if v.is_str() {
                Some(v.str().to_string())
            } else {
                None
            }
        };
        let err = j.get("error");
        Row {
            id: j.get("game_id").str().to_string(),
            seed: j.get("seed").i64(),
            tag: j.get("tag").str().to_string(),
            target_world: os(j.get("target_world")),
            world: os(j.get("world")),
            shops: j
                .get("shops")
                .arr()
                .iter()
                .map(|v| v.str().to_string())
                .collect(),
            agents: two(j.get("agents")),
            roles: two(j.get("roles")),
            banks: pair(j.get("banks")),
            scores: pair(j.get("scores")),
            steps: j.get("steps").i64(),
            secs: j.get("secs").f64(),
            act_ms_mean: pair(j.get("act_ms_mean")).unwrap_or([0.0, 0.0]),
            act_ms_max: pair(j.get("act_ms_max")).unwrap_or([0.0, 0.0]),
            error: if err.is_obj() {
                let s = err.get("seat");
                Some((
                    if s.is_num() {
                        Some(s.i64() as usize)
                    } else {
                        None
                    },
                    err.get("message").str().to_string(),
                ))
            } else {
                None
            },
            forfeit: j.get("forfeit").bool(),
        }
    }
}

enum Seat {
    Builtin(Policy),
    Tape(Arc<Vec<String>>),
    /// A Python agent in slot `.0` of the worker's host.
    Host(usize),
}

/// Per-thread resources: the Python host (one per worker, reused across
/// games) and parsed tapes.
#[derive(Default)]
pub struct Worker {
    host: Option<PyHost>,
    tapes: HashMap<String, Arc<Vec<String>>>,
}

impl Worker {
    fn tape(&mut self, path: &str) -> Result<Arc<Vec<String>>, String> {
        if let Some(t) = self.tapes.get(path) {
            return Ok(t.clone());
        }
        load_tape(path)?; // validates the header
        let raw = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let lines: Vec<String> = raw.lines().skip(1).map(str::to_string).collect();
        let t = Arc::new(lines);
        self.tapes.insert(path.to_string(), t.clone());
        Ok(t)
    }

    /// Number of live Python host processes held by this worker (0 or 1).
    pub fn host_count(&self) -> usize {
        usize::from(self.host.is_some())
    }

    fn host(&mut self, cfg: &PythonCfg) -> Result<&mut PyHost, String> {
        if self.host.as_ref().is_some_and(|h| h.is_dead()) {
            self.host = None;
        }
        if self.host.is_none() {
            self.host = Some(PyHost::spawn(cfg)?);
        }
        Ok(self.host.as_mut().expect("host"))
    }
}

type ErrInfo = (Option<usize>, String);

/// Map a host failure to the row's error: agent faults name the seat,
/// infrastructure failures do not (and are retried on resume).
fn host_err(e: HostError, step: Option<usize>) -> ErrInfo {
    let at = step.map(|s| format!("step {s}: ")).unwrap_or_default();
    match e {
        HostError::Agent(slot, m) => (Some(slot.min(1)), format!("{at}{m}")),
        HostError::Host(m) => (None, format!("{at}{m}")),
    }
}

pub fn play_game(
    g: &Game,
    cfg: &RunCfg,
    w: &mut Worker,
    sample_txs: &[SyncSender<String>],
    sink_failed: &AtomicBool,
) -> Row {
    let t0 = Instant::now();
    let mut row = Row {
        id: g.id.clone(),
        seed: g.seed,
        tag: g.tag.clone(),
        target_world: g.target_world.clone(),
        world: None,
        shops: Vec::new(),
        agents: [g.agents[0].name.clone(), g.agents[1].name.clone()],
        roles: g.roles.clone(),
        banks: None,
        scores: None,
        steps: 0,
        secs: 0.0,
        act_ms_mean: [0.0; 2],
        act_ms_max: [0.0; 2],
        error: None,
        forfeit: false,
    };
    let echo = cfg.samples.as_ref().map(|s| s.action_json).unwrap_or(false);
    let needs_host = g.agents.iter().any(|a| a.needs_python());
    if needs_host {
        if let Err(e) = w.host(&cfg.python) {
            return finish(row, t0, cfg, Some((None, e)), [&[], &[]]);
        }
    }
    let mut seats: Vec<Seat> = Vec::with_capacity(2);
    for (i, a) in g.agents.iter().enumerate() {
        let r: Result<Seat, ErrInfo> = match a.kind.as_str() {
            "builtin" => Policy::new(a.json.get("kind").str(), a.policy_seed(g.seed) as u64)
                .map(Seat::Builtin)
                .map_err(|e| (Some(i), e)),
            "tape" => w
                .tape(a.json.get("path").str())
                .map(Seat::Tape)
                .map_err(|e| (Some(i), e)),
            _ => {
                let h = w.host.as_mut().expect("host");
                h.load(i, a, g.seed, echo)
                    .map(|_| Seat::Host(i))
                    .map_err(|e| host_err(e, None))
            }
        };
        match r {
            Ok(s) => seats.push(s),
            Err(e) => return finish(row, t0, cfg, Some(e), [&[], &[]]),
        }
    }
    // Both seats in one round trip, unless per-call deadlines are needed.
    let both_host =
        cfg.time_limits.is_none() && matches!(seats[..], [Seat::Host(_), Seat::Host(_)]);

    let mut buffer = cfg.samples.as_ref().and_then(|sc| {
        if sample_txs.is_empty() {
            return None;
        }
        let b = Buffer::new(
            sc,
            sc.seats_for(&g.roles, &row.agents),
            fnv64(g.id.as_bytes()),
        );
        if b.active() {
            Some(b)
        } else {
            None
        }
    });

    let mut st = State::new(g.seed);
    let mut ms: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut overage = cfg.time_limits.map(|t| [t.overage_s; 2]);
    let mut error: Option<ErrInfo> = None;
    'game: while st.step < FINAL_STEP {
        let step = st.step as usize;
        let mut lines: [String; 2] = [String::new(), String::new()];
        let mut ajson: [Option<String>; 2] = [None, None];
        if both_host {
            let h = w.host.as_mut().expect("host");
            match h.act2(&seat_obs_json(&st, 0), &seat_obs_json(&st, 1)) {
                Ok(acted) => {
                    for (s, a) in acted.into_iter().enumerate() {
                        ms[s].push(a.ms);
                        lines[s] = a.line;
                        ajson[s] = a.action;
                    }
                }
                Err(e) => {
                    error = Some(host_err(e, Some(step)));
                    break 'game;
                }
            }
        } else {
            for seat in 0..2 {
                lines[seat] = match &mut seats[seat] {
                    Seat::Builtin(p) => p.act_line(&st, seat),
                    Seat::Tape(t) => t.get(step).cloned().unwrap_or_else(|| "PASS\t\t".into()),
                    Seat::Host(slot) => {
                        let deadline = match (cfg.time_limits, overage) {
                            (Some(t), Some(ov)) => {
                                // wall-clock allowance incl. a small IPC grace
                                Some(Duration::from_secs_f64(t.act_s + ov[seat] + 0.25))
                            }
                            _ => None,
                        };
                        let h = w.host.as_mut().expect("host");
                        match h.act(*slot, &seat_obs_json(&st, seat), deadline) {
                            Ok(a) => {
                                ms[seat].push(a.ms);
                                if let (Some(t), Some(ov)) = (cfg.time_limits, overage.as_mut()) {
                                    let over = a.ms / 1000.0 - t.act_s;
                                    if over > 0.0 {
                                        ov[seat] -= over;
                                        if ov[seat] < 0.0 {
                                            error =
                                                Some((Some(seat), format!("step {step}: timeout")));
                                            break 'game;
                                        }
                                    }
                                }
                                ajson[seat] = a.action;
                                a.line
                            }
                            Err(e) => {
                                error = Some(host_err(e, Some(step)));
                                break 'game;
                            }
                        }
                    }
                };
            }
        }
        if let Some(b) = buffer.as_mut() {
            b.record(&st, &lines, &ajson);
        }
        let before = [st.farms[0].money, st.farms[1].money];
        let actions = [parse_action_line(&lines[0]), parse_action_line(&lines[1])];
        engine::step(&mut st, &actions);
        if let Some(b) = buffer.as_mut() {
            b.reward([st.farms[0].money - before[0], st.farms[1].money - before[1]]);
        }
    }
    row.shops = st.town.unlocked_shops.clone();
    row.world = world_key(&row.shops, cfg.key_depth);
    row.steps = st.step;
    if error.is_none() {
        row.banks = Some([st.farms[0].money, st.farms[1].money]);
    }
    let row = finish(row, t0, cfg, error, [&ms[0], &ms[1]]);
    if let (Some(b), Some(banks)) = (buffer, row.banks) {
        let meta: [String; 2] = [0, 1].map(|s| sample_meta(&row, s));
        for line in b.finish(banks, &meta) {
            for tx in sample_txs {
                if tx.send(line.clone()).is_err() {
                    sink_failed.store(true, Ordering::Relaxed);
                }
            }
        }
    }
    row
}

fn sample_meta(row: &Row, seat: usize) -> String {
    format!(
        "\"game_id\": {}, \"seed\": {}, \"tag\": {}, \"seat\": {seat}, \"agent\": {}, \
         \"opponent\": {}, \"role\": {}, \"world\": {}, \"target_world\": {}",
        quote(&row.id),
        row.seed,
        quote(&row.tag),
        quote(&row.agents[seat]),
        quote(&row.agents[1 - seat]),
        quote(&row.roles[seat]),
        opt_str(&row.world),
        opt_str(&row.target_world)
    )
}

fn finish(mut row: Row, t0: Instant, cfg: &RunCfg, error: Option<ErrInfo>, ms: [&[f64]; 2]) -> Row {
    for s in 0..2 {
        if !ms[s].is_empty() {
            row.act_ms_mean[s] = (stats::mean(ms[s]) * 1000.0).round() / 1000.0;
            row.act_ms_max[s] =
                (ms[s].iter().cloned().fold(0.0, f64::max) * 1000.0).round() / 1000.0;
        }
    }
    row.error = error;
    match (&row.error, row.banks) {
        (None, Some(b)) => row.scores = Some([stats::score(b[0], b[1]), stats::score(b[1], b[0])]),
        (Some((Some(seat), _)), _) if cfg.on_error == OnError::Forfeit => {
            row.forfeit = true;
            row.scores = Some(if *seat == 0 { [0.0, 1.0] } else { [1.0, 0.0] });
        }
        _ => row.scores = None,
    }
    row.secs = t0.elapsed().as_secs_f64();
    row
}

/// Every parseable row of a results file, in FILE ORDER (later lines are
/// newer; a game id may appear more than once after resumed runs).
pub fn load_rows_ordered(path: &str) -> Vec<Row> {
    let mut out = Vec::new();
    if let Ok(raw) = std::fs::read_to_string(path) {
        for line in raw.lines() {
            if let Ok(j) = json::parse(line) {
                out.push(Row::from_json(&j));
            }
        }
    }
    out
}

/// Latest row per game id.
pub fn load_rows(path: &str) -> HashMap<String, Row> {
    load_rows_ordered(path)
        .into_iter()
        .map(|r| (r.id.clone(), r))
        .collect()
}

/// Make sure an existing file ends with a newline, so an append after a
/// crash that left a partial last line does not glue two records together.
fn terminate_last_line(path: &str) -> Result<(), String> {
    let Ok(mut f) = std::fs::OpenOptions::new()
        .read(true)
        .append(true)
        .open(path)
    else {
        return Ok(());
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len == 0 {
        return Ok(());
    }
    let mut last = [0u8; 1];
    f.seek(SeekFrom::Start(len - 1))
        .map_err(|e| e.to_string())?;
    f.read_exact(&mut last).map_err(|e| e.to_string())?;
    if last[0] != b'\n' {
        f.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Play `games` (resumable) and return all rows in schedule order. Rows
/// carried over from an earlier run get their world key recomputed with
/// the current `key_depth`.
pub fn run(games: &[Game], cfg: &RunCfg, sinks: &Sinks) -> Result<Vec<Row>, String> {
    let mut seen = HashSet::new();
    for g in games {
        if !seen.insert(&g.id) {
            return Err(format!("duplicate game id {} in schedule", g.id));
        }
    }
    if let Some(parent) = std::path::Path::new(&cfg.results_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let done: HashMap<String, Row> = if cfg.resume {
        terminate_last_line(&cfg.results_path)?;
        load_rows(&cfg.results_path)
            .into_iter()
            .filter(|(_, r)| !matches!(r.error, Some((None, _))))
            .collect()
    } else {
        HashMap::new()
    };
    let todo: Vec<&Game> = games.iter().filter(|g| !done.contains_key(&g.id)).collect();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(cfg.resume)
        .write(true)
        .truncate(!cfg.resume)
        .open(&cfg.results_path)
        .map_err(|e| format!("{}: {e}", cfg.results_path))?;
    let game_txs = sinks.game_senders();
    let sample_txs = if cfg.samples.is_some() {
        sinks.sample_senders()
    } else {
        Vec::new()
    };
    let total = games.len();
    let mut finished = total - todo.len();
    let mut new_rows: HashMap<String, Row> = HashMap::new();
    let next = AtomicUsize::new(0);
    let sink_failed = AtomicBool::new(false);
    let nthreads = cfg.workers.max(1).min(todo.len().max(1));
    let (tx, rx) = mpsc::channel::<Row>();
    let t_start = Instant::now();
    let loop_result = std::thread::scope(|s| -> Result<(), String> {
        for _ in 0..nthreads {
            let tx = tx.clone();
            let sample_txs = sample_txs.clone();
            let todo = &todo;
            let next = &next;
            let sink_failed = &sink_failed;
            s.spawn(move || {
                let mut w = Worker::default();
                loop {
                    if sink_failed.load(Ordering::Relaxed) {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= todo.len() {
                        break;
                    }
                    let row = play_game(todo[i], cfg, &mut w, &sample_txs, sink_failed);
                    if tx.send(row).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut write_err: Option<String> = None;
        for row in rx {
            if write_err.is_none() {
                if let Err(e) = writeln!(file, "{}", row.to_json()).and_then(|_| file.flush()) {
                    write_err = Some(format!("{}: {e}", cfg.results_path));
                    // Stop the workers; keep draining so they can exit.
                    next.store(usize::MAX / 2, Ordering::Relaxed);
                }
            }
            let rec = row.to_json_record(Some("game"));
            for gtx in &game_txs {
                if gtx.send(rec.clone()).is_err() {
                    sink_failed.store(true, Ordering::Relaxed);
                }
            }
            finished += 1;
            if cfg.progress && (finished.is_multiple_of(25) || finished == total) {
                let el = t_start.elapsed().as_secs_f64().max(1e-9);
                eprint!(
                    "\r{finished}/{total} games ({:.1} games/s)   ",
                    (finished - (total - todo.len())) as f64 / el
                );
            }
            new_rows.insert(row.id.clone(), row);
        }
        match write_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    });
    if cfg.progress && total > 0 {
        eprintln!();
    }
    loop_result?;
    if sink_failed.load(Ordering::Relaxed) {
        return Err("an output sink stopped accepting records (see its error); \
                    results.jsonl is intact and the run can be resumed"
            .into());
    }
    Ok(games
        .iter()
        .filter_map(|g| new_rows.get(&g.id).or_else(|| done.get(&g.id)).cloned())
        .map(|mut r| {
            r.world = world_key(&r.shops, cfg.key_depth);
            r
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::{Kind, SinkSpec};
    use kagg_engine::json;

    pub fn builtin(name: &str, kind: &str, seed: i64) -> Arc<AgentSpec> {
        let j = json::parse(&format!(
            r#"{{"name": "{name}", "type": "builtin", "kind": "{kind}", "seed": {seed}}}"#
        ))
        .unwrap();
        Arc::new(AgentSpec::from_json(&j, name).unwrap())
    }

    fn cfg(path: &str, samples: Option<&str>) -> RunCfg {
        RunCfg {
            workers: 2,
            key_depth: 2,
            on_error: OnError::Forfeit,
            results_path: path.into(),
            resume: true,
            python: PythonCfg::default(),
            samples: samples.map(|s| SampleCfg::from_json(&json::parse(s).unwrap()).unwrap()),
            progress: false,
            time_limits: None,
        }
    }

    fn tmp(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("kagg_runner_{}_{name}", std::process::id()))
            .to_string_lossy()
            .to_string()
    }

    fn games(n: i64) -> Vec<Game> {
        let a = builtin("a", "random", 1);
        let b = builtin("b", "chaos", 2);
        (0..n)
            .map(|s| Game {
                id: Game::make_id(&a, &b, s, ""),
                seed: s,
                agents: [a.clone(), b.clone()],
                roles: ["candidate".into(), "panel".into()],
                target_world: None,
                tag: String::new(),
            })
            .collect()
    }

    #[test]
    fn row_json_round_trip() {
        let g = &games(1)[0];
        let mut w = Worker::default();
        let r = play_game(g, &cfg("x", None), &mut w, &[], &AtomicBool::new(false));
        assert_eq!(r.steps, FINAL_STEP);
        assert!(r.scores.is_some() && r.error.is_none());
        let back = Row::from_json(&json::parse(&r.to_json()).unwrap());
        assert_eq!(back.id, r.id);
        assert_eq!(back.banks, r.banks);
        assert_eq!(back.world, r.world);
        let rec = json::parse(&r.to_json_record(Some("game"))).unwrap();
        assert_eq!(rec.get("record").str(), "game");
    }

    #[test]
    fn deterministic_across_workers_and_resumable() {
        let gs = games(6);
        let p1 = tmp("r1.jsonl");
        let p2 = tmp("r2.jsonl");
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
        let sinks = Sinks::start(&[], false).unwrap();
        let mut c1 = cfg(&p1, None);
        c1.workers = 1;
        let r1 = run(&gs, &c1, &sinks).unwrap();
        let r2 = run(&gs, &cfg(&p2, None), &sinks).unwrap();
        let banks = |rs: &[Row]| rs.iter().map(|r| r.banks).collect::<Vec<_>>();
        assert_eq!(banks(&r1), banks(&r2));
        // resume: nothing new is played, file does not grow
        let before = std::fs::read_to_string(&p2).unwrap().lines().count();
        let r3 = run(&gs, &cfg(&p2, None), &sinks).unwrap();
        assert_eq!(banks(&r3), banks(&r2));
        assert_eq!(
            std::fs::read_to_string(&p2).unwrap().lines().count(),
            before
        );
        sinks.finish().unwrap();
        // duplicate ids are refused
        let mut dup = gs.clone();
        dup.push(gs[0].clone());
        assert!(run(&dup, &cfg(&p2, None), &Sinks::start(&[], false).unwrap()).is_err());
        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn samples_reach_sinks() {
        let gs = games(2);
        let res = tmp("r3.jsonl");
        let out = tmp("s3.jsonl");
        let _ = std::fs::remove_file(&res);
        let _ = std::fs::remove_file(&out);
        let spec = SinkSpec {
            kind: Kind::Jsonl(out.clone()),
            games: true,
            samples: true,
        };
        let sinks = Sinks::start(&[spec], false).unwrap();
        let rows = run(
            &gs,
            &cfg(&res, Some(r#"{"stride": 100, "features": ["money"]}"#)),
            &sinks,
        )
        .unwrap();
        sinks.finish().unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        let recs: Vec<Json> = text.lines().map(|l| json::parse(l).unwrap()).collect();
        let n_games = recs
            .iter()
            .filter(|r| r.get("record").str() == "game")
            .count();
        let n_samples = recs
            .iter()
            .filter(|r| r.get("record").str() == "sample")
            .count();
        assert_eq!(n_games, 2);
        // steps 0,100,...,700 = 8 per seat, 2 seats, 2 games
        assert_eq!(n_samples, 32);
        let s = recs
            .iter()
            .find(|r| r.get("record").str() == "sample")
            .unwrap();
        assert!(s.get("features").get("money_me").is_num());
        let id = s.get("game_id").str();
        let row = rows.iter().find(|r| r.id == id).unwrap();
        let seat = s.get("seat").i64() as usize;
        assert_eq!(
            s.get("labels").get("outcome").f64(),
            row.scores.unwrap()[seat]
        );
        let _ = std::fs::remove_file(&res);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn forfeit_on_python_load_failure() {
        let j = json::parse(r#"{"name": "bad", "type": "python", "path": "Cargo.toml"}"#).unwrap();
        let bad = Arc::new(AgentSpec::from_json(&j, "bad").unwrap());
        let good = builtin("g", "idle", 0);
        let g = Game {
            id: "x".into(),
            seed: 1,
            agents: [bad, good],
            roles: ["candidate".into(), "panel".into()],
            target_world: None,
            tag: String::new(),
        };
        let mut c = cfg("unused", None);
        c.python.exe = "definitely-not-python-xyz".into();
        let r = play_game(&g, &c, &mut Worker::default(), &[], &AtomicBool::new(false));
        // host could not start: infrastructure error, no scores
        assert!(matches!(r.error, Some((None, _))));
        assert!(r.scores.is_none());
    }

    #[test]
    fn torn_last_line_is_terminated_before_append() {
        let gs = games(2);
        let p = tmp("torn.jsonl");
        let _ = std::fs::remove_file(&p);
        let sinks = Sinks::start(&[], false).unwrap();
        run(&gs[..1], &cfg(&p, None), &sinks).unwrap();
        // simulate a crash that left half a line
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(b"{\"game_id\": \"partial").unwrap();
        drop(f);
        let rows = run(&gs, &cfg(&p, None), &sinks).unwrap();
        assert_eq!(rows.len(), 2);
        let text = std::fs::read_to_string(&p).unwrap();
        let parsed = text.lines().filter(|l| json::parse(l).is_ok()).count();
        assert_eq!(
            parsed, 2,
            "both real rows parse; only the torn one does not"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn later_rows_win_and_world_is_recomputed() {
        let gs = games(1);
        let p = tmp("later.jsonl");
        let _ = std::fs::remove_file(&p);
        let sinks = Sinks::start(&[], false).unwrap();
        let mut c = cfg(&p, None);
        c.key_depth = 2;
        let r2 = run(&gs, &c, &sinks).unwrap();
        c.key_depth = 1;
        let r1 = run(&gs, &c, &sinks).unwrap();
        assert_eq!(
            r1[0].world.as_deref(),
            r2[0].world.as_deref().map(|w| w.split('|').next().unwrap())
        );
        let mut stale = r2[0].clone();
        stale.banks = Some([1.0, 2.0]);
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        writeln!(f, "{}", stale.to_json()).unwrap();
        drop(f);
        assert_eq!(load_rows(&p)[&stale.id].banks, Some([1.0, 2.0]));
        assert_eq!(load_rows_ordered(&p).len(), 2);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn failing_sink_aborts_the_run() {
        use crate::sink::Kind;
        let gs = games(3);
        let p = tmp("sinkfail.jsonl");
        let _ = std::fs::remove_file(&p);
        let argv: Vec<String> = if cfg!(windows) {
            vec!["cmd".into(), "/C".into(), "exit 3".into()]
        } else {
            vec!["sh".into(), "-c".into(), "exit 3".into()]
        };
        let sinks = Sinks::start(
            &[SinkSpec {
                kind: Kind::Command(argv),
                games: true,
                samples: true,
            }],
            false,
        )
        .unwrap();
        // give the command a moment to exit so sends start failing
        std::thread::sleep(std::time::Duration::from_millis(300));
        let r = run(&gs, &cfg(&p, Some(r#"{"stride": 50}"#)), &sinks);
        let fin = sinks.finish();
        assert!(r.is_err() || fin.is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn time_limits_parse() {
        assert_eq!(TimeLimits::from_json(&Json::Null).unwrap(), None);
        let t = TimeLimits::from_json(&json::parse(r#"{"act_s": 0.5}"#).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!((t.act_s, t.overage_s), (0.5, 60.0));
        assert!(TimeLimits::from_json(&json::parse(r#"{"act_s": 0}"#).unwrap()).is_err());
    }
}
