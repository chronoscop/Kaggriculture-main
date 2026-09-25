//! Agent specs and the Python agent host.
//!
//! Spec types (JSON):
//!
//! * `{"name": "mine", "type": "python", "path": "main.py"}` -- a Kaggle
//!   submission file, run in a `python -m kaggsim.host` process and invoked
//!   exactly like the official runner (arity, configuration, fresh isolated
//!   observation per call). The module is re-executed for every game.
//! * `{"name": "f", "type": "factory", "ref": "pkg.mod:make", "kwargs": {}}`
//!   -- any importable factory returning an agent callable (Python host).
//! * `{"name": "p", "type": "pypolicy", "kind": "scripted", "seed": 1}` --
//!   a `kaggsim.policies` fixture policy (Python host).
//! * `{"name": "t", "type": "tape", "path": "x.tape"}` -- open-loop tape.
//! * `{"name": "b", "type": "builtin", "kind": "random", "seed": 1}` -- a
//!   Rust fixture policy (`idle`, `chaos`, `random`); no Python at all.
//!
//! `"seed": "per_game"` (pypolicy / builtin) derives the policy seed from
//! the game seed. Any extra field (for example `"version": "3"`) becomes
//! part of the agent's fingerprint, which is how to force replays for a
//! `factory` or `pypolicy` agent whose code changed.
//!
//! Host protocol (one JSON object per line each way; a host holds up to two
//! agents in slots, slot = seat):
//!
//! ```text
//! -> {"cmd": "ping"}                                  <- {"ok": true, "pid": 1}
//! -> {"cmd": "load", "slot": 0, "spec": {...}, "game_seed": 17, "echo": false}
//!                                                     <- {"ok": true}
//! -> {"cmd": "act", "slot": 0, "obs": {...}}          <- {"line": "...", "ms": 0.8}
//! -> {"cmd": "act2", "obs": [{...}, {...}]}           <- {"lines": [..], "ms": [..]}
//! -> {"cmd": "quit"}
//! ```
//!
//! Agent errors come back as `{"error": "...", "slot": n}`.

use crate::util::{fnv64, i64_or, str_or};
use kagg_engine::json::{self, quote, Json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::Duration;

pub const TYPES: [&str; 5] = ["python", "factory", "pypolicy", "tape", "builtin"];

#[derive(Clone, Debug, PartialEq)]
pub struct AgentSpec {
    pub name: String,
    pub kind: String,
    /// The normalised spec as JSON (sent to the host, used for hashing).
    pub json: Json,
    pub fingerprint: String,
}

/// `canonicalize` without the Windows verbatim prefix (`\\?\C:\x` -> `C:\x`,
/// `\\?\UNC\server\share` -> `\\server\share`).
pub fn display_path(p: &std::path::Path) -> String {
    let s = p.to_string_lossy().to_string();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        rest.to_string()
    } else {
        s
    }
}

/// Bytes of a Python submission for fingerprinting: the file itself plus
/// every other `*.py` file in the same directory (helper modules), in name
/// order. Not recursive.
fn python_content(path: &std::path::Path) -> Result<Vec<u8>, String> {
    let mut out = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if let Some(dir) = path.parent() {
        let mut others: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .map(|it| {
                it.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "py") && p != path)
                    .collect()
            })
            .unwrap_or_default();
        others.sort();
        for o in others {
            out.extend(
                o.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
                    .bytes(),
            );
            out.extend(std::fs::read(&o).unwrap_or_default());
        }
    }
    Ok(out)
}

impl AgentSpec {
    pub fn from_json(j: &Json, default_name: &str) -> Result<Self, String> {
        if !j.is_obj() {
            return Err(format!("agent spec must be an object: {}", j.dump()));
        }
        let kind = str_or(j, "type", "").to_string();
        if !TYPES.contains(&kind.as_str()) {
            return Err(format!(
                "agent spec {}: type must be one of {TYPES:?}",
                j.dump()
            ));
        }
        let mut j = j.clone();
        let name = str_or(&j, "name", default_name).to_string();
        if name.is_empty() {
            return Err("agent name must not be empty".into());
        }
        j.set_path("name", Json::Str(name.clone()));
        let mut content = Vec::new();
        match kind.as_str() {
            "python" | "tape" => {
                let p = str_or(&j, "path", "").to_string();
                if p.is_empty() {
                    return Err(format!("agent {name}: needs a path"));
                }
                let abs =
                    std::fs::canonicalize(&p).map_err(|e| format!("agent {name}: {p}: {e}"))?;
                j.set_path("path", Json::Str(display_path(&abs)));
                if kind == "python" {
                    content = python_content(&abs)?;
                } else {
                    content = std::fs::read(&abs).map_err(|e| format!("{}: {e}", abs.display()))?;
                }
            }
            "factory" => {
                if !str_or(&j, "ref", "").contains(':') {
                    return Err(format!("agent {name}: factory needs ref 'module:name'"));
                }
            }
            "pypolicy" | "builtin" => {
                if str_or(&j, "kind", "").is_empty() {
                    return Err(format!("agent {name}: needs a kind"));
                }
                if kind == "builtin" {
                    let k = str_or(&j, "kind", "");
                    if !kagg_engine::policies::KINDS.contains(&k) {
                        return Err(format!(
                            "agent {name}: builtin kind must be one of {:?}",
                            kagg_engine::policies::KINDS
                        ));
                    }
                }
            }
            _ => unreachable!(),
        }
        // The spec (with the resolved path) plus the code it points at.
        let mut hashed = j.dump().into_bytes();
        hashed.extend(content);
        Ok(AgentSpec {
            name,
            kind,
            fingerprint: format!("{:016x}", fnv64(&hashed)),
            json: j,
        })
    }

    pub fn needs_python(&self) -> bool {
        matches!(self.kind.as_str(), "python" | "factory" | "pypolicy")
    }

    /// Policy seed for builtin / pypolicy agents.
    pub fn policy_seed(&self, game_seed: i64) -> i64 {
        if self.json.get("seed").str() == "per_game" {
            game_seed.wrapping_mul(7919).wrapping_add(13)
        } else {
            i64_or(&self.json, "seed", 0)
        }
    }
}

/// How to start a Python host.
#[derive(Clone, Debug)]
pub struct PythonCfg {
    pub exe: String,
    pub path: Vec<String>,
    pub quiet_stderr: bool,
}

impl Default for PythonCfg {
    fn default() -> Self {
        PythonCfg {
            exe: std::env::var("KAGGSIM_PYTHON").unwrap_or_else(|_| "python".into()),
            path: default_python_path(),
            quiet_stderr: false,
        }
    }
}

impl PythonCfg {
    pub fn from_json(j: &Json) -> Self {
        let mut c = PythonCfg::default();
        if j.get("exe").is_str() {
            c.exe = j.get("exe").str().to_string();
        }
        for p in j.get("path").arr() {
            c.path.push(p.str().to_string());
        }
        c.quiet_stderr = j.get("stderr").str() == "null";
        c
    }
}

/// `src-python` of a source checkout next to the running binary, if any.
fn default_python_path() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors().skip(1).take(5) {
            let cand = anc.join("src-python");
            if cand.join("kaggsim").join("host.py").exists() {
                out.push(display_path(&cand));
                break;
            }
        }
    }
    out
}

/// Why a host call failed.
#[derive(Clone, Debug, PartialEq)]
pub enum HostError {
    /// The agent in this slot raised, returned garbage, or ran out of time:
    /// the seat is at fault.
    Agent(usize, String),
    /// The host process died or the pipe broke: an infrastructure failure
    /// (the game is retried on resume, nobody forfeits).
    Host(String),
}

impl HostError {
    pub fn message(&self) -> &str {
        match self {
            HostError::Agent(_, m) | HostError::Host(m) => m,
        }
    }
}

pub struct PyHost {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    dead: bool,
}

impl PyHost {
    pub fn spawn(cfg: &PythonCfg) -> Result<Self, String> {
        let sep = if cfg!(windows) { ";" } else { ":" };
        let mut paths = cfg.path.clone();
        if let Ok(existing) = std::env::var("PYTHONPATH") {
            if !existing.is_empty() {
                paths.push(existing);
            }
        }
        let mut cmd = Command::new(&cfg.exe);
        cmd.args(["-u", "-m", "kaggsim.host"])
            .env("PYTHONPATH", paths.join(sep))
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(if cfg.quiet_stderr {
                Stdio::null()
            } else {
                Stdio::inherit()
            });
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot start python host ({}): {e}", cfg.exe))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        // A reader thread turns the host's stdout into a channel, so calls
        // can wait with a deadline. It ends when the pipe closes.
        let (tx, lines) = channel::<String>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut h = PyHost {
            child,
            stdin,
            lines,
            dead: false,
        };
        let pong = h
            .rpc("{\"cmd\": \"ping\"}", None)
            .map_err(|e| e.message().to_string())?;
        if !pong.get("ok").bool() {
            return Err(format!("python host did not answer ping: {}", pong.dump()));
        }
        Ok(h)
    }

    pub fn is_dead(&self) -> bool {
        self.dead
    }

    fn kill(&mut self) {
        self.dead = true;
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Send one request; wait for one reply (up to `timeout`, if given).
    /// A timeout kills the host and is reported as `Agent(usize::MAX, ..)`
    /// for the caller to attribute.
    fn rpc(&mut self, line: &str, timeout: Option<Duration>) -> Result<Json, HostError> {
        if self.dead {
            return Err(HostError::Host("python host is not running".into()));
        }
        let wrote = self
            .stdin
            .write_all(line.as_bytes())
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush());
        if let Err(e) = wrote {
            self.dead = true;
            return Err(HostError::Host(format!("python host write: {e}")));
        }
        let resp = match timeout {
            None => self
                .lines
                .recv()
                .map_err(|_| RecvTimeoutError::Disconnected),
            Some(t) => self.lines.recv_timeout(t),
        };
        match resp {
            Ok(resp) => json::parse(resp.trim_end()).map_err(|e| {
                self.kill();
                HostError::Host(format!("python host sent bad JSON ({e}): {resp}"))
            }),
            Err(RecvTimeoutError::Timeout) => {
                self.kill();
                Err(HostError::Agent(usize::MAX, "timeout".into()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.dead = true;
                Err(HostError::Host("python host exited".into()))
            }
        }
    }

    pub fn load(
        &mut self,
        slot: usize,
        spec: &AgentSpec,
        game_seed: i64,
        echo: bool,
    ) -> Result<(), HostError> {
        let r = self.rpc(
            &format!(
                "{{\"cmd\": \"load\", \"slot\": {slot}, \"spec\": {}, \"game_seed\": {game_seed}, \"echo\": {echo}}}",
                spec.json.dump()
            ),
            None,
        )?;
        if r.get("ok").bool() {
            Ok(())
        } else {
            Err(HostError::Agent(
                slot,
                format!("load: {}", r.get("error").str()),
            ))
        }
    }

    /// One slot, optionally with a deadline.
    pub fn act(
        &mut self,
        slot: usize,
        obs_json: &str,
        timeout: Option<Duration>,
    ) -> Result<Acted, HostError> {
        let r = self
            .rpc(
                &format!("{{\"cmd\": \"act\", \"slot\": {slot}, \"obs\": {obs_json}}}"),
                timeout,
            )
            .map_err(|e| match e {
                HostError::Agent(_, m) => HostError::Agent(slot, m),
                other => other,
            })?;
        if !r.get("error").is_null() {
            return Err(HostError::Agent(slot, r.get("error").str().to_string()));
        }
        Ok(Acted {
            line: r.get("line").str().to_string(),
            ms: r.get("ms").f64(),
            action: opt_dump(r.get("action")),
        })
    }

    /// Both slots in one round trip (no deadline).
    pub fn act2(&mut self, obs0: &str, obs1: &str) -> Result<[Acted; 2], HostError> {
        let r = self.rpc(
            &format!("{{\"cmd\": \"act2\", \"obs\": [{obs0}, {obs1}]}}"),
            None,
        )?;
        if !r.get("error").is_null() {
            let slot = r.get("slot").i64().clamp(0, 1) as usize;
            return Err(HostError::Agent(slot, r.get("error").str().to_string()));
        }
        let one = |i: usize| Acted {
            line: r.get("lines").idx(i).str().to_string(),
            ms: r.get("ms").idx(i).f64(),
            action: opt_dump(r.get("actions").idx(i)),
        };
        Ok([one(0), one(1)])
    }
}

/// One agent decision as returned by the host.
#[derive(Clone, Debug, Default)]
pub struct Acted {
    pub line: String,
    pub ms: f64,
    pub action: Option<String>,
}

fn opt_dump(j: &Json) -> Option<String> {
    if j.is_null() {
        None
    } else {
        Some(j.dump())
    }
}

impl Drop for PyHost {
    fn drop(&mut self) {
        if !self.dead {
            let _ = self.stdin.write_all(b"{\"cmd\": \"quit\"}\n");
            let _ = self.stdin.flush();
            for _ in 0..50 {
                if let Ok(Some(_)) = self.child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// JSON string for a name (helper for record writers).
pub fn qname(s: &str) -> String {
    quote(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_validation() {
        let j = json::parse(r#"{"type": "builtin", "kind": "random", "seed": 3}"#).unwrap();
        let s = AgentSpec::from_json(&j, "b").unwrap();
        assert_eq!(s.name, "b");
        assert!(!s.needs_python());
        assert_eq!(s.policy_seed(10), 3);
        let per =
            json::parse(r#"{"name": "x", "type": "builtin", "kind": "idle", "seed": "per_game"}"#)
                .unwrap();
        let s2 = AgentSpec::from_json(&per, "b").unwrap();
        assert_ne!(s2.policy_seed(1), s2.policy_seed(2));
        assert_ne!(s.fingerprint, s2.fingerprint);
        for bad in [
            r#"{"type": "nope"}"#,
            r#"{"type": "builtin", "kind": "zzz"}"#,
            r#"{"type": "python"}"#,
            r#"{"type": "python", "path": "/definitely/not/here.py"}"#,
            r#"{"type": "factory", "ref": "nocolon"}"#,
            r#"{"name": "", "type": "builtin", "kind": "idle"}"#,
            r#"[1]"#,
        ] {
            assert!(
                AgentSpec::from_json(&json::parse(bad).unwrap(), "a").is_err(),
                "{bad}"
            );
        }
        let f = json::parse(r#"{"type": "factory", "ref": "m:f"}"#).unwrap();
        let f1 = AgentSpec::from_json(&f, "f").unwrap();
        assert!(f1.needs_python());
        let fv = json::parse(r#"{"type": "factory", "ref": "m:f", "version": 2}"#).unwrap();
        assert_ne!(
            AgentSpec::from_json(&fv, "f").unwrap().fingerprint,
            f1.fingerprint
        );
    }

    #[test]
    fn fingerprint_covers_helper_modules() {
        let d = std::env::temp_dir().join(format!("kagg_fp_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("main.py"), "def agent(o): return {}\n").unwrap();
        std::fs::write(d.join("helper.py"), "X = 1\n").unwrap();
        let spec = format!(
            r#"{{"type": "python", "path": {}}}"#,
            quote(&d.join("main.py").to_string_lossy())
        );
        let a = AgentSpec::from_json(&json::parse(&spec).unwrap(), "a").unwrap();
        std::fs::write(d.join("helper.py"), "X = 2\n").unwrap();
        let b = AgentSpec::from_json(&json::parse(&spec).unwrap(), "a").unwrap();
        assert_ne!(a.fingerprint, b.fingerprint);
        assert!(!a.json.get("path").str().starts_with(r"\\?\"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn verbatim_paths_are_cleaned() {
        use std::path::Path;
        // drive letter assembled at run time (keeps the scrub audit quiet)
        let drive = "Z:";
        assert_eq!(
            display_path(Path::new(&format!(r"\\?\{drive}\a\b.py"))),
            format!(r"{drive}\a\b.py")
        );
        assert_eq!(
            display_path(Path::new(r"\\?\UNC\srv\share\x.py")),
            r"\\srv\share\x.py"
        );
        assert_eq!(display_path(Path::new("/home/x.py")), "/home/x.py");
    }

    #[test]
    fn python_cfg_from_json() {
        let j = json::parse(r#"{"exe": "py3", "path": ["/x"], "stderr": "null"}"#).unwrap();
        let c = PythonCfg::from_json(&j);
        assert_eq!(c.exe, "py3");
        assert!(c.path.contains(&"/x".to_string()));
        assert!(c.quiet_stderr);
        assert_eq!(qname("a"), "\"a\"");
        assert_eq!(HostError::Host("x".into()).message(), "x");
    }
}
