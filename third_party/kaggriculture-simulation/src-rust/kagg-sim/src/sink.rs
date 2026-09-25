//! Output hooks ("sinks") for game records and training samples.
//!
//! Every record is one JSON line with a `"record"` field (`"game"` or
//! `"sample"`). A sink receives the record kinds it asks for:
//!
//! ```json
//! {"type": "jsonl",   "path": "out/samples.jsonl", "records": ["sample"]}
//! {"type": "stdout",  "records": ["game"]}
//! {"type": "command", "argv": ["python", "-m", "kaggsim.processor",
//!                              "--out", "feat.jsonl"], "records": ["sample"]}
//! ```
//!
//! A `command` sink starts ONE external process for the run and streams
//! the records to its stdin (language-agnostic: any program that reads
//! JSON lines is a processor). Each sink has its own writer thread fed by a
//! bounded channel, so a slow processor applies back-pressure instead of
//! growing memory.

use crate::util::{str_or, strings};
use kagg_engine::json::Json;
use std::io::{BufWriter, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Jsonl(String),
    Stdout,
    Command(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct SinkSpec {
    pub kind: Kind,
    pub games: bool,
    pub samples: bool,
}

impl SinkSpec {
    pub fn from_json(j: &Json) -> Result<Self, String> {
        let recs = strings(j.get("records"));
        let (games, samples) = if recs.is_empty() {
            (true, true)
        } else {
            for r in &recs {
                if r != "game" && r != "sample" {
                    return Err(format!(
                        "sink records must be 'game' or 'sample', got {r:?}"
                    ));
                }
            }
            (
                recs.iter().any(|r| r == "game"),
                recs.iter().any(|r| r == "sample"),
            )
        };
        let kind = match str_or(j, "type", "") {
            "jsonl" => {
                let p = str_or(j, "path", "");
                if p.is_empty() {
                    return Err("jsonl sink needs a path".into());
                }
                Kind::Jsonl(p.to_string())
            }
            "stdout" => Kind::Stdout,
            "command" => {
                let argv = strings(j.get("argv"));
                if argv.is_empty() {
                    return Err("command sink needs argv".into());
                }
                Kind::Command(argv)
            }
            other => {
                return Err(format!(
                    "unknown sink type {other:?} (jsonl, stdout, command)"
                ))
            }
        };
        Ok(SinkSpec {
            kind,
            games,
            samples,
        })
    }
}

enum Out {
    File(BufWriter<std::fs::File>),
    Stdout,
    Proc(std::process::Child, BufWriter<std::process::ChildStdin>),
}

/// A running sink: a channel into its writer thread.
pub struct Sink {
    pub spec: SinkSpec,
    tx: Option<SyncSender<String>>,
    handle: Option<JoinHandle<Result<u64, String>>>,
}

impl Sink {
    pub fn start(spec: SinkSpec, append: bool) -> Result<Self, String> {
        let out = match &spec.kind {
            Kind::Jsonl(p) => {
                if let Some(parent) = std::path::Path::new(p).parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent).map_err(|e| format!("{p}: {e}"))?;
                    }
                }
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(append)
                    .write(true)
                    .truncate(!append)
                    .open(p)
                    .map_err(|e| format!("{p}: {e}"))?;
                Out::File(BufWriter::new(f))
            }
            Kind::Stdout => Out::Stdout,
            Kind::Command(argv) => {
                let mut child = Command::new(&argv[0])
                    .args(&argv[1..])
                    .stdin(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("cannot start sink command {argv:?}: {e}"))?;
                let stdin = child.stdin.take().ok_or("sink command has no stdin")?;
                Out::Proc(child, BufWriter::new(stdin))
            }
        };
        let (tx, rx) = sync_channel::<String>(4096);
        let handle = std::thread::spawn(move || writer(out, rx));
        Ok(Sink {
            spec,
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    pub fn sender(&self) -> SyncSender<String> {
        self.tx.clone().expect("sink closed")
    }

    /// Close the channel, wait for the writer; returns lines written.
    pub fn finish(mut self) -> Result<u64, String> {
        self.close()
    }

    fn close(&mut self) -> Result<u64, String> {
        self.tx.take();
        match self.handle.take() {
            Some(h) => h.join().map_err(|_| "sink writer panicked".to_string())?,
            None => Ok(0),
        }
    }
}

impl Drop for Sink {
    /// A sink dropped on an error path still flushes and reaps its command.
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn write_line(out: &mut Out, stdout: &std::io::Stdout, line: &str) -> std::io::Result<()> {
    match out {
        Out::File(w) => writeln!(w, "{line}"),
        Out::Stdout => writeln!(stdout.lock(), "{line}"),
        Out::Proc(_, w) => writeln!(w, "{line}"),
    }
}

fn flush(out: &mut Out, stdout: &std::io::Stdout) -> std::io::Result<()> {
    match out {
        Out::File(w) => w.flush(),
        Out::Stdout => stdout.lock().flush(),
        Out::Proc(_, w) => w.flush(),
    }
}

/// Write every record; flush after each burst (whenever the channel is
/// momentarily empty) so data written so far survives a crash.
fn writer(mut out: Out, rx: Receiver<String>) -> Result<u64, String> {
    let mut n = 0u64;
    let stdout = std::io::stdout();
    let mut body = || -> Result<(), String> {
        while let Ok(first) = rx.recv() {
            write_line(&mut out, &stdout, &first).map_err(|e| format!("sink write failed: {e}"))?;
            n += 1;
            while let Ok(line) = rx.try_recv() {
                write_line(&mut out, &stdout, &line)
                    .map_err(|e| format!("sink write failed: {e}"))?;
                n += 1;
            }
            flush(&mut out, &stdout).map_err(|e| format!("sink flush failed: {e}"))?;
        }
        Ok(())
    };
    let result = body();
    drop(rx);
    match out {
        Out::Proc(mut child, w) => {
            drop(w); // close the child's stdin
            if result.is_err() {
                let _ = child.kill();
            }
            let status = child.wait().map_err(|e| e.to_string())?;
            result?;
            if !status.success() {
                return Err(format!("sink command exited with {status}"));
            }
        }
        _ => result?,
    }
    Ok(n)
}

/// All sinks of a run, split by record kind.
pub struct Sinks {
    pub sinks: Vec<Sink>,
}

impl Sinks {
    pub fn start(specs: &[SinkSpec], append: bool) -> Result<Self, String> {
        let mut sinks = Vec::new();
        for s in specs {
            sinks.push(Sink::start(s.clone(), append)?);
        }
        Ok(Sinks { sinks })
    }

    pub fn game_senders(&self) -> Vec<SyncSender<String>> {
        self.sinks
            .iter()
            .filter(|s| s.spec.games)
            .map(|s| s.sender())
            .collect()
    }

    pub fn sample_senders(&self) -> Vec<SyncSender<String>> {
        self.sinks
            .iter()
            .filter(|s| s.spec.samples)
            .map(|s| s.sender())
            .collect()
    }

    pub fn wants_samples(&self) -> bool {
        self.sinks.iter().any(|s| s.spec.samples)
    }

    /// Finish every sink (all are joined even if one failed); returns the
    /// lines written per sink, or the first error.
    pub fn finish(self) -> Result<Vec<u64>, String> {
        let results: Vec<Result<u64, String>> =
            self.sinks.into_iter().map(|s| s.finish()).collect();
        results.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kagg_engine::json;

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("kagg_sink_{}_{name}", std::process::id()));
        d.to_string_lossy().to_string()
    }

    #[test]
    fn spec_parsing() {
        let s = SinkSpec::from_json(&json::parse(r#"{"type": "stdout"}"#).unwrap()).unwrap();
        assert!(s.games && s.samples);
        let s = SinkSpec::from_json(
            &json::parse(r#"{"type": "jsonl", "path": "x", "records": ["sample"]}"#).unwrap(),
        )
        .unwrap();
        assert!(!s.games && s.samples);
        for bad in [
            r#"{"type": "jsonl"}"#,
            r#"{"type": "command"}"#,
            r#"{"type": "zip"}"#,
            r#"{"type": "stdout", "records": ["x"]}"#,
        ] {
            assert!(SinkSpec::from_json(&json::parse(bad).unwrap()).is_err());
        }
    }

    #[test]
    fn jsonl_sink_writes_and_appends() {
        let p = tmp("a.jsonl");
        let _ = std::fs::remove_file(&p);
        let spec = SinkSpec {
            kind: Kind::Jsonl(p.clone()),
            games: true,
            samples: false,
        };
        let sinks = Sinks::start(std::slice::from_ref(&spec), false).unwrap();
        assert!(!sinks.wants_samples());
        for tx in sinks.game_senders() {
            tx.send("{\"record\": \"game\"}".into()).unwrap();
        }
        assert!(sinks.sample_senders().is_empty());
        assert_eq!(sinks.finish().unwrap(), vec![1]);
        let s2 = Sink::start(spec, true).unwrap();
        s2.sender().send("{}".into()).unwrap();
        s2.finish().unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap().lines().count(), 2);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn command_sink_streams_to_stdin() {
        // `sort` exists on every CI platform; write its output to a file via
        // the shell-free route: use the running test binary? Keep it simple:
        // a command that consumes stdin and exits 0.
        let argv: Vec<String> = if cfg!(windows) {
            vec!["cmd".into(), "/C".into(), "more > NUL".into()]
        } else {
            vec!["sh".into(), "-c".into(), "cat > /dev/null".into()]
        };
        let spec = SinkSpec {
            kind: Kind::Command(argv),
            games: true,
            samples: true,
        };
        let s = Sink::start(spec, false).unwrap();
        for i in 0..100 {
            s.sender().send(format!("{{\"i\": {i}}}")).unwrap();
        }
        assert_eq!(s.finish().unwrap(), 100);
        let bad = SinkSpec {
            kind: Kind::Command(vec!["definitely-not-a-program-xyz".into()]),
            games: true,
            samples: true,
        };
        assert!(Sink::start(bad, false).is_err());
    }
}
