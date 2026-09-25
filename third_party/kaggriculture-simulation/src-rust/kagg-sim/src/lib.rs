//! `kagg-sim`: tournaments, self-play and training-data export on top of
//! the byte-identical `kagg-engine`.
//!
//! Everything that is not agent code runs here, in Rust: scheduling, world
//! selection, the parallel game loop, labelling, feature extraction,
//! output sinks and statistics. Python only hosts Python agents (one small
//! `python -m kaggsim.host` process per worker thread, holding up to two
//! agents, one per seat), so a run's memory footprint is the engine (a few
//! kB per game) plus those hosts; games between built-in policies or tapes
//! need no Python at all.
//!
//! * [`agent`] -- agent specs (python file / factory / tape / builtin) and
//!   the Python agent host protocol.
//! * [`seeding`] -- world-selection strategies.
//! * [`runner`] -- the parallel game runner.
//! * [`samples`] -- per-step samples: features, labels, sampling.
//! * [`sink`] -- output hooks (jsonl file, stdout, external command).
//! * [`tournament`], [`selfplay`] -- the two front ends.
//! * [`stats`] -- scores, confidence intervals, McNemar exact.

// Two-seat games index per-seat arrays by seat number throughout.
#![allow(clippy::needless_range_loop)]

pub mod agent;
pub mod runner;
pub mod samples;
pub mod seeding;
pub mod selfplay;
pub mod sink;
pub mod stats;
pub mod tournament;
pub mod util;
