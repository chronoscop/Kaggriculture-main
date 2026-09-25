//! `kagg-engine`: a byte-identical Rust port of the Kaggle *Kaggriculture*
//! interpreter (`kaggle-environments` 1.32.7, `envs/kaggriculture`).
//!
//! * [`state`] -- the state model, mirroring the official observation.
//! * [`engine`] -- the step function (unit actions, market, town, decay,
//!   end-of-day), statement order preserved.
//! * [`market`], [`rules`], [`mt19937`] -- price curve, rule tables and a
//!   bit-exact CPython `random.Random`.
//! * [`tape`] -- the tape action format and whole-episode rollouts.
//! * [`obsjson`] / [`loadstate`] / [`obsstate`] -- state <-> JSON.
//!
//! Episode length follows the official runner: actions are applied for
//! steps `0..=718` and the final bank is read from the step-719 state
//! ([`state::FINAL_STEP`]).

// The step function is a statement-order-preserving port of the Python
// interpreter; index loops over seats and mirrored branches are kept on
// purpose so the two implementations can be read side by side.
#![allow(clippy::needless_range_loop, clippy::if_same_then_else)]

pub mod engine;
pub mod features;
pub mod json;
pub mod loadstate;
pub mod market;
pub mod mt19937;
pub mod obsjson;
pub mod obsstate;
pub mod policies;
pub mod rules;
pub mod state;
pub mod tape;
pub mod world;
