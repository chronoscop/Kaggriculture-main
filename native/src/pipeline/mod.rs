//! Independent mixed-production pipeline. No baseline source, route tape or opponent private state.
#[cfg(feature = "train")]
pub mod behavior;
pub mod encoding;
pub mod executor;
#[cfg(feature = "train")]
pub mod league;
#[cfg(feature = "train")]
pub mod matchmaking;
pub mod planner;
pub mod public_supply;
#[cfg(feature = "train")]
pub mod rollout;
pub mod trading;
pub const CONTEXT: usize = 320;
pub const GROUPS: usize = 19;
pub const TRAINING_REVISION: &str = "v8-market-3";
pub const SCHEMA: &str = "mixed-production-v8";
pub const ENCODING: &str = "mixed-market-320x32-independent-v3";

#[cfg(test)]
mod tests;
