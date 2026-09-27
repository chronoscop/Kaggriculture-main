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
#[cfg(feature = "train")]
pub mod rollout;
pub const TRAINING_REVISION: &str = "v7-independent-6";
pub const SCHEMA: &str = "mixed-production-v7";
pub const ENCODING: &str = "mixed-routes-96x32-hierarchy-cash-independent-v4";

#[cfg(test)]
mod tests;
