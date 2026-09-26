//! Independent mixed-production pipeline. No baseline source, route tape or opponent private state.
pub mod encoding;
pub mod executor;
pub mod planner;
#[cfg(feature = "train")]
pub mod rollout;
#[cfg(feature = "train")]
pub mod league;
pub const SCHEMA: &str = "mixed-production-v7";
pub const ENCODING: &str = "mixed-routes-96x32-hierarchy-cash-v3";

#[cfg(test)]
mod tests;
