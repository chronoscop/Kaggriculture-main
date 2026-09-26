//! Independent mixed-production pipeline. No baseline source, route tape or opponent private state.
pub mod encoding;
pub mod executor;
pub mod planner;
#[cfg(feature = "train")]
pub mod rollout;
pub const SCHEMA: &str = "mixed-production-v5";
pub const ENCODING: &str = "mixed-routes-96x32-v1";

#[cfg(test)]
mod tests;
