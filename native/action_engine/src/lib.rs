//! kagg_engine: Rust port of the kaggriculture engine.
//!
//! `engine`/`state` are dependency-free (usable for pure-Rust self-play);
//! `py_api` adds the pyo3 `RustEnv` class behind the `python` feature.

pub mod catalog;
pub mod engine;
pub mod exp28_features;
pub mod features;
pub mod fixed_tape_features;
pub mod legal_masks;
pub mod patch_masks;
pub mod patch_sales;
pub mod py_random;
pub mod state;
pub mod tracker;
pub mod owned_prefix;

#[cfg(feature = "python")]
mod py_api;
