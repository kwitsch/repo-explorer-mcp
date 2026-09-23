//! Dev-only training-data generator for the Stage-10 Laya judge. Builds
//! labelled judge rows programmatically from pinned public repositories — no
//! teacher LLM, no provider call. A library plus a thin binary (`src/main.rs`)
//! so integration tests can reach the generator core directly.
pub mod cli;
pub mod corpus;
pub mod fetch;
pub mod generate;
pub mod label;
pub mod rng;
pub mod rows;
pub mod stats;
pub mod symbols;
pub mod templates;
