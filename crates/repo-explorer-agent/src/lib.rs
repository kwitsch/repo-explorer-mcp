//! The exploration agent loop: an LLM-driven loop over memory/search tools,
//! transport-free and generic over its collaborators.
//!
//! This crate owns `serde_json` (tool-schema literals, argument parsing, result
//! rendering); `repo-explorer-core`'s domain/llm types stay serde-free —
//! continuing the one-impure-dependency-per-crate convention that keeps `rmcp`
//! in `repo-explorer-memory` and `genai` in `repo-explorer-llm`.

mod dispatch;
mod tools;

pub use tools::tool_catalog;
