//! The exploration orchestrator: a deterministic retrieval pre-stage (symbol
//! lookup + grep fanout + ranking, no LLM), an LLM verification stage over the
//! top-k candidates, and an explorative fallback loop for low-confidence
//! queries — plus repo-fingerprint-keyed result caches. Transport-free and
//! generic over its collaborators.
//!
//! This crate owns `serde_json` (tool-schema literals, argument parsing, result
//! rendering); `repo-explorer-core`'s domain/llm types stay serde-free —
//! continuing the one-impure-dependency-per-crate convention that keeps `rmcp`
//! in `repo-explorer-memory` and `genai` in `repo-explorer-llm`.

mod agent;
mod cache;
mod dispatch;
mod pipeline;
mod render;
mod skeleton;
mod tools;
mod verify;

pub use agent::{AgentLoop, AgentLoopError};
