//! The exploration orchestrator: a deterministic retrieval pre-stage (symbol
//! lookup + grep fanout + ranking, no LLM), an LLM verification stage over the
//! top-k candidates, and an explorative fallback loop for low-confidence
//! queries — plus repo-fingerprint-keyed result caches. Transport-free and
//! generic over its collaborators.
//!
//! This crate owns `serde_json` (tool-schema literals, argument parsing, result
//! rendering, and the persistent result cache's on-disk entries); of
//! `repo-explorer-core`'s types only the three result-shaped domain structs
//! carry serde derives (for that cache), and core itself still does no I/O —
//! continuing the one-impure-dependency-per-crate convention that keeps `rmcp`
//! in `repo-explorer-memory` and `genai` in `repo-explorer-llm`.

mod agent;
mod brief;
mod cache;
/// The on-disk L2 behind the in-memory query cache. Public so the binary's
/// `cache stats` / `cache clear` subcommands can report on and empty the store
/// without constructing an `AgentLoop`.
pub mod disk_cache;
mod dispatch;
pub mod judge_input;
mod judge_verify;
mod pipeline;
mod render;
mod skeleton;
mod snapshot;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
mod tools;
mod verify;

pub use agent::{AgentLoop, AgentLoopError};
pub use snapshot::{RetrievalSnapshot, SnapshotStage, retrieval_snapshot};

/// Do two 1-based, inclusive `[start, end]` line ranges intersect (one shared
/// line is enough)? The one "do two ranges overlap" primitive shared by
/// `render::symbols_for`, `tools::snaps_to_candidate` and
/// `judge_verify::llm_overlap_set` — each still applies its own guards
/// (unknown-location checks, inverted-range clamping, a strictly-inside
/// carve-out) around this call.
pub(crate) fn ranges_overlap(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> bool {
    a_start <= b_end && b_start <= a_end
}
