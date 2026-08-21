//! The `MemoryBackend` contract: an async trait plus its typed error, status,
//! and request value types. This module defines the boundary that the
//! `repo-explorer-memory` crate implements against `codebase-memory-mcp`.
//!
//! Core stays free of `rmcp`/transport concerns: the `rmcp` cause of any
//! failure is formatted into a `String`/message field at the memory-crate
//! boundary, so every type here is serde-free and fully comparable.

use crate::domain::{ExplorationQuery, ExplorationResult};
use std::path::{Path, PathBuf};

/// Outcome of the session-start freshness check. Carried upward as a value:
/// a failed indexing attempt is NOT an error — exploration still proceeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexStatus {
    /// The index was freshly (re)built during this call.
    Reindexed,
    /// An existing index is current within the staleness threshold.
    UpToDate,
    /// Re-index was attempted but failed; caller may explore against the
    /// existing (possibly stale/absent) index anyway.
    IndexingFailed { reason: String },
}

/// A lean graph-search request (maps onto the upstream `search_graph` tool).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphQuery {
    pub name_pattern: Option<String>,
    pub file_pattern: Option<String>,
    pub label: Option<String>,
    pub max_results: Option<u32>,
}

/// How to address a snippet (maps onto `get_code_snippet`'s two modes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnippetTarget {
    QualifiedName(String),
    FileRange {
        file: PathBuf,
        start_line: Option<u32>,
        end_line: Option<u32>,
    },
}

/// Backend/transport-level failures. Fully comparable so mock-based tests can
/// `assert_eq!` on error values (mirrors `config::ValidationError`).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MemoryError {
    /// Connection, spawn, protocol, or timeout failure. The `rmcp` cause is
    /// formatted into the message at the crate boundary (core stays rmcp-free).
    #[error("codebase-memory transport error: {0}")]
    Transport(String),
    /// A tool returned `is_error = true`.
    #[error("codebase-memory tool `{tool}` failed: {message}")]
    ToolFailed { tool: &'static str, message: String },
    /// A tool response could not be decoded into the expected shape.
    #[error("failed to decode `{tool}` response: {message}")]
    Decode { tool: &'static str, message: String },
    /// The chosen transport is not supported in this stage (e.g. network endpoint).
    #[error("unsupported codebase-memory transport: {0}")]
    UnsupportedTransport(String),
}

/// The exploration contract implemented by a concrete memory backend.
///
/// Native `async fn` in trait (AFIT) — no `async-trait` dependency in core.
/// Static dispatch (generics) suffices for Stage 2; a future `dyn` need can be
/// met with enum dispatch without changing this surface. The `allow` silences
/// the warn-by-default `async_fn_in_trait` lint (auto-trait bounds cannot be
/// named on the returned futures), which `-D warnings` would otherwise reject.
#[allow(async_fn_in_trait)]
pub trait MemoryBackend {
    /// Ensure `repo_root` has a current index. Sequences index_status ->
    /// detect_changes -> conditional index_repository against the configured
    /// staleness threshold. Soft failures come back as `Ok(IndexingFailed)`;
    /// only an unusable backend is `Err`.
    async fn ensure_fresh_index(&self, repo_root: &Path) -> Result<IndexStatus, MemoryError>;

    async fn search_code(
        &self,
        repo_root: &Path,
        query: &ExplorationQuery,
    ) -> Result<ExplorationResult, MemoryError>;

    async fn search_graph(
        &self,
        repo_root: &Path,
        query: &GraphQuery,
    ) -> Result<ExplorationResult, MemoryError>;

    async fn query_graph(
        &self,
        repo_root: &Path,
        query: &str,
        max_results: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError>;

    async fn trace_path(
        &self,
        repo_root: &Path,
        from: &str,
        to: &str,
        max_depth: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError>;

    async fn get_architecture(
        &self,
        repo_root: &Path,
        depth: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError>;

    async fn get_code_snippet(
        &self,
        repo_root: &Path,
        target: &SnippetTarget,
    ) -> Result<ExplorationResult, MemoryError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn index_status_is_comparable_and_clones() {
        let a = IndexStatus::Reindexed;
        assert_eq!(a, a.clone());
        assert_ne!(IndexStatus::Reindexed, IndexStatus::UpToDate);
        let failed = IndexStatus::IndexingFailed {
            reason: "boom".to_string(),
        };
        assert_eq!(failed.clone(), failed);
        assert_ne!(failed, IndexStatus::UpToDate);
    }

    #[test]
    fn graph_query_default_is_all_none() {
        let q = GraphQuery::default();
        assert_eq!(q.name_pattern, None);
        assert_eq!(q.file_pattern, None);
        assert_eq!(q.label, None);
        assert_eq!(q.max_results, None);
        assert_eq!(q, GraphQuery::default());
    }

    #[test]
    fn snippet_target_variants_compare() {
        let by_name = SnippetTarget::QualifiedName("foo::bar".to_string());
        assert_eq!(by_name, by_name.clone());
        let by_range = SnippetTarget::FileRange {
            file: PathBuf::from("src/lib.rs"),
            start_line: Some(1),
            end_line: Some(10),
        };
        assert_ne!(by_name, by_range);
    }

    #[test]
    fn memory_error_display_and_eq() {
        let t = MemoryError::Transport("spawn failed".to_string());
        assert_eq!(t, MemoryError::Transport("spawn failed".to_string()));
        assert_eq!(
            t.to_string(),
            "codebase-memory transport error: spawn failed"
        );

        let tf = MemoryError::ToolFailed {
            tool: "search_code",
            message: "no project".to_string(),
        };
        assert_eq!(
            tf.to_string(),
            "codebase-memory tool `search_code` failed: no project"
        );

        let d = MemoryError::Decode {
            tool: "get_architecture",
            message: "bad json".to_string(),
        };
        assert_eq!(
            d.to_string(),
            "failed to decode `get_architecture` response: bad json"
        );

        let u = MemoryError::UnsupportedTransport("network endpoint".to_string());
        assert_eq!(
            u.to_string(),
            "unsupported codebase-memory transport: network endpoint"
        );
        assert_ne!(t, u);
    }
}
