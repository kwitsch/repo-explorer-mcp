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
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
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
    /// `repo_root` could not be turned into a valid input (e.g. a project
    /// name), independent of any connection to the backend.
    #[error("invalid codebase-memory input: {0}")]
    InvalidInput(String),
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

/// In-memory `MemoryBackend` for tests: returns per-method canned results and
/// records each call for assertion. Gated so it compiles for core's own tests
/// and for downstream crates that enable `features = ["test-support"]`.
#[cfg(any(test, feature = "test-support"))]
pub mod mock {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// One recorded invocation, capturing the method and its key arguments.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Call {
        EnsureFreshIndex {
            repo_root: PathBuf,
        },
        SearchCode {
            repo_root: PathBuf,
            query: ExplorationQuery,
        },
        SearchGraph {
            repo_root: PathBuf,
            query: GraphQuery,
        },
        QueryGraph {
            repo_root: PathBuf,
            query: String,
            max_results: Option<u32>,
        },
        TracePath {
            repo_root: PathBuf,
            from: String,
            to: String,
            max_depth: Option<u32>,
        },
        GetArchitecture {
            repo_root: PathBuf,
            depth: Option<u32>,
        },
        GetCodeSnippet {
            repo_root: PathBuf,
            target: SnippetTarget,
        },
    }

    fn empty_result() -> ExplorationResult {
        ExplorationResult {
            findings: Vec::new(),
            summary: String::new(),
        }
    }

    /// Programmable, call-recording `MemoryBackend`.
    #[derive(Clone)]
    pub struct MockMemoryBackend {
        ensure_fresh_index: Result<IndexStatus, MemoryError>,
        search_code: Result<ExplorationResult, MemoryError>,
        search_graph: Result<ExplorationResult, MemoryError>,
        query_graph: Result<ExplorationResult, MemoryError>,
        trace_path: Result<ExplorationResult, MemoryError>,
        get_architecture: Result<ExplorationResult, MemoryError>,
        get_code_snippet: Result<ExplorationResult, MemoryError>,
        calls: Arc<Mutex<Vec<Call>>>,
    }

    impl Default for MockMemoryBackend {
        fn default() -> Self {
            Self {
                ensure_fresh_index: Ok(IndexStatus::UpToDate),
                search_code: Ok(empty_result()),
                search_graph: Ok(empty_result()),
                query_graph: Ok(empty_result()),
                trace_path: Ok(empty_result()),
                get_architecture: Ok(empty_result()),
                get_code_snippet: Ok(empty_result()),
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl MockMemoryBackend {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_ensure_fresh_index_result(
            mut self,
            r: Result<IndexStatus, MemoryError>,
        ) -> Self {
            self.ensure_fresh_index = r;
            self
        }
        pub fn with_search_code_result(
            mut self,
            r: Result<ExplorationResult, MemoryError>,
        ) -> Self {
            self.search_code = r;
            self
        }
        pub fn with_search_graph_result(
            mut self,
            r: Result<ExplorationResult, MemoryError>,
        ) -> Self {
            self.search_graph = r;
            self
        }
        pub fn with_query_graph_result(
            mut self,
            r: Result<ExplorationResult, MemoryError>,
        ) -> Self {
            self.query_graph = r;
            self
        }
        pub fn with_trace_path_result(mut self, r: Result<ExplorationResult, MemoryError>) -> Self {
            self.trace_path = r;
            self
        }
        pub fn with_get_architecture_result(
            mut self,
            r: Result<ExplorationResult, MemoryError>,
        ) -> Self {
            self.get_architecture = r;
            self
        }
        pub fn with_get_code_snippet_result(
            mut self,
            r: Result<ExplorationResult, MemoryError>,
        ) -> Self {
            self.get_code_snippet = r;
            self
        }

        /// Snapshot of recorded calls, in order.
        pub fn calls(&self) -> Vec<Call> {
            self.calls.lock().expect("mock call log poisoned").clone()
        }

        fn record(&self, call: Call) {
            self.calls
                .lock()
                .expect("mock call log poisoned")
                .push(call);
        }
    }

    impl MemoryBackend for MockMemoryBackend {
        async fn ensure_fresh_index(&self, repo_root: &Path) -> Result<IndexStatus, MemoryError> {
            self.record(Call::EnsureFreshIndex {
                repo_root: repo_root.to_path_buf(),
            });
            self.ensure_fresh_index.clone()
        }

        async fn search_code(
            &self,
            repo_root: &Path,
            query: &ExplorationQuery,
        ) -> Result<ExplorationResult, MemoryError> {
            self.record(Call::SearchCode {
                repo_root: repo_root.to_path_buf(),
                query: query.clone(),
            });
            self.search_code.clone()
        }

        async fn search_graph(
            &self,
            repo_root: &Path,
            query: &GraphQuery,
        ) -> Result<ExplorationResult, MemoryError> {
            self.record(Call::SearchGraph {
                repo_root: repo_root.to_path_buf(),
                query: query.clone(),
            });
            self.search_graph.clone()
        }

        async fn query_graph(
            &self,
            repo_root: &Path,
            query: &str,
            max_results: Option<u32>,
        ) -> Result<ExplorationResult, MemoryError> {
            self.record(Call::QueryGraph {
                repo_root: repo_root.to_path_buf(),
                query: query.to_string(),
                max_results,
            });
            self.query_graph.clone()
        }

        async fn trace_path(
            &self,
            repo_root: &Path,
            from: &str,
            to: &str,
            max_depth: Option<u32>,
        ) -> Result<ExplorationResult, MemoryError> {
            self.record(Call::TracePath {
                repo_root: repo_root.to_path_buf(),
                from: from.to_string(),
                to: to.to_string(),
                max_depth,
            });
            self.trace_path.clone()
        }

        async fn get_architecture(
            &self,
            repo_root: &Path,
            depth: Option<u32>,
        ) -> Result<ExplorationResult, MemoryError> {
            self.record(Call::GetArchitecture {
                repo_root: repo_root.to_path_buf(),
                depth,
            });
            self.get_architecture.clone()
        }

        async fn get_code_snippet(
            &self,
            repo_root: &Path,
            target: &SnippetTarget,
        ) -> Result<ExplorationResult, MemoryError> {
            self.record(Call::GetCodeSnippet {
                repo_root: repo_root.to_path_buf(),
                target: target.clone(),
            });
            self.get_code_snippet.clone()
        }
    }
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

    use mock::{Call, MockMemoryBackend};

    #[tokio::test]
    async fn mock_returns_canned_ensure_fresh_index_and_records_call() {
        let backend =
            MockMemoryBackend::new().with_ensure_fresh_index_result(Ok(IndexStatus::Reindexed));
        let root = PathBuf::from("/repo");
        let got = backend.ensure_fresh_index(&root).await;
        assert_eq!(got, Ok(IndexStatus::Reindexed));
        assert_eq!(
            backend.calls(),
            vec![Call::EnsureFreshIndex {
                repo_root: PathBuf::from("/repo")
            }]
        );
    }

    #[tokio::test]
    async fn mock_default_results_are_empty_and_uptodate() {
        let backend = MockMemoryBackend::new();
        let root = PathBuf::from("/repo");
        assert_eq!(
            backend.ensure_fresh_index(&root).await,
            Ok(IndexStatus::UpToDate)
        );
        let q = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let res = backend.search_code(&root, &q).await.unwrap();
        assert!(res.findings.is_empty());
    }

    #[tokio::test]
    async fn mock_returns_canned_error() {
        let backend =
            MockMemoryBackend::new().with_search_graph_result(Err(MemoryError::ToolFailed {
                tool: "search_graph",
                message: "bad".to_string(),
            }));
        let root = PathBuf::from("/repo");
        let gq = GraphQuery::default();
        let got = backend.search_graph(&root, &gq).await;
        assert_eq!(
            got,
            Err(MemoryError::ToolFailed {
                tool: "search_graph",
                message: "bad".to_string()
            })
        );
    }

    #[tokio::test]
    async fn mock_records_all_seven_methods() {
        let backend = MockMemoryBackend::new();
        let root = PathBuf::from("/repo");
        let q = ExplorationQuery {
            text: "t".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let gq = GraphQuery::default();
        let target = SnippetTarget::QualifiedName("a::b".to_string());
        let _ = backend.ensure_fresh_index(&root).await;
        let _ = backend.search_code(&root, &q).await;
        let _ = backend.search_graph(&root, &gq).await;
        let _ = backend.query_graph(&root, "MATCH", Some(5)).await;
        let _ = backend.trace_path(&root, "a", "b", Some(3)).await;
        let _ = backend.get_architecture(&root, Some(2)).await;
        let _ = backend.get_code_snippet(&root, &target).await;
        let calls = backend.calls();
        assert_eq!(calls.len(), 7);
        assert_eq!(
            calls[3],
            Call::QueryGraph {
                repo_root: root.clone(),
                query: "MATCH".to_string(),
                max_results: Some(5)
            }
        );
        assert_eq!(
            calls[4],
            Call::TracePath {
                repo_root: root.clone(),
                from: "a".to_string(),
                to: "b".to_string(),
                max_depth: Some(3)
            }
        );
    }
}
