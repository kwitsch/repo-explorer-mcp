//! The `SearchBackend` contract: an async trait plus its typed error and option
//! value types, defining the boundary the Stage 3 rtk/ripgrep implementation
//! fills. Only the contract and a test mock live here; no subprocess code.
//!
//! Mirrors `memory.rs`: serde-free, fully comparable value types; native AFIT;
//! a `mock` module gated for downstream `test-support`. `repo_root` leads the
//! call for consistency with every `MemoryBackend` method; the LLM never
//! supplies it.

use crate::domain::ExplorationFinding;
use std::path::Path;
#[cfg(any(test, feature = "test-support"))]
use std::path::PathBuf;

/// Whether a search matches file contents or file names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Content,
    FileName,
}

/// Options controlling a single search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchOptions {
    pub mode: SearchMode,
    pub max_results: Option<u32>,
}

/// Search-tool failures. Fully comparable so mock-based tests can `assert_eq!`
/// on error values (mirrors `MemoryError`).
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum SearchError {
    #[error("search transport error: {0}")]
    Transport(String),
    #[error("failed to decode search output: {message}")]
    Decode { message: String },
    #[error("search tool unavailable: {0}")]
    NotAvailable(String),
    #[error("invalid search input: {0}")]
    InvalidInput(String),
}

/// The text/file search contract implemented by a concrete search backend.
///
/// Native `async fn` in trait (AFIT) — no `async-trait` dependency in core,
/// mirroring `MemoryBackend`/`LlmProvider`. The `allow` silences the
/// warn-by-default `async_fn_in_trait` lint that `-D warnings` would reject.
#[allow(async_fn_in_trait)]
pub trait SearchBackend {
    async fn search(
        &self,
        repo_root: &Path,
        pattern: &str,
        scope: Option<&Path>,
        options: &SearchOptions,
    ) -> Result<Vec<ExplorationFinding>, SearchError>;
}

/// In-memory `SearchBackend` for tests: returns a canned result and records each
/// call for assertion. Gated so it compiles for core's own tests and for
/// downstream crates that enable `features = ["test-support"]`.
#[cfg(any(test, feature = "test-support"))]
pub mod mock {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// One recorded `search` invocation.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Call {
        Search {
            repo_root: PathBuf,
            pattern: String,
            scope: Option<PathBuf>,
            options: SearchOptions,
        },
    }

    /// Programmable, call-recording `SearchBackend`.
    #[derive(Clone)]
    pub struct MockSearchBackend {
        search_result: Arc<Result<Vec<ExplorationFinding>, SearchError>>,
        calls: Arc<Mutex<Vec<Call>>>,
    }

    impl Default for MockSearchBackend {
        fn default() -> Self {
            Self {
                search_result: Arc::new(Ok(Vec::new())),
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl MockSearchBackend {
        pub fn new() -> Self {
            Self::default()
        }

        /// Set the canned result returned by every `search` call.
        pub fn with_search_result(self, r: Result<Vec<ExplorationFinding>, SearchError>) -> Self {
            Self {
                search_result: Arc::new(r),
                ..self
            }
        }

        /// Snapshot of recorded calls, in order.
        pub fn calls(&self) -> Vec<Call> {
            self.calls.lock().expect("mock call log poisoned").clone()
        }
    }

    impl SearchBackend for MockSearchBackend {
        async fn search(
            &self,
            repo_root: &Path,
            pattern: &str,
            scope: Option<&Path>,
            options: &SearchOptions,
        ) -> Result<Vec<ExplorationFinding>, SearchError> {
            self.calls
                .lock()
                .expect("mock call log poisoned")
                .push(Call::Search {
                    repo_root: repo_root.to_path_buf(),
                    pattern: pattern.to_string(),
                    scope: scope.map(|p| p.to_path_buf()),
                    options: options.clone(),
                });
            (*self.search_result).clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::FileLocation;
    use mock::{Call, MockSearchBackend};

    fn finding() -> ExplorationFinding {
        ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from("src/lib.rs"),
                line_start: 1,
                line_end: 2,
            },
            snippet: Some("x".to_string()),
            note: None,
        }
    }

    #[test]
    fn search_error_display_and_eq() {
        let t = SearchError::Transport("boom".to_string());
        assert_eq!(t, SearchError::Transport("boom".to_string()));
        assert_eq!(t.to_string(), "search transport error: boom");

        let d = SearchError::Decode {
            message: "bad".to_string(),
        };
        assert_eq!(d.to_string(), "failed to decode search output: bad");

        let n = SearchError::NotAvailable("rtk".to_string());
        assert_eq!(n.to_string(), "search tool unavailable: rtk");

        let i = SearchError::InvalidInput("empty".to_string());
        assert_eq!(i.to_string(), "invalid search input: empty");

        assert_ne!(t, n);
    }

    #[tokio::test]
    async fn mock_records_call_and_returns_canned() {
        let backend = MockSearchBackend::new().with_search_result(Ok(vec![finding()]));
        let root = PathBuf::from("/repo");
        let scope = PathBuf::from("src");
        let opts = SearchOptions {
            mode: SearchMode::Content,
            max_results: Some(10),
        };
        let got = backend.search(&root, "needle", Some(&scope), &opts).await;
        assert_eq!(got, Ok(vec![finding()]));
        assert_eq!(
            backend.calls(),
            vec![Call::Search {
                repo_root: PathBuf::from("/repo"),
                pattern: "needle".to_string(),
                scope: Some(PathBuf::from("src")),
                options: SearchOptions {
                    mode: SearchMode::Content,
                    max_results: Some(10),
                },
            }]
        );
    }

    #[tokio::test]
    async fn mock_default_returns_empty() {
        let backend = MockSearchBackend::new();
        let root = PathBuf::from("/repo");
        let opts = SearchOptions {
            mode: SearchMode::FileName,
            max_results: None,
        };
        let got = backend.search(&root, "p", None, &opts).await.unwrap();
        assert!(got.is_empty());
    }
}
