//! The `SearchBackend` contract: an async trait plus its typed error and
//! options value type. This module defines the boundary that the
//! `repo-explorer-search` crate implements against `rtk`/`ripgrep`.
//!
//! Core stays free of subprocess/transport concerns: the spawn/IO cause of any
//! failure is formatted into a `String`/message field at the search-crate
//! boundary, so every type here is serde-free and fully comparable. This module
//! is a structural clone of [`crate::memory`].

use crate::domain::ExplorationFinding;
use std::path::Path;

/// Options narrowing a search. A serde-free options bag with all-public fields
/// for struct-update construction; `Default` is all-`None`/`false`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SearchOptions {
    /// Cap on findings returned (applied as a post-parse truncation for uniform
    /// behavior across both tools).
    pub max_results: Option<u32>,
    /// Case-sensitive match. `false` (default) lets the tool use its own
    /// smart/insensitive default; `true` forces `-s`.
    pub case_sensitive: bool,
    /// Lines of surrounding context to include in the snippet (rg `-C`).
    pub context_lines: Option<u32>,
    /// Restrict to files matching this glob (rg `-g`).
    pub file_glob: Option<String>,
}

/// Search-backend failures. Fully comparable so mock-based tests can
/// `assert_eq!` on error values (mirrors [`crate::memory::MemoryError`]).
/// Spawn/IO causes are formatted to `String`, never stored as `std::io::Error`
/// (which is not `Eq`).
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum SearchError {
    /// No usable search binary was found (rtk and ripgrep both absent/unresolvable).
    #[error("no search backend available: {0}")]
    BackendNotFound(String),
    /// The chosen backend failed to spawn or exited with a real error code.
    #[error("search backend `{backend}` failed: {message}")]
    BackendFailed {
        backend: &'static str,
        message: String,
    },
    /// The backend's output could not be parsed into findings.
    #[error("failed to decode `{backend}` output: {message}")]
    Decode {
        backend: &'static str,
        message: String,
    },
    /// The search exceeded the configured timeout.
    #[error("search backend `{backend}` timed out after {seconds}s")]
    Timeout { backend: &'static str, seconds: u64 },
    /// The request itself was invalid (e.g. an empty pattern).
    #[error("invalid search input: {0}")]
    InvalidInput(String),
}

/// The text-search contract implemented by a concrete CLI backend.
///
/// Native `async fn` in trait (AFIT) — no `async-trait` dependency in core,
/// mirroring [`crate::memory::MemoryBackend`]. The `allow` silences the
/// warn-by-default `async_fn_in_trait` lint that `-D warnings` would reject.
#[allow(async_fn_in_trait)]
pub trait SearchBackend {
    /// Search `pattern` under `repo_root`, optionally narrowed to `scope`.
    ///
    /// `repo_root` is the working directory / search base the subprocess runs
    /// against; `scope`, when set, narrows the search to a sub-path (absolute,
    /// or relative to `repo_root`). `pattern` is treated as a literal-capable
    /// regex passed straight to rtk/rg.
    async fn search(
        &self,
        repo_root: &Path,
        pattern: &str,
        scope: Option<&Path>,
        options: &SearchOptions,
    ) -> Result<Vec<ExplorationFinding>, SearchError>;
}

/// In-memory `SearchBackend` for tests: returns a canned result and records
/// each call for assertion. Gated so it compiles for core's own tests and for
/// downstream crates that enable `features = ["test-support"]`.
#[cfg(any(test, feature = "test-support"))]
pub mod mock {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    /// One recorded invocation, capturing the method and its key arguments.
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
        search_result: Result<Vec<ExplorationFinding>, SearchError>,
        calls: Arc<Mutex<Vec<Call>>>,
    }

    impl Default for MockSearchBackend {
        fn default() -> Self {
            Self {
                search_result: Ok(Vec::new()),
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl MockSearchBackend {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_search_result(
            mut self,
            r: Result<Vec<ExplorationFinding>, SearchError>,
        ) -> Self {
            self.search_result = r;
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

    impl SearchBackend for MockSearchBackend {
        async fn search(
            &self,
            repo_root: &Path,
            pattern: &str,
            scope: Option<&Path>,
            options: &SearchOptions,
        ) -> Result<Vec<ExplorationFinding>, SearchError> {
            self.record(Call::Search {
                repo_root: repo_root.to_path_buf(),
                pattern: pattern.to_string(),
                scope: scope.map(|p| p.to_path_buf()),
                options: options.clone(),
            });
            self.search_result.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mock::{Call, MockSearchBackend};
    use std::path::PathBuf;

    #[test]
    fn search_options_default_is_all_none_false() {
        let o = SearchOptions::default();
        assert_eq!(o.max_results, None);
        assert!(!o.case_sensitive);
        assert_eq!(o.context_lines, None);
        assert_eq!(o.file_glob, None);
        assert_eq!(o, SearchOptions::default());
    }

    #[test]
    fn search_error_display_and_eq() {
        let nf = SearchError::BackendNotFound("none".to_string());
        assert_eq!(nf, SearchError::BackendNotFound("none".to_string()));
        assert_eq!(nf.to_string(), "no search backend available: none");

        let bf = SearchError::BackendFailed {
            backend: "rtk",
            message: "boom".to_string(),
        };
        assert_eq!(bf.to_string(), "search backend `rtk` failed: boom");

        let d = SearchError::Decode {
            backend: "ripgrep",
            message: "bad json".to_string(),
        };
        assert_eq!(d.to_string(), "failed to decode `ripgrep` output: bad json");

        let t = SearchError::Timeout {
            backend: "rtk",
            seconds: 30,
        };
        assert_eq!(t.to_string(), "search backend `rtk` timed out after 30s");

        let ii = SearchError::InvalidInput("empty".to_string());
        assert_eq!(ii.to_string(), "invalid search input: empty");
        assert_ne!(nf, ii);
    }

    #[tokio::test]
    async fn mock_default_returns_empty_and_records_call() {
        let backend = MockSearchBackend::new();
        let root = PathBuf::from("/repo");
        let opts = SearchOptions::default();
        let got = backend.search(&root, "needle", None, &opts).await;
        assert_eq!(got, Ok(Vec::new()));
        assert_eq!(
            backend.calls(),
            vec![Call::Search {
                repo_root: PathBuf::from("/repo"),
                pattern: "needle".to_string(),
                scope: None,
                options: SearchOptions::default(),
            }]
        );
    }

    #[tokio::test]
    async fn mock_returns_canned_error() {
        let backend = MockSearchBackend::new()
            .with_search_result(Err(SearchError::BackendNotFound("nothing".to_string())));
        let root = PathBuf::from("/repo");
        let opts = SearchOptions::default();
        let got = backend
            .search(&root, "x", Some(std::path::Path::new("src")), &opts)
            .await;
        assert_eq!(
            got,
            Err(SearchError::BackendNotFound("nothing".to_string()))
        );
        assert_eq!(
            backend.calls()[0],
            Call::Search {
                repo_root: PathBuf::from("/repo"),
                pattern: "x".to_string(),
                scope: Some(PathBuf::from("src")),
                options: SearchOptions::default(),
            }
        );
    }
}
