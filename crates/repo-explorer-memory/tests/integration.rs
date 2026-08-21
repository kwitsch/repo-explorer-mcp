//! Live-server integration test for `MemoryClientBackend`.
//!
//! Every test here is `#[ignore]`d: it needs a real `codebase-memory-mcp`
//! binary on PATH and indexes THIS repository. Run explicitly with:
//!
//! ```text
//! cargo test -p repo-explorer-memory -- --ignored
//! ```
//!
//! Override the launch command/args via env `REPO_EXPLORER_MEMORY_CMD`
//! (default `codebase-memory-mcp`) and `REPO_EXPLORER_MEMORY_ARGS`
//! (space-separated, default empty).

use repo_explorer_core::config::CodebaseMemoryConfig;
use repo_explorer_core::domain::ExplorationQuery;
use repo_explorer_core::memory::MemoryBackend;
use repo_explorer_memory::MemoryClientBackend;
use std::path::PathBuf;

fn live_config() -> CodebaseMemoryConfig {
    let command = std::env::var("REPO_EXPLORER_MEMORY_CMD")
        .unwrap_or_else(|_| "codebase-memory-mcp".to_string());
    let args = std::env::var("REPO_EXPLORER_MEMORY_ARGS")
        .ok()
        .map(|s| s.split_whitespace().map(|w| w.to_string()).collect())
        .unwrap_or_default();
    CodebaseMemoryConfig {
        command: Some(command),
        args,
        endpoint: None,
        staleness_seconds: 3600,
    }
}

fn repo_root() -> PathBuf {
    // crate manifest dir is crates/repo-explorer-memory; go up two levels.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[tokio::test]
#[ignore = "requires a live codebase-memory-mcp server"]
async fn ensure_fresh_index_then_search_code() {
    let mut backend = MemoryClientBackend::connect(&live_config())
        .await
        .expect("connect to live server");
    let root = repo_root();
    backend
        .ensure_fresh_index(&root)
        .await
        .expect("ensure_fresh_index should not hard-fail");
    let query = ExplorationQuery {
        text: "MemoryBackend".to_string(),
        scope_hint: None,
        max_results: Some(5),
    };
    let result = backend
        .search_code(&root, &query)
        .await
        .expect("search_code should return a non-error result");
    // A non-error result is enough; findings may be empty depending on index.
    let _ = result.summary;
    backend.close().await;
}

#[tokio::test]
async fn endpoint_config_is_unsupported_without_server() {
    let cfg = CodebaseMemoryConfig {
        command: None,
        args: vec![],
        endpoint: Some("http://localhost:1234".to_string()),
        staleness_seconds: 3600,
    };
    let err = MemoryClientBackend::connect(&cfg).await.unwrap_err();
    assert_eq!(
        err,
        repo_explorer_core::memory::MemoryError::UnsupportedTransport(
            "network endpoint".to_string()
        )
    );
}
