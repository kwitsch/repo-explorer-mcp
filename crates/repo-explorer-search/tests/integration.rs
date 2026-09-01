//! Integration test for `CliSearchBackend` against a bundled fixture repo.
//!
//! Not `#[ignore]`d: it runs wherever `rtk` is present (a common CI tool) and
//! skips cleanly (early return, no panic) when it is not. It needs no network
//! and no live server.

use repo_explorer_core::config::SearchConfig;
use repo_explorer_core::search::{SearchBackend, SearchOptions};
use repo_explorer_search::CliSearchBackend;
use std::path::Path;

fn sample_repo() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sample_repo")
}

#[tokio::test]
async fn searches_bundled_repo_for_literal() {
    if which::which("rtk").is_err() {
        eprintln!("skipping: `rtk` is not available on PATH");
        return;
    }
    let backend = CliSearchBackend::new(&SearchConfig::default(), None);
    let root = sample_repo();
    let findings = backend
        .search(&root, "needle", None, &SearchOptions::default())
        .await
        .expect("search should succeed against the sample repo");

    // alpha.txt and gamma.txt each contain "needle" once.
    assert!(
        findings.len() >= 2,
        "expected at least 2 findings, got {}",
        findings.len()
    );
    assert!(
        findings
            .iter()
            .any(|f| f.snippet.as_deref().unwrap_or("").contains("needle")),
        "expected a finding whose snippet contains the matched literal"
    );
    assert!(
        findings.iter().all(|f| f.location.line_start >= 1),
        "line numbers should be 1-based and present"
    );
}

#[tokio::test]
async fn empty_pattern_is_invalid_input() {
    let backend = CliSearchBackend::new(&SearchConfig::default(), None);
    let root = sample_repo();
    let err = backend
        .search(&root, "", None, &SearchOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        repo_explorer_core::search::SearchError::InvalidInput(_)
    ));
}
