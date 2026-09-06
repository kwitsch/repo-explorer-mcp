//! Integration test for `CliSearchBackend` against a bundled fixture repo.
//!
//! Not `#[ignore]`d: it runs wherever `rg` is present (a common CI tool) and
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
    if which::which("rg").is_err() {
        eprintln!("skipping: `rg` is not available on PATH");
        return;
    }
    let backend = CliSearchBackend::new(&SearchConfig::default(), None).await;
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
    let backend = CliSearchBackend::new(&SearchConfig::default(), None).await;
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

#[tokio::test]
async fn back_to_back_searches_are_path_ordered_and_identical() {
    // F-03: with more than the 20-hit truncation cliff, which matches survive
    // the client-side `truncate(max_results)` depends on `rg` traversal order.
    // `--sort path` pins that order, so two runs over an unchanged tree return
    // identical, path-sorted findings (the eval-plan `pre_stage_identical`
    // metric). A flat directory makes `rg --sort path` order equal plain
    // lexicographic filename order, sidestepping the nested-path caveat.
    if which::which("rg").is_err() {
        eprintln!("skipping: `rg` is not available on PATH");
        return;
    }

    // Unique temp dir under the system temp root — no `tempfile` dependency.
    let dir = std::env::temp_dir().join(format!(
        "repo_explorer_f03_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp fixture dir");

    // 25 flat files (> the 20-hit truncation cliff), each containing `needle`.
    for i in 0..25 {
        std::fs::write(dir.join(format!("f{i:02}.txt")), "needle\n").expect("write fixture file");
    }

    let backend = CliSearchBackend::new(&SearchConfig::default(), None).await;
    let opts = SearchOptions {
        max_results: Some(20),
        ..Default::default()
    };

    let first = backend
        .search(&dir, "needle", None, &opts)
        .await
        .expect("first search should succeed");
    let second = backend
        .search(&dir, "needle", None, &opts)
        .await
        .expect("second search should succeed");

    let first_paths: Vec<_> = first.iter().map(|f| f.location.path.clone()).collect();
    let second_paths: Vec<_> = second.iter().map(|f| f.location.path.clone()).collect();

    // Best-effort cleanup before assertions can unwind the test.
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        first_paths.len(),
        20,
        "truncation to max_results=20 must be exercised (fixture has 25 matches)"
    );
    assert_eq!(
        first_paths, second_paths,
        "two back-to-back searches must return identical, order-stable paths (pre_stage_identical)"
    );

    let mut sorted = first_paths.clone();
    sorted.sort();
    assert_eq!(
        first_paths, sorted,
        "flat-dir `rg --sort path` output must be lexicographically non-decreasing"
    );
}
