//! Determinism + counts for `generate_repo` on a one-file, non-git fixture.

use repo_explorer_core::domain::{ExplorationFinding, ExplorationResult, FileLocation};
use repo_explorer_core::memory::mock::MockMemoryBackend;
use repo_explorer_core::search::mock::MockSearchBackend;
use repo_explorer_datagen::corpus::CorpusRepo;
use repo_explorer_datagen::generate::{GenerateOptions, generate_repo};
use std::path::PathBuf;

fn finding(path: &str, s: u32, e: u32, note: &str) -> ExplorationFinding {
    ExplorationFinding {
        location: FileLocation {
            path: PathBuf::from(path),
            line_start: s,
            line_end: e,
        },
        snippet: None,
        note: Some(note.to_string()),
    }
}

fn fixture() -> PathBuf {
    // One plain source file, 25 lines, no comments/errors. NOT a git repo.
    let dir = std::env::temp_dir().join(format!("datagen_generate_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let body = (1..=25)
        .map(|i| format!("let x{i} = {i};"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(dir.join("a.rs"), body).unwrap();
    dir
}

fn mocks() -> (MockMemoryBackend, MockSearchBackend) {
    // file_outline: one symbol render_widget at 10-20 in a.rs.
    // search_graph (symbol + bm25 legs): an exact match in a.rs (positive) and
    // one in b.rs (negative) -> ambiguous -> Verify.
    let memory = MockMemoryBackend::new()
        .with_file_outline_result(Ok(ExplorationResult {
            findings: vec![finding("a.rs", 10, 20, "module::render_widget")],
            summary: "1".to_string(),
        }))
        .with_search_graph_result(Ok(ExplorationResult {
            findings: vec![
                finding("a.rs", 10, 10, "module::render_widget"),
                finding("b.rs", 5, 5, "other::render_widget"),
            ],
            summary: "2".to_string(),
        }));
    let search = MockSearchBackend::new().with_search_result(Ok(vec![]));
    (memory, search)
}

fn repo() -> CorpusRepo {
    CorpusRepo {
        name: "fix".into(),
        url: "https://github.com/x/fix".into(),
        rev: "0".repeat(40),
        license: "MIT".into(),
        lang: "rust".into(),
        split: "train".into(),
    }
}

fn opts() -> GenerateOptions {
    GenerateOptions {
        seed: 20260923,
        max_queries_per_repo: 10,
        max_negatives_per_query: 4,
    }
}

#[tokio::test]
async fn generate_repo_is_deterministic_and_labels_the_positive() {
    let dir = fixture();
    let (memory, search) = mocks();
    let r = repo();
    let o = opts();

    let a = generate_repo(&memory, &search, &r, &dir, &o).await;
    let b = generate_repo(&memory, &search, &r, &dir, &o).await;
    std::fs::remove_dir_all(&dir).ok();

    let ja = serde_json::to_string(&a.rows).unwrap();
    let jb = serde_json::to_string(&b.rows).unwrap();
    assert_eq!(ja, jb, "two identical runs must be byte-identical");

    assert_eq!(a.stats.files, 1);
    assert_eq!(a.stats.outline_errors, 0);
    assert_eq!(a.stats.symbols_total, 1);
    assert_eq!(a.stats.symbols_sampled, 1);
    assert_eq!(a.stats.queries_generated, 1);
    assert_eq!(a.stats.filtered_early_exit, 0);
    assert_eq!(a.stats.filtered_fallback, 0);
    assert_eq!(a.stats.queries_with_positive, 1);
    assert_eq!(a.stats.queries_without_positive, 0);
    assert_eq!(a.stats.rows_positive, 1);
    assert_eq!(a.stats.rows_negative, 1);
    assert_eq!(a.stats.templates.values().sum::<u32>(), 1);
    assert_eq!(a.rows.len(), 2);

    // Exactly one A and one B, and the A row points into a.rs.
    let labels: Vec<&str> = a.rows.iter().map(|r| r.label.as_str()).collect();
    assert!(labels.contains(&"A") && labels.contains(&"B"));
    let positive = a.rows.iter().find(|r| r.label == "A").unwrap();
    let state: String = serde_json::from_str(&positive.state).unwrap();
    assert!(state.contains("candidate: a.rs:10-10"), "{state}");
}
