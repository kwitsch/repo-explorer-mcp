//! The deterministic retrieval pre-stage taken in isolation — no index refresh,
//! no cache, no LLM — plus the stage classification (`EarlyExit`/`Verify`/
//! `Fallback`) the 10a datagen generator labels rows against. The disk
//! authorization of the real early exit is intentionally NOT simulated: an
//! accepted approximation.

use repo_explorer_core::config::AgentSettings;
use repo_explorer_core::domain::{Candidate, ExplorationQuery};
use repo_explorer_core::memory::MemoryBackend;
use repo_explorer_core::search::SearchBackend;
use std::path::Path;

use crate::agent::early_exit_route;
use crate::pipeline;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotStage {
    EarlyExit,
    Verify,
    Fallback,
}

#[derive(Debug, Clone)]
pub struct RetrievalSnapshot {
    pub candidates: Vec<Candidate>,
    pub confidence: u32,
    pub stage: SnapshotStage,
}

/// The deterministic pre-stage alone. `leg_cache` is `None` (no cache), and no
/// index refresh runs — the caller (datagen) refreshes once per repo.
pub async fn retrieval_snapshot<M: MemoryBackend, S: SearchBackend>(
    memory: &M,
    search: &S,
    repo_root: &Path,
    query: &ExplorationQuery,
    settings: &AgentSettings,
) -> RetrievalSnapshot {
    let outcome = pipeline::retrieve(memory, search, repo_root, query, settings.top_k, None).await;
    let stage = if early_exit_route(&outcome, settings).is_some() {
        SnapshotStage::EarlyExit
    } else if outcome.confidence >= settings.fallback_confidence && !outcome.candidates.is_empty() {
        SnapshotStage::Verify
    } else {
        SnapshotStage::Fallback
    };
    RetrievalSnapshot {
        candidates: outcome.candidates,
        confidence: outcome.confidence,
        stage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::config::AgentSettings;
    use repo_explorer_core::domain::{
        ExplorationFinding, ExplorationQuery, ExplorationResult, FileLocation,
    };
    use repo_explorer_core::memory::mock::MockMemoryBackend;
    use repo_explorer_core::search::mock::MockSearchBackend;
    use std::path::{Path, PathBuf};

    fn finding(path: &str, line: u32, note: Option<&str>) -> ExplorationFinding {
        ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: line,
                line_end: line,
            },
            snippet: None,
            note: note.map(str::to_string),
        }
    }

    fn query(text: &str) -> ExplorationQuery {
        ExplorationQuery {
            text: text.to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        }
    }

    #[tokio::test]
    async fn exact_symbol_is_early_exit() {
        let memory = MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
            findings: vec![finding(
                "crates/x/src/freshness.rs",
                12,
                Some("decide_freshness"),
            )],
            summary: "1".to_string(),
        }));
        let search = MockSearchBackend::new().with_search_result(Ok(vec![]));
        let snap = retrieval_snapshot(
            &memory,
            &search,
            Path::new("/repo"),
            &query("decide_freshness"),
            &AgentSettings::default(),
        )
        .await;
        assert_eq!(snap.stage, SnapshotStage::EarlyExit);
        assert!(!snap.candidates.is_empty());
    }

    #[tokio::test]
    async fn no_candidates_is_fallback() {
        let memory = MockMemoryBackend::new();
        let search = MockSearchBackend::new().with_search_result(Ok(vec![]));
        let snap = retrieval_snapshot(
            &memory,
            &search,
            Path::new("/repo"),
            &query("nothing matches here"),
            &AgentSettings::default(),
        )
        .await;
        assert_eq!(snap.stage, SnapshotStage::Fallback);
        assert_eq!(snap.confidence, 0);
    }

    #[tokio::test]
    async fn ambiguous_exact_symbols_go_to_verify() {
        // Same trusted name in two files: has_exact_symbol_match is true but not
        // unique and confidence < 90, so no early exit; confidence >= 30 -> Verify.
        let memory = MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
            findings: vec![
                finding("a.rs", 12, Some("m::decide_freshness")),
                finding("b.rs", 3, Some("other::decide_freshness")),
            ],
            summary: "2".to_string(),
        }));
        let search = MockSearchBackend::new().with_search_result(Ok(vec![]));
        let snap = retrieval_snapshot(
            &memory,
            &search,
            Path::new("/repo"),
            &query("decide_freshness"),
            &AgentSettings::default(),
        )
        .await;
        assert_eq!(snap.stage, SnapshotStage::Verify);
        assert!(
            snap.confidence >= 30 && snap.confidence < 90,
            "got {}",
            snap.confidence
        );
    }
}
