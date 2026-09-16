//! Pure domain value types describing exploration queries and results.
//!
//! Only the result-shaped types — [`FileLocation`], [`ExplorationFinding`],
//! [`ExplorationResult`] and the [`ExplorationOutcome`]/[`StageExit`] wrapper
//! around them — carry serde derives, and only because the agent crate's
//! persistent result cache writes them to disk:
//! the "if and when it is actually needed" condition the original YAGNI note
//! named is met by that cache, not by an MCP boundary. The on-disk shape is
//! versioned in exactly one place, `repo_explorer_agent::disk_cache::
//! SCHEMA_VERSION`. No other domain type gains derives (the query itself is
//! covered by the cache key), and core still performs no I/O.

use std::path::PathBuf;

/// Saturating `u64` -> `u32`: a value beyond `u32::MAX` (e.g. a malformed or
/// huge line number reported by an upstream tool) clamps to `u32::MAX` rather
/// than silently wrapping to a small, wrong value via a bare `as` cast.
/// Shared by every backend that builds a [`FileLocation`] from externally
/// reported line numbers (`repo-explorer-search`'s parsers,
/// `repo-explorer-memory`'s response mapping).
pub fn saturate_u32(n: u64) -> u32 {
    n.min(u32::MAX as u64) as u32
}

/// A span of lines within a single file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FileLocation {
    pub path: PathBuf,
    pub line_start: u32,
    pub line_end: u32,
}

/// A single finding produced while exploring a repository.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExplorationFinding {
    pub location: FileLocation,
    pub snippet: Option<String>,
    pub note: Option<String>,
}

/// A request to explore a repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplorationQuery {
    pub text: String,
    pub scope_hint: Option<PathBuf>,
    pub max_results: Option<u32>,
    /// Response-side rendering only: `true` caps the *final* findings'
    /// snippets at `agent.snippet_max_chars_detailed` instead of the default
    /// `agent.snippet_max_chars` (as a floor — never narrower than the
    /// concise cap). LLM *prompt* rendering keeps `agent.snippet_max_chars`
    /// either way — raising the prompt cap would inflate token cost, the
    /// opposite of what this flag is for. Part of the query cache key, so a
    /// detailed call can't be served a concise entry.
    ///
    /// Known limit: findings produced by the fallback loop's tool dispatch
    /// are already capped at `agent.snippet_max_chars` when they are
    /// dispatched, so on that loop's budget-exhausted-without-finish exit
    /// this flag cannot widen them.
    pub detailed_snippets: bool,
}

/// The outcome of running an [`ExplorationQuery`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExplorationResult {
    pub findings: Vec<ExplorationFinding>,
    pub summary: String,
}

/// Which stage of the pipeline produced an answer. The wire spellings are the
/// exact literals `QueryMetrics::path` already logs, so a response can be
/// cross-checked against the run's own log line for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StageExit {
    EarlyExit,
    Verify,
    Fallback,
    Cache,
}

impl StageExit {
    /// The same string serde writes — `&'static str` so metrics (which want
    /// exactly that) need no allocation.
    pub fn as_str(&self) -> &'static str {
        match self {
            StageExit::EarlyExit => "early-exit",
            StageExit::Verify => "verify",
            StageExit::Fallback => "fallback",
            StageExit::Cache => "cache",
        }
    }
}

/// An [`ExplorationResult`] plus the provenance the MCP boundary reports next
/// to it. Built exactly once, at the end of `AgentLoop::run` — every backend
/// leg and every intermediate stage keeps passing the bare
/// [`ExplorationResult`], for which this data is meaningless.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExplorationOutcome {
    pub result: ExplorationResult,
    /// The deterministic pre-stage's score (0-100) for its CANDIDATE SET —
    /// deliberately not named `confidence`: it does not score the answer. A
    /// fallback run has a low one by definition (that is what sent it to the
    /// fallback loop) even when the answer is good.
    pub retrieval_confidence: u32,
    pub stage_exit: StageExit,
    /// Qualified symbol per finding location, for the findings where exactly
    /// one ranked candidate covered the location and knew a symbol. Sparse and
    /// keyed by location rather than a field on [`ExplorationFinding`], which
    /// has no symbol on any backend leg.
    pub symbols: Vec<(FileLocation, String)>,
}

/// How a retrieval candidate was found, ordered by intrinsic strength
/// (strongest first). The ranking in `retrieval` keys base scores off this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateKind {
    /// A symbol whose name equals a query token exactly.
    SymbolExact,
    /// A symbol whose name merely contains a query token.
    SymbolFuzzy,
    /// A file whose name/path matches a query path token.
    FileNameHit,
    /// A semantic-search hit from the memory backend.
    SemanticHit,
    /// A plain text (grep) content match.
    ContentHit,
}

/// One location produced by the deterministic retrieval pre-stage, before
/// ranking has selected the top-k. `score` is an integer (0-1000 scale) so the
/// type stays `Eq`/`Hash` like the rest of the domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub location: FileLocation,
    /// Qualified symbol name, when the source knows one.
    pub symbol: Option<String>,
    pub kind: CandidateKind,
    pub score: u32,
    pub snippet: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_types_smoke() {
        let location = FileLocation {
            path: PathBuf::from("src").join("lib.rs"),
            line_start: 10,
            line_end: 20,
        };
        let finding = ExplorationFinding {
            location: location.clone(),
            snippet: Some("fn main() {}".to_string()),
            note: None,
        };
        let query = ExplorationQuery {
            text: "where is main".to_string(),
            scope_hint: Some(PathBuf::from("src")),
            max_results: Some(5),
            detailed_snippets: false,
        };
        let result = ExplorationResult {
            findings: vec![finding.clone()],
            summary: "one match".to_string(),
        };

        // Clone yields an equal value.
        assert_eq!(location, location.clone());
        assert_eq!(finding, finding.clone());
        assert_eq!(query, query.clone());
        assert_eq!(result, result.clone());

        // Distinct field values compare unequal.
        let other = FileLocation {
            path: PathBuf::from("src").join("main.rs"),
            line_start: 1,
            line_end: 2,
        };
        assert_ne!(location, other);

        // Nested access works and holds the expected data.
        assert_eq!(result.findings[0].location.line_start, 10);
        assert_eq!(result.summary, "one match");
    }
}
