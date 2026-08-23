//! Pure domain value types describing exploration queries and results.
//!
//! These types carry no serde derives and perform no I/O — only
//! `Debug + Clone + PartialEq + Eq`. Serialization is added later, at an MCP
//! boundary, if and when it is actually needed (YAGNI).

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileLocation {
    pub path: PathBuf,
    pub line_start: u32,
    pub line_end: u32,
}

/// A single finding produced while exploring a repository.
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

/// The outcome of running an [`ExplorationQuery`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplorationResult {
    pub findings: Vec<ExplorationFinding>,
    pub summary: String,
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
