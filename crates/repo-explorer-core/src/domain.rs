//! Pure domain value types describing exploration queries and results.
//!
//! These types carry no serde derives and perform no I/O — only
//! `Debug + Clone + PartialEq + Eq`. Serialization is added later, at an MCP
//! boundary, if and when it is actually needed (YAGNI).

use std::path::PathBuf;

/// A span of lines within a single file.
#[derive(Debug, Clone, PartialEq, Eq)]
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
