//! Graph-based skeleton views: a file's symbol outline (names + line ranges)
//! from the memory backend, instead of feeding whole files to the model. The
//! symbol name rides on `ExplorationFinding.note` (set by the memory crate).

use repo_explorer_core::memory::MemoryBackend;
use repo_explorer_core::retrieval::is_unknown_location;
use std::path::Path;

/// Symbols listed per skeleton (files rarely have more worth showing).
const MAX_OUTLINE_SYMBOLS: usize = 30;

/// A compact outline of `path`, or `None` when the graph knows nothing about
/// it (caller falls back to the candidate's snippet). Backed by
/// `MemoryBackend::file_outline` — exact path, source order, no file/module
/// container rows — rather than a `search_graph{file_pattern}` regex match,
/// whose answer came in arbitrary order and carried a `__file__` row that
/// rendered as a bogus `(location unknown)` line in every verify prompt.
pub(crate) async fn skeleton_for<M: MemoryBackend>(
    memory: &M,
    repo_root: &Path,
    path: &Path,
) -> Option<String> {
    let res = memory
        .file_outline(repo_root, path, Some(MAX_OUTLINE_SYMBOLS as u32))
        .await
        .ok()?;
    let mut lines: Vec<String> = res
        .findings
        .iter()
        .take(MAX_OUTLINE_SYMBOLS)
        .filter_map(|f| {
            f.note.as_deref().map(|name| {
                if is_unknown_location(&f.location) {
                    format!("  {name} (location unknown)")
                } else {
                    format!(
                        "  {name} @ {}-{}",
                        f.location.line_start, f.location.line_end
                    )
                }
            })
        })
        .collect();
    lines.dedup();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::{ExplorationFinding, ExplorationResult, FileLocation};
    use repo_explorer_core::memory::mock::{Call, MockMemoryBackend};
    use std::path::PathBuf;

    fn symbol(path: &str, name: &str, start: u32, end: u32) -> ExplorationFinding {
        ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: start,
                line_end: end,
            },
            snippet: None,
            note: Some(name.to_string()),
        }
    }

    #[tokio::test]
    async fn outline_lists_symbols_with_ranges() {
        let memory = MockMemoryBackend::new().with_file_outline_result(Ok(ExplorationResult {
            findings: vec![
                symbol("a.rs", "foo", 1, 10),
                symbol("a.rs", "Bar::baz", 12, 30),
            ],
            summary: "2 rows".to_string(),
        }));
        let got = skeleton_for(&memory, Path::new("/repo"), Path::new("a.rs"))
            .await
            .unwrap();
        assert_eq!(got, "  foo @ 1-10\n  Bar::baz @ 12-30");
        // The exact path and the symbol cap reach the backend.
        assert_eq!(
            memory.calls(),
            vec![Call::FileOutline {
                repo_root: PathBuf::from("/repo"),
                path: PathBuf::from("a.rs"),
                limit: Some(MAX_OUTLINE_SYMBOLS as u32),
            }]
        );
    }

    #[tokio::test]
    async fn unknown_location_symbol_renders_without_line_range() {
        let memory = MockMemoryBackend::new().with_file_outline_result(Ok(ExplorationResult {
            findings: vec![
                symbol("a.rs", "helper", 12, 30),
                symbol("a.rs", "unresolved", 0, 0),
            ],
            summary: "2 rows".to_string(),
        }));
        let got = skeleton_for(&memory, Path::new("/repo"), Path::new("a.rs"))
            .await
            .unwrap();
        assert_eq!(got, "  helper @ 12-30\n  unresolved (location unknown)");
    }

    #[tokio::test]
    async fn empty_or_nameless_outline_is_none() {
        let memory = MockMemoryBackend::new();
        assert_eq!(
            skeleton_for(&memory, Path::new("/repo"), Path::new("a.rs")).await,
            None
        );
        let nameless = MockMemoryBackend::new().with_file_outline_result(Ok(ExplorationResult {
            findings: vec![ExplorationFinding {
                location: FileLocation {
                    path: PathBuf::from("a.rs"),
                    line_start: 1,
                    line_end: 1,
                },
                snippet: None,
                note: None,
            }],
            summary: "1 row".to_string(),
        }));
        assert_eq!(
            skeleton_for(&nameless, Path::new("/repo"), Path::new("a.rs")).await,
            None
        );
    }
}
