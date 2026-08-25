//! Compressed rendering of findings for LLM consumption: path normalization
//! (no `./` spellings), location-level dedupe, snippet/character caps, and
//! stable per-file ordering. Every byte rendered here is a prompt token — the
//! caps are the token-diet half of the retrieval-pipeline design.

use repo_explorer_core::domain::{ExplorationFinding, ExplorationResult};
use repo_explorer_core::retrieval::normalize_rel_path;
use serde::Serialize;
use std::collections::HashSet;

/// Output-size caps applied when rendering tool results and prompts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RenderCaps {
    pub snippet_max_chars: usize,
    pub read_file_max_lines: usize,
}

impl Default for RenderCaps {
    fn default() -> Self {
        Self {
            snippet_max_chars: 400,
            read_file_max_lines: 200,
        }
    }
}

const TRUNCATION_MARKER: &str = "…[truncated]";

/// Cap `s` to at most `max_chars` characters (char-boundary safe), appending a
/// marker when anything was cut. `max_chars == 0` disables the cap.
pub(crate) fn cap_snippet(s: &str, max_chars: usize) -> String {
    if max_chars == 0 || s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str(TRUNCATION_MARKER);
    out
}

/// Cap file contents to `max_lines` lines, appending an explicit marker so the
/// model knows to request a narrower range instead of assuming EOF.
pub(crate) fn cap_file_lines(contents: String, max_lines: usize) -> String {
    if max_lines == 0 || contents.lines().count() <= max_lines {
        return contents;
    }
    let mut out: Vec<&str> = contents.lines().take(max_lines).collect();
    let marker = format!("…[truncated after {max_lines} lines; request a narrower line range]");
    out.push(&marker);
    out.join("\n")
}

/// Normalize, dedupe (by normalized location; first seen wins), and cap
/// snippets — preserving the input order. Use where order carries meaning
/// (rank order of the retrieval pre-stage, the model's own finish order).
pub(crate) fn tidy_findings(
    findings: Vec<ExplorationFinding>,
    caps: &RenderCaps,
) -> Vec<ExplorationFinding> {
    let mut seen = HashSet::new();
    let mut out: Vec<ExplorationFinding> = Vec::with_capacity(findings.len());
    for mut f in findings {
        f.location.path = normalize_rel_path(&f.location.path);
        if seen.contains(&f.location) {
            continue;
        }
        seen.insert(f.location.clone());
        f.snippet = f
            .snippet
            .map(|s| cap_snippet(&s, caps.snippet_max_chars))
            .filter(|s| !s.is_empty());
        f.note = f.note.filter(|n| !n.trim().is_empty());
        out.push(f);
    }
    out
}

/// [`tidy_findings`], then ordered by (path, line) so per-file findings render
/// adjacently — for tool results fed back to the model, where grouping beats
/// arrival order.
pub(crate) fn compress_findings(
    findings: Vec<ExplorationFinding>,
    caps: &RenderCaps,
) -> Vec<ExplorationFinding> {
    let mut out = tidy_findings(findings, caps);
    out.sort_by(|a, b| {
        a.location
            .path
            .cmp(&b.location.path)
            .then_with(|| a.location.line_start.cmp(&b.location.line_start))
    });
    out
}

#[derive(Serialize)]
struct FindingDto<'a> {
    path: String,
    line_start: u32,
    line_end: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    snippet: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<&'a str>,
}

#[derive(Serialize)]
struct ResultDto<'a> {
    findings: Vec<FindingDto<'a>>,
    summary: &'a str,
}

fn finding_dto(f: &ExplorationFinding) -> FindingDto<'_> {
    FindingDto {
        path: f.location.path.display().to_string(),
        line_start: f.location.line_start,
        line_end: f.location.line_end,
        snippet: f.snippet.as_deref(),
        note: f.note.as_deref(),
    }
}

/// Serialize `value`, falling back to `fallback` (a literal empty-JSON shape,
/// e.g. `"{}"`/`"[]"`) on the practically-unreachable serialize failure.
fn serialize_or_empty<T: Serialize>(value: &T, fallback: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| fallback.to_string())
}

/// Render a memory-tool `ExplorationResult` compressed; returns the rendered
/// text plus the compressed findings for accumulation.
pub(crate) fn render_result(
    res: ExplorationResult,
    caps: &RenderCaps,
) -> (String, Vec<ExplorationFinding>) {
    let findings = compress_findings(res.findings, caps);
    let content = serialize_or_empty(
        &ResultDto {
            findings: findings.iter().map(finding_dto).collect(),
            summary: &res.summary,
        },
        "{}",
    );
    (content, findings)
}

/// Render bare grep/find findings compressed.
pub(crate) fn render_findings(
    findings: Vec<ExplorationFinding>,
    caps: &RenderCaps,
) -> (String, Vec<ExplorationFinding>) {
    let findings = compress_findings(findings, caps);
    let dtos: Vec<FindingDto> = findings.iter().map(finding_dto).collect();
    let content = serialize_or_empty(&dtos, "[]");
    (content, findings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::FileLocation;
    use std::path::PathBuf;

    fn finding(path: &str, line: u32, snippet: Option<&str>) -> ExplorationFinding {
        ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: line,
                line_end: line,
            },
            snippet: snippet.map(str::to_string),
            note: None,
        }
    }

    #[test]
    fn cap_snippet_boundaries() {
        assert_eq!(cap_snippet("short", 10), "short");
        assert_eq!(cap_snippet("short", 0), "short", "0 disables the cap");
        let capped = cap_snippet("abcdef", 3);
        assert!(capped.starts_with("abc") && capped.ends_with(TRUNCATION_MARKER));
        // Multi-byte chars must not split.
        let capped = cap_snippet("äöüß", 2);
        assert!(capped.starts_with("äö"));
    }

    #[test]
    fn cap_file_lines_appends_marker() {
        let contents = "a\nb\nc\nd".to_string();
        assert_eq!(cap_file_lines(contents.clone(), 4), "a\nb\nc\nd");
        let capped = cap_file_lines(contents, 2);
        assert!(capped.starts_with("a\nb\n…[truncated after 2 lines"));
    }

    #[test]
    fn compress_dedupes_dot_slash_spellings_and_sorts() {
        let caps = RenderCaps::default();
        let out = compress_findings(
            vec![
                finding("./b.rs", 5, Some("x")),
                finding("a.rs", 9, None),
                finding("b.rs", 5, Some("first wins? no — first seen wins")),
                finding("a.rs", 2, None),
            ],
            &caps,
        );
        // ./b.rs:5 and b.rs:5 collapse (first seen wins); output sorted.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].location.path, PathBuf::from("a.rs"));
        assert_eq!(out[0].location.line_start, 2);
        assert_eq!(out[2].location.path, PathBuf::from("b.rs"));
        assert_eq!(out[2].snippet.as_deref(), Some("x"));
    }

    #[test]
    fn render_result_caps_snippets() {
        let caps = RenderCaps {
            snippet_max_chars: 4,
            read_file_max_lines: 10,
        };
        let (content, findings) = render_result(
            ExplorationResult {
                findings: vec![finding("a.rs", 1, Some("0123456789"))],
                summary: "s".to_string(),
            },
            &caps,
        );
        assert!(findings[0].snippet.as_deref().unwrap().starts_with("0123"));
        assert!(content.contains("truncated"));
    }
}
