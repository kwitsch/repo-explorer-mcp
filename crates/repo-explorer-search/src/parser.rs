//! Two tolerant parsers turning raw tool output into `ExplorationFinding`s.
//!
//! `parse_rtk` reads the fixed `rtk rg -H -n` line grammar (`path:line:content`
//! for matches, `path-line-content` for context); `parse_rg_json` reads
//! `rg --json` JSON-lines. Both are tolerant of malformed individual rows (skip,
//! not fatal) and saturate `u64`->`u32` line numbers (never a bare `as` cast).

use repo_explorer_core::domain::{ExplorationFinding, FileLocation, saturate_u32};
use repo_explorer_core::search::SearchError;
use serde_json::Value;
use std::path::PathBuf;

/// Find the leftmost `<sep><digits><sep>` run for one separator byte.
fn find_sep_run(bytes: &[u8], sep: u8) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == sep && i > 0 {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            // require at least one digit AND a matching closing separator
            if j > i + 1 && j < bytes.len() && bytes[j] == sep {
                return Some((i, j));
            }
        }
        i += 1;
    }
    None
}

/// Split one grep-style line into `(path, line, content, is_match)`.
///
/// The path itself may contain both `:` and `-` (e.g. `repo-explorer-core`, or
/// a date-like file name such as `2024-01-01.rs`), and the content after the
/// separator may too (e.g. a timestamp like `12:30:05`), so we can't just
/// scan for the leftmost `<sep><digits><sep>` run of either separator kind:
/// a date-like `-NN-` run inside a *path* can sit to the left of the real
/// `:NN:` match separator further right, and would be picked instead (as it
/// is here, since `2024-01-01.rs` is the doc-cited example). A literal `:`
/// essentially never occurs inside a real file path (Windows forbids it
/// outright), so unlike `-` it can't be shadowed by an earlier false run.
/// We therefore look for the leftmost `:<digits>:` run first -- finding one
/// identifies a match line unambiguously -- and only fall back to the
/// leftmost `-<digits>-` run (a context line) when no `:` run exists at all.
fn split_grep_line(line: &str) -> Option<(&str, u32, &str, bool)> {
    let bytes = line.as_bytes();
    let (sep, i, j) = find_sep_run(bytes, b':')
        .map(|(i, j)| (b':', i, j))
        .or_else(|| find_sep_run(bytes, b'-').map(|(i, j)| (b'-', i, j)))?;
    let path = &line[..i];
    let num: u64 = line[i + 1..j].parse().ok()?;
    let content = &line[j + 1..];
    Some((path, saturate_u32(num), content, sep == b':'))
}

/// Build one finding for a match line, prepending any buffered before-context
/// collected since the last group boundary (and clearing the buffer). Shared
/// by `parse_rtk` and `parse_rg_json`.
fn push_finding(
    findings: &mut Vec<ExplorationFinding>,
    pending_before: &mut Vec<String>,
    path: &str,
    line: u32,
    content: Option<&str>,
) {
    let snippet = if pending_before.is_empty() {
        content.map(str::to_string)
    } else {
        let mut combined = pending_before.join("\n");
        if let Some(c) = content {
            combined.push('\n');
            combined.push_str(c);
        }
        pending_before.clear();
        Some(combined)
    };
    findings.push(ExplorationFinding {
        location: FileLocation {
            path: PathBuf::from(path),
            line_start: line,
            line_end: line,
        },
        snippet,
        note: None,
    });
}

/// Append a context line's text to the previous finding's snippet (joined by
/// `\n`), or start a fresh snippet if there is none yet; on a best-effort
/// basis, dropped if there is no previous finding. Only for context that
/// trails the finding it follows -- context that leads the next, not yet
/// created, finding is buffered by the caller instead (see `push_finding`).
fn append_context(findings: &mut [ExplorationFinding], text: &str) {
    if let Some(last) = findings.last_mut() {
        match last.snippet.as_mut() {
            Some(s) => {
                s.push('\n');
                s.push_str(text);
            }
            None => last.snippet = Some(text.to_string()),
        }
    }
}

/// Route one context line's text to the right place depending on where it
/// falls relative to a group boundary: text seen since a boundary leads the
/// next, not yet created, finding and is buffered; otherwise it trails the
/// previous finding. Shared by `parse_rtk` and `parse_rg_json`.
fn handle_context_line(
    findings: &mut [ExplorationFinding],
    pending_before: &mut Vec<String>,
    at_boundary: bool,
    text: &str,
) {
    if at_boundary {
        pending_before.push(text.to_string());
    } else {
        append_context(findings, text);
    }
}

/// Parse `rtk rg -H -n` output. Each match line becomes one finding. A context
/// line before any group boundary (`--`) appends to the previous finding's
/// snippet on a best-effort basis (dropped if there is no previous finding);
/// a context line after a boundary is buffered and prepended to the next
/// finding instead, since it leads that match rather than trailing the one
/// before the boundary.
pub(crate) fn parse_rtk(stdout: &str) -> Vec<ExplorationFinding> {
    let mut findings: Vec<ExplorationFinding> = Vec::new();
    let mut pending_before: Vec<String> = Vec::new();
    let mut at_boundary = false;
    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        if line == "--" {
            at_boundary = true;
            continue;
        }
        match split_grep_line(line) {
            Some((path, num, content, true)) => {
                push_finding(&mut findings, &mut pending_before, path, num, Some(content));
                at_boundary = false;
            }
            Some((_, _, content, false)) => {
                handle_context_line(&mut findings, &mut pending_before, at_boundary, content);
            }
            None => {}
        }
    }
    findings
}

/// Parse `rg --json` JSON-lines output. Each non-empty line must be valid JSON
/// (a non-JSON line means the whole stream shape is wrong -> `Decode`); a
/// well-formed line missing expected fields (including a `bytes`-only binary
/// row) is skipped. Only `"match"` produces a finding; `"context"` appends to
/// the previous finding's snippet unless a `"begin"` (new file) has been seen
/// since, in which case it is buffered and prepended to that file's next
/// finding instead; all other event types are ignored.
fn row_line_text(data: &Value) -> Option<&str> {
    data.get("lines")
        .and_then(|l| l.get("text"))
        .and_then(Value::as_str)
        .map(|s| s.strip_suffix('\n').unwrap_or(s))
}

pub(crate) fn parse_rg_json(stdout: &str) -> Result<Vec<ExplorationFinding>, SearchError> {
    let mut findings: Vec<ExplorationFinding> = Vec::new();
    let mut pending_before: Vec<String> = Vec::new();
    let mut at_boundary = false;
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|e| SearchError::Decode {
            backend: "ripgrep",
            message: format!("invalid JSON line: {e}"),
        })?;
        let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
        let Some(data) = value.get("data") else {
            continue;
        };
        match kind {
            "begin" => {
                at_boundary = true;
            }
            "match" => {
                // bytes-only / binary row: drop gracefully
                let Some(path) = data
                    .get("path")
                    .and_then(|p| p.get("text"))
                    .and_then(Value::as_str)
                else {
                    continue;
                };
                let line_number = data
                    .get("line_number")
                    .and_then(Value::as_u64)
                    .map(saturate_u32)
                    .unwrap_or(0);
                let snippet = row_line_text(data);
                push_finding(
                    &mut findings,
                    &mut pending_before,
                    path,
                    line_number,
                    snippet,
                );
                at_boundary = false;
            }
            "context" => {
                if let Some(text) = row_line_text(data) {
                    handle_context_line(&mut findings, &mut pending_before, at_boundary, text);
                }
            }
            _ => {} // end / summary / unknown: ignored, not a decode error
        }
    }
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name);
        std::fs::read_to_string(path).expect("read fixture")
    }

    #[test]
    fn parse_rtk_extracts_matches_and_appends_context() {
        let findings = parse_rtk(&fixture("rtk_rg_output.txt"));
        // Two match lines (70, 71); the leading context line (69) is dropped
        // (no prior match), the trailing context line (72) appends to 71.
        assert_eq!(findings.len(), 2);
        assert_eq!(
            findings[0].location.path,
            PathBuf::from("crates/repo-explorer-core/src/config.rs")
        );
        assert_eq!(findings[0].location.line_start, 70);
        assert_eq!(findings[0].location.line_end, 70);
        assert_eq!(
            findings[0].snippet.as_deref(),
            Some("    #[serde(default = \"default_prefer_rtk\")]")
        );
        assert_eq!(findings[1].location.line_start, 71);
        let s1 = findings[1].snippet.as_deref().unwrap();
        assert!(s1.contains("pub prefer_rtk: bool,"));
        assert!(s1.contains('}')); // trailing context appended
    }

    #[test]
    fn parse_rtk_skips_group_separators_and_garbage() {
        let input = "--\nnotavalidline\nsrc/x.rs:5:hello\n";
        let findings = parse_rtk(input);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].location.line_start, 5);
        assert_eq!(findings[0].snippet.as_deref(), Some("hello"));
    }

    #[test]
    fn split_grep_line_handles_date_like_file_name() {
        // The `-01-` run inside the file name must not be mistaken for the
        // real `:70:` match separator further right.
        let result = split_grep_line("2024-01-01.rs:70:some content");
        assert_eq!(result, Some(("2024-01-01.rs", 70, "some content", true)));
    }

    #[test]
    fn parse_rg_json_extracts_matches_ignores_non_match_events() {
        let findings = parse_rg_json(&fixture("rg_output.jsonl")).unwrap();
        // begin/end/summary are ignored; only the two match rows produce findings.
        assert_eq!(findings.len(), 2);
        assert_eq!(
            findings[0].location.path,
            PathBuf::from("crates/repo-explorer-core/src/config.rs")
        );
        assert_eq!(findings[0].location.line_start, 70);
        assert_eq!(findings[0].location.line_end, 70);
        assert_eq!(
            findings[0].snippet.as_deref(),
            Some("    #[serde(default = \"default_prefer_rtk\")]")
        );
        assert_eq!(findings[1].location.line_start, 71);
    }

    #[test]
    fn parse_rg_json_drops_binary_bytes_only_row() {
        // A match row whose path/lines carry `bytes` (base64) instead of `text`
        // must be dropped gracefully, not panic on `.as_str()`.
        let input = concat!(
            "{\"type\":\"match\",\"data\":{\"path\":{\"bytes\":\"AAAA\"},",
            "\"lines\":{\"bytes\":\"BBBB\"},\"line_number\":3}}\n"
        );
        let findings = parse_rg_json(input).unwrap();
        assert!(findings.is_empty());
    }

    #[test]
    fn parse_rg_json_non_json_line_is_decode_error() {
        let err = parse_rg_json("not json at all\n").unwrap_err();
        assert!(matches!(
            err,
            repo_explorer_core::search::SearchError::Decode {
                backend: "ripgrep",
                ..
            }
        ));
    }

    #[test]
    fn parse_rg_json_saturates_huge_line_number() {
        let big = u64::MAX;
        let input = format!(
            "{{\"type\":\"match\",\"data\":{{\"path\":{{\"text\":\"a.rs\"}},\
             \"lines\":{{\"text\":\"x\\n\"}},\"line_number\":{big}}}}}\n"
        );
        let findings = parse_rg_json(&input).unwrap();
        assert_eq!(findings[0].location.line_start, u32::MAX);
    }
}
