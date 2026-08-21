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

// `#[allow(dead_code)]` on these items: this task lands the parsers on their
// own (fixture-tested in isolation), before `backend` (a later task in this
// stage) wires them into `CliSearchBackend`. Non-test builds have no caller
// yet, which `-D warnings` would otherwise reject as dead code.

/// Split one grep-style line into `(path, line, content, is_match)`.
///
/// The path itself may contain both `:` and `-` (e.g. `repo-explorer-core`, or
/// a date-like file name such as `2024-01-01.rs`), so we cannot scan for
/// whichever separator (`:` or `-`) shows up first in byte order — a `-`
/// buried inside the path could be mistaken for the context separator before
/// the real `:line:` match separator further right is ever reached. Instead
/// we scan for each separator kind independently, over the *whole* line, and
/// prefer `:` (match lines) over `-` (context lines): a match line's path
/// never legitimately contains a `:NN:` run, so this can't misfire the other
/// way.
#[allow(dead_code)]
fn split_grep_line(line: &str) -> Option<(&str, u32, &str, bool)> {
    scan_for_separator(line, b':').or_else(|| scan_for_separator(line, b'-'))
}

/// Scan `line` for the first `<sep><digits><sep>` run using exactly `sep` as
/// both delimiters: that middle run is the line number, everything before is
/// the path, everything after is the content.
fn scan_for_separator(line: &str, sep: u8) -> Option<(&str, u32, &str, bool)> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == sep && i > 0 {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            // require at least one digit AND a matching closing separator
            if j > i + 1 && j < bytes.len() && bytes[j] == sep {
                let path = &line[..i];
                let num: u64 = line[i + 1..j].parse().ok()?;
                let content = &line[j + 1..];
                return Some((path, saturate_u32(num), content, sep == b':'));
            }
        }
        i += 1;
    }
    None
}

/// Append a context line's text to the previous finding's snippet (joined by
/// `\n`), or start a fresh snippet if there is none yet; on a best-effort
/// basis, dropped if there is no previous finding. Shared by `parse_rtk` and
/// `parse_rg_json`, whose context-line handling is otherwise identical.
fn append_context(findings: &mut [ExplorationFinding], text: &str) {
    if let Some(last) = findings.last_mut() {
        let combined = match last.snippet.take() {
            Some(s) => format!("{s}\n{text}"),
            None => text.to_string(),
        };
        last.snippet = Some(combined);
    }
}

/// Parse `rtk rg -H -n` output. Each match line becomes one finding; a context
/// line appends its content to the previous finding's snippet on a best-effort
/// basis (dropped if there is no previous finding).
#[allow(dead_code)]
pub(crate) fn parse_rtk(stdout: &str) -> Vec<ExplorationFinding> {
    let mut findings: Vec<ExplorationFinding> = Vec::new();
    for line in stdout.lines() {
        if line.is_empty() || line == "--" {
            continue;
        }
        match split_grep_line(line) {
            Some((path, num, content, true)) => {
                findings.push(ExplorationFinding {
                    location: FileLocation {
                        path: PathBuf::from(path),
                        line_start: num,
                        line_end: num,
                    },
                    snippet: Some(content.to_string()),
                    note: None,
                });
            }
            Some((_, _, content, false)) => {
                append_context(&mut findings, content);
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
/// the previous finding's snippet; all other event types are ignored.
#[allow(dead_code)]
pub(crate) fn parse_rg_json(stdout: &str) -> Result<Vec<ExplorationFinding>, SearchError> {
    let mut findings: Vec<ExplorationFinding> = Vec::new();
    for line in stdout.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|e| SearchError::Decode {
            backend: "ripgrep",
            message: format!("invalid JSON line: {e}"),
        })?;
        let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
        let data = match value.get("data") {
            Some(d) => d,
            None => continue,
        };
        match kind {
            "match" => {
                let path = match data
                    .get("path")
                    .and_then(|p| p.get("text"))
                    .and_then(Value::as_str)
                {
                    Some(p) => p,
                    None => continue, // bytes-only / binary row: drop gracefully
                };
                let line_number = data
                    .get("line_number")
                    .and_then(Value::as_u64)
                    .map(saturate_u32)
                    .unwrap_or(0);
                let snippet = data
                    .get("lines")
                    .and_then(|l| l.get("text"))
                    .and_then(Value::as_str)
                    .map(|s| s.strip_suffix('\n').unwrap_or(s).to_string());
                findings.push(ExplorationFinding {
                    location: FileLocation {
                        path: PathBuf::from(path),
                        line_start: line_number,
                        line_end: line_number,
                    },
                    snippet,
                    note: None,
                });
            }
            "context" => {
                if let Some(text) = data
                    .get("lines")
                    .and_then(|l| l.get("text"))
                    .and_then(Value::as_str)
                {
                    let text = text.strip_suffix('\n').unwrap_or(text);
                    append_context(&mut findings, text);
                }
            }
            _ => {} // begin / end / summary / unknown: ignored, not a decode error
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
