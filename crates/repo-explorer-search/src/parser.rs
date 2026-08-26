//! Two tolerant parsers turning raw tool output into `ExplorationFinding`s.
//!
//! `parse_rtk` reads the fixed `rtk rg -H -n` line grammar (`path:line:content`
//! for matches, `path-line-content` for context); `parse_rg_json` reads
//! `rg --json` JSON-lines. Both are tolerant of malformed individual rows (skip,
//! not fatal) and saturate `u64`->`u32` line numbers (never a bare `as` cast).
//! The one row `parse_rtk` does *not* tolerate is rtk's own truncation-footer
//! line (see `is_truncation_marker`): that line is proof rtk silently dropped
//! real matches for a file, so it is a decode error, not a skippable row.

use repo_explorer_core::domain::{ExplorationFinding, FileLocation, saturate_u32};
use repo_explorer_core::search::SearchError;
use serde_json::Value;
use std::path::PathBuf;

/// Find the leftmost *plausible* `<sep><digits><sep>` run for one separator
/// byte in `bytes[start..end]`: the digit span must not be zero-padded,
/// since real rg/rtk line numbers are always printed unpadded. A
/// zero-padded span (e.g. the `01` in a date-like path segment `-01-`) is
/// therefore not a real line number and is skipped in favor of a later run
/// rather than accepted as a fabricated one. `end` lets a caller who only
/// cares about runs strictly before some other position skip scanning
/// past it — a run's closing separator can never be found beyond `end`,
/// and its digit span (bounded by the first non-digit byte, `end`'s own
/// content included) naturally stops there too, so bounding is
/// behavior-preserving for that use, not just an optimization by coincidence.
fn find_sep_run(bytes: &[u8], sep: u8, start: usize, end: usize) -> Option<(usize, usize)> {
    let mut i = start;
    while i < end {
        if bytes[i] == sep && i > 0 {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let digits = j - (i + 1);
            let zero_padded = digits > 1 && bytes[i + 1] == b'0';
            // require at least one digit, a matching closing separator, and a
            // plausible (unpadded) line number
            if digits > 0 && j < bytes.len() && bytes[j] == sep && !zero_padded {
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
/// take whichever separator kind turns up a run first: a date-like `-NN-` run
/// inside a *path* can sit to the left of the real `:NN:` match separator
/// further right (as in the `2024-01-01.rs` case above), while a timestamp-like
/// `:NN:` run inside *content* can sit to the right of the real `-NN-` context
/// separator (e.g. `src/log.rs-69-12:30:05 boot`). `find_sep_run` already
/// rules out the first case for zero-padded digit spans (`01` is never a real
/// line number), but an unpadded dash-numbered path segment (`issue-42-fix.rs`)
/// still yields a plausible `-` run left of the real `:` one, so position alone
/// can't disambiguate: `src/log.rs-69-12:30:05 boot` and
/// `issue-42-fix.rs:10:the fix` both have a plausible `-` run starting left of
/// a plausible `:` run, yet the former is genuinely `-`-delimited (dash real)
/// and the latter is genuinely `:`-delimited (dash is inside the path). What
/// differs is what sits between the two runs: a real trailing path segment
/// (`fix.rs`, `old.rs`) always carries a file extension made of nothing but
/// letters, so the unbroken token butting right up against the candidate
/// `:` run ends in `.` followed by only ASCII-alphabetic bytes, while a
/// coincidental `:NN:` run inside genuine `-`-delimited content never looks
/// like that -- whether that content is a bare word (`status`), a bare
/// timestamp (`12:30:05`), a word/timestamp mix (`boot 12:30:05`,
/// `data[10:20:2]`), prose containing an unrelated decimal number further
/// back (`Release 1.2.0 shipped at 10:30:00 UTC`, where the `.` in `1.2.0`
/// sits well outside the whitespace-delimited token -- `10` -- immediately
/// before the `:30:` run), or a token that does carry a `.` right up against
/// the run but whose suffix isn't extension-shaped (`self.matrix[10:20:2]`,
/// where the token is `self.matrix[10` and what follows the last `.` is
/// `matrix[10` -- letters, digits, and a bracket, not a plausible
/// extension; or a version-like `v1.2:30:00`, where the suffix is the lone
/// digit `2`). (Checking only the byte right before the `:` run for a digit
/// is not enough: that misses the bare-word case, e.g. `status:42:ok`, where
/// nothing but a `.` distinguishes it from a real trailing path segment; and
/// treating any `.` in the trailing token as proof of a real extension is
/// too loose: that misclassifies the bracket-index and version-token cases
/// above.) So when both runs exist and the `-` run starts first, only trust
/// it if the token immediately preceding the `:` run -- the span back to
/// the previous whitespace, or to the `-` run's end if there is none --
/// does *not* end in a plausible extension (`.` followed by one or more
/// ASCII-alphabetic bytes and nothing else); otherwise the `:` run sits
/// right after a real path segment and is the genuine separator.
///
/// One more wrinkle: an extension-less real file (`issue-42-fix`, or
/// `Makefile`/`README`/`LICENSE`-style names) can produce a plausible `-N-`
/// run left of the genuine `:N:` separator with no extension anywhere to
/// find -- not on the trailing token, not earlier in the path -- so
/// `trailing_token_has_extension` alone can only rule the dash run out, not
/// confirm it. What still tells them apart: genuine dash-delimited content
/// that happens to be a bare word (`status` in
/// `src/log.rs-69-status:42:ok`) sits after a path that already carries its
/// own extension (`.rs`), while an extension-less file's trailing segment
/// (`fix` in `issue-42-fix`) sits after a path prefix (`issue`) with none.
/// So a bare-letters trailing token (`token_is_bare_word`) only flips the
/// verdict to `:` when the path before the dash run also has no `.`
/// anywhere.
fn trailing_token(bytes: &[u8], start: usize, end: usize) -> &[u8] {
    let span = &bytes[start..end];
    let token_start = span
        .iter()
        .rposition(u8::is_ascii_whitespace)
        .map_or(0, |p| p + 1);
    &span[token_start..]
}

fn token_has_extension(token: &[u8]) -> bool {
    match token.iter().rposition(|&b| b == b'.') {
        Some(dot) => {
            let ext = &token[dot + 1..];
            !ext.is_empty() && ext.iter().all(u8::is_ascii_alphabetic)
        }
        None => false,
    }
}

fn trailing_token_has_extension(bytes: &[u8], start: usize, end: usize) -> bool {
    token_has_extension(trailing_token(bytes, start, end))
}

/// True when the trailing token (see `trailing_token`) is a non-empty run of
/// ASCII letters with no `.` at all -- shaped like a bare, extension-less
/// path segment (`fix` in `issue-42-fix`) rather than incidental content
/// (`status`, a bare number like `10`, or a bracket/mixed token like
/// `data[10`). On its own this is not enough to trust the `:` run -- `status`
/// in `src/log.rs-69-status:42:ok` has the same shape but is genuine
/// dash-delimited content -- so `split_grep_line` only acts on it once the
/// path *before* the dash run also carries no extension of its own (see
/// there for why).
fn token_is_bare_word(token: &[u8]) -> bool {
    !token.is_empty() && token.iter().all(u8::is_ascii_alphabetic)
}

/// Starting from the leftmost plausible `-N-` run, walk later `-N-` runs on
/// the line and adopt the first one found whose token immediately before it
/// (back to the last whitespace, or to the prior run examined) ends in a
/// plausible extension, marking a real file extension -- the same
/// disambiguation `split_grep_line` applies once between a dash run and a
/// colon run, generalized to walk past every such run on a pure
/// dash-delimited (no colon) line. A run whose preceding token has no
/// extension (a coincidental in-path `-N-` segment) is skipped over rather
/// than ending the walk -- needed for a path with two or more such segments
/// before its extension, e.g. `component-1-item-2-view.tsx-45-body`, where
/// neither the coincidental `-1-` nor `-2-` run has an extension-bearing
/// token before it, but the real `-45-` separator further right (preceded by
/// `view.tsx`) does; ending the walk at the first non-extension run would
/// wrongly settle on `-1-`.
///
/// Once a run *is* adopted, though, that's the genuine path/line separator
/// and the walk stops right there instead of continuing into the content
/// that follows it: that content can itself contain a coincidental
/// extension-shaped token immediately before a later `-N-` run (e.g. a
/// mentioned filename like `see other.log-42-more`, or `readme.md-20-line`),
/// which would otherwise be misadopted as a "more real" separator and
/// corrupt the path/line/content split. The same reasoning applies to the
/// leftmost run itself: if the token before *it* already looks like a real
/// extension (e.g. `src/log.rs-69-...`, where `log.rs` precedes `-69-`), it
/// is already the genuine separator and the walk never starts.
///
/// The walk is bounded to the line's first whitespace byte: a real path
/// never contains whitespace (these tools always print `path<sep>line<sep>
/// content`, and the path portion is exactly what `find_sep_run`/this walk
/// are trying to delimit), so any `-N-` run at or after that point is
/// definitely inside free-text content, not a candidate separator at all --
/// e.g. `README-1-see item-2-here`: the coincidental `-2-` run sits inside
/// `item-2-here`, which only appears after the space following `see`, so it
/// is never even examined and the genuine `-1-` run is returned untouched.
///
/// If the walk exhausts every later in-bound run without any preceding
/// token ever looking extension-shaped, the *last* run examined is
/// returned rather than `first`: within the whitespace-bounded path region
/// a later run is always at least as plausible as an earlier one (neither
/// has positive extension evidence, so position is the only signal left,
/// and the separator is what immediately precedes the content that
/// follows it) -- e.g. `issue-42-fix-1-line one context`: both `-42-` and
/// `-1-` sit before the line's first whitespace (inside "line one
/// context"), so both are in-bound candidates, and `-1-` (examined last)
/// is the genuine separator, not the coincidental `-42-` inside the file
/// name.
fn resolve_dash_sep_run(bytes: &[u8], first: (usize, usize)) -> (usize, usize) {
    if trailing_token_has_extension(bytes, 0, first.0) {
        return first;
    }
    let ws_bound = bytes
        .iter()
        .position(u8::is_ascii_whitespace)
        .unwrap_or(bytes.len());
    let mut candidate = first;
    let mut cursor = first.1 + 1;
    while let Some(next) = find_sep_run(bytes, b'-', cursor, ws_bound) {
        candidate = next;
        if trailing_token_has_extension(bytes, cursor, next.0) {
            return next;
        }
        cursor = next.1 + 1;
    }
    candidate
}

fn split_grep_line(line: &str) -> Option<(&str, u32, &str, bool)> {
    let bytes = line.as_bytes();
    let colon = find_sep_run(bytes, b':', 0, bytes.len());
    // Only a dash run starting before the colon run can change the outcome
    // (see the match arms below), so skip scanning past it once found.
    let dash_end = colon.map_or(bytes.len(), |c| c.0);
    let dash = find_sep_run(bytes, b'-', 0, dash_end);
    let (sep, i, j) = match (colon, dash) {
        (Some(c), Some(d))
            if d.0 < c.0 && {
                // Computed once and reused by both checks below instead of
                // each re-deriving the same trailing_token(bytes, d.1+1, c.0).
                let token = trailing_token(bytes, d.1 + 1, c.0);
                // Strip a leading `./` before checking for an extension dot: it's
                // the tools' relative-prefix marker, not evidence of a real
                // extension elsewhere in the path (see `split_grep_line_handles_*`
                // extensionless-path tests with a `./` prefix).
                let prefix = bytes[..d.0].strip_prefix(b"./").unwrap_or(&bytes[..d.0]);
                !token_has_extension(token)
                    && !(token_is_bare_word(token) && !prefix.contains(&b'.'))
            } =>
        {
            let d = resolve_dash_sep_run(bytes, d);
            (b'-', d.0, d.1)
        }
        (Some(c), _) => (b':', c.0, c.1),
        (None, Some(d)) => {
            let d = resolve_dash_sep_run(bytes, d);
            (b'-', d.0, d.1)
        }
        (None, None) => return None,
    };
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

/// True for rtk's own truncation-footer line, e.g. `  +35 more in many.txt
/// [see remaining: tail -n +26 ~/.local/share/rtk/tee/....log]`, emitted in
/// place of the remaining match lines once a file's real match count exceeds
/// rtk's undocumented, non-configurable per-file cap. This shape never
/// matches `split_grep_line`'s `path:line:content` / `path-line-content`
/// grammar (no plausible `<sep><digits><sep>` run exists in it), so it must
/// be recognized explicitly -- see `parse_rtk` for why silently falling
/// through to its catch-all `None` arm, as for ordinary unparsable garbage,
/// is wrong here specifically: unlike garbage, this line is proof that real
/// matches were dropped before `parse_rtk` ever saw them.
fn is_truncation_marker(line: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix('+') else {
        return false;
    };
    let digits_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    digits_end > 0 && rest[digits_end..].starts_with(" more in ")
}

/// Parse `rtk rg -H -n` output. Each match line becomes one finding. A context
/// line before any group boundary (`--`) appends to the previous finding's
/// snippet; a context line after a boundary -- including before the very
/// first match, since the stream starts at a boundary -- is buffered and
/// prepended to the next finding instead, since it leads that match rather
/// than trailing the one before the boundary.
///
/// Errors with `SearchError::Decode` -- mirroring `parse_rg_json`'s reaction
/// to a line that violates rg's JSON grammar -- the moment a truncation-
/// footer line (see `is_truncation_marker`) is seen: that line means rtk
/// itself dropped real matches for a file, so the findings collected so far
/// (from this file and any other) are known-incomplete and must not be
/// reported as a complete, silent `Ok` result.
pub(crate) fn parse_rtk(stdout: &str) -> Result<Vec<ExplorationFinding>, SearchError> {
    let mut findings: Vec<ExplorationFinding> = Vec::new();
    let mut pending_before: Vec<String> = Vec::new();
    // Starts true, mirroring parse_rg_json: any context before the first
    // match in the whole stream is leading, not trailing a prior finding.
    let mut at_boundary = true;
    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        if line == "--" {
            at_boundary = true;
            continue;
        }
        if is_truncation_marker(line) {
            return Err(SearchError::Decode {
                backend: "rtk",
                message: format!("rtk truncated its own results and dropped matches: {line}"),
            });
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
    Ok(findings)
}

/// Parse `rg --json` JSON-lines output. Each non-empty line must be valid JSON
/// (a non-JSON line means the whole stream shape is wrong -> `Decode`); a
/// well-formed line missing expected fields (including a `bytes`-only binary
/// row) is skipped. Only `"match"` produces a finding; `"context"` appends to
/// the previous finding's snippet while contiguous with the last line seen in
/// this file (so it trails that match), or is buffered and prepended to the
/// next finding instead once a gap is seen (so it leads the next match) --
/// unlike rtk's explicit `--` separator, `rg --json` has no marker between
/// match groups within one file, so line-number contiguity is the only signal
/// available to tell trailing context from leading context. That leading/
/// trailing mode is tracked as persistent state (`at_boundary`, mirroring
/// rtk's boundary flag) rather than recomputed from each row's immediate
/// predecessor alone: once a gap flips it into leading mode it stays there
/// across any further *contiguous* context rows, so a whole multi-line
/// leading-context run (`context_lines >= 2`) buffers correctly instead of
/// only its first row; a match row resets it back to trailing mode. All other
/// event types are ignored.
fn row_line_text(data: &Value) -> Option<&str> {
    data.get("lines")
        .and_then(|l| l.get("text"))
        .and_then(Value::as_str)
        // rg's `lines.text` carries the line's original terminator; strip a
        // trailing LF and then, for CRLF-terminated source files, the CR left
        // behind so it doesn't end up baked into the returned snippet.
        .map(|s| s.strip_suffix('\n').unwrap_or(s))
        .map(|s| s.strip_suffix('\r').unwrap_or(s))
}

fn row_line_number(data: &Value) -> Option<u32> {
    data.get("line_number")
        .and_then(Value::as_u64)
        .map(saturate_u32)
}

pub(crate) fn parse_rg_json(stdout: &str) -> Result<Vec<ExplorationFinding>, SearchError> {
    let mut findings: Vec<ExplorationFinding> = Vec::new();
    let mut pending_before: Vec<String> = Vec::new();
    // Line number of the last match/context row seen in the current file;
    // `None` right after `begin` (no rows seen yet) or once `saturate_u32`
    // pins a `line_number` at `u32::MAX` (no successor can be contiguous).
    let mut last_line_number: Option<u32> = None;
    // Persistent leading/trailing-context mode, like rtk's boundary flag:
    // starts (and resets on `begin`) true since any context before the first
    // match is leading; a match row sets it false; a context row can only
    // set it back to true on a line-number gap, and it then stays true
    // through subsequent contiguous context rows until the next match.
    let mut at_boundary = true;
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
                // drop any leading-context buffered for a dropped match in the
                // previous file, so it can't leak into this file's first finding
                pending_before.clear();
                last_line_number = None;
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
                // missing line_number: rg is always run with `-H` line numbering
                // on, so a real match row always has one -- drop it like the
                // missing-path case above rather than fabricate line 0
                let Some(line_number) = row_line_number(data) else {
                    continue;
                };
                let snippet = row_line_text(data);
                push_finding(
                    &mut findings,
                    &mut pending_before,
                    path,
                    line_number,
                    snippet,
                );
                last_line_number = Some(line_number);
                at_boundary = false;
            }
            "context" => {
                // Compute the gap and advance `last_line_number` from this row's
                // `line_number` even when `lines` is undecodable (`bytes`, no
                // `text`): otherwise an unrelated undecodable row between two
                // decodable ones would leave `last_line_number` stale, making the
                // next contiguous row look like it has a gap and get misrouted as
                // leading context for a later, unrelated match.
                let line_number = row_line_number(data);
                let gap = match (last_line_number, line_number) {
                    (Some(prev), Some(n)) => prev.checked_add(1) != Some(n),
                    _ => true,
                };
                at_boundary |= gap;
                if let Some(text) = row_line_text(data) {
                    handle_context_line(&mut findings, &mut pending_before, at_boundary, text);
                }
                last_line_number = line_number.or(last_line_number);
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
        let findings = parse_rtk(&fixture("rtk_rg_output.txt")).unwrap();
        // Two match lines (70, 71); the leading context line (69), coming
        // before any "--" boundary, is prepended as findings[0]'s leading
        // context, and the trailing context line (72) appends to 71.
        assert_eq!(findings.len(), 2);
        assert_eq!(
            findings[0].location.path,
            PathBuf::from("crates/repo-explorer-core/src/config.rs")
        );
        assert_eq!(findings[0].location.line_start, 70);
        assert_eq!(findings[0].location.line_end, 70);
        assert_eq!(
            findings[0].snippet.as_deref(),
            Some("    pub timeout_seconds: u64,\n    #[serde(default = \"default_prefer_rtk\")]")
        );
        assert_eq!(findings[1].location.line_start, 71);
        let s1 = findings[1].snippet.as_deref().unwrap();
        assert!(s1.contains("pub prefer_rtk: bool,"));
        assert!(s1.contains('}')); // trailing context appended
    }

    #[test]
    fn parse_rtk_skips_group_separators_and_garbage() {
        let input = "--\nnotavalidline\nsrc/x.rs:5:hello\n";
        let findings = parse_rtk(input).unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].location.line_start, 5);
        assert_eq!(findings[0].snippet.as_deref(), Some("hello"));
    }

    #[test]
    fn parse_rtk_errors_on_truncation_footer() {
        // Real rtk shape: a header, 25 real match rows (elided here), then a
        // footer in place of the remaining rows once a file's match count
        // exceeds rtk's internal per-file cap. The footer must not be
        // silently dropped like ordinary garbage -- it is proof matches were
        // lost, so it must surface as an error instead of a falsely-complete
        // `Ok` result.
        let input = "60 matches in 1 files:\nmany.txt:1:needle\n  +35 more in many.txt [see remaining: tail -n +26 ~/.local/share/rtk/tee/x.log]\n";
        let err = parse_rtk(input).unwrap_err();
        assert_eq!(
            err,
            SearchError::Decode {
                backend: "rtk",
                message: "rtk truncated its own results and dropped matches:   \
                          +35 more in many.txt [see remaining: tail -n +26 ~/.local/share/rtk/tee/x.log]"
                    .to_string(),
            }
        );
    }

    #[test]
    fn is_truncation_marker_matches_rtk_footer_shape() {
        assert!(is_truncation_marker(
            "  +35 more in many.txt [see remaining: tail -n +26 ~/x.log]"
        ));
        assert!(is_truncation_marker("+1 more in a.rs"));
    }

    #[test]
    fn is_truncation_marker_rejects_non_footer_lines() {
        // The summary header shares the digits-then-word shape but not the
        // `+N more in ` prefix, and must stay a tolerated, skippable row.
        assert!(!is_truncation_marker("60 matches in 1 files:"));
        assert!(!is_truncation_marker("src/x.rs:5:hello"));
        assert!(!is_truncation_marker(""));
        assert!(!is_truncation_marker("notavalidline"));
        // A `+`-prefixed content line whose digits aren't followed by the
        // exact `" more in "` marker must not false-positive.
        assert!(!is_truncation_marker("+35 more info"));
        assert!(!is_truncation_marker("+more in x.rs"));
    }

    #[test]
    fn split_grep_line_handles_date_like_file_name() {
        // The `-01-` run inside the file name must not be mistaken for the
        // real `:70:` match separator further right.
        let result = split_grep_line("2024-01-01.rs:70:some content");
        assert_eq!(result, Some(("2024-01-01.rs", 70, "some content", true)));
    }

    #[test]
    fn split_grep_line_handles_context_line_with_colon_in_content() {
        // The `:30:` run inside the content must not be mistaken for the
        // real `-69-` context separator further left.
        let result = split_grep_line("src/log.rs-69-12:30:05 boot");
        assert_eq!(result, Some(("src/log.rs", 69, "12:30:05 boot", false)));
    }

    #[test]
    fn split_grep_line_handles_unpadded_dash_numbered_path_in_context_line() {
        // No colon anywhere on the line (a context row), so the
        // colon-vs-dash disambiguation never fires; the coincidental `-2-`
        // run inside the file name (leftmost) must not be mistaken for the
        // real `-45-` separator further right.
        let result = split_grep_line("component-2-renderer.tsx-45-body");
        assert_eq!(
            result,
            Some(("component-2-renderer.tsx", 45, "body", false))
        );
    }

    #[test]
    fn split_grep_line_handles_extensionless_file_in_context_line() {
        // No colon anywhere (a context row) and the file itself has no
        // extension, so no run's preceding token is ever extension-shaped.
        // Both `-42-` (coincidental, inside the file name) and `-1-` (the
        // genuine separator) sit before the line's first whitespace, so both
        // are in-bound candidates; the genuine one is the one examined last,
        // immediately before the real content (`line one context`) begins.
        let result = split_grep_line("./issue-42-fix-1-line one context");
        assert_eq!(
            result,
            Some(("./issue-42-fix", 1, "line one context", false))
        );
    }

    #[test]
    fn split_grep_line_handles_extensionless_file_with_coincidental_dash_run_in_content() {
        // No colon anywhere and the file (`README`) has no extension. The
        // coincidental `-2-` run sits inside `item-2-here`, which only
        // appears after the space following `see` -- past the walk's
        // whitespace bound -- so it is never even examined, and the
        // genuine, leftmost `-1-` separator is returned untouched.
        let result = split_grep_line("README-1-see item-2-here");
        assert_eq!(result, Some(("README", 1, "see item-2-here", false)));

        // Same shape, but the extensionless path itself also contains a
        // (non-digit) dash (`run-tests`), and both `-3-` and the coincidental
        // `-2-` inside `step-2-verify output` sit before the line's first
        // whitespace (between "verify" and "output") -- so, unlike the
        // README case above, the whitespace bound alone can't rule `-2-`
        // out. With no extension evidence to prefer one over the other
        // either, this specific shape is genuinely ambiguous from the bytes
        // alone (`run-tests` at line 3 vs. `run-tests-3-step` at line 2 are
        // both equally plausible extensionless file names) -- this pins
        // down the walk's actual, deterministic choice (the run examined
        // last within bounds) rather than asserting one reading is somehow
        // provably correct.
        let result = split_grep_line("run-tests-3-step-2-verify output");
        assert_eq!(
            result,
            Some(("run-tests-3-step", 2, "verify output", false))
        );
    }

    #[test]
    fn split_grep_line_handles_unpadded_dash_numbered_path() {
        // The `-42-` run inside the file name (unpadded, so not caught by the
        // zero-padding check) must not be mistaken for the real `:10:` match
        // separator further right.
        let result = split_grep_line("issue-42-fix.rs:10:the fix");
        assert_eq!(result, Some(("issue-42-fix.rs", 10, "the fix", true)));
    }

    #[test]
    fn split_grep_line_handles_unpadded_dash_numbered_extensionless_path() {
        // Same shape as the extension-bearing case above, but the real file
        // (`issue-42-fix`) has no extension at all -- neither on the
        // trailing token (`fix`) nor earlier in the path (`issue`) -- so the
        // `-42-` run must still lose to the real `:10:` match separator.
        let result = split_grep_line("issue-42-fix:10:the fix");
        assert_eq!(result, Some(("issue-42-fix", 10, "the fix", true)));
    }

    #[test]
    fn split_grep_line_handles_unpadded_dash_numbered_extensionless_path_with_relative_prefix() {
        // Same shape as above, but with the `./` relative prefix these tools
        // always print (target is always `.`, see backend.rs). The leading
        // `.` in `./` must not be mistaken for a real extension dot earlier
        // in the path, or the `-42-` run wrongly wins over the real `:10:`
        // match separator.
        let result = split_grep_line("./issue-42-fix:10:the fix");
        assert_eq!(result, Some(("./issue-42-fix", 10, "the fix", true)));
    }

    #[test]
    fn split_grep_line_handles_dash_numbered_path_with_more_segments() {
        let result = split_grep_line("src/file-42-old.rs:70:content");
        assert_eq!(result, Some(("src/file-42-old.rs", 70, "content", true)));
    }

    #[test]
    fn split_grep_line_handles_context_line_with_prefixed_colon_run_in_content() {
        // A coincidental `:NN:` run preceded by non-digit text (not sitting
        // right after the `-N-` separator) must still not be mistaken for the
        // real match separator: the byte right before the found `:` is a
        // digit either way, so the genuine `-5-` context separator wins.
        let result = split_grep_line("file-5-abc 10:20:rest");
        assert_eq!(result, Some(("file", 5, "abc 10:20:rest", false)));

        let result = split_grep_line("src/log.rs-69-boot 12:30:05");
        assert_eq!(result, Some(("src/log.rs", 69, "boot 12:30:05", false)));

        let result = split_grep_line("src/log.rs-69-data[10:20:2]");
        assert_eq!(result, Some(("src/log.rs", 69, "data[10:20:2]", false)));
    }

    #[test]
    fn split_grep_line_handles_context_line_with_bare_word_colon_run_in_content() {
        // A coincidental `:NN:` run preceded by a bare word (no digit chain,
        // no space) must still not be mistaken for the real match separator:
        // unlike a real trailing path segment (`fix.rs`), the word `status`
        // carries no file extension, so the genuine `-69-` context separator
        // wins.
        let result = split_grep_line("src/log.rs-69-status:42:ok");
        assert_eq!(result, Some(("src/log.rs", 69, "status:42:ok", false)));
    }

    #[test]
    fn split_grep_line_handles_context_line_with_decimal_before_coincidental_colon_run() {
        // A `.` earlier in the content (inside the decimal `1.2.0`, well
        // outside the whitespace-delimited token -- `10` -- immediately
        // before the coincidental `:30:` run) must not be mistaken for a
        // real trailing path segment's extension: the genuine `-2-` context
        // separator wins, and the whole timestamp-bearing sentence is
        // preserved as content rather than truncated at the coincidental
        // colon run.
        let result = split_grep_line("./CHANGELOG.md-2-Release 1.2.0 shipped at 10:30:00 UTC");
        assert_eq!(
            result,
            Some((
                "./CHANGELOG.md",
                2,
                "Release 1.2.0 shipped at 10:30:00 UTC",
                false
            ))
        );
    }

    #[test]
    fn split_grep_line_handles_dash_numbered_path_with_coincidental_colon_in_content() {
        // Both a coincidental in-path `-2-` run (left of the real `-45-`
        // context separator) and a coincidental `:30:` run in the content
        // exist on this line; the mixed colon+dash branch must resolve the
        // dash run the same way the no-colon branch does rather than
        // stopping at the first in-path dash run.
        let result = split_grep_line("component-2-renderer.tsx-45-time 12:30:00");
        assert_eq!(
            result,
            Some(("component-2-renderer.tsx", 45, "time 12:30:00", false))
        );
    }

    #[test]
    fn split_grep_line_handles_bracket_index_dot_before_coincidental_colon_run() {
        // The trailing token before the coincidental `:20:` run is
        // `self.matrix[10` -- it has a `.`, but what follows it (`matrix[10`)
        // is not extension-shaped (letters, digits, and a bracket), so it
        // must not be mistaken for a real trailing path segment: the genuine
        // `-15-` context separator wins.
        let result = split_grep_line("code.py-15-    self.matrix[10:20:2]");
        assert_eq!(
            result,
            Some(("code.py", 15, "    self.matrix[10:20:2]", false))
        );
    }

    #[test]
    fn split_grep_line_handles_dash_numbered_path_with_two_intermediate_segments() {
        // Two coincidental in-path `-N-` runs (`-1-`, `-2-`) sit left of the
        // real `-45-` separator, neither directly preceded by a dotted
        // extension -- the walk must skip past both rather than stopping at
        // the first and settling on `-1-`.
        let result = split_grep_line("component-1-item-2-view.tsx-45-body");
        assert_eq!(
            result,
            Some(("component-1-item-2-view.tsx", 45, "body", false))
        );
    }

    #[test]
    fn split_grep_line_handles_extension_shaped_token_in_content_after_dash_separator() {
        // The genuine `-10-` separator is directly preceded by an
        // extension-bearing path (`foo.rs`), so it must be adopted
        // immediately rather than walked past into the content, where the
        // coincidental `-20-` run (preceded by the extension-shaped
        // `readme.md`) must not be mistaken for a "more real" separator.
        let result = split_grep_line("foo.rs-10-see readme.md-20-line");
        assert_eq!(result, Some(("foo.rs", 10, "see readme.md-20-line", false)));

        let result = split_grep_line("src/parser.rs-10-see utils.rs-20-also");
        assert_eq!(
            result,
            Some(("src/parser.rs", 10, "see utils.rs-20-also", false))
        );

        let result = split_grep_line("src/log.rs-69-see other.log-42-more text");
        assert_eq!(
            result,
            Some(("src/log.rs", 69, "see other.log-42-more text", false))
        );
    }

    #[test]
    fn parse_rg_json_routes_context_between_two_match_groups_in_one_file() {
        // Two matches (3, 9) far enough apart that their context windows
        // (radius 1) don't overlap: context(4) trails match A, context(8)
        // leads match B -- there is no `begin`/boundary marker between them,
        // so this must be inferred from line-number contiguity alone.
        let input = concat!(
            "{\"type\":\"begin\",\"data\":{\"path\":{\"text\":\"f.rs\"}}}\n",
            "{\"type\":\"context\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"line2\\n\"},\"line_number\":2}}\n",
            "{\"type\":\"match\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"MATCH_A\\n\"},\"line_number\":3}}\n",
            "{\"type\":\"context\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"line4\\n\"},\"line_number\":4}}\n",
            "{\"type\":\"context\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"line8\\n\"},\"line_number\":8}}\n",
            "{\"type\":\"match\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"MATCH_B\\n\"},\"line_number\":9}}\n",
            "{\"type\":\"context\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"line10\\n\"},\"line_number\":10}}\n",
            "{\"type\":\"end\",\"data\":{\"path\":{\"text\":\"f.rs\"}}}\n",
        );
        let findings = parse_rg_json(input).unwrap();
        assert_eq!(findings.len(), 2);
        assert_eq!(
            findings[0].snippet.as_deref(),
            Some("line2\nMATCH_A\nline4")
        );
        assert_eq!(
            findings[1].snippet.as_deref(),
            Some("line8\nMATCH_B\nline10")
        );
    }

    #[test]
    fn parse_rg_json_buffers_multi_line_leading_context() {
        // context_lines >= 2: a match at line 7 preceded by two leading
        // context rows (5, 6) with no prior match in the file. Both rows must
        // land in the *upcoming* finding's snippet, not be dropped or glued
        // onto an unrelated already-emitted finding.
        let input = concat!(
            "{\"type\":\"begin\",\"data\":{\"path\":{\"text\":\"f.rs\"}}}\n",
            "{\"type\":\"context\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"line5\\n\"},\"line_number\":5}}\n",
            "{\"type\":\"context\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"line6\\n\"},\"line_number\":6}}\n",
            "{\"type\":\"match\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"MATCH\\n\"},\"line_number\":7}}\n",
            "{\"type\":\"end\",\"data\":{\"path\":{\"text\":\"f.rs\"}}}\n",
        );
        let findings = parse_rg_json(input).unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].location.line_start, 7);
        assert_eq!(findings[0].snippet.as_deref(), Some("line5\nline6\nMATCH"));
    }

    #[test]
    fn row_line_text_strips_crlf_line_ending() {
        let data: Value =
            serde_json::from_str(r#"{"lines":{"text":"    let x = 1;\r\n"},"line_number":1}"#)
                .unwrap();
        assert_eq!(row_line_text(&data), Some("    let x = 1;"));
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
    fn parse_rg_json_undecodable_context_row_keeps_line_bookkeeping() {
        // An undecodable (`bytes`-only) context row between two decodable ones
        // must still update `last_line_number`/`at_boundary` from its own
        // `line_number` even though it contributes no text itself: otherwise
        // the next contiguous context row would see a spurious gap and get
        // misrouted as leading context for a later, unrelated match instead of
        // trailing context for the match it actually follows.
        let input = concat!(
            "{\"type\":\"begin\",\"data\":{\"path\":{\"text\":\"f.rs\"}}}\n",
            "{\"type\":\"match\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"MATCH_A\\n\"},\"line_number\":3}}\n",
            "{\"type\":\"context\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"bytes\":\"AAAA\"},\"line_number\":4}}\n",
            "{\"type\":\"context\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"line5\\n\"},\"line_number\":5}}\n",
            "{\"type\":\"match\",\"data\":{\"path\":{\"text\":\"f.rs\"},",
            "\"lines\":{\"text\":\"MATCH_B\\n\"},\"line_number\":20}}\n",
            "{\"type\":\"end\",\"data\":{\"path\":{\"text\":\"f.rs\"}}}\n",
        );
        let findings = parse_rg_json(input).unwrap();
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].snippet.as_deref(), Some("MATCH_A\nline5"));
        assert_eq!(findings[1].snippet.as_deref(), Some("MATCH_B"));
    }

    #[test]
    fn parse_rg_json_drops_match_row_missing_line_number() {
        // A match row without `line_number` must be dropped like the missing-path
        // case, not fabricate a bogus `line 0` finding.
        let input = concat!(
            "{\"type\":\"match\",\"data\":{\"path\":{\"text\":\"a.rs\"},",
            "\"lines\":{\"text\":\"x\\n\"}}}\n"
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
