//! A tolerant parser turning raw `rtk rg -H -n` output into `ExplorationFinding`s.
//!
//! `parse_rtk` reads the fixed `rtk rg -H -n` line grammar (`path:line:content`
//! for matches, `path-line-content` for context). It is tolerant of malformed
//! individual rows (skip, not fatal) and saturates `u64`->`u32` line numbers
//! (never a bare `as` cast).

use repo_explorer_core::domain::{ExplorationFinding, FileLocation, saturate_u32};
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
/// So a bare-letters trailing token (`trailing_token_is_bare_word`) only
/// flips the verdict to `:` when the path before the dash run also has no
/// `.` anywhere.
fn trailing_token(bytes: &[u8], start: usize, end: usize) -> &[u8] {
    let span = &bytes[start..end];
    let token_start = span
        .iter()
        .rposition(u8::is_ascii_whitespace)
        .map_or(0, |p| p + 1);
    &span[token_start..]
}

fn trailing_token_has_extension(bytes: &[u8], start: usize, end: usize) -> bool {
    let token = trailing_token(bytes, start, end);
    match token.iter().rposition(|&b| b == b'.') {
        Some(dot) => {
            let ext = &token[dot + 1..];
            !ext.is_empty() && ext.iter().all(u8::is_ascii_alphabetic)
        }
        None => false,
    }
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
fn trailing_token_is_bare_word(bytes: &[u8], start: usize, end: usize) -> bool {
    let token = trailing_token(bytes, start, end);
    !token.is_empty() && token.iter().all(u8::is_ascii_alphabetic)
}

/// Starting from the leftmost plausible `-N-` run, walk every later `-N-`
/// run on the line and adopt one as the new candidate whenever the token
/// immediately before it (back to the last whitespace, or to the prior run
/// examined) ends in a plausible extension, marking a real file extension --
/// the same disambiguation `split_grep_line` applies once between a dash run
/// and a colon run, generalized to walk past every such run on a pure
/// dash-delimited (no colon) line. The scan position advances past every run
/// examined regardless of whether it was adopted, so a run whose preceding
/// token has no extension (a coincidental in-path `-N-` segment) is skipped
/// over rather than ending the walk -- needed for a path with two or more
/// such segments before its extension, e.g.
/// `component-1-item-2-view.tsx-45-body`, where neither the coincidental
/// `-1-` nor `-2-` run has an extension-bearing token before it, but the
/// real `-45-` separator further right (preceded by `view.tsx`) does; ending
/// the walk at the first non-extension run would wrongly settle on `-1-`.
fn resolve_dash_sep_run(bytes: &[u8], first: (usize, usize)) -> (usize, usize) {
    let mut candidate = first;
    let mut cursor = first.1 + 1;
    while let Some(next) = find_sep_run(bytes, b'-', cursor, bytes.len()) {
        if trailing_token_has_extension(bytes, cursor, next.0) {
            candidate = next;
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
            if d.0 < c.0
                && !trailing_token_has_extension(bytes, d.1 + 1, c.0)
                && !(trailing_token_is_bare_word(bytes, d.1 + 1, c.0)
                    && !bytes[..d.0].contains(&b'.')) =>
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
/// collected since the last group boundary (and clearing the buffer). Used by
/// `parse_rtk`.
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
/// previous finding. Used by `parse_rtk`.
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
/// snippet; a context line after a boundary -- including before the very
/// first match, since the stream starts at a boundary -- is buffered and
/// prepended to the next finding instead, since it leads that match rather
/// than trailing the one before the boundary.
pub(crate) fn parse_rtk(stdout: &str) -> Vec<ExplorationFinding> {
    let mut findings: Vec<ExplorationFinding> = Vec::new();
    let mut pending_before: Vec<String> = Vec::new();
    // Starts true: any context before the first match in the whole stream is
    // leading, not trailing a prior finding.
    let mut at_boundary = true;
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
            Some(
                "    pub timeout_seconds: u64,\n    #[serde(default = \"default_search_timeout_seconds\")]"
            )
        );
        assert_eq!(findings[1].location.line_start, 71);
        let s1 = findings[1].snippet.as_deref().unwrap();
        assert!(s1.contains("pub max_depth: usize,"));
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
}
