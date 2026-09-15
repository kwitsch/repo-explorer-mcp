//! The deterministic repository brief: parse `get_architecture`'s raw
//! multi-section plain text and render it into a token-budgeted markdown
//! block that Stage 5 injects as a second system message.
//!
//! Pure — no backend, no async, no serde. The one CBM round trip that feeds
//! it lives in `agent.rs`.

use std::fmt::Write as _;

/// How many entry points the brief lists; the tail is orientation noise.
const MAX_ENTRYPOINTS: usize = 10;

/// The prompt-hardening line. It ships *with* the brief rather than in
/// `FALLBACK_SYSTEM_PROMPT` so the static cache prefix stays byte-stable and
/// the model is never told about a brief it did not get.
const HARDENING: &str = "A deterministic repository brief follows. Do NOT call get_architecture to re-derive it; call it only if the brief is insufficient for this query.";

/// One `name: N  (cols: a b c)` section plus its indented rows.
struct Section<'a> {
    name: &'a str,
    cols: Vec<&'a str>,
    rows: Vec<Vec<&'a str>>,
}

impl<'a> Section<'a> {
    fn col(&self, name: &str) -> Option<usize> {
        self.cols.iter().position(|c| *c == name)
    }

    fn cell(&self, row: &[&'a str], name: &str) -> Option<&'a str> {
        self.col(name).and_then(|i| row.get(i).copied())
    }
}

/// Walk the plain-text tables `codebase-memory-mcp` answers with — the same
/// shape `repo_explorer_memory`'s finding mapper walks, minus its "drop every
/// section without a `file` column" rule, which is what discards the
/// `node_labels:`/`edge_types:`/`packages:` data this brief is made of.
/// Column names are alias-stripped (`n.file` -> `file`) like that parser.
fn parse_sections(text: &str) -> Vec<Section<'_>> {
    let mut sections: Vec<Section> = Vec::new();
    // Any non-indented line closes the open section, so rows under a
    // column-less header (e.g. `total_grep_matches: 44`) are dropped instead
    // of being adopted by the section above — same as `text_table_findings`.
    let mut open = false;
    for line in text.lines() {
        if !line.starts_with(' ') {
            open = false;
            let Some((name, rest)) = line.split_once(':') else {
                continue;
            };
            let Some((_, tail)) = rest.split_once("(cols:") else {
                continue;
            };
            sections.push(Section {
                name: name.trim(),
                cols: tail
                    .trim_end_matches(')')
                    .split_whitespace()
                    .map(|c| c.rsplit('.').next().unwrap_or(c))
                    .collect(),
                rows: Vec::new(),
            });
            open = true;
            continue;
        }
        let Some(section) = sections.last_mut().filter(|_| open) else {
            continue;
        };
        let cells: Vec<&str> = line.split_whitespace().collect();
        if cells.len() == section.cols.len() {
            section.rows.push(cells);
        }
    }
    sections
}

/// ponytail: chars/4 token estimate — no tokenizer dependency, ±10% is all
/// the budget knob needs. Upgrade to a real tokenizer only if the budget ever
/// has to be tight enough for that error to matter.
pub(crate) fn estimate_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

/// Append `line` plus a newline when the result still fits the budget.
fn push_if_fits(out: &mut String, line: &str, max_tokens: u32) -> bool {
    if estimate_tokens(out) + estimate_tokens(line) + 1 > max_tokens as usize {
        return false;
    }
    out.push_str(line);
    out.push('\n');
    true
}

/// Render `get_architecture`'s raw text into the brief, or `None` when the
/// payload carried nothing usable (Stage 5 then runs exactly as before, with
/// a single system message). A `max_tokens` too small for a single row yields
/// `None` the same way — the caller's own `max_tokens == 0` opt-out short-
/// circuits before the round trip that feeds this.
pub(crate) fn render_brief(text: &str, max_tokens: u32) -> Option<String> {
    let sections = parse_sections(text);
    let find = |name: &str| sections.iter().find(|s| s.name == name);
    let mut out = String::with_capacity(text.len().min(max_tokens as usize * 4));
    out.push_str(HARDENING);
    out.push_str("\n\n");
    let mut content = false;

    // Graph vocabulary: one folded line each, cheapest orientation there is.
    for (section, label) in [
        ("node_labels", "Graph nodes"),
        ("edge_types", "Graph edges"),
    ] {
        let Some(s) = find(section) else { continue };
        let names: Vec<&str> = s
            .rows
            .iter()
            .filter_map(|r| {
                s.cell(r, "label")
                    .or_else(|| s.cell(r, "type"))
                    .or(r.first().copied())
            })
            .collect();
        if !names.is_empty()
            && push_if_fits(
                &mut out,
                &format!("{label}: {}", names.join(", ")),
                max_tokens,
            )
        {
            content = true;
        }
    }

    if let Some(s) = find("entry_points") {
        // Rolled back below if the header is all that fit — a heading with no
        // rows under it is pure budget waste.
        let mark = out.len();
        let mut rows_written = 0;
        for row in s.rows.iter().take(MAX_ENTRYPOINTS) {
            let Some(file) = s.cell(row, "file").or_else(|| s.cell(row, "path")) else {
                continue;
            };
            if rows_written == 0 && !push_if_fits(&mut out, "\nEntry points:", max_tokens) {
                break;
            }
            let line = match s.cell(row, "qn") {
                Some(qn) => format!("- {file} ({qn})"),
                None => format!("- {file}"),
            };
            if !push_if_fits(&mut out, &line, max_tokens) {
                break;
            }
            rows_written += 1;
            content = true;
        }
        if rows_written == 0 {
            out.truncate(mark);
        }
    }

    // Modules, biggest first — that ordering *is* the budget rule: when the
    // table runs out of room the smallest packages are the ones left out.
    if let Some(s) = find("packages") {
        let num = |row: &[&str], col: &str| {
            s.cell(row, col)
                .and_then(|c| c.parse::<u64>().ok())
                .unwrap_or(0)
        };
        let mut rows: Vec<&Vec<&str>> = s
            .rows
            .iter()
            .filter(|r| s.cell(r, "name").is_some())
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(num(r, "nodes")));
        let mark = out.len();
        let mut rows_written = 0;
        for row in rows {
            let name = s.cell(row, "name").expect("filtered above");
            if rows_written == 0
                && !push_if_fits(
                    &mut out,
                    "\n| module | symbols | fan_in |\n| --- | ---: | ---: |",
                    max_tokens,
                )
            {
                break;
            }
            let mut line = String::new();
            let _ = write!(
                line,
                "| {name} | {} | {} |",
                num(row, "nodes"),
                num(row, "fan_in")
            );
            if !push_if_fits(&mut out, &line, max_tokens) {
                break;
            }
            rows_written += 1;
            content = true;
        }
        if rows_written == 0 {
            out.truncate(mark);
        }
    }

    content.then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The live `get_architecture` payload shape, copied from
    /// `repo_explorer_memory`'s own decoder test.
    const PAYLOAD: &str = "\
node_labels: 2  (cols: label count)\n  Function 120\n  Method 136\n\
packages: 2  (cols: name nodes fan_in fan_out)\n  repo-explorer-core 256 0 0\n  repo-explorer-agent 512 3 7\n\
entry_points: 2  (cols: qn file)\n  \
repo.crates.repo-explorer-mcp.src.main.main crates/repo-explorer-mcp/src/main.rs\n  \
repo.crates.repo-explorer-core.src.lib.run crates/repo-explorer-core/src/lib.rs\n";

    #[test]
    fn parses_get_architecture_sections() {
        let sections = parse_sections(PAYLOAD);
        let names: Vec<&str> = sections.iter().map(|s| s.name).collect();
        assert_eq!(names, ["node_labels", "packages", "entry_points"]);
        let packages = &sections[1];
        assert_eq!(packages.cols, ["name", "nodes", "fan_in", "fan_out"]);
        assert_eq!(packages.rows.len(), 2);
        assert_eq!(packages.cell(&packages.rows[0], "nodes"), Some("256"));
        assert_eq!(sections[2].rows.len(), 2);
    }

    #[test]
    fn render_starts_with_the_do_not_call_line() {
        let brief = render_brief(PAYLOAD, 3000).expect("payload is usable");
        assert!(brief.starts_with(HARDENING));
        assert!(brief.contains("Do NOT call get_architecture"));
        assert!(brief.contains("repo-explorer-core"));
        assert!(brief.contains("crates/repo-explorer-mcp/src/main.rs"));
        assert!(brief.contains("Graph nodes: Function, Method"));
    }

    #[test]
    fn render_keeps_biggest_modules_within_budget() {
        let mut text = String::from("packages: 20  (cols: name nodes fan_in fan_out)\n");
        for i in 1..=20u32 {
            let _ = writeln!(text, "  package-number-{i:02} {} 0 0", i * 100);
        }
        let brief = render_brief(&text, 120).expect("packages section is usable");
        assert!(estimate_tokens(&brief) <= 120, "budget exceeded:\n{brief}");
        assert!(
            brief.contains("package-number-20"),
            "biggest dropped:\n{brief}"
        );
        assert!(
            !brief.contains("package-number-01"),
            "smallest survived a tight budget:\n{brief}"
        );
    }

    #[test]
    fn render_never_leaves_a_dangling_heading() {
        // Every budget where the brief is cut mid-section must still produce a
        // heading only when a row survived under it.
        for budget in 40..=120u32 {
            let Some(brief) = render_brief(PAYLOAD, budget) else {
                continue;
            };
            assert!(
                estimate_tokens(&brief) <= budget as usize,
                "{budget}: {brief}"
            );
            if brief.contains("Entry points:") {
                assert!(brief.contains("\n- "), "{budget}: {brief}");
            }
            if brief.contains("| module |") {
                assert!(brief.contains("\n| repo-explorer"), "{budget}: {brief}");
            }
        }
    }

    #[test]
    fn render_returns_none_for_unusable_payload() {
        assert!(render_brief("", 3000).is_none());
        assert!(render_brief("total_grep_matches: 44\n  stray row\n", 3000).is_none());
        assert!(render_brief(PAYLOAD, 0).is_none(), "no row fits a 0 budget");
    }

    #[test]
    fn rows_under_a_column_less_header_are_dropped() {
        let sections = parse_sections(
            "packages: 1  (cols: name nodes fan_in fan_out)\n  repo-explorer-core 256 0 0\n\
             hotspots: 1\n  src/foo.rs 10 2 3\n",
        );
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].rows, [["repo-explorer-core", "256", "0", "0"]]);
    }

    #[test]
    fn estimate_tokens_is_chars_over_four() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2, "must round up");
    }
}
