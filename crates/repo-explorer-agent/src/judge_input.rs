//! The plain-text judge state (Stage 10): the byte-exact input every judge
//! backend sees for ONE retrieval candidate. Format version =
//! `repo_explorer_core::judge::JUDGE_STATE_VERSION`. Bump that constant on any
//! change here — the datagen generator and the 10b runtime both call this, so
//! training and serving stay byte-identical by construction.

use futures_util::future::join_all;
use repo_explorer_core::domain::Candidate;
use repo_explorer_core::memory::MemoryBackend;
use repo_explorer_core::retrieval::{is_unknown_location, kind_label, normalize_location};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Lines of context kept before the candidate's own start line.
pub const BODY_CONTEXT_BEFORE: u32 = 3;
/// Lines of context kept after the candidate's own end line.
pub const BODY_CONTEXT_AFTER: u32 = 5;
/// Cap on rendered body lines.
pub const BODY_MAX_LINES: usize = 60;
/// Cap on characters (not bytes) per rendered body line.
pub const LINE_MAX_CHARS: usize = 200;
/// Cap on rendered outline lines.
pub const OUTLINE_MAX_LINES: usize = 30;

/// Everything the judge sees for one candidate.
pub struct JudgeCandidateView<'a> {
    pub query: &'a str,
    pub candidate: &'a Candidate,
    /// `skeleton_for` output for the candidate's file, if the graph knows it.
    pub outline: Option<&'a str>,
    /// Raw file lines of the body window (see `render_judge_states`).
    pub body: Option<&'a str>,
}

/// Pure, deterministic. Format version = `core::judge::JUDGE_STATE_VERSION`.
/// Builds the lines and joins them with `\n`, no trailing newline.
pub fn render_judge_state(view: &JudgeCandidateView<'_>) -> String {
    let mut lines: Vec<String> = Vec::new();

    // 1. Query line: newlines flattened to spaces, then trimmed.
    let flattened: String = view
        .query
        .chars()
        .map(|c| if c == '\r' || c == '\n' { ' ' } else { c })
        .collect();
    lines.push(format!("query: {}", flattened.trim()));

    // 2. Candidate line.
    let loc = normalize_location(view.candidate.location.clone());
    let path = loc.path.to_string_lossy().replace('\\', "/");
    let sym = match &view.candidate.symbol {
        Some(symbol) => format!(", symbol `{symbol}`"),
        None => String::new(),
    };
    lines.push(format!(
        "candidate: {path}:{}-{} ({}{sym})",
        loc.line_start,
        loc.line_end,
        kind_label(view.candidate.kind),
    ));

    // 3. Code section: prefer body, else snippet, else "(unavailable)".
    let source = view
        .body
        .filter(|s| !s.is_empty())
        .or_else(|| view.candidate.snippet.as_deref().filter(|s| !s.is_empty()));
    match source {
        Some(text) => {
            lines.push("code:".to_string());
            for raw in text.split('\n').take(BODY_MAX_LINES) {
                let stripped = raw.strip_suffix('\r').unwrap_or(raw);
                let capped: String = stripped.chars().take(LINE_MAX_CHARS).collect();
                lines.push(format!("  {capped}"));
            }
        }
        None => lines.push("code: (unavailable)".to_string()),
    }

    // 4. Outline section (last: Laya truncates the tail).
    if let Some(outline) = view.outline.filter(|s| !s.is_empty()) {
        lines.push("outline:".to_string());
        for line in outline.split('\n').take(OUTLINE_MAX_LINES) {
            lines.push(line.to_string());
        }
    }

    lines.join("\n")
}

/// One entry per candidate, same order. `None` for an unknown-location
/// candidate (`is_unknown_location`): those are never judged or labelled.
pub async fn render_judge_states<M: MemoryBackend>(
    memory: &M,
    repo_root: &Path,
    query: &str,
    candidates: &[Candidate],
) -> Vec<Option<String>> {
    // Canonicalize once; on failure every body is None.
    let canonical_root = crate::dispatch::canonical_repo_root(repo_root).await.ok();

    // Distinct known-location paths, first-appearance order.
    let mut distinct_paths: Vec<PathBuf> = Vec::new();
    for c in candidates {
        if is_unknown_location(&c.location) {
            continue;
        }
        let path = normalize_location(c.location.clone()).path;
        if !distinct_paths.contains(&path) {
            distinct_paths.push(path);
        }
    }

    // One outline per distinct path, concurrently.
    let outlines = join_all(
        distinct_paths
            .iter()
            .map(|p| crate::skeleton::skeleton_for(memory, repo_root, p)),
    )
    .await;
    let outline_by_path: HashMap<PathBuf, Option<String>> =
        distinct_paths.into_iter().zip(outlines).collect();

    // One body per known-location candidate, concurrently.
    let bodies = join_all(candidates.iter().map(|c| {
        let canonical_root = canonical_root.clone();
        async move {
            if is_unknown_location(&c.location) {
                return None;
            }
            let canonical_root = canonical_root?;
            let loc = normalize_location(c.location.clone());
            let start = loc.line_start.saturating_sub(BODY_CONTEXT_BEFORE).max(1);
            let end = loc.line_end.saturating_add(BODY_CONTEXT_AFTER);
            let path_str = loc.path.to_string_lossy();
            crate::dispatch::read_file_canonical(
                repo_root,
                &canonical_root,
                &path_str,
                Some(start),
                Some(end),
            )
            .await
            .ok()
        }
    }))
    .await;

    candidates
        .iter()
        .zip(bodies)
        .map(|(c, body)| {
            if is_unknown_location(&c.location) {
                return None;
            }
            let path = normalize_location(c.location.clone()).path;
            let outline = outline_by_path.get(&path).and_then(|o| o.as_deref());
            let view = JudgeCandidateView {
                query,
                candidate: c,
                outline,
                body: body.as_deref(),
            };
            Some(render_judge_state(&view))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_repo_with;
    use repo_explorer_core::domain::{
        Candidate, CandidateKind, ExplorationFinding, ExplorationResult, FileLocation,
    };
    use repo_explorer_core::memory::mock::MockMemoryBackend;
    use std::path::PathBuf;

    fn cand(
        path: &str,
        start: u32,
        end: u32,
        kind: CandidateKind,
        symbol: Option<&str>,
        snippet: Option<&str>,
    ) -> Candidate {
        Candidate {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: start,
                line_end: end,
            },
            symbol: symbol.map(str::to_string),
            kind,
            score: 0,
            snippet: snippet.map(str::to_string),
        }
    }

    #[test]
    fn known_location_with_symbol_body_and_outline() {
        let c = cand(
            "src/lib.rs",
            10,
            12,
            CandidateKind::SymbolExact,
            Some("foo::bar"),
            None,
        );
        let view = JudgeCandidateView {
            query: "where is bar",
            candidate: &c,
            outline: Some("  foo @ 1-3\n  bar @ 10-12"),
            body: Some("fn bar() {\n    baz()\n}"),
        };
        assert_eq!(
            render_judge_state(&view),
            "query: where is bar\n\
             candidate: src/lib.rs:10-12 (exact symbol match, symbol `foo::bar`)\n\
             code:\n\
             \x20\x20fn bar() {\n\
             \x20\x20\x20\x20\x20\x20baz()\n\
             \x20\x20}\n\
             outline:\n\
             \x20\x20foo @ 1-3\n\
             \x20\x20bar @ 10-12"
        );
    }

    #[test]
    fn no_symbol_uses_snippet_when_body_absent() {
        let c = cand("a.rs", 5, 5, CandidateKind::ContentHit, None, Some("x"));
        let view = JudgeCandidateView {
            query: "q",
            candidate: &c,
            outline: None,
            body: None,
        };
        assert_eq!(
            render_judge_state(&view),
            "query: q\ncandidate: a.rs:5-5 (text match)\ncode:\n  x"
        );
    }

    #[test]
    fn body_none_falls_back_to_snippet_with_symbol() {
        let c = cand(
            "a.rs",
            1,
            1,
            CandidateKind::SemanticHit,
            Some("m::f"),
            Some("snippet line"),
        );
        let view = JudgeCandidateView {
            query: "q",
            candidate: &c,
            outline: None,
            body: None,
        };
        assert_eq!(
            render_judge_state(&view),
            "query: q\ncandidate: a.rs:1-1 (semantic search match, symbol `m::f`)\ncode:\n  snippet line"
        );
    }

    #[test]
    fn neither_body_nor_snippet_is_unavailable() {
        let c = cand("a.rs", 1, 1, CandidateKind::FileNameHit, None, None);
        let view = JudgeCandidateView {
            query: "q",
            candidate: &c,
            outline: None,
            body: None,
        };
        assert_eq!(
            render_judge_state(&view),
            "query: q\ncandidate: a.rs:1-1 (file name match)\ncode: (unavailable)"
        );
    }

    #[test]
    fn body_over_60_lines_is_truncated() {
        let body = (1..=70)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let c = cand("a.rs", 1, 1, CandidateKind::ContentHit, None, None);
        let view = JudgeCandidateView {
            query: "q",
            candidate: &c,
            outline: None,
            body: Some(&body),
        };
        let out = render_judge_state(&view);
        let code: Vec<&str> = out.lines().filter(|l| l.starts_with("  line")).collect();
        assert_eq!(code.len(), 60);
        assert_eq!(*code.last().unwrap(), "  line60");
        assert!(!out.contains("line61"));
    }

    #[test]
    fn long_multibyte_line_is_cut_at_200_chars() {
        let long = "é".repeat(250);
        let c = cand("a.rs", 1, 1, CandidateKind::ContentHit, None, None);
        let view = JudgeCandidateView {
            query: "q",
            candidate: &c,
            outline: None,
            body: Some(&long),
        };
        let out = render_judge_state(&view);
        let code_line = out.lines().find(|l| l.starts_with("  é")).unwrap();
        assert_eq!(code_line.chars().count(), 202); // 2-space prefix + 200 chars
        assert_eq!(code_line.chars().filter(|&c| c == 'é').count(), 200);
    }

    #[test]
    fn query_newlines_become_spaces_and_trim() {
        let c = cand("a.rs", 1, 1, CandidateKind::ContentHit, None, Some("x"));
        let view = JudgeCandidateView {
            query: "  where is\nthe\r\ncode  ",
            candidate: &c,
            outline: None,
            body: None,
        };
        let out = render_judge_state(&view);
        assert_eq!(out.lines().next().unwrap(), "query: where is the  code");
    }

    #[test]
    fn windows_backslash_path_is_normalized() {
        let c = cand(
            "src\\win\\file.rs",
            3,
            4,
            CandidateKind::SymbolFuzzy,
            None,
            Some("y"),
        );
        let view = JudgeCandidateView {
            query: "q",
            candidate: &c,
            outline: None,
            body: None,
        };
        let out = render_judge_state(&view);
        assert!(
            out.contains("candidate: src/win/file.rs:3-4 (symbol name match)"),
            "{out}"
        );
    }

    #[test]
    fn inverted_range_is_normalized() {
        let c = cand("a.rs", 20, 10, CandidateKind::ContentHit, None, Some("z"));
        let view = JudgeCandidateView {
            query: "q",
            candidate: &c,
            outline: None,
            body: None,
        };
        let out = render_judge_state(&view);
        assert!(out.contains("candidate: a.rs:10-20 (text match)"), "{out}");
    }

    #[tokio::test]
    async fn render_judge_states_reads_bodies_and_outlines() {
        let file = (1..=20)
            .map(|i| format!("l{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let repo = temp_repo_with("judge_input", "render_states", &[("a.rs", &file)]);

        let memory = MockMemoryBackend::new().with_file_outline_result(Ok(ExplorationResult {
            findings: vec![ExplorationFinding {
                location: FileLocation {
                    path: PathBuf::from("a.rs"),
                    line_start: 6,
                    line_end: 8,
                },
                snippet: None,
                note: Some("thing".to_string()),
            }],
            summary: String::new(),
        }));

        let cands = vec![
            cand(
                "a.rs",
                2,
                3,
                CandidateKind::SymbolExact,
                Some("thing"),
                None,
            ),
            cand("a.rs", 10, 10, CandidateKind::ContentHit, None, None),
            cand(
                "a.rs",
                0,
                0,
                CandidateKind::SymbolFuzzy,
                Some("unknown"),
                None,
            ),
        ];

        let states = render_judge_states(&memory, &repo, "find thing", &cands).await;
        std::fs::remove_dir_all(&repo).ok();

        assert_eq!(states.len(), 3);
        assert!(
            states[2].is_none(),
            "unknown-location candidate must be None"
        );

        let s0 = states[0].as_ref().unwrap();
        let code0: Vec<&str> = s0.lines().filter(|l| l.starts_with("  l")).collect();
        assert_eq!(
            code0,
            vec![
                "  l1", "  l2", "  l3", "  l4", "  l5", "  l6", "  l7", "  l8"
            ]
        );
        assert!(s0.contains("outline:\n  thing @ 6-8"), "{s0}");

        let s1 = states[1].as_ref().unwrap();
        let code1: Vec<&str> = s1.lines().filter(|l| l.starts_with("  l")).collect();
        assert_eq!(
            code1,
            vec![
                "  l7", "  l8", "  l9", "  l10", "  l11", "  l12", "  l13", "  l14", "  l15"
            ]
        );
        assert!(s1.contains("outline:\n  thing @ 6-8"), "{s1}");
    }
}
