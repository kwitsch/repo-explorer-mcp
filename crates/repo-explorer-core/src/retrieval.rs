//! Pure decision logic for the deterministic retrieval pre-stage: query →
//! search patterns, raw candidates → merged/ranked top-k, ranked list →
//! confidence score. No I/O, no LLM — the orchestration (backend fanout) lives
//! in `repo-explorer-agent`; this module mirrors the pure-logic pattern of
//! `repo-explorer-memory`'s `freshness.rs`.

use crate::domain::{Candidate, CandidateKind, ExplorationFinding, FileLocation};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Search inputs derived deterministically from a query text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct QueryPatterns {
    /// Identifier-like tokens (symbol lookup + coverage scoring), deduped in
    /// first-appearance order.
    pub identifiers: Vec<String>,
    /// Quoted literals, to be searched verbatim.
    pub literals: Vec<String>,
    /// Path-like tokens (contain a separator or a file extension).
    pub path_tokens: Vec<String>,
    /// Regex-escaped patterns for the grep fanout (literals first, then
    /// identifiers), capped at [`MAX_GREP_PATTERNS`].
    pub grep_patterns: Vec<String>,
}

/// Upper bound on grep fanout width per query.
pub const MAX_GREP_PATTERNS: usize = 6;

/// Score scale: candidate scores live in `0..=MAX_SCORE`.
pub const MAX_SCORE: u32 = 1000;

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "where", "what", "when", "which", "does", "how",
    "are", "was", "were", "will", "would", "should", "could", "into", "from", "used", "uses",
    "using", "have", "has", "been", "there", "find", "show", "code", "file", "files", "function",
    "method", "class", "der", "die", "das", "und", "wird", "wie", "was", "welche", "wo",
];

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// True for tokens worth treating as identifiers: snake_case, camelCase,
/// digit-bearing, or any sufficiently long non-stopword word.
fn is_identifier_like(token: &str) -> bool {
    if token.len() < 3 {
        return false;
    }
    let has_underscore = token.contains('_');
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    let has_lower = token.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = token.chars().any(|c| c.is_ascii_uppercase());
    if has_underscore || has_digit || (has_lower && has_upper) {
        return true;
    }
    token.len() >= 4 && !STOPWORDS.contains(&token.to_ascii_lowercase().as_str())
}

/// Escape a literal so grep backends treat it verbatim. Hand-rolled: core has
/// no regex dependency, and the target dialect (rust/regex via ripgrep) treats
/// exactly these ASCII metacharacters specially.
pub fn escape_regex(literal: &str) -> String {
    let mut out = String::with_capacity(literal.len());
    for c in literal.chars() {
        if matches!(
            c,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn push_unique(list: &mut Vec<String>, value: String) {
    if !value.is_empty() && !list.contains(&value) {
        list.push(value);
    }
}

/// Derive deterministic search inputs from a free-text query.
pub fn derive_patterns(query: &str) -> QueryPatterns {
    let mut patterns = QueryPatterns::default();

    // Quoted literals: content of "…", '…', and `…` pairs.
    let mut rest = query;
    let mut unquoted = String::new();
    while let Some(open) = rest.find(['"', '\'', '`']) {
        let quote = rest.as_bytes()[open] as char;
        unquoted.push_str(&rest[..open]);
        unquoted.push(' ');
        let after = &rest[open + 1..];
        match after.find(quote) {
            Some(close) => {
                push_unique(&mut patterns.literals, after[..close].trim().to_string());
                rest = &after[close + 1..];
            }
            None => {
                // Unbalanced quote: treat the rest as plain text.
                rest = after;
            }
        }
    }
    unquoted.push_str(rest);

    // Word-level tokens from the unquoted remainder. `::`/`.`/`/`-joined
    // compounds contribute both the compound's segments and, for paths, the
    // compound itself.
    for raw in unquoted.split_whitespace() {
        let trimmed = raw.trim_matches(|c: char| !is_word_char(c) && !matches!(c, '/' | '.'));
        if trimmed.is_empty() {
            continue;
        }
        let looks_like_path = trimmed.contains('/')
            || Path::new(trimmed)
                .extension()
                .is_some_and(|e| e.to_str().is_some_and(|e| !e.is_empty()));
        if looks_like_path {
            push_unique(&mut patterns.path_tokens, trimmed.to_string());
        }
        for segment in trimmed.split(|c: char| !is_word_char(c)) {
            if is_identifier_like(segment) {
                push_unique(&mut patterns.identifiers, segment.to_string());
            }
        }
    }

    for literal in &patterns.literals {
        if patterns.grep_patterns.len() >= MAX_GREP_PATTERNS {
            break;
        }
        push_unique(&mut patterns.grep_patterns, escape_regex(literal));
    }
    for identifier in &patterns.identifiers {
        if patterns.grep_patterns.len() >= MAX_GREP_PATTERNS {
            break;
        }
        push_unique(&mut patterns.grep_patterns, escape_regex(identifier));
    }

    patterns
}

/// Strip a leading `./` so the same file never appears under two spellings
/// (grep emits `./x` without a scope and `x` with one).
pub fn normalize_rel_path(path: &Path) -> PathBuf {
    path.strip_prefix(".")
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| path.to_path_buf())
}

fn kind_base_score(kind: CandidateKind) -> u32 {
    match kind {
        CandidateKind::SymbolExact => 700,
        CandidateKind::SymbolFuzzy => 400,
        CandidateKind::FileNameHit => 300,
        CandidateKind::SemanticHit => 260,
        CandidateKind::ContentHit => 150,
    }
}

fn kind_rank(kind: CandidateKind) -> u32 {
    // Higher = stronger; used to pick the surviving kind when merging.
    kind_base_score(kind)
}

/// Number of distinct query identifiers/literals appearing (case-insensitive)
/// in the candidate's symbol, path, or snippet.
fn coverage(candidate: &Candidate, patterns: &QueryPatterns) -> u32 {
    let haystack = format!(
        "{} {} {}",
        candidate.symbol.as_deref().unwrap_or(""),
        candidate.location.path.to_string_lossy(),
        candidate.snippet.as_deref().unwrap_or("")
    )
    .to_ascii_lowercase();
    patterns
        .identifiers
        .iter()
        .chain(patterns.literals.iter())
        .filter(|token| !token.is_empty() && haystack.contains(&token.to_ascii_lowercase()))
        .count() as u32
}

/// Normalize a location: strip `./`, swap an inverted line range, and widen a
/// zero `line_end` to `line_start`.
fn normalize_location(location: FileLocation) -> FileLocation {
    let path = normalize_rel_path(&location.path);
    let (mut start, mut end) = (location.line_start, location.line_end);
    if end < start {
        std::mem::swap(&mut start, &mut end);
    }
    FileLocation {
        path,
        line_start: start,
        line_end: end,
    }
}

fn overlaps(a: &FileLocation, b: &FileLocation) -> bool {
    a.line_start <= b.line_end && b.line_start <= a.line_end
}

/// Merge `b` into `a`: widen the range, keep the stronger kind's identity.
fn merge_into(a: &mut Candidate, b: Candidate) {
    a.location.line_start = a.location.line_start.min(b.location.line_start);
    a.location.line_end = a.location.line_end.max(b.location.line_end);
    if kind_rank(b.kind) > kind_rank(a.kind) {
        a.kind = b.kind;
        a.symbol = b.symbol.or(a.symbol.take());
        if b.snippet.is_some() {
            a.snippet = b.snippet;
        }
    } else {
        if a.symbol.is_none() {
            a.symbol = b.symbol;
        }
        if a.snippet.is_none() {
            a.snippet = b.snippet;
        }
    }
}

/// Merge overlapping candidates per file, score, rank, and truncate to
/// `top_k`. Deterministic: stable ordering by (score desc, path, line_start).
pub fn merge_and_rank(raw: Vec<Candidate>, patterns: &QueryPatterns, top_k: u32) -> Vec<Candidate> {
    let mut per_file: HashMap<PathBuf, Vec<Candidate>> = HashMap::new();
    for mut candidate in raw {
        candidate.location = normalize_location(candidate.location);
        per_file
            .entry(candidate.location.path.clone())
            .or_default()
            .push(candidate);
    }

    let mut merged: Vec<Candidate> = Vec::new();
    for (_, mut candidates) in per_file {
        candidates.sort_by_key(|c| (c.location.line_start, c.location.line_end));
        let mut file_merged: Vec<Candidate> = Vec::new();
        for candidate in candidates {
            match file_merged.last_mut() {
                Some(last) if overlaps(&last.location, &candidate.location) => {
                    merge_into(last, candidate);
                }
                _ => file_merged.push(candidate),
            }
        }
        // Per-file hit density: files with several distinct hits are likelier
        // to be the answer; boost each surviving candidate.
        let density_boost = 15 * (file_merged.len().saturating_sub(1)).min(4) as u32;
        for mut candidate in file_merged {
            let base = kind_base_score(candidate.kind);
            let coverage_boost = (30 * coverage(&candidate, patterns)).min(120);
            candidate.score = (base + coverage_boost + density_boost).min(MAX_SCORE);
            merged.push(candidate);
        }
    }

    merged.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.location.path.cmp(&b.location.path))
            .then_with(|| a.location.line_start.cmp(&b.location.line_start))
    });
    merged.truncate(top_k as usize);
    merged
}

/// Confidence (0-100) that the ranked list already answers the query: the top
/// score plus its margin over the runner-up. Empty list → 0.
pub fn confidence(ranked: &[Candidate]) -> u32 {
    let Some(top) = ranked.first() else {
        return 0;
    };
    let runner_up = ranked.get(1).map(|c| c.score).unwrap_or(0);
    let margin = top.score.saturating_sub(runner_up);
    (top.score / 10 + margin / 20).min(100)
}

/// Render a candidate as a finding, preserving its provenance in the note.
pub fn finding_from_candidate(candidate: Candidate) -> ExplorationFinding {
    let kind = match candidate.kind {
        CandidateKind::SymbolExact => "exact symbol match",
        CandidateKind::SymbolFuzzy => "symbol name match",
        CandidateKind::FileNameHit => "file name match",
        CandidateKind::SemanticHit => "semantic search match",
        CandidateKind::ContentHit => "text match",
    };
    let note = match &candidate.symbol {
        Some(symbol) => format!("{kind}: `{symbol}`"),
        None => kind.to_string(),
    };
    ExplorationFinding {
        location: candidate.location,
        snippet: candidate.snippet,
        note: Some(note),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(path: &str, start: u32, end: u32, kind: CandidateKind) -> Candidate {
        Candidate {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: start,
                line_end: end,
            },
            symbol: None,
            kind,
            score: 0,
            snippet: None,
        }
    }

    // --- derive_patterns boundaries ---

    #[test]
    fn empty_query_derives_nothing() {
        assert_eq!(derive_patterns(""), QueryPatterns::default());
        assert_eq!(derive_patterns("   \t\n"), QueryPatterns::default());
    }

    #[test]
    fn stopwords_and_short_words_are_dropped() {
        let p = derive_patterns("where is the and for");
        assert!(p.identifiers.is_empty(), "got {:?}", p.identifiers);
        assert!(p.grep_patterns.is_empty());
    }

    #[test]
    fn snake_camel_and_long_words_are_identifiers() {
        let p = derive_patterns("how does decide_freshness handle StalenessWindow timeouts");
        assert_eq!(
            p.identifiers,
            vec!["decide_freshness", "handle", "StalenessWindow", "timeouts"]
        );
    }

    #[test]
    fn quoted_literals_are_extracted_and_escaped() {
        let p = derive_patterns(r#"find "foo.bar(x)" usages"#);
        assert_eq!(p.literals, vec!["foo.bar(x)"]);
        assert_eq!(p.grep_patterns[0], r"foo\.bar\(x\)");
        assert!(p.identifiers.contains(&"usages".to_string()));
    }

    #[test]
    fn unbalanced_quote_degrades_to_plain_text() {
        let p = derive_patterns("what is \"decide_freshness");
        assert!(p.literals.is_empty());
        assert_eq!(p.identifiers, vec!["decide_freshness"]);
    }

    #[test]
    fn path_and_module_compounds_split_into_segments() {
        let p = derive_patterns("look at crates/core/src/llm.rs and memory::freshness::decide");
        assert!(
            p.path_tokens
                .contains(&"crates/core/src/llm.rs".to_string())
        );
        assert!(p.identifiers.contains(&"freshness".to_string()));
        assert!(p.identifiers.contains(&"memory".to_string()));
    }

    #[test]
    fn grep_fanout_is_capped_and_deduped() {
        let p = derive_patterns("alpha_a beta_b gamma_c delta_d epsilon_e zeta_f eta_g alpha_a");
        assert_eq!(p.grep_patterns.len(), MAX_GREP_PATTERNS);
        assert_eq!(p.identifiers.len(), 7);
    }

    // --- merge_and_rank boundaries ---

    #[test]
    fn empty_input_ranks_empty() {
        let p = derive_patterns("anything");
        assert!(merge_and_rank(vec![], &p, 10).is_empty());
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let p = QueryPatterns::default();
        let raw = vec![candidate("a.rs", 1, 2, CandidateKind::ContentHit)];
        assert!(merge_and_rank(raw, &p, 0).is_empty());
    }

    #[test]
    fn inverted_range_is_swapped_not_dropped() {
        let p = QueryPatterns::default();
        let ranked = merge_and_rank(
            vec![candidate("a.rs", 20, 10, CandidateKind::ContentHit)],
            &p,
            10,
        );
        assert_eq!(ranked[0].location.line_start, 10);
        assert_eq!(ranked[0].location.line_end, 20);
    }

    #[test]
    fn dot_slash_prefix_dedupes_with_bare_path() {
        let p = QueryPatterns::default();
        let ranked = merge_and_rank(
            vec![
                candidate("./src/a.rs", 5, 5, CandidateKind::ContentHit),
                candidate("src/a.rs", 5, 5, CandidateKind::ContentHit),
            ],
            &p,
            10,
        );
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].location.path, PathBuf::from("src/a.rs"));
    }

    #[test]
    fn overlapping_ranges_merge_and_keep_stronger_kind() {
        let p = QueryPatterns::default();
        let mut sym = candidate("a.rs", 15, 25, CandidateKind::SymbolExact);
        sym.symbol = Some("foo".to_string());
        let ranked = merge_and_rank(
            vec![candidate("a.rs", 10, 20, CandidateKind::ContentHit), sym],
            &p,
            10,
        );
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].location.line_start, 10);
        assert_eq!(ranked[0].location.line_end, 25);
        assert_eq!(ranked[0].kind, CandidateKind::SymbolExact);
        assert_eq!(ranked[0].symbol.as_deref(), Some("foo"));
    }

    #[test]
    fn adjacent_but_disjoint_ranges_stay_separate() {
        let p = QueryPatterns::default();
        let ranked = merge_and_rank(
            vec![
                candidate("a.rs", 10, 20, CandidateKind::ContentHit),
                candidate("a.rs", 21, 30, CandidateKind::ContentHit),
            ],
            &p,
            10,
        );
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn symbol_exact_outranks_content_hits() {
        let p = derive_patterns("decide_freshness");
        let mut sym = candidate("b.rs", 1, 1, CandidateKind::SymbolExact);
        sym.symbol = Some("decide_freshness".to_string());
        let ranked = merge_and_rank(
            vec![candidate("a.rs", 1, 1, CandidateKind::ContentHit), sym],
            &p,
            10,
        );
        assert_eq!(ranked[0].kind, CandidateKind::SymbolExact);
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn tie_scores_order_stably_by_path_then_line() {
        let p = QueryPatterns::default();
        let ranked = merge_and_rank(
            vec![
                candidate("b.rs", 1, 1, CandidateKind::ContentHit),
                candidate("a.rs", 9, 9, CandidateKind::ContentHit),
                candidate("a.rs", 2, 2, CandidateKind::ContentHit),
            ],
            &p,
            10,
        );
        let order: Vec<_> = ranked
            .iter()
            .map(|c| (c.location.path.clone(), c.location.line_start))
            .collect();
        // a.rs has two hits → density boost puts both above b.rs; within the
        // file, ordering is by line.
        assert_eq!(
            order,
            vec![
                (PathBuf::from("a.rs"), 2),
                (PathBuf::from("a.rs"), 9),
                (PathBuf::from("b.rs"), 1),
            ]
        );
    }

    #[test]
    fn coverage_boost_is_capped() {
        let p = derive_patterns("alpha_x beta_x gamma_x delta_x epsilon_x zeta_x");
        let mut c = candidate("a.rs", 1, 1, CandidateKind::ContentHit);
        c.snippet = Some("alpha_x beta_x gamma_x delta_x epsilon_x zeta_x".to_string());
        let ranked = merge_and_rank(vec![c], &p, 10);
        assert_eq!(
            ranked[0].score,
            kind_base_score(CandidateKind::ContentHit) + 120
        );
    }

    // --- confidence boundaries ---

    #[test]
    fn confidence_of_empty_is_zero() {
        assert_eq!(confidence(&[]), 0);
    }

    #[test]
    fn single_exact_symbol_clears_early_exit_threshold() {
        let p = derive_patterns("decide_freshness");
        let mut sym = candidate("b.rs", 1, 1, CandidateKind::SymbolExact);
        sym.symbol = Some("decide_freshness".to_string());
        let ranked = merge_and_rank(vec![sym], &p, 10);
        assert!(confidence(&ranked) >= 90, "got {}", confidence(&ranked));
    }

    #[test]
    fn ambiguous_equal_candidates_have_no_margin() {
        let p = QueryPatterns::default();
        let ranked = merge_and_rank(
            vec![
                candidate("a.rs", 1, 1, CandidateKind::SymbolExact),
                candidate("b.rs", 1, 1, CandidateKind::SymbolExact),
            ],
            &p,
            10,
        );
        let c = confidence(&ranked);
        assert!(c < 90, "equal rivals must not early-exit, got {c}");
        assert!(
            c >= 30,
            "strong rivals should still reach verification, got {c}"
        );
    }

    #[test]
    fn weak_content_hit_falls_below_verify_threshold() {
        let p = QueryPatterns::default();
        let ranked = merge_and_rank(
            vec![candidate("a.rs", 1, 1, CandidateKind::ContentHit)],
            &p,
            10,
        );
        assert!(confidence(&ranked) < 30, "got {}", confidence(&ranked));
    }

    #[test]
    fn score_is_capped_at_max() {
        let p = derive_patterns("alpha_x beta_x gamma_x delta_x");
        let mut candidates = Vec::new();
        for line in [1u32, 5, 9, 13, 17, 21] {
            let mut c = candidate("a.rs", line, line, CandidateKind::SymbolExact);
            c.symbol = Some("alpha_x beta_x gamma_x delta_x".to_string());
            candidates.push(c);
        }
        let ranked = merge_and_rank(candidates, &p, 10);
        assert!(ranked.iter().all(|c| c.score <= MAX_SCORE));
    }

    #[test]
    fn finding_from_candidate_preserves_provenance() {
        let mut c = candidate("a.rs", 1, 2, CandidateKind::SymbolExact);
        c.symbol = Some("foo::bar".to_string());
        c.snippet = Some("fn bar() {}".to_string());
        let f = finding_from_candidate(c);
        assert_eq!(f.note.as_deref(), Some("exact symbol match: `foo::bar`"));
        assert_eq!(f.snippet.as_deref(), Some("fn bar() {}"));
    }
}
