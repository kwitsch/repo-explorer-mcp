//! Pure decision logic for the deterministic retrieval pre-stage: query →
//! search patterns, raw candidates → merged/ranked top-k, ranked list →
//! confidence score. No I/O, no LLM — the orchestration (backend fanout) lives
//! in `repo-explorer-agent`; this module mirrors the pure-logic pattern of
//! `repo-explorer-memory`'s `freshness.rs`.

use crate::domain::{Candidate, CandidateKind, ExplorationFinding, FileLocation};
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
    "method", "class", "der", "die", "das", "und", "wird", "wie", "welche", "wo",
];

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// True for tokens worth treating as identifiers: snake_case, camelCase,
/// digit-bearing, or any sufficiently long non-stopword word.
fn is_identifier_like(token: &str) -> bool {
    if token.len() < 3 {
        return false;
    }
    // Stopword check must precede the shape checks below: a capitalized
    // stopword (e.g. "How") still has_lower && has_upper and would otherwise
    // short-circuit past the filter.
    if STOPWORDS.iter().any(|s| s.eq_ignore_ascii_case(token)) {
        return false;
    }
    let has_underscore = token.contains('_');
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    let has_lower = token.chars().any(|c| c.is_ascii_lowercase());
    let has_upper = token.chars().any(|c| c.is_ascii_uppercase());
    if has_underscore || has_digit || (has_lower && has_upper) {
        return true;
    }
    token.len() >= 4
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

/// Locate the next quote delimiter in `s`. `"` and `` ` `` always count; `'`
/// only counts when not preceded by a word char, so contractions/possessives
/// (`user's`, `isn't`) are left as plain text instead of opening a literal
/// that some later, unrelated apostrophe would then close.
fn find_quote_start(s: &str) -> Option<(usize, char)> {
    for (i, c) in s.char_indices() {
        match c {
            '"' | '`' => return Some((i, c)),
            '\'' if !s[..i].chars().next_back().is_some_and(is_word_char) => {
                return Some((i, c));
            }
            _ => {}
        }
    }
    None
}

/// Locate the delimiter that closes a literal opened with `quote` in `s`.
/// `"` and `` ` `` always count; `'` is skipped when it sits inside a word
/// (a word char on both sides, e.g. the possessive in `cache's`) so a
/// contraction/possessive apostrophe within the literal doesn't close it
/// early — the genuine closing quote is itself preceded by a word char (the
/// literal's last letter), so, unlike `find_quote_start`, only a word char on
/// *both* sides marks it as non-delimiting.
fn find_quote_end(s: &str, quote: char) -> Option<usize> {
    for (i, c) in s.char_indices() {
        if c != quote {
            continue;
        }
        if quote == '\''
            && s[..i].chars().next_back().is_some_and(is_word_char)
            && s[i + 1..].chars().next().is_some_and(is_word_char)
        {
            continue;
        }
        return Some(i);
    }
    None
}

/// Derive deterministic search inputs from a free-text query.
pub fn derive_patterns(query: &str) -> QueryPatterns {
    let mut patterns = QueryPatterns::default();

    // Quoted literals: content of "…", '…', and `…` pairs.
    let mut rest = query;
    let mut unquoted = String::new();
    while let Some((open, quote)) = find_quote_start(rest) {
        unquoted.push_str(&rest[..open]);
        unquoted.push(' ');
        let after = &rest[open + 1..];
        match find_quote_end(after, quote) {
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
        // Leading '.' is exempted (preserves `./x` relative-path tokens);
        // trailing '.' is not, so a sentence-ending period doesn't stick to
        // the token (e.g. "…/retrieval.rs." at a sentence's end).
        let trimmed = raw
            .trim_start_matches(|c: char| !is_word_char(c) && !matches!(c, '/' | '.'))
            .trim_end_matches(|c: char| !is_word_char(c) && c != '/');
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

    for token in patterns.literals.iter().chain(patterns.identifiers.iter()) {
        if patterns.grep_patterns.len() >= MAX_GREP_PATTERNS {
            break;
        }
        push_unique(&mut patterns.grep_patterns, escape_regex(token));
    }

    patterns
}

/// Strip a leading `./` so the same file never appears under two spellings
/// (grep emits `./x` without a scope and `x` with one).
pub fn normalize_rel_path(path: PathBuf) -> PathBuf {
    match path.strip_prefix(".") {
        Ok(stripped) => stripped.to_path_buf(),
        Err(_) => path,
    }
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

/// Number of distinct query identifiers/literals (pre-lowercased by the
/// caller) appearing in the candidate's symbol, path, or snippet.
fn coverage(candidate: &Candidate, lowered_patterns: &[String]) -> u32 {
    let mut haystack = format!(
        "{} {} {}",
        candidate.symbol.as_deref().unwrap_or(""),
        candidate.location.path.to_string_lossy(),
        candidate.snippet.as_deref().unwrap_or("")
    );
    haystack.make_ascii_lowercase();
    lowered_patterns
        .iter()
        .filter(|token| haystack.contains(token.as_str()))
        .count() as u32
}

/// Normalize a location: strip `./`, swap an inverted line range, and widen a
/// zero `line_end` to `line_start`. `pub` so callers outside the deterministic
/// retrieval path (e.g. the agent crate's `finish`-argument parsing) can apply
/// the same normalization to model-supplied locations.
pub fn normalize_location(location: FileLocation) -> FileLocation {
    let path = normalize_rel_path(location.path);
    let (mut start, mut end) = (location.line_start, location.line_end);
    if end == 0 {
        end = start;
    } else if end < start {
        std::mem::swap(&mut start, &mut end);
    }
    FileLocation {
        path,
        line_start: start,
        line_end: end,
    }
}

/// `(0, 0)` is `normalize_location`'s "location unknown" sentinel, not a real
/// one-line span at line 0 — two candidates that both merely lack line data
/// must not be treated as overlapping just because they share that sentinel.
fn is_unknown_location(loc: &FileLocation) -> bool {
    loc.line_start == 0 && loc.line_end == 0
}

fn overlaps(a: &FileLocation, b: &FileLocation) -> bool {
    if is_unknown_location(a) || is_unknown_location(b) {
        return false;
    }
    a.line_start <= b.line_end && b.line_start <= a.line_end
}

/// Merge `b` into `a`: widen the range, keep the stronger kind's identity.
fn merge_into(a: &mut Candidate, b: Candidate) {
    a.location.line_start = a.location.line_start.min(b.location.line_start);
    a.location.line_end = a.location.line_end.max(b.location.line_end);
    if kind_base_score(b.kind) > kind_base_score(a.kind) {
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
    // Lowercase pattern tokens once, deduped post-lowercasing so a term
    // repeated under different casing (or as both identifier and literal)
    // isn't double-counted by coverage()'s distinct-match contract.
    let mut lowered_patterns: Vec<String> = Vec::new();
    for token in patterns.identifiers.iter().chain(patterns.literals.iter()) {
        push_unique(&mut lowered_patterns, token.to_ascii_lowercase());
    }

    let mut normalized: Vec<Candidate> = raw
        .into_iter()
        .map(|mut candidate| {
            candidate.location = normalize_location(candidate.location);
            candidate
        })
        .collect();
    // Single sort by (path, line_start, line_end) doubles as the per-file
    // grouping key (consecutive runs below) and the line ordering merge()
    // needs, replacing a HashMap groupby (one PathBuf clone per candidate)
    // plus a redundant per-file sort.
    normalized.sort_by(|a, b| {
        a.location
            .path
            .cmp(&b.location.path)
            .then_with(|| a.location.line_start.cmp(&b.location.line_start))
            .then_with(|| a.location.line_end.cmp(&b.location.line_end))
    });

    let mut merged: Vec<Candidate> = Vec::new();
    let mut candidates = normalized.into_iter().peekable();
    while let Some(first) = candidates.next() {
        let mut file_merged: Vec<Candidate> = vec![first];
        loop {
            let same_file = candidates
                .peek()
                .is_some_and(|c| c.location.path == file_merged[0].location.path);
            if !same_file {
                break;
            }
            let candidate = candidates.next().expect("peeked Some above");
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
            let coverage_boost = (30 * coverage(&candidate, &lowered_patterns)).min(120);
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

/// Human-readable label for a candidate's provenance, shared by the
/// finding renderer and the verification-stage candidate-block renderer.
pub fn kind_label(kind: CandidateKind) -> &'static str {
    match kind {
        CandidateKind::SymbolExact => "exact symbol match",
        CandidateKind::SymbolFuzzy => "symbol name match",
        CandidateKind::FileNameHit => "file name match",
        CandidateKind::SemanticHit => "semantic search match",
        CandidateKind::ContentHit => "text match",
    }
}

/// Render a candidate as a finding, preserving its provenance in the note.
pub fn finding_from_candidate(candidate: Candidate) -> ExplorationFinding {
    let kind = kind_label(candidate.kind);
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
    fn possessive_and_contraction_apostrophes_are_not_quote_delimiters() {
        let p = derive_patterns("how does the cache's ttl work and isn't it stale");
        assert!(p.literals.is_empty(), "got {:?}", p.literals);
        assert!(p.identifiers.contains(&"cache".to_string()));
        assert!(p.identifiers.contains(&"work".to_string()));
        assert!(p.identifiers.contains(&"stale".to_string()));
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
    fn unknown_location_sentinels_do_not_merge_distinct_symbols() {
        // Two symbols in the same file both missing line data (backend
        // omitted the `lines` column) must survive as distinct candidates,
        // not collapse into one via the (0, 0) "location unknown" sentinel.
        let p = QueryPatterns::default();
        let mut a = candidate("a.rs", 0, 0, CandidateKind::SymbolExact);
        a.symbol = Some("decide_freshness".to_string());
        let mut b = candidate("a.rs", 0, 0, CandidateKind::SymbolExact);
        b.symbol = Some("StalenessWindow".to_string());
        let ranked = merge_and_rank(vec![a, b], &p, 10);
        assert_eq!(ranked.len(), 2);
        let symbols: Vec<_> = ranked.iter().map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&Some("decide_freshness")));
        assert!(symbols.contains(&Some("StalenessWindow")));
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
    fn coverage_boost_does_not_double_count_case_variants() {
        // "Cache" and "cache" are distinct identifiers pre-lowercasing but
        // must collapse to one distinct term for coverage scoring.
        let p = derive_patterns("check Cache and cache items");
        assert_eq!(p.identifiers, vec!["check", "Cache", "cache", "items"]);
        let mut c = candidate("a.rs", 1, 1, CandidateKind::ContentHit);
        c.snippet = Some("check cache items".to_string());
        let ranked = merge_and_rank(vec![c], &p, 10);
        // 3 distinct terms ("check", "cache", "items") -> boost of 90, not 120.
        assert_eq!(
            ranked[0].score,
            kind_base_score(CandidateKind::ContentHit) + 90
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
