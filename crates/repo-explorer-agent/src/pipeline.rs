//! The deterministic retrieval pre-stage: derive patterns from the query, fan
//! out concurrently over the memory and search backends, and rank the merged
//! candidates — no LLM involved. Backend failures are soft: a failed leg
//! contributes nothing.

use futures_util::future::join_all;
use repo_explorer_core::domain::{
    Candidate, CandidateKind, ExplorationFinding, ExplorationQuery, FileLocation,
};
use repo_explorer_core::fingerprint::RepoFingerprint;
use repo_explorer_core::memory::{GraphQuery, MemoryBackend};
use repo_explorer_core::retrieval::{confidence, derive_patterns, merge_and_rank};
use repo_explorer_core::search::{SearchBackend, SearchOptions};
use std::collections::HashSet;
use std::path::Path;

use crate::cache::{ResultCache, encode_field_into, opt_to_string, scope_display};
use crate::dispatch::{MATCH_ANY_NON_EMPTY_LINE, escapes_repo_root};

/// How many identifier tokens get their own symbol-lookup leg.
const SYMBOL_LOOKUP_TOKENS: usize = 4;
/// How many path tokens get their own file-name leg.
const FILE_LOOKUP_TOKENS: usize = 3;
/// Raw results requested per leg (ranking truncates to top-k afterwards).
const PER_LEG_MAX_RESULTS: u32 = 20;

/// What the pre-stage produced for one query.
pub(crate) struct RetrievalOutcome {
    /// Ranked, merged, top-k candidates (strongest first).
    pub candidates: Vec<Candidate>,
    /// 0-100; see `repo_explorer_core::retrieval::confidence`.
    pub confidence: u32,
}

/// Memoization handle for the fanout legs: only active when both a cache and
/// a repository fingerprint exist.
pub(crate) type LegCache<'a> = Option<(&'a ResultCache, &'a RepoFingerprint)>;

pub(crate) async fn retrieve<M: MemoryBackend, S: SearchBackend>(
    memory: &M,
    search: &S,
    repo_root: &Path,
    query: &ExplorationQuery,
    top_k: u32,
    leg_cache: LegCache<'_>,
) -> RetrievalOutcome {
    let patterns = derive_patterns(&query.text);
    // Unlike the LLM's own grep/find scope argument (validated in dispatch.rs
    // via `reject_escaping_path`), this top-level query's `scope_hint` comes
    // straight from the MCP caller with no validation applied upstream. Drop
    // an escaping hint rather than handing it to `search.search` unchecked —
    // fall back to unscoped (still repo_root-bounded) instead of leaking
    // content from outside the repository.
    let scope = query
        .scope_hint
        .as_deref()
        .filter(|p| !escapes_repo_root(p));

    let symbol_legs = join_all(patterns.identifiers.iter().take(SYMBOL_LOOKUP_TOKENS).map(
        |token| {
            memoized(
                leg_cache,
                move || leg_key("symbol", token, scope),
                async move {
                    let graph_query = GraphQuery {
                        name_pattern: Some(token.clone()),
                        file_pattern: scope.map(|p| p.to_string_lossy().into_owned()),
                        max_results: Some(PER_LEG_MAX_RESULTS),
                        ..GraphQuery::default()
                    };
                    soft_leg(
                        "symbol",
                        token,
                        memory.search_graph(repo_root, &graph_query),
                        |res| symbol_candidates(res.findings, token),
                    )
                    .await
                },
            )
        },
    ));

    let semantic_leg = memoized(
        leg_cache,
        move || {
            let mut key = leg_key("semantic", &query.text, scope);
            encode_field_into(&mut key, &opt_to_string(query.max_results));
            key
        },
        async move {
            // Reuses `query.text`/`max_results` but swaps in the already-
            // filtered `scope`, not `query.scope_hint` — an escaping hint
            // must not reach `search_code`'s RPC any more than it reaches
            // `search.search` in the other three legs.
            let sanitized_query = ExplorationQuery {
                text: query.text.clone(),
                scope_hint: scope.map(Path::to_path_buf),
                max_results: query.max_results,
            };
            soft_leg(
                "semantic",
                query.text.as_str(),
                memory.search_code(repo_root, &sanitized_query),
                |res| candidates_of_kind(res.findings, CandidateKind::SemanticHit),
            )
            .await
        },
    );

    let grep_legs = join_all(patterns.grep_patterns.iter().map(|pattern| {
        memoized(
            leg_cache,
            move || leg_key("grep", pattern, scope),
            async move {
                let options = SearchOptions {
                    max_results: Some(PER_LEG_MAX_RESULTS),
                    ..SearchOptions::default()
                };
                soft_leg(
                    "grep",
                    pattern,
                    search.search(repo_root, pattern, scope, &options),
                    |findings| candidates_of_kind(findings, CandidateKind::ContentHit),
                )
                .await
            },
        )
    }));

    let file_legs = join_all(
        patterns
            .path_tokens
            .iter()
            .take(FILE_LOOKUP_TOKENS)
            .map(|token| {
                memoized(
                    leg_cache,
                    move || leg_key("file", token, scope),
                    async move {
                        let options = SearchOptions {
                            max_results: Some(PER_LEG_MAX_RESULTS),
                            file_glob: Some(file_glob_for(token)),
                            ..SearchOptions::default()
                        };
                        soft_leg(
                            "file",
                            token,
                            search.search(repo_root, MATCH_ANY_NON_EMPTY_LINE, scope, &options),
                            file_candidates,
                        )
                        .await
                    },
                )
            }),
    );

    let (symbols, semantic, greps, files) =
        futures_util::future::join4(symbol_legs, semantic_leg, grep_legs, file_legs).await;

    let mut raw: Vec<Candidate> = Vec::new();
    raw.extend(symbols.into_iter().flatten());
    raw.extend(semantic);
    raw.extend(greps.into_iter().flatten());
    raw.extend(files.into_iter().flatten());

    let candidates = merge_and_rank(raw, &patterns, top_k);
    let confidence = confidence(&candidates);
    RetrievalOutcome {
        candidates,
        confidence,
    }
}

/// Wrap a leg future with the leg cache: return the memoized candidates when
/// present, otherwise run the leg and store its result. `leg` is only called
/// when a cache is actually active, so an inactive cache costs no key-format
/// work. Only a successful run (`Some`) is memoized — a swallowed backend
/// failure (`None`, see `soft_leg`) must not poison the cache with a
/// permanent-looking empty result for what was really a transient hiccup.
async fn memoized(
    leg_cache: LegCache<'_>,
    leg: impl FnOnce() -> String,
    fut: impl Future<Output = Option<Vec<Candidate>>>,
) -> Vec<Candidate> {
    let key = leg_cache.map(|(cache, fp)| (cache, ResultCache::leg_key(fp, &leg())));
    if let Some((cache, key)) = &key
        && let Some(hit) = cache.get_leg(key)
    {
        return hit;
    }
    let out = fut.await;
    if let Some((cache, key)) = key
        && let Some(candidates) = &out
    {
        cache.put_leg(key, candidates.clone());
    }
    out.unwrap_or_default()
}

/// Run one fanout leg's backend call: classify a success, log-and-drop a
/// failure. Failures are soft — a failed leg contributes nothing rather than
/// failing the whole query — but stay distinguishable (`None`) from a
/// legitimate zero-hit success (`Some(vec![])`) so `memoized` never caches a
/// transient failure as a permanent empty answer.
async fn soft_leg<T, E: std::fmt::Display>(
    leg: &'static str,
    ctx: impl std::fmt::Display,
    fut: impl Future<Output = Result<T, E>>,
    classify: impl FnOnce(T) -> Vec<Candidate>,
) -> Option<Vec<Candidate>> {
    match fut.await {
        Ok(res) => Some(classify(res)),
        Err(e) => {
            tracing::debug!(leg, %ctx, error = %e, "retrieval leg failed");
            None
        }
    }
}

/// Build a leg-cache key from a leg prefix, its value, and the scope hint.
/// Shared by every fanout leg so a future change to the (value, scope)
/// encoding can't be applied to some legs and missed on others.
fn leg_key(prefix: &str, value: &str, scope: Option<&Path>) -> String {
    let scope = scope_display(scope);
    let mut key = String::with_capacity(prefix.len() + value.len() + scope.len() + 16);
    key.push_str(prefix);
    encode_field_into(&mut key, value);
    encode_field_into(&mut key, &scope);
    key
}

/// Build candidates from findings, with `kind_of` classifying each finding
/// individually. Shared by `symbol_candidates` (kind depends on the finding)
/// and `candidates_of_kind` (kind is fixed) so the two never drift apart on
/// how a `Candidate` is otherwise assembled from an `ExplorationFinding`.
fn candidates_with(
    findings: Vec<ExplorationFinding>,
    kind_of: impl Fn(&ExplorationFinding) -> CandidateKind,
) -> Vec<Candidate> {
    findings
        .into_iter()
        .map(|f| {
            let kind = kind_of(&f);
            Candidate {
                location: f.location,
                symbol: f.note,
                kind,
                score: 0,
                snippet: f.snippet,
            }
        })
        .collect()
}

/// Classify symbol-lookup findings: the memory backend carries the symbol name
/// in `note`; an exact (last-segment) name match is `SymbolExact`, everything
/// else `SymbolFuzzy`.
fn symbol_candidates(findings: Vec<ExplorationFinding>, token: &str) -> Vec<Candidate> {
    candidates_with(findings, |f| match f.note.as_deref() {
        Some(name) if last_segment(name) == token => CandidateKind::SymbolExact,
        _ => CandidateKind::SymbolFuzzy,
    })
}

/// A qualified name's final segment (`a::b::c` → `c`, `a.b` → `b`).
fn last_segment(name: &str) -> &str {
    // rsplit always yields at least the whole string, so this is infallible.
    name.rsplit(&[':', '.'][..]).next().unwrap()
}

fn candidates_of_kind(findings: Vec<ExplorationFinding>, kind: CandidateKind) -> Vec<Candidate> {
    candidates_with(findings, |_| kind)
}

/// Collapse content rows from the `find` emulation to one file-level candidate
/// per unique path.
fn file_candidates(findings: Vec<ExplorationFinding>) -> Vec<Candidate> {
    let mut seen = HashSet::new();
    // Dedup by borrowed path first so the owned pass below can move each
    // survivor's path straight into its Candidate instead of cloning it.
    let keep: Vec<bool> = findings
        .iter()
        .map(|f| seen.insert(&f.location.path))
        .collect();
    findings
        .into_iter()
        .zip(keep)
        .filter(|(_, keep)| *keep)
        .map(|(f, _)| Candidate {
            location: FileLocation {
                path: f.location.path,
                line_start: 1,
                line_end: 1,
            },
            symbol: None,
            kind: CandidateKind::FileNameHit,
            score: 0,
            snippet: None,
        })
        .collect()
}

/// Glob for a path-like token: a bare file name matches by basename; a token
/// without an extension matches as a substring. Shared with `dispatch`'s
/// `find` tool, which emulates the same filename search over an LLM-supplied
/// pattern.
pub(crate) fn file_glob_for(token: &str) -> String {
    // rsplit always yields at least the whole string, so this is infallible.
    let name = token.rsplit('/').next().unwrap();
    if name.contains('.') {
        name.to_string()
    } else {
        format!("*{name}*")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::ExplorationResult;
    use repo_explorer_core::memory::mock::MockMemoryBackend;
    use repo_explorer_core::search::mock::{Call as SearchCall, MockSearchBackend};
    use std::path::PathBuf;

    fn finding(path: &str, line: u32, note: Option<&str>) -> ExplorationFinding {
        ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: line,
                line_end: line,
            },
            snippet: None,
            note: note.map(str::to_string),
        }
    }

    #[test]
    fn symbol_classification_exact_vs_fuzzy() {
        let out = symbol_candidates(
            vec![
                finding("a.rs", 1, Some("module::decide_freshness")),
                finding("b.rs", 2, Some("decide_freshness_v2")),
                finding("c.rs", 3, None),
            ],
            "decide_freshness",
        );
        assert_eq!(out[0].kind, CandidateKind::SymbolExact);
        assert_eq!(out[1].kind, CandidateKind::SymbolFuzzy);
        assert_eq!(out[2].kind, CandidateKind::SymbolFuzzy);
    }

    #[test]
    fn file_glob_shapes() {
        assert_eq!(file_glob_for("crates/core/src/llm.rs"), "llm.rs");
        assert_eq!(file_glob_for("freshness"), "*freshness*");
    }

    #[test]
    fn file_candidates_collapse_per_path() {
        let out = file_candidates(vec![
            finding("x.rs", 3, None),
            finding("x.rs", 9, None),
            finding("y.rs", 1, None),
        ]);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|c| c.kind == CandidateKind::FileNameHit));
        assert!(
            out.iter()
                .all(|c| c.location.line_start == 1 && c.location.line_end == 1)
        );
    }

    #[tokio::test]
    async fn exact_symbol_hit_yields_high_confidence() {
        let memory = MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
            findings: vec![finding(
                "crates/x/src/freshness.rs",
                12,
                Some("decide_freshness"),
            )],
            summary: "1 row".to_string(),
        }));
        let search = MockSearchBackend::new();
        let query = ExplorationQuery {
            text: "decide_freshness".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let out = retrieve(&memory, &search, Path::new("/repo"), &query, 12, None).await;
        assert_eq!(out.candidates[0].kind, CandidateKind::SymbolExact);
        assert!(out.confidence >= 90, "got {}", out.confidence);
    }

    #[tokio::test]
    async fn backend_failures_are_soft_and_yield_empty() {
        let memory = MockMemoryBackend::new()
            .with_search_graph_result(Err(repo_explorer_core::memory::MemoryError::Transport(
                "down".to_string(),
            )))
            .with_search_code_result(Err(repo_explorer_core::memory::MemoryError::Transport(
                "down".to_string(),
            )));
        let search = MockSearchBackend::new().with_search_result(Err(
            repo_explorer_core::search::SearchError::BackendNotFound("none".to_string()),
        ));
        let query = ExplorationQuery {
            text: "decide_freshness".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let out = retrieve(&memory, &search, Path::new("/repo"), &query, 12, None).await;
        assert!(out.candidates.is_empty());
        assert_eq!(out.confidence, 0);
    }

    #[tokio::test]
    async fn leg_cache_does_not_cross_scopes() {
        let search = MockSearchBackend::new().with_search_result(Ok(vec![]));
        let memory = MockMemoryBackend::new();
        let cache = ResultCache::new(16);
        let fp = RepoFingerprint {
            head_sha: "sha".to_string(),
            dirty_hash: "clean".to_string(),
        };

        let unscoped = ExplorationQuery {
            text: "decide_freshness".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let _ = retrieve(
            &memory,
            &search,
            Path::new("/repo"),
            &unscoped,
            12,
            Some((&cache, &fp)),
        )
        .await;

        let scoped = ExplorationQuery {
            text: "decide_freshness".to_string(),
            scope_hint: Some(PathBuf::from("crates/api")),
            max_results: None,
        };
        let _ = retrieve(
            &memory,
            &search,
            Path::new("/repo"),
            &scoped,
            12,
            Some((&cache, &fp)),
        )
        .await;

        // A second query differing only by scope_hint must not be served from
        // the first query's cached (wrongly-scoped) leg entry — the grep leg
        // must re-run search for its own scope.
        let scopes: Vec<Option<PathBuf>> = search
            .calls()
            .into_iter()
            .map(|SearchCall::Search { scope, .. }| scope)
            .collect();
        assert!(
            scopes.contains(&None),
            "unscoped search never ran: {scopes:?}"
        );
        assert!(
            scopes.contains(&Some(PathBuf::from("crates/api"))),
            "scoped search never ran: {scopes:?}"
        );
    }

    #[tokio::test]
    async fn grep_leg_cache_key_does_not_collide_across_pattern_scope_boundary() {
        // Regression: a bare `#`-joined leg key let pattern `x#y` (unscoped)
        // and pattern `x` (scope `y#`) hash to the identical key
        // `"grep#x#y#"`. Query B must not be served query A's cached rows.
        let cache = ResultCache::new(16);
        let fp = RepoFingerprint {
            head_sha: "sha".to_string(),
            dirty_hash: "clean".to_string(),
        };
        let memory = MockMemoryBackend::new();

        let search_a = MockSearchBackend::new().with_search_result(Ok(vec![finding(
            "from_query_a.rs",
            1,
            None,
        )]));
        let query_a = ExplorationQuery {
            text: r#"grep for "x#y""#.to_string(),
            scope_hint: None,
            max_results: None,
        };
        let _ = retrieve(
            &memory,
            &search_a,
            Path::new("/repo"),
            &query_a,
            12,
            Some((&cache, &fp)),
        )
        .await;

        let search_b = MockSearchBackend::new().with_search_result(Ok(vec![finding(
            "from_query_b.rs",
            1,
            None,
        )]));
        let query_b = ExplorationQuery {
            text: r#"grep for "x""#.to_string(),
            scope_hint: Some(PathBuf::from("y#")),
            max_results: None,
        };
        let out_b = retrieve(
            &memory,
            &search_b,
            Path::new("/repo"),
            &query_b,
            12,
            Some((&cache, &fp)),
        )
        .await;

        assert!(
            !search_b.calls().is_empty(),
            "query B's grep leg was served from query A's cache instead of running its own search"
        );
        assert!(
            out_b
                .candidates
                .iter()
                .all(|c| c.location.path.as_path() != Path::new("from_query_a.rs")),
            "query B's candidates were contaminated by query A's cached grep results: {:?}",
            out_b.candidates
        );
    }

    #[tokio::test]
    async fn fanout_passes_scope_and_leg_caps_to_search() {
        let search = MockSearchBackend::new().with_search_result(Ok(vec![]));
        let memory = MockMemoryBackend::new();
        let query = ExplorationQuery {
            text: "\"exact phrase\" plus_token".to_string(),
            scope_hint: Some(PathBuf::from("crates")),
            max_results: None,
        };
        let _ = retrieve(&memory, &search, Path::new("/repo"), &query, 12, None).await;
        let calls = search.calls();
        assert!(!calls.is_empty());
        for call in &calls {
            let SearchCall::Search { scope, options, .. } = call;
            assert_eq!(scope.as_deref(), Some(Path::new("crates")));
            assert_eq!(options.max_results, Some(PER_LEG_MAX_RESULTS));
        }
    }

    #[tokio::test]
    async fn escaping_scope_hint_is_dropped_instead_of_reaching_search() {
        let search = MockSearchBackend::new().with_search_result(Ok(vec![]));
        let memory = MockMemoryBackend::new();
        let query = ExplorationQuery {
            text: "main".to_string(),
            scope_hint: Some(PathBuf::from("/etc")),
            max_results: None,
        };
        let _ = retrieve(&memory, &search, Path::new("/repo"), &query, 12, None).await;
        let calls = search.calls();
        assert!(!calls.is_empty());
        for call in &calls {
            let SearchCall::Search { scope, .. } = call;
            assert_eq!(
                scope, &None,
                "escaping scope_hint must not reach the search backend"
            );
        }
        assert_escaping_scope_hint_dropped_from_memory(&memory);
    }

    #[tokio::test]
    async fn relative_escaping_scope_hint_is_dropped_instead_of_reaching_search() {
        let search = MockSearchBackend::new().with_search_result(Ok(vec![]));
        let memory = MockMemoryBackend::new();
        let query = ExplorationQuery {
            text: "main".to_string(),
            scope_hint: Some(PathBuf::from("../../etc")),
            max_results: None,
        };
        let _ = retrieve(&memory, &search, Path::new("/repo"), &query, 12, None).await;
        let calls = search.calls();
        assert!(!calls.is_empty());
        for call in &calls {
            let SearchCall::Search { scope, .. } = call;
            assert_eq!(
                scope, &None,
                "escaping scope_hint must not reach the search backend"
            );
        }
        assert_escaping_scope_hint_dropped_from_memory(&memory);
    }

    /// Regression for the semantic leg forwarding the raw, unsanitized
    /// `query.scope_hint` to `memory.search_code` (it must instead see the
    /// same filtered `scope` the other three legs get).
    fn assert_escaping_scope_hint_dropped_from_memory(memory: &MockMemoryBackend) {
        let calls = memory.calls();
        let search_code_calls: Vec<_> = calls
            .iter()
            .filter_map(|c| match c {
                repo_explorer_core::memory::mock::Call::SearchCode { query, .. } => {
                    Some(query.scope_hint.as_ref())
                }
                _ => None,
            })
            .collect();
        assert!(
            !search_code_calls.is_empty(),
            "semantic leg never called search_code"
        );
        for scope_hint in search_code_calls {
            assert_eq!(
                scope_hint, None,
                "escaping scope_hint must not reach memory.search_code"
            );
        }
    }
}
