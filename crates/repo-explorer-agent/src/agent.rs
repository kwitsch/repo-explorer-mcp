//! The exploration orchestrator. Everything deterministic runs in Rust; the
//! LLM only selects/verifies:
//!
//! 1. query-cache lookup (repo fingerprint, diff-based invalidation) — a hit
//!    costs nothing;
//! 2. deterministic retrieval pre-stage (symbol lookup + grep fanout +
//!    ranking) — high confidence answers directly with **zero** LLM calls;
//! 3. verification stage — 1 turn (plus an optional expand turn) over the
//!    top-k candidates;
//! 4. explorative fallback loop — only for low-confidence queries, hardened
//!    with a shared token budget, batch enforcement, and a forced final
//!    `finish`.
//!
//! Tool/backend failures and malformed model output degrade into `Role::Tool`
//! messages fed back to the model; only a `RouterError` in the fallback loop
//! is a hard failure.

use repo_explorer_core::config::{AgentSettings, CacheKeyMode, CacheSettings, RepoBriefKey};
use repo_explorer_core::domain::{
    Candidate, ExplorationFinding, ExplorationOutcome, ExplorationQuery, ExplorationResult,
    FileLocation, StageExit,
};
use repo_explorer_core::fingerprint::{RepoFingerprint, RepoStateProbe};
use repo_explorer_core::llm::{
    CallOptions, Clock, LlmProvider, Message, ProviderResponse, ProviderRouter, SystemClock,
    TokenUsage, ToolCall,
};
use repo_explorer_core::memory::{IndexStatus, MemoryBackend, MemoryError};
use repo_explorer_core::retrieval::{
    finding_from_candidate, is_unknown_location, normalize_location, normalize_rel_path,
};
use repo_explorer_core::search::SearchBackend;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::brief;
use crate::cache::{CappedMap, QueryEntry, ResultCache};
use crate::disk_cache::{self, FileDep};
use crate::dispatch::{canonical_repo_root, clamp_location, dispatch_inner, read_verified_file};
use crate::pipeline;
use crate::render::{RenderCaps, dedupe_key, symbols_for, tidy_findings};
use crate::tools::{finish_only_catalog, parse_finish_lenient, resolve_finish, tool_catalog};
use crate::verify::{VerifyOutcome, verify};

/// The only hard-failure mode: the provider router could not produce a response
/// (all providers exhausted, or a non-failover provider error). Flattened to a
/// `String` to stay comparable, matching the crate-boundary convention.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum AgentLoopError {
    #[error("llm provider error: {0}")]
    Provider(String),
}

/// Shared token accounting across the verification stage and the fallback
/// loop. `limit == 0` means "no budget".
#[derive(Debug, Clone)]
pub(crate) struct TokenBudget {
    limit: u64,
    spent: u64,
    /// Number of `Completion`s returned to the agent so far this run —
    /// exists purely to be logged on `exploration complete`.
    llm_calls: u32,
    /// Prompt tokens this run was served from the provider's prompt cache,
    /// and prompt tokens it wrote into that cache. Telemetry only — both are
    /// already inside `spent` (see `TokenUsage::total`).
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    /// This run's `CallOptions::rotation_seed` — one value picked by `run`
    /// per top-level conversation and reused by every raw
    /// `complete_with_tools` call the verification stage and the fallback
    /// loop make for it (piggybacked on `TokenBudget` since it's already
    /// threaded to every call site that builds `CallOptions`), so a rotating
    /// entry's start model stays stable across that conversation's turns —
    /// see `CallOptions::rotation_seed`'s doc comment for why that matters.
    rotation_seed: usize,
}

impl TokenBudget {
    pub(crate) fn new(limit: u64, rotation_seed: usize) -> Self {
        Self {
            limit,
            spent: 0,
            llm_calls: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            rotation_seed,
        }
    }

    pub(crate) fn rotation_seed(&self) -> usize {
        self.rotation_seed
    }

    pub(crate) fn add(&mut self, usage: Option<TokenUsage>) {
        self.llm_calls += 1;
        if let Some(usage) = usage {
            self.spent = self.spent.saturating_add(usage.total());
            self.cache_read_tokens = self.cache_read_tokens.saturating_add(usage.cached_tokens);
            self.cache_write_tokens = self
                .cache_write_tokens
                .saturating_add(usage.cache_creation_tokens);
        }
    }

    pub(crate) fn exhausted(&self) -> bool {
        self.limit != 0 && self.spent >= self.limit
    }

    pub(crate) fn spent(&self) -> u64 {
        self.spent
    }

    pub(crate) fn llm_calls(&self) -> u32 {
        self.llm_calls
    }

    pub(crate) fn cache_read_tokens(&self) -> u64 {
        self.cache_read_tokens
    }

    pub(crate) fn cache_write_tokens(&self) -> u64 {
        self.cache_write_tokens
    }
}

/// Consecutive single-call turns rejected before one is executed anyway (the
/// 2-strike batching rule; the escape hatch keeps weak models from
/// deadlocking).
const MAX_SINGLE_CALL_REJECTIONS: u32 = 2;

/// Optional JSONL metrics sink: when `REPO_EXPLORER_METRICS` names a path,
/// every query appends one `QueryMetrics` line to it. Resolved once per
/// process, never per query.
static METRICS_SINK: std::sync::LazyLock<Option<PathBuf>> =
    std::sync::LazyLock::new(|| sink_path(std::env::var_os));

/// Resolve the sink path from an env accessor. An empty value counts as
/// unset — in an MCP server's `env` map that is the usual way to disable a
/// variable, and a `""` path would otherwise WARN on every single query.
/// Split out from the `LazyLock` so a test can pin the documented variable
/// name without mutating the process environment.
fn sink_path(get_env: impl FnOnce(&'static str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    get_env("REPO_EXPLORER_METRICS")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// One per-query telemetry record, emitted from **every** `run` exit path
/// (cache hit, early exit, verify, fallback, provider error).
///
/// Built up as the run progresses, starting at the "nothing ran yet" values,
/// so an exit taken before a stage executed still yields a complete record.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct QueryMetrics {
    /// Run start, for `total_ms`. Not part of the record.
    #[serde(skip)]
    started: Instant,
    pub ts_unix_ms: u64,
    pub repo_path: String,
    pub query: String,
    pub scope_hint: Option<String>,
    pub max_results: Option<u32>,
    /// Exit path: `early-exit` | `verify` | `fallback` | `cache` | `error`.
    pub path: &'static str,
    pub index_status: &'static str,
    /// `None` until the retrieval pre-stage runs — the cache path exits
    /// before it, and a fabricated `0` there would read as a real score.
    pub confidence: Option<u32>,
    pub candidate_count: Option<usize>,
    /// Which Stage-3 gate let the run skip the LLM: `confidence` |
    /// `unique-symbol` | `none`.
    pub early_exit_route: &'static str,
    pub tokens: u64,
    pub llm_calls: u32,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub forced_finish: bool,
    pub findings_count: usize,
    pub summary_len: usize,
    pub git_probe_ms: u64,
    pub total_ms: u64,
    /// Estimated size of the repo brief injected into Stage 5. `None` when
    /// Stage 5 never ran or produced no brief — never a fabricated `0`.
    pub brief_tokens: Option<u32>,
    /// In-loop `get_architecture` calls actually executed. Seeded to `Some(0)`
    /// on Stage-5 entry, so a real zero is distinguishable from "Stage 5
    /// never ran" (`None`).
    pub orientation_calls_in_loop: Option<u32>,
    /// Which cache layer served this run: `l1` (in-memory) | `l2` (on-disk).
    /// `None` on every non-cache path — never a fabricated value. `eval`'s
    /// scorer derives the `cache_hit_l1` / `cache_hit_l2` rates from it; two
    /// always-false booleans would ride every non-hit log line instead.
    pub cache_layer: Option<&'static str>,
    /// `llm_calls` the cached run spent, i.e. the turns this hit avoided.
    /// `None` unless served from cache.
    pub turns_saved_by_cache: Option<u32>,
    /// Tokens the cached run spent. `None` unless served from cache, and
    /// deliberately *not* folded into `tokens`, which has to keep meaning
    /// "spend" or the cost aggregate breaks.
    pub tokens_saved_by_cache: Option<u64>,
    /// How many findings of the accepted `finish` call carried a
    /// `candidate_id` that resolved into the numbered registry AND supplied
    /// the location — counted at parse time, i.e. BEFORE `finalize`'s dedupe
    /// and `max_results` cap, so it can exceed `findings_count`. `None` on
    /// every path where no `finish` call ran (cache, early exit, the
    /// no-finish synthesis). A run of zeroes across an eval says the models
    /// ignore the field and it should be reverted.
    pub cited_candidate_ids: Option<u32>,
}

impl QueryMetrics {
    /// `started` doubles as the run-start instant (it is the same `Instant`
    /// the git probe is timed from, the first statement of `run`).
    fn new(repo_root: &Path, query: &ExplorationQuery, started: Instant) -> Self {
        Self {
            started,
            ts_unix_ms: 0,
            repo_path: repo_root.to_string_lossy().into_owned(),
            query: query.text.clone(),
            scope_hint: query
                .scope_hint
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            max_results: query.max_results,
            path: "none",
            // The cache path exits before Stage 1 ever runs.
            index_status: "NotChecked",
            confidence: None,
            candidate_count: None,
            early_exit_route: "none",
            tokens: 0,
            llm_calls: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            forced_finish: false,
            findings_count: 0,
            summary_len: 0,
            git_probe_ms: 0,
            total_ms: 0,
            brief_tokens: None,
            orientation_calls_in_loop: None,
            cache_layer: None,
            turns_saved_by_cache: None,
            tokens_saved_by_cache: None,
            cited_candidate_ids: None,
        }
    }

    fn record_result(&mut self, path: &'static str, result: &ExplorationResult) {
        self.path = path;
        self.findings_count = result.findings.len();
        self.summary_len = result.summary.len();
    }

    fn record_budget(&mut self, budget: &TokenBudget, forced_finish: bool) {
        self.tokens = budget.spent();
        self.llm_calls = budget.llm_calls();
        self.cache_read_tokens = budget.cache_read_tokens();
        self.cache_write_tokens = budget.cache_write_tokens();
        self.forced_finish = forced_finish;
    }
}

/// Stamp the wall-clock fields and emit the record: the headline fields on
/// `msg`'s INFO line (what `eval/run.py` parses), and the full record to the
/// optional `REPO_EXPLORER_METRICS` JSONL file.
async fn emit_metrics(metrics: &mut QueryMetrics, msg: &'static str) {
    metrics.ts_unix_ms = now_unix_ms();
    metrics.total_ms = metrics.started.elapsed().as_millis() as u64;
    tracing::info!(
        path = metrics.path,
        tokens = metrics.tokens,
        llm_calls = metrics.llm_calls,
        forced_finish = metrics.forced_finish,
        index_status = metrics.index_status,
        git_probe_ms = metrics.git_probe_ms,
        confidence = metrics.confidence,
        candidate_count = metrics.candidate_count,
        early_exit_route = metrics.early_exit_route,
        cache_read_tokens = metrics.cache_read_tokens,
        cache_write_tokens = metrics.cache_write_tokens,
        brief_tokens = metrics.brief_tokens,
        orientation_calls_in_loop = metrics.orientation_calls_in_loop,
        cache_layer = metrics.cache_layer,
        turns_saved_by_cache = metrics.turns_saved_by_cache,
        tokens_saved_by_cache = metrics.tokens_saved_by_cache,
        cited_candidate_ids = metrics.cited_candidate_ids,
        total_ms = metrics.total_ms,
        "{}",
        msg
    );
    // Skip the serialization entirely when the sink is disabled (the common
    // case): `append_metrics_line` would otherwise discard the string it
    // just paid to build on every single query.
    if METRICS_SINK.is_some() {
        let json = serde_json::to_string(&metrics).unwrap_or_default();
        append_metrics_line(json).await;
    }
    #[cfg(test)]
    EMITTED.with(|v| v.borrow_mut().push(metrics.clone()));
}

/// Stamp every distinct file the answer references, for `key_mode = "paths"`.
/// At most `max_results` `stat` calls, and only after the run has already
/// produced its result — never on the request path.
async fn collect_deps(repo_root: &Path, result: &ExplorationResult) -> Vec<FileDep> {
    let mut paths: Vec<String> = Vec::new();
    for finding in &result.findings {
        let path = finding.location.path.to_string_lossy().into_owned();
        if !paths.contains(&path) {
            paths.push(path);
        }
    }
    if paths.is_empty() {
        return Vec::new();
    }
    let repo_root = repo_root.to_path_buf();
    tokio::task::spawn_blocking(move || {
        // All or nothing: a partial stamp set would be indistinguishable from
        // a complete one at validation time (`entry_still_valid` only tests
        // for emptiness), so the unstamped files would never be checked again.
        // Empty falls back to `strict`, which is the safe policy.
        paths
            .iter()
            .map(|p| disk_cache::file_dep(&repo_root, p))
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default()
    })
    .await
    .unwrap_or_default()
}

/// Every stamped file still has its store-time `(len, mtime)`. A file that
/// vanished or became unreadable counts as changed.
async fn deps_unchanged(repo_root: &Path, deps: &[FileDep]) -> bool {
    let repo_root = repo_root.to_path_buf();
    let deps = deps.to_vec();
    tokio::task::spawn_blocking(move || {
        deps.iter()
            .all(|dep| disk_cache::file_dep(&repo_root, &dep.path).as_ref() == Some(dep))
    })
    .await
    .unwrap_or(false)
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Append one JSONL record to `REPO_EXPLORER_METRICS`, when set. Non-fatal:
/// a sink failure must never fail a query. Never stdout — that is the MCP
/// JSON-RPC channel.
///
/// Runs the blocking `open`+`write_all` via `spawn_blocking`, like the
/// filesystem syscalls in `dispatch.rs` (`canonical_repo_root`,
/// `read_file_canonical`): a slow or contended sink path must not stall the
/// tokio worker thread and therefore every other task sharing it, including
/// unrelated concurrent `explore_repository` calls.
async fn append_metrics_line(json: String) {
    let Some(path) = METRICS_SINK.clone() else {
        return;
    };
    let result = tokio::task::spawn_blocking(move || {
        let result = append_json_line(&path, &json);
        (path, result)
    })
    .await;
    let (path, result) = match result {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "metrics sink write task panicked");
            return;
        }
    };
    let _ = result.inspect_err(
        |e| tracing::warn!(path = %path.display(), error = %e, "metrics sink write failed"),
    );
}

/// Create-or-append one `\n`-terminated line. Never truncates: a restarted
/// server keeps extending the same JSONL file. Record and newline go out in
/// ONE `write_all`: `O_APPEND` makes a single write atomic, but not a pair,
/// so two concurrent queries would otherwise interleave as `{a}{b}\n\n`.
fn append_json_line(path: &Path, json: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(format!("{json}\n").as_bytes())
}

#[cfg(test)]
thread_local! {
    /// Test-only tap on `emit_metrics`, so each exit path can assert the
    /// record it produced without standing up a tracing subscriber.
    /// `#[tokio::test]` is single-threaded, so this stays per-test.
    static EMITTED: std::cell::RefCell<Vec<QueryMetrics>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The generic exploration orchestrator. Owns `memory`, `search`, the
/// `router` (which owns its providers), and the repo-state `probe` — static
/// dispatch, mirroring `ProviderRouter`.
pub struct AgentLoop<M, S, P, R, C = SystemClock>
where
    M: MemoryBackend,
    S: SearchBackend,
    P: LlmProvider,
    R: RepoStateProbe,
    C: Clock,
{
    memory: M,
    search: S,
    router: ProviderRouter<P, C>,
    probe: R,
    settings: AgentSettings,
    cache: Option<ResultCache>,
    caps: RenderCaps,
    /// Advanced once per `run` call to hand each conversation a fresh
    /// `CallOptions::rotation_seed` (see `TokenBudget::rotation_seed`).
    rotation_seed: std::sync::atomic::AtomicUsize,
    /// Per-`repo_root` git fingerprint + monotonic instant of the last
    /// usable `ensure_fresh_index`, so repeat calls on an unchanged repo
    /// skip the upstream freshness round-trips. Bounded/FIFO-evicting like
    /// the sibling caches in `cache.rs`, capped by the same
    /// `cache_settings.max_entries` (independent of whether the query cache
    /// itself is enabled).
    index_refresh_seen: std::sync::Mutex<CappedMap<IndexRefreshMark>>,
    /// Trust window for `index_refresh_seen`, sourced from
    /// `codebase_memory.staleness_seconds`.
    index_trust_ttl: Duration,
    /// How a cached entry is proved still valid once the fingerprint moved.
    /// One policy, applied identically to both cache layers.
    cache_key_mode: CacheKeyMode,
}

impl<M, S, P, R, C> AgentLoop<M, S, P, R, C>
where
    M: MemoryBackend,
    S: SearchBackend,
    P: LlmProvider,
    R: RepoStateProbe,
    C: Clock,
{
    pub fn new(
        memory: M,
        search: S,
        router: ProviderRouter<P, C>,
        probe: R,
        settings: AgentSettings,
        cache_settings: CacheSettings,
        index_trust_ttl: Duration,
    ) -> Self {
        // The on-disk L2 is opt-in twice over (`enabled` kills both layers,
        // `persistent` only the disk one) and degrades to `None` — i.e. exactly
        // today's L1-only behaviour — whenever the directory is unset or
        // unusable. `dir` arrives already resolved: XDG lookup is a
        // binary-boundary concern, so no crate below it touches `dirs`.
        let disk = (cache_settings.enabled && cache_settings.persistent)
            .then(|| {
                disk_cache::DiskCache::open(
                    &cache_settings.dir,
                    cache_settings.persistent_max_bytes,
                )
            })
            .flatten()
            .map(std::sync::Arc::new);
        let cache = cache_settings
            .enabled
            .then(|| ResultCache::new(cache_settings.max_entries, disk));
        let caps = RenderCaps {
            snippet_max_chars: settings.snippet_max_chars as usize,
            ..RenderCaps::default()
        };
        Self {
            memory,
            search,
            router,
            probe,
            settings,
            cache,
            caps,
            rotation_seed: std::sync::atomic::AtomicUsize::new(0),
            index_refresh_seen: std::sync::Mutex::new(CappedMap::new(cache_settings.max_entries)),
            index_trust_ttl,
            cache_key_mode: cache_settings.key_mode,
        }
    }

    /// Deterministic per-query cache key, exposed so a caller (the MCP
    /// server's `explore` observability span) can derive a short
    /// request-correlation id from the same normalization the query cache
    /// uses internally. Correlation only, so it deliberately omits the
    /// config-derived suffix [`Self::run_query_key`] adds: a `req_id` must
    /// stay stable per query, not per cache entry.
    pub fn query_cache_key(repo_root: &Path, query: &ExplorationQuery) -> String {
        ResultCache::query_key(repo_root, query)
    }

    /// The loop's own key: [`Self::query_cache_key`] plus the snippet cap this
    /// loop would apply. A stored result is already truncated by
    /// `tidy_and_truncate`, and an L2 entry outlives the config file that set
    /// that cap — so without this, raising (or lowering) `snippet_max_chars`
    /// would be silently ignored for every persisted query, with `cache clear`
    /// as the only way out. `SCHEMA_VERSION` versions the stored *shape*, not
    /// the settings that produced the value.
    fn run_query_key(&self, repo_root: &Path, query: &ExplorationQuery) -> String {
        let mut key = ResultCache::query_key(repo_root, query);
        crate::cache::encode_field_into(
            &mut key,
            &self.response_caps(query).snippet_max_chars.to_string(),
        );
        key
    }

    pub async fn run(
        &self,
        repo_root: &Path,
        query: &ExplorationQuery,
    ) -> Result<ExplorationOutcome, AgentLoopError> {
        // Stage 0: query cache. `git_probe_start` doubles as the run-start
        // instant for `QueryMetrics::total_ms`.
        let git_probe_start = std::time::Instant::now();
        let fingerprint = self.probe.fingerprint(repo_root).await;
        let mut metrics = QueryMetrics::new(repo_root, query, git_probe_start);
        metrics.git_probe_ms = git_probe_start.elapsed().as_millis() as u64;
        let query_key = self.run_query_key(repo_root, query);
        if let Some((hit, layer)) = self
            .query_cache_lookup(repo_root, &query_key, &fingerprint)
            .await
        {
            metrics.record_result("cache", &hit.result.result);
            metrics.cache_layer = Some(layer);
            metrics.turns_saved_by_cache = Some(hit.llm_turns);
            metrics.tokens_saved_by_cache = Some(hit.tokens);
            emit_metrics(&mut metrics, "exploration served from query cache").await;
            // Only the exit path is rewritten: `retrieval_confidence` and
            // `symbols` are the stored run's, which is exactly what a hit
            // replays.
            return Ok(ExplorationOutcome {
                stage_exit: StageExit::Cache,
                ..hit.result
            });
        }

        // Stage 1: ensure a fresh index (once). Failures are non-fatal notes.
        // Repeat calls on an unchanged repo (same git fingerprint, still within
        // the trust window) skip the upstream freshness round-trip.
        let now = Instant::now();
        let index_key = repo_root.to_string_lossy().into_owned();
        let skip_candidate = {
            let seen = self
                .index_refresh_seen
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            may_skip_index_refresh(
                fingerprint.as_ref(),
                seen.get(&index_key).as_ref(),
                now,
                self.index_trust_ttl,
            )
        };
        // A skip candidate is still safety-netted by a cheap existence probe:
        // if the upstream index was lost/invalidated for a reason the local
        // git fingerprint can't see (daemon restart, external deletion), fall
        // back to the full flow instead of trusting the fingerprint alone.
        let skip =
            skip_candidate && matches!(self.memory.probe_index_ready(repo_root).await, Ok(true));
        let index_result = if skip {
            Ok(IndexStatus::UpToDate)
        } else {
            let r = self.memory.ensure_fresh_index(repo_root).await;
            if let (Ok(IndexStatus::UpToDate | IndexStatus::Reindexed), Some(fp)) =
                (&r, &fingerprint)
            {
                self.index_refresh_seen
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(
                        index_key,
                        IndexRefreshMark {
                            fingerprint: fp.clone(),
                            at: now,
                        },
                    );
            }
            r
        };
        // Distinguish a synthesized skip from a real backend round-trip in the
        // "exploration complete" log line — both otherwise map to "UpToDate".
        metrics.index_status = if skip {
            "UpToDateSkipped"
        } else {
            index_status_label(&index_result)
        };
        let index_note = match index_result {
            Ok(IndexStatus::Reindexed) | Ok(IndexStatus::UpToDate) => None,
            Ok(IndexStatus::IndexingFailed { reason }) => Some(format!(
                "Note: the memory index could not be refreshed ({reason}); memory results may be stale."
            )),
            Err(e) => Some(format!(
                "Note: the memory backend is unavailable ({e}); rely on the grep/find/read_file search tools."
            )),
        };

        // Stage 2: deterministic retrieval — no LLM.
        let leg_cache = self.cache_for(fingerprint.as_ref());
        let outcome = pipeline::retrieve(
            &self.memory,
            &self.search,
            repo_root,
            query,
            self.settings.top_k,
            leg_cache,
        )
        .await;
        tracing::debug!(
            candidates = %candidates_json(&outcome.candidates),
            "retrieval candidates"
        );
        tracing::info!(
            candidates = outcome.candidates.len(),
            confidence = outcome.confidence,
            "retrieval pre-stage complete"
        );
        metrics.confidence = Some(outcome.confidence);
        metrics.candidate_count = Some(outcome.candidates.len());

        // If the top-level scope_hint escaped the repo root, `outcome` already
        // dropped it for every leg (`pipeline::retrieve` computes this once);
        // surface a generic caveat so the returned summary/prose stops
        // pretending the hint was honored. Generic (not the raw value)
        // because the cache key coalesces all escaping hints, so a
        // value-specific note could be served for a different input; the
        // specific value stays in the per-run WARN log.
        let scope_note = outcome.scope_hint_escaped.then(|| {
            "Note: the requested scope hint escapes the repository root and was \
             ignored; the search covered the entire repository."
                .to_string()
        });
        let note: Option<String> = {
            let combined = [index_note, scope_note]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" ");
            (!combined.is_empty()).then_some(combined)
        };

        // One fresh seed per conversation — reused (via `budget`) by every
        // raw provider call the verification stage and the fallback loop
        // make for this `run`, so a rotating entry's start model can't move
        // mid-conversation (see `TokenBudget::rotation_seed`).
        let rotation_seed = self
            .rotation_seed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut budget = TokenBudget::new(self.settings.token_budget, rotation_seed);

        // Stage 3: early exit — the pre-stage already answered. Two ways in,
        // both requiring a trusted exact symbol match: high confidence alone
        // isn't proof the query named a real code symbol — a coincidental
        // path/prose match must not early-exit, even though a genuine exact
        // symbol match should, whatever the matched token's case/shape (F-16).
        //
        // - "confidence": confidence clears the threshold and *some* ranked
        //   candidate is a trusted exact symbol match.
        // - "unique-symbol" (QW-2): *exactly one* ranked candidate is, at a
        //   known location. That is unambiguous by construction, so the
        //   confidence score — which a strong SymbolFuzzy runner-up deflates
        //   well below the threshold — is not consulted at all.
        let early_exit_route = early_exit_route(&outcome, &self.settings);
        if let Some(route) = early_exit_route {
            let result = self
                .result_from_candidates(
                    &outcome.candidates,
                    query,
                    outcome.confidence,
                    note.as_deref(),
                    repo_root,
                )
                .await;
            // The unique-symbol route is authorized by ONE specific
            // candidate. If disk verification or the response caps dropped
            // exactly that one (stale line past EOF, renamed file,
            // `max_results` truncation), the survivors are unrelated matches
            // nothing vetted — so the authorization is void and the run has
            // to pay for verification after all.
            let authorized = match outcome.unique_trusted_symbol {
                Some(index) if route == "unique-symbol" => {
                    candidate_survived(&result.findings, &outcome.candidates[index])
                }
                _ => !result.findings.is_empty(),
            };
            if authorized {
                // Logged only now that verification is actually being
                // skipped: the "unique-symbol" route above is provisional
                // until its one authorizing candidate is confirmed to have
                // survived disk verification / the response caps (just
                // above) — logging any earlier would over-count runs that
                // actually fell through to Stage 4 (see `eval/run.py`'s
                // early-exit log parsing).
                if route == "unique-symbol" {
                    let index = outcome
                        .unique_trusted_symbol
                        .expect("route == \"unique-symbol\" implies Some(index)");
                    tracing::info!(
                        confidence = outcome.confidence,
                        symbol = outcome.candidates[index].symbol.as_deref().unwrap_or(""),
                        "early-exit: sole trusted exact symbol match, skipping verification"
                    );
                }
                metrics.early_exit_route = route;
                return Ok(self
                    .complete_run(
                        repo_root,
                        &mut metrics,
                        StageExit::EarlyExit,
                        &budget,
                        false,
                        &query_key,
                        fingerprint,
                        result,
                        &outcome.candidates,
                    )
                    .await);
            }
            tracing::info!(
                "early-exit produced no filesystem-verified candidate; falling through to verification"
            );
            // fall through to Stage 4
        } else if outcome.confidence >= self.settings.early_exit_confidence
            && !outcome.candidates.is_empty()
        {
            tracing::info!(
                confidence = outcome.confidence,
                "early-exit vetoed: no trusted exact symbol match"
            );
        }

        // Stage 4: LLM verification over the candidates.
        if outcome.confidence >= self.settings.fallback_confidence && !outcome.candidates.is_empty()
        {
            if let VerifyOutcome::Finished(result, cited) = verify(
                &self.memory,
                &self.router,
                repo_root,
                query,
                outcome.scope_hint_escaped,
                note.as_deref(),
                &outcome.candidates,
                self.settings.max_verify_iterations,
                &mut budget,
                &self.caps,
            )
            .await
            {
                metrics.cited_candidate_ids = Some(cited);
                return Ok(self
                    .finalize_and_complete(
                        repo_root,
                        &mut metrics,
                        StageExit::Verify,
                        result,
                        query,
                        &budget,
                        false,
                        &query_key,
                        fingerprint,
                        &outcome.candidates,
                    )
                    .await);
            }
            tracing::info!("verification escalated to the fallback loop");
        }

        // Stage 5: explorative fallback loop. The only hard-error exit —
        // matched rather than `?`d so it emits a metrics record too.
        let looped = self
            .fallback_loop(
                repo_root,
                query,
                outcome.scope_hint_escaped,
                note.as_deref(),
                &outcome.candidates,
                fingerprint.as_ref(),
                &mut budget,
                &mut metrics,
            )
            .await;
        let (result, forced_finish, cited) = match looped {
            Ok(v) => v,
            Err(e) => {
                metrics.path = "error";
                metrics.record_budget(&budget, false);
                emit_metrics(&mut metrics, "exploration complete").await;
                return Err(e);
            }
        };
        metrics.cited_candidate_ids = cited;
        Ok(self
            .finalize_and_complete(
                repo_root,
                &mut metrics,
                StageExit::Fallback,
                result,
                query,
                &budget,
                forced_finish,
                &query_key,
                fingerprint,
                &outcome.candidates,
            )
            .await)
    }

    /// The shared dedupe-then-truncate contract: dedupe first so a run of
    /// colliding-sentinel findings can't push a legitimate distinct finding
    /// out of the `max_results` cap. Used by both `finalize` (verify/fallback
    /// results) and `result_from_candidates` (the early-exit path) so the two
    /// can't silently diverge.
    fn tidy_and_truncate(
        &self,
        findings: Vec<ExplorationFinding>,
        query: &ExplorationQuery,
    ) -> Vec<ExplorationFinding> {
        let mut findings = tidy_findings(findings, &self.response_caps(query));
        if let Some(max) = query.max_results {
            findings.truncate(max as usize);
        }
        findings
    }

    /// Caps for the FINAL response only. `self.caps` (the `snippet_max_chars`
    /// knob) stays the prompt-side cap on every path — a "detailed" request
    /// must not widen what the LLM is shown, or it would inflate the token
    /// cost this whole feature exists to cut.
    ///
    /// The detailed cap is a FLOOR, never a ceiling: `detailed` must not
    /// return less than `concise` when `snippet_max_chars` is configured
    /// above `snippet_max_chars_detailed` (or set to `0`, "no cap").
    fn response_caps(&self, query: &ExplorationQuery) -> RenderCaps {
        if !query.detailed_snippets {
            return self.caps;
        }
        let detailed = self.settings.snippet_max_chars_detailed as usize;
        let snippet_max_chars = if detailed == 0 || self.caps.snippet_max_chars == 0 {
            0
        } else {
            detailed.max(self.caps.snippet_max_chars)
        };
        RenderCaps {
            snippet_max_chars,
            ..self.caps
        }
    }

    /// Normalize/dedupe/cap the final findings once, whatever stage produced
    /// them — preserving their order (rank order / the model's finish order).
    /// `max_results` is enforced here (after dedupe) so it bounds every path
    /// that doesn't early-exit: verify's finish, the fallback loop's finish,
    /// its forced finish, and its no-finish synthesis all funnel through this
    /// single choke point via `finalize_and_complete`.
    fn finalize(
        &self,
        mut result: ExplorationResult,
        query: &ExplorationQuery,
    ) -> ExplorationResult {
        result.findings = self.tidy_and_truncate(result.findings, query);
        result
    }

    /// The shared tail of every `run()` branch: emit the run's metrics
    /// record, persist to the query cache, and hand back the result for the
    /// caller to wrap in `Ok`.
    #[allow(clippy::too_many_arguments)]
    async fn complete_run(
        &self,
        repo_root: &Path,
        metrics: &mut QueryMetrics,
        stage: StageExit,
        budget: &TokenBudget,
        forced_finish: bool,
        query_key: &str,
        fingerprint: Option<RepoFingerprint>,
        result: ExplorationResult,
        candidates: &[Candidate],
    ) -> ExplorationOutcome {
        metrics.record_result(stage.as_str(), &result);
        metrics.record_budget(budget, forced_finish);
        emit_metrics(metrics, "exploration complete").await;
        // The one place an `ExplorationOutcome` is built. `retrieval_confidence`
        // is the pre-stage score already recorded in `metrics.confidence` — the
        // only path where that is `None` is the cache hit, which returns long
        // before here.
        let outcome = ExplorationOutcome {
            retrieval_confidence: metrics.confidence.unwrap_or(0),
            stage_exit: stage,
            symbols: symbols_for(&result.findings, candidates),
            result,
        };
        // Awaited, not detached: deterministic for tests, and no task racing
        // process exit. Sub-millisecond, except on the one run per process
        // that also carries `DiskCache`'s eviction sweep (see its comment) —
        // never on a cache hit, which returns long before here.
        self.store_query_cache(
            repo_root,
            query_key,
            fingerprint,
            &outcome,
            budget.llm_calls(),
            budget.spent(),
        )
        .await;
        outcome
    }

    /// The shared tail of the verify/fallback branches: finalize the result,
    /// then run it through `complete_run` with the tokens spent so far.
    #[allow(clippy::too_many_arguments)]
    async fn finalize_and_complete(
        &self,
        repo_root: &Path,
        metrics: &mut QueryMetrics,
        stage: StageExit,
        result: ExplorationResult,
        query: &ExplorationQuery,
        budget: &TokenBudget,
        forced_finish: bool,
        query_key: &str,
        fingerprint: Option<RepoFingerprint>,
        candidates: &[Candidate],
    ) -> ExplorationOutcome {
        let result = self.finalize(result, query);
        self.complete_run(
            repo_root,
            metrics,
            stage,
            budget,
            forced_finish,
            query_key,
            fingerprint,
            result,
            candidates,
        )
        .await
    }

    /// The cache is usable this call only when caching is enabled and a
    /// fingerprint was obtainable this run — shared by every read path.
    /// (`store_query_cache` needs an owned fingerprint to move into the
    /// entry, so it keeps its own guard.)
    fn cache_for<'a>(&'a self, fingerprint: Option<&'a RepoFingerprint>) -> pipeline::LegCache<'a> {
        match (&self.cache, fingerprint) {
            (Some(cache), Some(fp)) => Some((cache, fp)),
            _ => None,
        }
    }

    /// The deterministic Stage-5 repo brief: one `get_architecture` round trip
    /// per cold fingerprint, rendered locally and memoized in the existing
    /// result cache. Every failure mode (disabled, backend error, unusable
    /// payload) degrades to `None`, which leaves the loop exactly as it was —
    /// a prefetch must never fail a query.
    async fn repo_brief(
        &self,
        repo_root: &Path,
        fingerprint: Option<&RepoFingerprint>,
    ) -> Option<String> {
        // `max_tokens == 0` is the documented budget opt-out — honour it here,
        // before the round trip it is meant to avoid.
        if !self.settings.repo_brief.enabled || self.settings.repo_brief.max_tokens == 0 {
            return None;
        }
        let cached = self.cache_for(fingerprint).map(|(cache, fp)| {
            (
                cache,
                ResultCache::brief_key(
                    repo_root,
                    fp,
                    self.settings.repo_brief.key == RepoBriefKey::Head,
                ),
            )
        });
        if let Some((cache, key)) = &cached
            && let Some(hit) = cache.get_brief(key)
        {
            return Some(hit);
        }
        let started = Instant::now();
        let text = match self.memory.get_architecture_text(repo_root).await {
            Ok(text) => text,
            Err(e) => {
                tracing::debug!(error = %e, "repo brief prefetch failed");
                return None;
            }
        };
        let brief = brief::render_brief(&text, self.settings.repo_brief.max_tokens)?;
        tracing::debug!(
            brief_tokens = brief::estimate_tokens(&brief),
            build_ms = started.elapsed().as_millis() as u64,
            "repo brief built"
        );
        if let Some((cache, key)) = cached {
            cache.put_brief(key, brief.clone());
        }
        Some(brief)
    }

    /// The one validity policy, shared by both cache layers so they can never
    /// diverge.
    ///
    /// `strict` (the default) serves only what is provably unchanged: the same
    /// fingerprint, or a fingerprint change with a provably empty diff.
    /// Checking the diff against only the *entry's own* contributing paths is
    /// unsound in general — retrieval scans the whole repo, so a path outside
    /// those paths (not least a newly added file) can still turn into a better
    /// match that the stale entry never saw — so under `strict` any actual
    /// diff invalidates.
    ///
    /// `paths` accepts that recall ceiling knowingly, in exchange for surviving
    /// an unrelated working-tree edit: the answer is served when every file it
    /// references still has its store-time `(len, mtime)`. A newly added file
    /// that would have been the better match is missed — which is why it is
    /// opt-in, not the default. An entry with no stamps (stored under `strict`,
    /// or an answer with no findings) falls back to `strict`, so a
    /// summary-only entry can never be served forever.
    ///
    /// A `head`-only mode is deliberately absent: it is the one policy that
    /// serves snippets an uncommitted edit already invalidated.
    async fn entry_still_valid(
        &self,
        repo_root: &Path,
        entry: &QueryEntry,
        fp: &RepoFingerprint,
    ) -> bool {
        if entry.fingerprint == *fp {
            return true;
        }
        if self.cache_key_mode == CacheKeyMode::Paths && !entry.deps.is_empty() {
            return deps_unchanged(repo_root, &entry.deps).await;
        }
        matches!(
            self.probe
                .changed_paths(repo_root, &entry.fingerprint, fp)
                .await,
            Some(changed) if changed.is_empty()
        )
    }

    /// Serve from the query cache, L1 first and the on-disk L2 behind it.
    /// Returns the whole entry (so the caller can report the turns/tokens the
    /// hit saved) plus which layer served it.
    async fn query_cache_lookup(
        &self,
        repo_root: &Path,
        query_key: &str,
        fingerprint: &Option<RepoFingerprint>,
    ) -> Option<(QueryEntry, &'static str)> {
        let (cache, fp) = self.cache_for(fingerprint.as_ref())?;
        if let Some(entry) = cache.get_query(query_key) {
            if entry.fingerprint == *fp {
                return Some((entry, "l1"));
            }
            if self.entry_still_valid(repo_root, &entry, fp).await {
                cache.refresh_query_fingerprint(query_key, &entry.fingerprint, fp.clone());
                return Some((entry, "l1"));
            }
            cache.remove_query(query_key, &entry.fingerprint);
        }
        let entry = cache.get_query_l2(query_key).await?;
        if !self.entry_still_valid(repo_root, &entry, fp).await {
            return None;
        }
        // Promote into L1, relabelled to the current fingerprint (validation
        // just proved the answer holds for it). The dep stamps are left alone:
        // they are store-time truth about the files the answer references.
        cache.put_query(
            query_key.to_string(),
            QueryEntry {
                fingerprint: fp.clone(),
                ..entry.clone()
            },
        );
        Some((entry, "l2"))
    }

    async fn store_query_cache(
        &self,
        repo_root: &Path,
        query_key: &str,
        fingerprint: Option<RepoFingerprint>,
        outcome: &ExplorationOutcome,
        llm_turns: u32,
        tokens: u64,
    ) {
        let (Some(cache), Some(fingerprint)) = (&self.cache, fingerprint) else {
            return;
        };
        let deps = if self.cache_key_mode == CacheKeyMode::Paths {
            collect_deps(repo_root, &outcome.result).await
        } else {
            Vec::new()
        };
        let entry = QueryEntry {
            fingerprint,
            result: outcome.clone(),
            llm_turns,
            tokens,
            deps,
        };
        cache.put_query(query_key.to_string(), entry.clone());
        cache.put_query_l2(query_key, &entry).await;
    }

    /// Build the early-exit result from the ranked candidates, verifying each
    /// against the live file first: drop any candidate whose path is missing
    /// or whose `line_start` is past EOF, and clamp `line_end` on survivors —
    /// but keep the backend-provided snippet (it is not LLM-authored, so the
    /// finish path's F-18 disk-re-derivation does not apply). The canonical
    /// root is resolved once for the whole batch; if it cannot be resolved
    /// the repo is unreadable, so every candidate is dropped and the caller
    /// falls through to the LLM stages. `index_note` (e.g. a failed reindex)
    /// is appended to the summary — this is the only stage that doesn't
    /// thread it into an LLM prompt.
    ///
    /// Several candidates commonly point at the same file (e.g. multiple
    /// symbol/content hits in one large source file), so the per-path
    /// read+line-count ([`read_verified_file`]) is cached for the batch
    /// instead of re-reading a shared file once per candidate.
    async fn result_from_candidates(
        &self,
        candidates: &[Candidate],
        query: &ExplorationQuery,
        confidence: u32,
        index_note: Option<&str>,
        repo_root: &Path,
    ) -> ExplorationResult {
        let canonical_root = canonical_repo_root(repo_root).await.ok();
        let mut verified = Vec::with_capacity(candidates.len());
        if let Some(root) = &canonical_root {
            let mut line_counts: HashMap<PathBuf, Result<u32, String>> = HashMap::new();
            for candidate in candidates {
                let location = normalize_location(candidate.location.clone());
                let line_count = match line_counts.get(&location.path) {
                    Some(cached) => cached.clone(),
                    None => {
                        let result = read_verified_file(&location.path, repo_root, root)
                            .await
                            .map(|(_content, line_count)| line_count);
                        line_counts.insert(location.path.clone(), result.clone());
                        result
                    }
                };
                match line_count.and_then(|line_count| clamp_location(location, line_count)) {
                    Ok(location) => {
                        // Keep the backend-provided snippet (not LLM-authored,
                        // so F-18 does not apply); only the location is verified.
                        let mut candidate = candidate.clone();
                        candidate.location = location;
                        verified.push(finding_from_candidate(candidate));
                    }
                    Err(reason) => {
                        tracing::debug!(reason = %reason, "early-exit dropped an unverifiable candidate")
                    }
                }
            }
        }
        let findings = self.tidy_and_truncate(verified, query);
        let mut summary = format!(
            "Resolved deterministically by the retrieval pre-stage (confidence {confidence}/100, no LLM involved): {} location(s) matching \"{}\".",
            findings.len(),
            query.text
        );
        if let Some(note) = index_note {
            summary.push(' ');
            summary.push_str(note);
        }
        ExplorationResult { findings, summary }
    }

    /// The explorative loop, now the low-confidence escalation path: hard turn
    /// and token budgets, 2-strike batch enforcement, concurrent batch
    /// execution, tool-result memoization, and a forced final `finish`.
    ///
    /// Returns the result plus whether it came from the forced-finish path or
    /// the deterministic synthesis fallback (`true`), as opposed to a normal
    /// in-loop `finish` call (`false`) — logged on `exploration complete`.
    #[allow(clippy::too_many_arguments)]
    async fn fallback_loop(
        &self,
        repo_root: &Path,
        query: &ExplorationQuery,
        scope_hint_escaped: bool,
        index_note: Option<&str>,
        candidates: &[Candidate],
        fingerprint: Option<&RepoFingerprint>,
        budget: &mut TokenBudget,
        metrics: &mut QueryMetrics,
    ) -> Result<(ExplorationResult, bool, Option<u32>), AgentLoopError> {
        // Stage 5 starts blind otherwise: the first turns go on orientation
        // (get_architecture) that one deterministic call answers for free.
        // Stage 4 deliberately gets nothing — it is 75% of runs. Skipped when
        // no exploratory turn will run (budget already spent upstream, or zero
        // iterations configured): the prefetch would only bill the
        // forced-finish call for a brief nothing can act on.
        let repo_brief = if budget.exhausted() || self.settings.max_fallback_iterations == 0 {
            None
        } else {
            self.repo_brief(repo_root, fingerprint).await
        };
        metrics.brief_tokens = repo_brief
            .as_deref()
            .map(|b| brief::estimate_tokens(b) as u32);
        // Seeded here so every exit path of this loop carries a truthful
        // count, and the paths that never reach Stage 5 leave it absent.
        metrics.orientation_calls_in_loop = Some(0);

        let tools = tool_catalog();
        // The registry `finish`'s `candidate_id` resolves against is exactly
        // the list the prompt numbers — ids past it were never shown to the
        // model, so resolving them would substitute a range nobody inspected.
        // The full slice is still used below for the deterministic synthesis.
        let seeds = &candidates[..SEED_CANDIDATES.min(candidates.len())];
        // A second system message, not part of the user prompt: every system
        // message carries its own cache breakpoint, and the brief is stable
        // per fingerprint — so it is re-read from the provider cache on every
        // turn instead of re-billed behind the conversation's only breakpoint.
        let mut messages: Vec<Message> = vec![Message::system(FALLBACK_SYSTEM_PROMPT)];
        if let Some(b) = &repo_brief {
            messages.push(Message::system(b));
        }
        messages.push(Message::user(user_prompt(
            query,
            scope_hint_escaped,
            index_note,
            seeds,
        )));

        let mut findings: Vec<ExplorationFinding> = Vec::new();
        let mut seen: HashSet<(FileLocation, Option<String>)> = HashSet::new();
        let mut single_call_rejections = 0u32;
        let mut turn_limit_hit = true;

        for turn in 0..self.settings.max_fallback_iterations {
            if budget.exhausted() {
                turn_limit_hit = false;
                break;
            }
            let options = CallOptions {
                rotation_seed: Some(budget.rotation_seed()),
                ..Default::default()
            };
            match self
                .router
                .complete_with_tools(&messages, tools, &options)
                .await
            {
                Ok(completion) => {
                    budget.add(completion.usage);
                    match completion.response {
                        ProviderResponse::ToolCalls(calls) if calls.is_empty() => {
                            tracing::debug!(
                                turn,
                                tool_names = "",
                                rejected_single_call = false,
                                strikes = single_call_rejections,
                                budget_spent = budget.spent(),
                                "fallback turn"
                            );
                            push_nudge(
                                &mut messages,
                                Message::assistant_tool_calls(Vec::new()),
                                "You must respond with a tool call; call finish when done.",
                            );
                        }
                        ProviderResponse::ToolCalls(calls) => {
                            // Deferred: `calls` is only read via `.iter()` below (never
                            // mutated), so it's moved into the assistant message once
                            // those borrows are done instead of cloned up front. On the
                            // common immediate-finish path we return before any of that
                            // is needed, skipping the allocation entirely.
                            let mut turn_messages: Vec<Message> =
                                match resolve_finish(&calls, repo_root, seeds).await {
                                    Ok((result, cited)) => {
                                        return Ok((result, false, Some(cited)));
                                    }
                                    Err(rejections) => rejections,
                                };
                            let non_finish: Vec<&ToolCall> =
                                calls.iter().filter(|c| c.name != "finish").collect();
                            if calls.len() == 1
                                && non_finish.len() == 1
                                && single_call_rejections < MAX_SINGLE_CALL_REJECTIONS
                            {
                                single_call_rejections += 1;
                                tracing::debug!(
                                    turn,
                                    tool_names = non_finish[0].name.as_str(),
                                    rejected_single_call = true,
                                    strikes = single_call_rejections,
                                    budget_spent = budget.spent(),
                                    "fallback turn"
                                );
                                turn_messages.push(Message::tool(
                                    &non_finish[0].id,
                                    "call rejected: batch ALL independent tool calls of a turn into one response (they execute concurrently); resend this call together with the other lookups you need",
                                ));
                                messages.push(Message::assistant_tool_calls(calls));
                                messages.extend(turn_messages);
                                continue;
                            }
                            if !non_finish.is_empty() {
                                single_call_rejections = 0;
                            }
                            // Counted where the calls actually execute, so the
                            // rejected single-call turn above (which
                            // `continue`s) is not counted twice.
                            *metrics.orientation_calls_in_loop.get_or_insert(0) += non_finish
                                .iter()
                                .filter(|c| c.name == "get_architecture")
                                .count()
                                as u32;
                            tracing::debug!(
                                turn,
                                tool_names = %tool_names_joined(&non_finish),
                                rejected_single_call = false,
                                strikes = single_call_rejections,
                                budget_spent = budget.spent(),
                                "fallback turn"
                            );
                            let results =
                                futures_util::future::join_all(non_finish.iter().map(|call| {
                                    self.cached_dispatch(repo_root, call, fingerprint)
                                }))
                                .await;
                            messages.push(Message::assistant_tool_calls(calls));
                            messages.extend(turn_messages);
                            for (message, new_findings) in results {
                                messages.push(message);
                                for f in new_findings {
                                    accumulate(&mut findings, &mut seen, f);
                                }
                            }
                        }
                        ProviderResponse::Text(text) => {
                            tracing::debug!(
                                turn,
                                tool_names = "",
                                rejected_single_call = false,
                                strikes = single_call_rejections,
                                budget_spent = budget.spent(),
                                "fallback turn"
                            );
                            push_nudge(
                                &mut messages,
                                Message::assistant_text(text),
                                "You must respond with a tool call; call finish when done.",
                            );
                        }
                    }
                }
                Err(router_err) => {
                    return Err(AgentLoopError::Provider(router_err.to_string()));
                }
            }
        }

        // Budget exhausted without finish: one forced final answer, then a
        // deterministic synthesis from everything gathered so far.
        if let Some((result, cited)) = self
            .forced_finish(&mut messages, budget, repo_root, seeds)
            .await
        {
            return Ok((result, true, Some(cited)));
        }
        for candidate in candidates {
            accumulate(
                &mut findings,
                &mut seen,
                finding_from_candidate(candidate.clone()),
            );
        }
        let cause = if turn_limit_hit {
            format!(
                "iteration limit ({})",
                self.settings.max_fallback_iterations
            )
        } else {
            format!("token budget ({})", self.settings.token_budget)
        };
        Ok((
            ExplorationResult {
                findings,
                summary: format!(
                    "Exploration stopped after reaching the {cause} without an explicit finish; returning best-effort findings gathered so far."
                ),
            },
            true,
            // No `finish` call parsed on this path, so there is nothing to
            // count — `Some(0)` would read as "the model cited nothing" and
            // pad the eval's citation aggregate with zeroes.
            None,
        ))
    }

    /// One last call offering only `finish`, with the tool choice forced.
    async fn forced_finish(
        &self,
        messages: &mut Vec<Message>,
        budget: &mut TokenBudget,
        repo_root: &Path,
        candidates: &[Candidate],
    ) -> Option<(ExplorationResult, u32)> {
        messages.push(Message::user(
            "The exploration budget is exhausted. Call finish NOW with the best findings gathered so far.",
        ));
        match self
            .router
            .complete_with_tools(
                messages,
                finish_only_catalog(),
                &force_finish_options(budget),
            )
            .await
        {
            Ok(completion) => {
                budget.add(completion.usage);
                if let ProviderResponse::ToolCalls(calls) = completion.response {
                    for call in &calls {
                        if call.name != "finish" {
                            continue;
                        }
                        // Lenient: a one-shot call has no retry path to feed a
                        // rejection back to the model, so keep whatever findings
                        // validate instead of discarding all of them over one bad
                        // path (see F-05 escalation on parse_finish's strictness).
                        match parse_finish_lenient(&call.arguments_json, repo_root, candidates)
                            .await
                        {
                            Ok(result) => return Some(result),
                            Err(reason) => {
                                tracing::debug!(reason = %reason, "forced finish call failed to parse")
                            }
                        }
                    }
                }
                None
            }
            Err(e) => {
                tracing::debug!(error = %e, "forced finish call failed; synthesizing result");
                None
            }
        }
    }

    /// `dispatch_inner` behind the tool-result memo (active only with a cache
    /// and a fingerprint); only a successful dispatch is cached.
    async fn cached_dispatch(
        &self,
        repo_root: &Path,
        call: &ToolCall,
        fingerprint: Option<&RepoFingerprint>,
    ) -> (Message, Vec<ExplorationFinding>) {
        let key = self.cache_for(fingerprint).map(|(cache, fp)| {
            (
                cache,
                ResultCache::tool_key(repo_root, fp, &call.name, &call.arguments_json),
            )
        });
        if let Some((cache, key)) = &key
            && let Some((content, findings)) = cache.get_tool(key)
        {
            return (Message::tool(&call.id, content), findings);
        }
        // Only a successful call is memoized — a failure (subprocess/RPC
        // error) is typically transient and must be retried, not replayed.
        match dispatch_inner(&self.memory, &self.search, repo_root, call, &self.caps).await {
            Ok((content, findings)) => {
                if let Some((cache, key)) = key {
                    cache.put_tool(key, (content.clone(), findings.clone()));
                }
                (Message::tool(&call.id, content), findings)
            }
            Err(msg) => (Message::tool(&call.id, msg), Vec::new()),
        }
    }
}

/// One recorded successful `ensure_fresh_index`: the git fingerprint at that
/// point and the monotonic instant it ran.
#[derive(Clone)]
struct IndexRefreshMark {
    fingerprint: RepoFingerprint,
    at: Instant,
}

/// True when a repeat call may skip `ensure_fresh_index`: the repo's git
/// fingerprint is known and identical to the last refresh, and that refresh
/// is still within the trust window. Any `None`/unknown input is
/// conservative (returns false -> run the full flow).
fn may_skip_index_refresh(
    current: Option<&RepoFingerprint>,
    mark: Option<&IndexRefreshMark>,
    now: Instant,
    ttl: Duration,
) -> bool {
    match (current, mark) {
        (Some(fp), Some(m)) => *fp == m.fingerprint && now.saturating_duration_since(m.at) < ttl,
        _ => false,
    }
}

/// Map `ensure_fresh_index`'s result onto a short label for the
/// `exploration complete` log line — `Unavailable` covers the `Err` arm,
/// which carries no `IndexStatus` value of its own.
fn index_status_label(result: &Result<IndexStatus, MemoryError>) -> &'static str {
    match result {
        Ok(IndexStatus::Reindexed) => "Reindexed",
        Ok(IndexStatus::UpToDate) => "UpToDate",
        Ok(IndexStatus::IndexingFailed { .. }) => "IndexingFailed",
        Err(_) => "Unavailable",
    }
}

/// The Stage-3 early-exit route choice, factored out of `AgentLoop::run` so the
/// deterministic `snapshot::retrieval_snapshot` classifier can reuse the exact
/// same rule. Pure; no behaviour change from the inlined form.
///
/// - `"confidence"`: confidence clears `early_exit_confidence` and some ranked
///   candidate is a trusted exact symbol match.
/// - `"unique-symbol"`: exactly one ranked candidate is, at a known location
///   (and the escape hatch `skip_verify_on_exact_symbol` is on).
/// - `None`: no early exit; the `authorized` re-check still stays in `run`.
pub(crate) fn early_exit_route(
    outcome: &pipeline::RetrievalOutcome,
    settings: &AgentSettings,
) -> Option<&'static str> {
    if outcome.candidates.is_empty() {
        None
    } else if outcome.confidence >= settings.early_exit_confidence && outcome.has_exact_symbol_match
    {
        Some("confidence")
    } else if settings.skip_verify_on_exact_symbol && outcome.unique_trusted_symbol.is_some() {
        Some("unique-symbol")
    } else {
        None
    }
}

/// Did `candidate` itself survive into the final findings? Path plus
/// `line_start` identify it: filesystem verification only clamps `line_end`
/// (a `line_start` past EOF drops the candidate outright) and tidying only
/// normalizes the path spelling. Used by the unique-symbol early exit, whose
/// authorization is void if the one candidate that granted it was dropped.
fn candidate_survived(findings: &[ExplorationFinding], candidate: &Candidate) -> bool {
    let location = normalize_location(candidate.location.clone());
    let path = normalize_rel_path(location.path);
    findings
        .iter()
        .any(|f| f.location.path == path && f.location.line_start == location.line_start)
}

/// One JSON-array line of the ranked candidate list — rank, kind, score,
/// path, line range, symbol — for the `retrieval candidates` debug log, so a
/// harness can compute candidate-recall-at-top-k without re-deriving the
/// ranking itself.
fn candidates_json(candidates: &[Candidate]) -> String {
    let dump: Vec<serde_json::Value> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            serde_json::json!({
                "rank": i + 1,
                "kind": format!("{:?}", c.kind),
                "score": c.score,
                "path": c.location.path.to_string_lossy(),
                "line_start": c.location.line_start,
                "line_end": c.location.line_end,
                "symbol": c.symbol,
            })
        })
        .collect();
    serde_json::to_string(&dump).unwrap_or_default()
}

/// Comma-joined tool names for the `fallback turn` debug log.
fn tool_names_joined(calls: &[&ToolCall]) -> String {
    calls
        .iter()
        .map(|c| c.name.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

/// Push `f` unless a finding with the same dedupe key is already present
/// (first-seen snippet/note wins). Keys on `render::dedupe_key` — location,
/// disambiguated by note for the "unknown location" `(0, 0)` sentinel — so
/// this can't diverge from `tidy_findings`'s later, note-aware dedup and
/// collapse distinct same-file findings that only lack line info. `seen`
/// mirrors the keys already in `findings`, so the check is O(1) rather than a
/// linear scan per incoming finding.
fn accumulate(
    findings: &mut Vec<ExplorationFinding>,
    seen: &mut HashSet<(FileLocation, Option<String>)>,
    f: ExplorationFinding,
) {
    if seen.insert(dedupe_key(&f)) {
        findings.push(f);
    }
}

/// Static — no per-run content, so the provider-side prompt cache gets a
/// stable prefix. The index note moved into the user message for this reason.
const FALLBACK_SYSTEM_PROMPT: &str = "You are a repository exploration agent. Use the provided tools to locate the code relevant to the user's query. \
The memory tools (search_code, search_graph, query_graph, trace_path, get_architecture, get_code_snippet) are primary and authoritative — prefer them first. \
The grep, find, and read_file tools are a supplement/fallback, to be used only when the memory tools are insufficient. \
Batch ALL independent tool calls of a turn into ONE response with multiple tool calls — they execute concurrently, and single-call turns are rejected. \
When you have gathered enough information, you MUST conclude by calling the finish tool with your findings and a summary. \
When a finding comes from one of the numbered starting points, pass its number as that finding's candidate_id.";

/// How many retrieval candidates are listed as starting points in the
/// fallback prompt — and therefore the exact registry `finish`'s
/// `candidate_id` resolves against on that leg (see `fallback_loop`).
const SEED_CANDIDATES: usize = 8;

/// Push an assistant turn followed by the user-facing nudge asking it to
/// retry with a tool call — the shape shared by the fallback loop's and the
/// verification stage's empty-tool-calls and stray-text arms.
pub(crate) fn push_nudge(messages: &mut Vec<Message>, assistant: Message, nudge: &str) {
    messages.push(assistant);
    messages.push(Message::user(nudge));
}

/// Call options that force the `finish` tool — the shape shared by this
/// loop's `forced_finish` and the verification stage's last-turn call.
/// Carries `budget`'s rotation seed so this call agrees with every other
/// call of the same conversation on a rotating entry's start model.
pub(crate) fn force_finish_options(budget: &TokenBudget) -> CallOptions {
    CallOptions {
        force_tool: Some("finish".to_string()),
        max_tokens: None,
        rotation_seed: Some(budget.rotation_seed()),
    }
}

/// The 4-part preamble shared by the fallback loop's and the verification
/// stage's user prompts: query text, scope hint, max_results, index note. A
/// scope hint that escapes the repository root (`scope_hint_escaped`, computed
/// once by `pipeline::retrieve`) is omitted rather than rendered as an
/// in-effect scope — the honest "was ignored" caveat is carried once by the
/// threaded `index_note`, so an inline tag here would duplicate (and could
/// contradict) it.
pub(crate) fn query_preamble(
    query: &ExplorationQuery,
    scope_hint_escaped: bool,
    index_note: Option<&str>,
) -> String {
    let mut s = format!("Exploration query: {}", query.text);
    if let Some(scope) = query.scope_hint.as_deref()
        && !scope_hint_escaped
    {
        s.push_str(&format!("\nScope hint: {}", scope.display()));
    }
    if let Some(max) = query.max_results {
        s.push_str(&format!("\nDesired maximum results: {max}"));
    }
    if let Some(note) = index_note {
        s.push('\n');
        s.push_str(note);
    }
    s
}

fn user_prompt(
    query: &ExplorationQuery,
    scope_hint_escaped: bool,
    index_note: Option<&str>,
    candidates: &[Candidate],
) -> String {
    let mut s = query_preamble(query, scope_hint_escaped, index_note);
    if !candidates.is_empty() {
        s.push_str(
            "\nStarting points found by the deterministic retrieval pre-stage (ranked, may be incomplete):",
        );
        // Numbered exactly like `verify::candidates_block`, so `finish`'s
        // `candidate_id` means the same `[n]` in both stages. The caller
        // passes the `SEED_CANDIDATES` prefix it also resolves ids against —
        // truncating here instead would number fewer than it resolves.
        for (idx, c) in candidates.iter().enumerate() {
            let symbol = c
                .symbol
                .as_deref()
                .map(|sym| format!(" `{sym}`"))
                .unwrap_or_default();
            if is_unknown_location(&c.location) {
                s.push_str(&format!(
                    "\n[{}] {} (location unknown){symbol}",
                    idx + 1,
                    c.location.path.display(),
                ));
            } else {
                s.push_str(&format!(
                    "\n[{}] {}:{}-{}{symbol}",
                    idx + 1,
                    c.location.path.display(),
                    c.location.line_start,
                    c.location.line_end
                ));
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::cache_prefix_fingerprint;
    use repo_explorer_core::config::AgentSettings;
    use repo_explorer_core::domain::CandidateKind;
    use repo_explorer_core::fingerprint::RepoFingerprint;
    use repo_explorer_core::fingerprint::mock::MockRepoStateProbe;
    use repo_explorer_core::llm::mock::{FakeClock, MockLlmProvider};
    use repo_explorer_core::llm::{Completion, ToolCall};
    use repo_explorer_core::memory::mock::{Call, MockMemoryBackend};
    use repo_explorer_core::search::SearchError;
    use repo_explorer_core::search::mock::MockSearchBackend;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    /// Trust-window TTL used across these tests — centralized so a future
    /// default change or edge-case TTL needs editing in one place.
    const TEST_INDEX_TRUST_TTL: Duration = Duration::from_secs(60);

    #[test]
    fn early_exit_route_selects_the_expected_branch() {
        use crate::pipeline::RetrievalOutcome;
        use repo_explorer_core::domain::{Candidate, CandidateKind, FileLocation};

        let settings = AgentSettings::default();
        let c = Candidate {
            location: FileLocation {
                path: std::path::PathBuf::from("a.rs"),
                line_start: 1,
                line_end: 1,
            },
            symbol: Some("foo".to_string()),
            kind: CandidateKind::SymbolExact,
            score: 700,
            snippet: None,
        };
        let mk =
            |cands: Vec<Candidate>, conf: u32, exact: bool, uniq: Option<usize>| RetrievalOutcome {
                candidates: cands,
                confidence: conf,
                has_exact_symbol_match: exact,
                unique_trusted_symbol: uniq,
                scope_hint_escaped: false,
            };

        // Empty candidates: never an early exit, whatever the flags say.
        assert_eq!(
            super::early_exit_route(&mk(vec![], 100, true, Some(0)), &settings),
            None
        );
        // High confidence + a trusted exact match -> "confidence".
        assert_eq!(
            super::early_exit_route(
                &mk(vec![c.clone()], settings.early_exit_confidence, true, None),
                &settings
            ),
            Some("confidence")
        );
        // Sole trusted symbol, sub-threshold confidence -> "unique-symbol".
        assert_eq!(
            super::early_exit_route(&mk(vec![c.clone()], 0, false, Some(0)), &settings),
            Some("unique-symbol")
        );
        // Neither -> None.
        assert_eq!(
            super::early_exit_route(&mk(vec![c], 0, false, None), &settings),
            None
        );
    }

    #[test]
    fn fallback_cache_prefix_is_byte_stable() {
        // Sibling of `verify::tests::verify_cache_prefix_is_byte_stable` for
        // the fallback loop's prefix; same reasoning, see that test.
        assert!(
            !FALLBACK_SYSTEM_PROMPT.contains('{'),
            "FALLBACK_SYSTEM_PROMPT must stay a plain const with no format \
             placeholder — per-run content in the prefix defeats the cache"
        );
        let names: Vec<&str> = tool_catalog().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "search_code",
                "search_graph",
                "query_graph",
                "trace_path",
                "get_architecture",
                "get_code_snippet",
                "grep",
                "find",
                "read_file",
                "finish",
            ]
        );
        // 5965 content bytes (~6.5 KB on the wire) is roughly 1.6-1.8k tokens
        // — over Anthropic's 1024-token minimum for Sonnet/Opus, under
        // Haiku's 2048.
        assert_eq!(
            cache_prefix_fingerprint(FALLBACK_SYSTEM_PROMPT, tool_catalog()),
            (5965, 4653669523128741792)
        );
    }

    fn finish_call() -> ToolCall {
        ToolCall {
            id: "c1".to_string(),
            name: "finish".to_string(),
            arguments_json:
                r#"{"findings":[{"location":{"path":"src/lib.rs","line_start":1,"line_end":2},"note":"here"}],"summary":"done"}"#
                    .to_string(),
            thought_signatures: None,
        }
    }

    /// Temp repo containing the finding paths the fallback `finish` tests
    /// reference, so path validation accepts them, via the crate's shared
    /// `test_support::temp_repo_with` fixture. Caller removes the dir.
    fn temp_repo(test: &str) -> PathBuf {
        crate::test_support::temp_repo_with(
            "agent_run",
            test,
            &[("src/lib.rs", "a\nb\nc\n"), ("src/other.rs", "d\ne\nf\n")],
        )
    }

    /// `n` numbered lines ("l1\nl2\n...") — a body long enough for a given
    /// line range to fall within the file's extent.
    fn numbered_lines(n: usize) -> String {
        (1..=n)
            .map(|i| format!("l{i}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// One single-line finding — the shape every graph-memory fixture here
    /// uses. Mirrors `pipeline`'s and `render`'s local `finding` helpers.
    fn finding(path: &str, line: u32, note: &str) -> ExplorationFinding {
        ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: line,
                line_end: line,
            },
            snippet: None,
            note: Some(note.to_string()),
        }
    }

    /// A memory backend whose graph leg returns exactly `findings`. The
    /// summary is never asserted (only `summary_len > 0`), so it is derived.
    fn graph_memory(findings: Vec<ExplorationFinding>) -> MockMemoryBackend {
        let summary = format!("{} rows", findings.len());
        MockMemoryBackend::new()
            .with_search_graph_result(Ok(ExplorationResult { findings, summary }))
    }

    fn tool_calls(
        calls: Vec<ToolCall>,
    ) -> Result<Completion, repo_explorer_core::llm::ProviderError> {
        Ok(Completion::from(ProviderResponse::ToolCalls(calls)))
    }

    fn agent_with(
        provider: MockLlmProvider,
    ) -> AgentLoop<
        MockMemoryBackend,
        MockSearchBackend,
        MockLlmProvider,
        MockRepoStateProbe,
        FakeClock,
    > {
        let router = ProviderRouter::with_clock(
            vec![("primary".to_string(), vec![("m".to_string(), provider)])],
            60,
            FakeClock::new(),
        );
        AgentLoop::new(
            MockMemoryBackend::new(),
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new(),
            AgentSettings::default(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        )
    }

    #[tokio::test]
    async fn immediate_finish_returns_its_payload() {
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]);
        let agent = agent_with(provider);
        let query = ExplorationQuery {
            text: "where is main".to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let dir = temp_repo("immediate_finish");
        let got = agent.run(&dir, &query).await.unwrap();
        assert_eq!(got.result.summary, "done");
        assert_eq!(got.result.findings.len(), 1);
        assert_eq!(got.result.findings[0].location.line_start, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn fallback_finish_is_capped_by_max_results() {
        // Regression: `max_results` must bound the fallback loop's own
        // `finish`, not just the deterministic early-exit path.
        let two_findings = ToolCall {
            id: "c1".to_string(),
            name: "finish".to_string(),
            arguments_json:
                r#"{"findings":[{"location":{"path":"src/lib.rs","line_start":1,"line_end":2},"note":"one"},{"location":{"path":"src/other.rs","line_start":3,"line_end":4},"note":"two"}],"summary":"done"}"#
                    .to_string(),
            thought_signatures: None,
        };
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![two_findings])]);
        let agent = agent_with(provider);
        let query = ExplorationQuery {
            text: "where is main".to_string(),
            scope_hint: None,
            max_results: Some(1),
            detailed_snippets: false,
        };
        let dir = temp_repo("fallback_capped");
        let got = agent.run(&dir, &query).await.unwrap();
        assert_eq!(got.result.findings.len(), 1, "capped to max_results");
        assert_eq!(
            got.result.findings[0].location.path,
            PathBuf::from("src/lib.rs")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn early_exit_dedupes_before_truncating_to_max_results() {
        // Regression: result_from_candidates must dedupe (collapsing true
        // duplicates at the "unknown location" (0, 0) sentinel) before
        // truncating to max_results, matching finalize()'s dedupe-then-
        // truncate contract — otherwise a duplicate consumes a truncation
        // slot a distinct 4th candidate should have had.
        fn candidate(
            path: &str,
            line_start: u32,
            line_end: u32,
            symbol: &str,
            kind: CandidateKind,
            score: u32,
        ) -> Candidate {
            Candidate {
                location: FileLocation {
                    path: PathBuf::from(path),
                    line_start,
                    line_end,
                },
                symbol: Some(symbol.to_string()),
                kind,
                score,
                snippet: None,
            }
        }
        let candidates = vec![
            candidate(
                "a.rs",
                10,
                20,
                "decide_freshness",
                CandidateKind::SymbolExact,
                900,
            ),
            candidate(
                "b.rs",
                0,
                0,
                "helper_thing",
                CandidateKind::SymbolFuzzy,
                430,
            ),
            candidate(
                "b.rs",
                0,
                0,
                "helper_thing",
                CandidateKind::SymbolFuzzy,
                430,
            ),
            candidate(
                "another.rs",
                5,
                5,
                "unrelated",
                CandidateKind::ContentHit,
                200,
            ),
        ];
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "early_exit_dedupe",
            &[
                ("a.rs", &numbered_lines(20)),
                ("b.rs", "x\n"),
                ("another.rs", &numbered_lines(5)),
            ],
        );
        let agent = agent_with(MockLlmProvider::new());
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: Some(3),
            detailed_snippets: false,
        };
        let result = agent
            .result_from_candidates(&candidates, &query, 100, None, &dir)
            .await;
        assert_eq!(
            result.findings.len(),
            3,
            "the distinct 4th candidate must survive once the true duplicate is collapsed"
        );
        let paths: Vec<_> = result
            .findings
            .iter()
            .map(|f| f.location.path.clone())
            .collect();
        assert!(
            paths.contains(&PathBuf::from("another.rs")),
            "pre-dedupe truncation must not drop the distinct 4th candidate"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn early_exit_drops_candidate_with_line_start_past_eof() {
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "early_exit_drop_past_eof",
            &[("f.rs", &numbered_lines(5))],
        );
        let candidates = vec![
            Candidate {
                location: FileLocation {
                    path: PathBuf::from("f.rs"),
                    line_start: 2,
                    line_end: 3,
                },
                symbol: Some("ok".to_string()),
                kind: CandidateKind::SymbolExact,
                score: 900,
                snippet: None,
            },
            Candidate {
                location: FileLocation {
                    path: PathBuf::from("f.rs"),
                    line_start: 99,
                    line_end: 120,
                },
                symbol: Some("bad".to_string()),
                kind: CandidateKind::SymbolExact,
                score: 900,
                snippet: None,
            },
        ];
        let agent = agent_with(MockLlmProvider::new());
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let result = agent
            .result_from_candidates(&candidates, &query, 100, None, &dir)
            .await;
        assert_eq!(
            result.findings.len(),
            1,
            "past-EOF candidate must be dropped"
        );
        assert_eq!(result.findings[0].location.line_start, 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn early_exit_drops_candidate_with_nonexistent_path() {
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "early_exit_drop_missing",
            &[("real.rs", &numbered_lines(5))],
        );
        let candidates = vec![Candidate {
            location: FileLocation {
                path: PathBuf::from("gone.rs"),
                line_start: 1,
                line_end: 2,
            },
            symbol: Some("x".to_string()),
            kind: CandidateKind::SymbolExact,
            score: 900,
            snippet: None,
        }];
        let agent = agent_with(MockLlmProvider::new());
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let result = agent
            .result_from_candidates(&candidates, &query, 100, None, &dir)
            .await;
        assert!(
            result.findings.is_empty(),
            "a candidate at a nonexistent path must be dropped"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn early_exit_clamps_candidate_line_end_past_eof() {
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "early_exit_clamp",
            &[("f.rs", &numbered_lines(3))],
        );
        let candidates = vec![Candidate {
            location: FileLocation {
                path: PathBuf::from("f.rs"),
                line_start: 2,
                line_end: 99,
            },
            symbol: Some("x".to_string()),
            kind: CandidateKind::SymbolExact,
            score: 900,
            snippet: None,
        }];
        let agent = agent_with(MockLlmProvider::new());
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let result = agent
            .result_from_candidates(&candidates, &query, 100, None, &dir)
            .await;
        assert_eq!(result.findings.len(), 1);
        assert_eq!(result.findings[0].location.line_start, 2);
        assert_eq!(
            result.findings[0].location.line_end, 3,
            "line_end past EOF must be clamped to the file length"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn early_exit_keeps_backend_snippet_without_rederiving_from_disk() {
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "early_exit_keep_snippet",
            &[("f.rs", &numbered_lines(5))],
        );
        let candidates = vec![Candidate {
            location: FileLocation {
                path: PathBuf::from("f.rs"),
                line_start: 2,
                line_end: 3,
            },
            symbol: Some("x".to_string()),
            kind: CandidateKind::SymbolExact,
            score: 900,
            snippet: Some("backend text".to_string()),
        }];
        let agent = agent_with(MockLlmProvider::new());
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let result = agent
            .result_from_candidates(&candidates, &query, 100, None, &dir)
            .await;
        assert_eq!(result.findings.len(), 1);
        assert_eq!(
            result.findings[0].snippet.as_deref(),
            Some("backend text"),
            "early-exit must keep the backend snippet, not re-derive it from disk (real lines 2-3 are \"l2\\nl3\")"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn early_exit_surfaces_stale_index_note_in_summary() {
        // Regression: Stage 3's early exit must not silently drop Stage 1's
        // index-freshness note — a confident answer served from a stale
        // index (reindex failed, memory backend still answers from the
        // previous index) must still say so.
        let memory = graph_memory(vec![finding(
            "crates/x/src/freshness.rs",
            12,
            "decide_freshness",
        )])
        .with_ensure_fresh_index_result(Ok(IndexStatus::IndexingFailed {
            reason: "boom".to_string(),
        }));
        let router = ProviderRouter::with_clock(
            vec![(
                "primary".to_string(),
                vec![("m".to_string(), MockLlmProvider::new())],
            )],
            60,
            FakeClock::new(),
        );
        let agent = AgentLoop::new(
            memory,
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new(),
            AgentSettings::default(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let query = ExplorationQuery {
            text: "decide_freshness".to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "early_exit_stale_note",
            &[("crates/x/src/freshness.rs", &numbered_lines(12))],
        );
        let got = agent.run(&dir, &query).await.unwrap();
        assert!(
            got.result
                .summary
                .contains("memory index could not be refreshed"),
            "early-exit summary must surface the index-freshness note: {}",
            got.result.summary
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn early_exit_surfaces_scope_hint_ignored_note_in_summary() {
        // F-06: an escaping top-level scope_hint is dropped for the legs, but
        // the deterministic early-exit summary must say so — not silently
        // pretend the scope was honored. Same high-confidence SymbolExact setup
        // as the stale-index-note test, but with an escaping scope_hint.
        let memory = graph_memory(vec![finding(
            "crates/x/src/freshness.rs",
            12,
            "decide_freshness",
        )]);
        let router = ProviderRouter::with_clock(
            vec![(
                "primary".to_string(),
                vec![("m".to_string(), MockLlmProvider::new())],
            )],
            60,
            FakeClock::new(),
        );
        let agent = AgentLoop::new(
            memory,
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new(),
            AgentSettings::default(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let query = ExplorationQuery {
            text: "decide_freshness".to_string(),
            scope_hint: Some(PathBuf::from("../../etc")),
            max_results: None,
            detailed_snippets: false,
        };
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "early_exit_scope_note",
            &[("crates/x/src/freshness.rs", &numbered_lines(12))],
        );
        let got = agent.run(&dir, &query).await.unwrap();
        assert!(
            got.result.summary.contains("escapes the repository root"),
            "early-exit summary must surface the ignored-scope note: {}",
            got.result.summary
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn symbol_free_query_is_vetoed_from_early_exit() {
        // Regression: F-16 — the self-P2-01 repro string names a file path,
        // not a symbol to look up ("what does this constant do"). The symbol
        // leg still comes back with an exact match on "crates" only because
        // it's a fragment of that path, not because the query deliberately
        // named a "crates" symbol — is_trusted_symbol_match excludes a match
        // that's just a path fragment, so has_exact_symbol_match stays false
        // and the Stage 3 gate is vetoed even though a lone SymbolExact
        // candidate here scores >= early_exit_confidence (90).
        // The note's last segment "crates" == the query's first symbol-lookup
        // token, so the symbol leg classifies this as SymbolExact ->
        // confidence clears early_exit_confidence (90). But "crates" is also a
        // fragment of the query's own path token
        // ("crates/repo-explorer-agent/src/verify.rs"), so the guard (not the
        // confidence check) is what blocks early-exit here.
        let memory = graph_memory(vec![finding(
            "crates/repo-explorer-agent/src/verify.rs",
            35,
            "crates",
        )]);
        // Prime the LLM to finish so the vetoed run completes via verify; the
        // finish payload points at src/lib.rs, which temp_repo creates.
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]);
        let router = ProviderRouter::with_clock(
            vec![("primary".to_string(), vec![("m".to_string(), provider)])],
            60,
            FakeClock::new(),
        );
        let agent = AgentLoop::new(
            memory,
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new(),
            AgentSettings::default(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let query = ExplorationQuery {
            text: "crates/repo-explorer-agent/src/verify.rs:35 what does this constant do"
                .to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let dir = temp_repo("symbol_free_no_early_exit");
        let got = agent.run(&dir, &query).await.unwrap();
        assert!(
            !got.result
                .summary
                .contains("Resolved deterministically by the retrieval pre-stage"),
            "symbol-free query must not early-exit: {}",
            got.result.summary
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn router_error_is_hard_fail() {
        // Empty provider list -> RouterError::NoProviders on the first turn.
        let router: ProviderRouter<MockLlmProvider, FakeClock> =
            ProviderRouter::with_clock(vec![], 60, FakeClock::new());
        let agent = AgentLoop::new(
            MockMemoryBackend::new(),
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new(),
            AgentSettings::default(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let got = agent.run(&PathBuf::from("/repo"), &query).await;
        assert!(matches!(got, Err(AgentLoopError::Provider(_))));
    }

    #[tokio::test]
    async fn failed_tool_call_is_not_cached() {
        // Regression: a transient backend error must not be memoized under the
        // tool-result key, or a later byte-identical call would replay the
        // stale failure forever instead of retrying.
        let search = MockSearchBackend::new().with_search_result(Err(SearchError::BackendFailed {
            backend: "rg",
            message: "boom".to_string(),
        }));
        let router = ProviderRouter::with_clock(
            vec![(
                "primary".to_string(),
                vec![("m".to_string(), MockLlmProvider::new())],
            )],
            60,
            FakeClock::new(),
        );
        let agent = AgentLoop::new(
            MockMemoryBackend::new(),
            search,
            router,
            MockRepoStateProbe::new(),
            AgentSettings::default(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let fp = RepoFingerprint {
            head_sha: "abc".to_string(),
            dirty_hash: "def".to_string(),
        };
        let call = ToolCall {
            id: "c1".to_string(),
            name: "grep".to_string(),
            arguments_json: r#"{"pattern":"fn main"}"#.to_string(),
            thought_signatures: None,
        };

        let (message, findings) = agent
            .cached_dispatch(&PathBuf::from("/repo"), &call, Some(&fp))
            .await;
        assert!(message.content.contains("failed"));
        assert!(findings.is_empty());

        let key = ResultCache::tool_key(Path::new("/repo"), &fp, &call.name, &call.arguments_json);
        let cached = agent
            .cache_for(Some(&fp))
            .and_then(|(cache, _)| cache.get_tool(&key));
        assert!(cached.is_none(), "a failed tool call must not be memoized");
    }

    #[test]
    fn accumulate_keeps_distinct_unknown_location_findings_separate() {
        // Regression: accumulate's dedup key must match render::dedupe_key's
        // note-based disambiguation at the "unknown location" (0, 0)
        // sentinel, or two genuinely distinct same-file findings that only
        // lack line info collide and the second is silently dropped before
        // finalize()'s later, note-aware dedup ever sees it.
        let mut findings = Vec::new();
        let mut seen = HashSet::new();
        let foo = ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from("a.rs"),
                line_start: 0,
                line_end: 0,
            },
            snippet: None,
            note: Some("Foo".to_string()),
        };
        let bar = ExplorationFinding {
            location: foo.location.clone(),
            snippet: None,
            note: Some("Bar".to_string()),
        };
        accumulate(&mut findings, &mut seen, foo.clone());
        accumulate(&mut findings, &mut seen, bar);
        assert_eq!(
            findings.len(),
            2,
            "distinct notes at the unknown-location sentinel must not collide"
        );

        // True duplicates (same location AND note) must still collapse.
        accumulate(&mut findings, &mut seen, foo);
        assert_eq!(findings.len(), 2, "a true duplicate must still be dropped");
    }

    #[test]
    fn agent_loop_error_is_comparable() {
        let a = AgentLoopError::Provider("x".to_string());
        assert_eq!(a, AgentLoopError::Provider("x".to_string()));
        assert_eq!(a.to_string(), "llm provider error: x");
    }

    #[test]
    fn token_budget_boundaries() {
        let mut b = TokenBudget::new(0, 0);
        b.add(Some(TokenUsage {
            prompt_tokens: u64::MAX,
            completion_tokens: 1,
            ..Default::default()
        }));
        assert!(!b.exhausted(), "0 means unlimited");

        let mut b = TokenBudget::new(10, 0);
        assert!(!b.exhausted());
        b.add(None);
        assert!(!b.exhausted());
        b.add(Some(TokenUsage {
            prompt_tokens: 7,
            completion_tokens: 3,
            ..Default::default()
        }));
        assert!(b.exhausted(), "exactly at the limit counts as exhausted");
    }

    #[test]
    fn token_budget_accumulates_cache_tokens_without_double_charging() {
        // Cache read/write are a breakdown of prompt_tokens, so they must sum
        // into their own counters while `spent` still tracks total() only.
        let mut b = TokenBudget::new(0, 0);
        b.add(Some(TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 10,
            cached_tokens: 60,
            cache_creation_tokens: 40,
        }));
        b.add(Some(TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 10,
            cached_tokens: 90,
            cache_creation_tokens: 0,
        }));
        b.add(None);
        assert_eq!(b.spent(), 220, "spent stays prompt+completion");
        assert_eq!(b.cache_read_tokens(), 150);
        assert_eq!(b.cache_write_tokens(), 40);
        assert_eq!(b.llm_calls(), 3);
    }

    #[test]
    fn query_preamble_omits_escaping_scope_hint() {
        // An escaping scope hint is dropped for the legs, so the preamble must
        // not render it as an in-effect "Scope hint:" — that would tell the LLM
        // the search was scoped when it was not. A valid in-root hint still is.
        let escaping = ExplorationQuery {
            text: "where is main".to_string(),
            scope_hint: Some(PathBuf::from("../../etc")),
            max_results: None,
            detailed_snippets: false,
        };
        let preamble = query_preamble(&escaping, true, None);
        assert!(
            !preamble.contains("Scope hint:"),
            "escaping scope hint must not be rendered as in-effect: {preamble}"
        );

        let valid = ExplorationQuery {
            text: "where is main".to_string(),
            scope_hint: Some(PathBuf::from("src")),
            max_results: None,
            detailed_snippets: false,
        };
        let preamble = query_preamble(&valid, false, None);
        assert!(
            preamble.contains("Scope hint: src"),
            "a valid in-root scope hint must still be rendered: {preamble}"
        );
    }

    #[tokio::test]
    async fn rotating_router_keeps_one_conversations_turns_on_the_same_model() {
        // Regression: a caller-supplied rotation seed (see
        // `ProviderRouter::with_clock_and_rotation`) must stay fixed for
        // every raw provider call one `run()` conversation makes. Before
        // `TokenBudget::rotation_seed` pinned it, each raw call re-derived
        // its own rotation, so a later turn of the SAME exploration could
        // land on a different rotated model than an earlier turn already
        // committed provider-specific continuation state to (e.g. Gemini
        // thought signatures) — exactly the failure this pins down.
        let bogus_call = ToolCall {
            id: "c0".to_string(),
            name: "not_a_real_tool".to_string(),
            arguments_json: "{}".to_string(),
            thought_signatures: None,
        };
        let a = MockLlmProvider::new().with_responses(vec![
            tool_calls(vec![bogus_call]),
            tool_calls(vec![finish_call()]),
        ]);
        let b = MockLlmProvider::new();
        let router = ProviderRouter::new_with_rotation(
            vec![(
                "gemini".to_string(),
                vec![("a".to_string(), a.clone()), ("b".to_string(), b.clone())],
                true,
            )],
            60,
        );
        let agent = AgentLoop::new(
            MockMemoryBackend::new(),
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new(),
            AgentSettings::default(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let query = ExplorationQuery {
            text: "where is main".to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let dir = temp_repo("rotation_stable_within_conversation");
        let got = agent.run(&dir, &query).await.unwrap();
        assert_eq!(got.result.summary, "done");
        assert_eq!(
            a.calls().len(),
            2,
            "both turns of one conversation must land on the model the seed picked"
        );
        assert_eq!(
            b.calls().len(),
            0,
            "the rotation seed must not move mid-conversation"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    fn fp(sha: &str) -> RepoFingerprint {
        RepoFingerprint {
            head_sha: sha.to_string(),
            dirty_hash: "d".to_string(),
        }
    }

    /// Build an `AgentLoop` over the mocks with an explicit probe and trust TTL,
    /// so a caller can hold the `MockMemoryBackend` clone to count refreshes.
    fn agent_with_probe_and_ttl(
        memory: MockMemoryBackend,
        provider: MockLlmProvider,
        probe: MockRepoStateProbe,
        ttl: Duration,
    ) -> AgentLoop<
        MockMemoryBackend,
        MockSearchBackend,
        MockLlmProvider,
        MockRepoStateProbe,
        FakeClock,
    > {
        let router = ProviderRouter::with_clock(
            vec![("primary".to_string(), vec![("m".to_string(), provider)])],
            60,
            FakeClock::new(),
        );
        AgentLoop::new(
            memory,
            MockSearchBackend::new(),
            router,
            probe,
            AgentSettings::default(),
            CacheSettings::default(),
            ttl,
        )
    }

    fn count_ensure_fresh(memory: &MockMemoryBackend) -> usize {
        memory
            .calls()
            .iter()
            .filter(|c| matches!(c, Call::EnsureFreshIndex { .. }))
            .count()
    }

    fn q(text: &str) -> ExplorationQuery {
        ExplorationQuery {
            text: text.to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        }
    }

    #[test]
    fn may_skip_no_mark_is_false() {
        // First call for a repo has no recorded mark -> must run the full flow.
        let now = Instant::now();
        assert!(!may_skip_index_refresh(
            Some(&fp("a")),
            None,
            now,
            TEST_INDEX_TRUST_TTL
        ));
    }

    #[test]
    fn may_skip_matching_within_ttl_is_true() {
        let now = Instant::now();
        let mark = IndexRefreshMark {
            fingerprint: fp("a"),
            at: now - Duration::from_secs(1),
        };
        assert!(may_skip_index_refresh(
            Some(&fp("a")),
            Some(&mark),
            now,
            TEST_INDEX_TRUST_TTL
        ));
    }

    #[test]
    fn may_skip_matching_past_ttl_is_false() {
        // Staleness backstop: an unchanged repo still re-checks after the window.
        let now = Instant::now();
        let mark = IndexRefreshMark {
            fingerprint: fp("a"),
            at: now - Duration::from_secs(100),
        };
        assert!(!may_skip_index_refresh(
            Some(&fp("a")),
            Some(&mark),
            now,
            Duration::from_secs(10)
        ));
    }

    #[test]
    fn may_skip_differing_fingerprint_is_false() {
        // A .git op / branch switch changes the fingerprint -> full flow.
        let now = Instant::now();
        let mark = IndexRefreshMark {
            fingerprint: fp("a"),
            at: now,
        };
        assert!(!may_skip_index_refresh(
            Some(&fp("b")),
            Some(&mark),
            now,
            TEST_INDEX_TRUST_TTL
        ));
    }

    #[test]
    fn may_skip_no_current_fingerprint_is_false() {
        // Probe failed / not a git repo -> never skip.
        let now = Instant::now();
        let mark = IndexRefreshMark {
            fingerprint: fp("a"),
            at: now,
        };
        assert!(!may_skip_index_refresh(
            None,
            Some(&mark),
            now,
            TEST_INDEX_TRUST_TTL
        ));
    }

    #[test]
    fn may_skip_zero_ttl_never_skips() {
        // staleness_seconds == 0 degrades to today's behavior.
        let now = Instant::now();
        let mark = IndexRefreshMark {
            fingerprint: fp("a"),
            at: now,
        };
        assert!(!may_skip_index_refresh(
            Some(&fp("a")),
            Some(&mark),
            now,
            Duration::ZERO
        ));
    }

    #[tokio::test]
    async fn repeat_call_same_fingerprint_skips_second_refresh() {
        // Regression: two explore calls for the same repo_path with distinct
        // queries (to miss the query cache) on an unchanged repo must run
        // ensure_fresh_index exactly once — the redundant upstream freshness
        // round-trip is skipped on the second call.
        let memory = MockMemoryBackend::new();
        let mem = memory.clone();
        let provider = MockLlmProvider::new().with_responses(vec![
            tool_calls(vec![finish_call()]),
            tool_calls(vec![finish_call()]),
        ]);
        let probe = MockRepoStateProbe::new().with_fingerprint(Some(fp("abc")));
        let agent = agent_with_probe_and_ttl(memory, provider, probe, Duration::from_secs(3600));
        let dir = temp_repo("skip_second_refresh");
        agent.run(&dir, &q("first query")).await.unwrap();
        agent.run(&dir, &q("second query")).await.unwrap();
        assert_eq!(
            count_ensure_fresh(&mem),
            1,
            "second call on an unchanged repo must skip the redundant refresh"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn fingerprint_change_forces_second_refresh() {
        // Regression: a `git checkout` / .git op between two calls changes the
        // fingerprint, which must force a full ensure_fresh_index on the second
        // call — closing the branch-switch correctness gap.
        let memory = MockMemoryBackend::new();
        let mem = memory.clone();
        let provider = MockLlmProvider::new().with_responses(vec![
            tool_calls(vec![finish_call()]),
            tool_calls(vec![finish_call()]),
        ]);
        let probe = MockRepoStateProbe::new().with_fingerprint(Some(fp("abc")));
        let probe_handle = probe.clone();
        let agent = agent_with_probe_and_ttl(memory, provider, probe, Duration::from_secs(3600));
        let dir = temp_repo("fp_change_refresh");
        agent.run(&dir, &q("first query")).await.unwrap();
        probe_handle.set_fingerprint(Some(fp("def")));
        agent.run(&dir, &q("second query")).await.unwrap();
        assert_eq!(
            count_ensure_fresh(&mem),
            2,
            "a changed fingerprint must force a second refresh"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn no_fingerprint_never_skips() {
        // Not a git repo / probe failure -> no fingerprint -> never skip; every
        // call runs the full flow, exactly as today.
        let memory = MockMemoryBackend::new();
        let mem = memory.clone();
        let provider = MockLlmProvider::new().with_responses(vec![
            tool_calls(vec![finish_call()]),
            tool_calls(vec![finish_call()]),
        ]);
        let probe = MockRepoStateProbe::new(); // fingerprint None
        let agent = agent_with_probe_and_ttl(memory, provider, probe, Duration::from_secs(3600));
        let dir = temp_repo("no_fp_no_skip");
        agent.run(&dir, &q("first query")).await.unwrap();
        agent.run(&dir, &q("second query")).await.unwrap();
        assert_eq!(
            count_ensure_fresh(&mem),
            2,
            "without a fingerprint, never skip"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn failed_refresh_is_not_recorded() {
        // Regression: a first call whose refresh failed must NOT record a skip
        // mark, so the second call retries the full flow (never serves against a
        // never-built index).
        let memory = MockMemoryBackend::new().with_ensure_fresh_index_result(Ok(
            IndexStatus::IndexingFailed {
                reason: "boom".to_string(),
            },
        ));
        let mem = memory.clone();
        let provider = MockLlmProvider::new().with_responses(vec![
            tool_calls(vec![finish_call()]),
            tool_calls(vec![finish_call()]),
        ]);
        let probe = MockRepoStateProbe::new().with_fingerprint(Some(fp("abc")));
        let agent = agent_with_probe_and_ttl(memory, provider, probe, Duration::from_secs(3600));
        let dir = temp_repo("failed_refresh_retries");
        agent.run(&dir, &q("first query")).await.unwrap();
        agent.run(&dir, &q("second query")).await.unwrap();
        assert_eq!(
            count_ensure_fresh(&mem),
            2,
            "a failed refresh must not be recorded; the next call retries"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Same as `agent_with_probe_and_ttl` but with an explicit `CacheSettings`,
    /// so a test can shrink `max_entries` to exercise `index_refresh_seen`'s
    /// eviction.
    fn agent_with_probe_ttl_and_cache(
        memory: MockMemoryBackend,
        provider: MockLlmProvider,
        probe: MockRepoStateProbe,
        ttl: Duration,
        cache_settings: CacheSettings,
    ) -> AgentLoop<
        MockMemoryBackend,
        MockSearchBackend,
        MockLlmProvider,
        MockRepoStateProbe,
        FakeClock,
    > {
        let router = ProviderRouter::with_clock(
            vec![("primary".to_string(), vec![("m".to_string(), provider)])],
            60,
            FakeClock::new(),
        );
        AgentLoop::new(
            memory,
            MockSearchBackend::new(),
            router,
            probe,
            AgentSettings::default(),
            cache_settings,
            ttl,
        )
    }

    #[tokio::test]
    async fn probe_index_ready_false_forces_refresh_despite_skip_candidate() {
        // Regression: a skip candidate (unchanged fingerprint, within TTL) must
        // still be safety-netted by a cheap existence probe — if the upstream
        // index was lost for a reason the git fingerprint can't see, the full
        // flow must run instead of trusting the local skip.
        let memory = MockMemoryBackend::new().with_probe_index_ready_result(Ok(false));
        let mem = memory.clone();
        let provider = MockLlmProvider::new().with_responses(vec![
            tool_calls(vec![finish_call()]),
            tool_calls(vec![finish_call()]),
        ]);
        let probe = MockRepoStateProbe::new().with_fingerprint(Some(fp("abc")));
        let agent = agent_with_probe_and_ttl(memory, provider, probe, Duration::from_secs(3600));
        let dir = temp_repo("probe_not_ready_forces_refresh");
        agent.run(&dir, &q("first query")).await.unwrap();
        agent.run(&dir, &q("second query")).await.unwrap();
        assert_eq!(
            count_ensure_fresh(&mem),
            2,
            "a not-ready safety probe must force the full flow even for a skip candidate"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn index_refresh_seen_evicts_oldest_beyond_cache_cap() {
        // Regression: index_refresh_seen is bounded by cache_settings.max_entries
        // (FIFO), not an unbounded per-repo map — a third call for the first repo,
        // after a second distinct repo pushed it out, forces the full flow again.
        let memory = MockMemoryBackend::new();
        let mem = memory.clone();
        let provider = MockLlmProvider::new().with_responses(vec![
            tool_calls(vec![finish_call()]),
            tool_calls(vec![finish_call()]),
            tool_calls(vec![finish_call()]),
        ]);
        let probe = MockRepoStateProbe::new().with_fingerprint(Some(fp("abc")));
        let cache_settings = CacheSettings {
            enabled: true,
            max_entries: 1,
            ..CacheSettings::default()
        };
        let agent = agent_with_probe_ttl_and_cache(
            memory,
            provider,
            probe,
            Duration::from_secs(3600),
            cache_settings,
        );
        let dir_a = temp_repo("evict_repo_a");
        let dir_b = temp_repo("evict_repo_b");
        agent.run(&dir_a, &q("a1")).await.unwrap();
        agent.run(&dir_b, &q("b1")).await.unwrap();
        agent.run(&dir_a, &q("a2")).await.unwrap();
        assert_eq!(
            count_ensure_fresh(&mem),
            3,
            "repo A's mark must be evicted once repo B's mark is inserted past the cap"
        );
        std::fs::remove_dir_all(&dir_a).ok();
        std::fs::remove_dir_all(&dir_b).ok();
    }

    // --- QW-0: every `run` exit path must emit exactly one metrics record ---

    /// The records `emit_metrics` taped on this thread, oldest first. Under
    /// `--test-threads=1` earlier tests can have left some behind, so callers
    /// assert on the last one — emission is in call order.
    fn taped_metrics() -> Vec<QueryMetrics> {
        EMITTED.with(|v| std::mem::take(&mut *v.borrow_mut()))
    }

    /// A memory backend whose graph leg returns one exact `decide_freshness`
    /// symbol hit — enough confidence for Stage 3's early exit.
    fn high_confidence_memory() -> MockMemoryBackend {
        graph_memory(vec![finding(
            "crates/x/src/freshness.rs",
            12,
            "decide_freshness",
        )])
    }

    #[tokio::test]
    async fn metrics_emitted_on_cache_path() {
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]);
        let probe = MockRepoStateProbe::new().with_fingerprint(Some(fp("abc")));
        let agent = agent_with_probe_and_ttl(
            MockMemoryBackend::new(),
            provider,
            probe,
            TEST_INDEX_TRUST_TTL,
        );
        let dir = temp_repo("metrics_cache");
        let query = q("where is main");
        agent.run(&dir, &query).await.unwrap();
        // Same query, same fingerprint -> served from the query cache, which
        // returns before Stage 1 and never reaches complete_run.
        agent.run(&dir, &query).await.unwrap();

        let taped = taped_metrics();
        let m = taped.last().expect("the cache path must emit a record");
        assert_eq!(m.path, "cache");
        assert_eq!(m.query, "where is main");
        assert_eq!(m.repo_path, dir.to_string_lossy());
        assert_eq!(m.tokens, 0);
        assert_eq!(m.llm_calls, 0);
        assert_eq!(
            (m.confidence, m.candidate_count),
            (None, None),
            "the pre-stage never ran, so neither was measured"
        );
        assert_eq!(m.early_exit_route, "none");
        assert_eq!(m.index_status, "NotChecked", "Stage 1 never ran");
        assert_eq!(m.findings_count, 1);
        assert!(m.ts_unix_ms > 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn metrics_emitted_on_early_exit_path() {
        let agent = agent_with_probe_and_ttl(
            high_confidence_memory(),
            MockLlmProvider::new(),
            MockRepoStateProbe::new(),
            TEST_INDEX_TRUST_TTL,
        );
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "metrics_early_exit",
            &[("crates/x/src/freshness.rs", &numbered_lines(12))],
        );
        let got = agent.run(&dir, &q("decide_freshness")).await.unwrap();

        let taped = taped_metrics();
        let m = taped
            .last()
            .expect("the early-exit path must emit a record");
        assert_eq!(m.path, "early-exit");
        assert_eq!(got.stage_exit, StageExit::EarlyExit);
        assert_eq!(
            got.stage_exit.as_str(),
            m.path,
            "the response agrees with the log"
        );
        assert_eq!(Some(got.retrieval_confidence), m.confidence);
        assert_eq!(m.cited_candidate_ids, None, "no finish call ran");
        assert_eq!(m.early_exit_route, "confidence");
        assert_eq!(m.llm_calls, 0, "early exit makes no provider call");
        assert!(m.confidence.unwrap() > 0);
        assert_eq!(m.candidate_count, Some(1));
        assert_eq!(m.findings_count, 1);
        assert!(m.summary_len > 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- QW-2: the unique-trusted-symbol Stage-3 route ---

    /// A graph leg returning one exact `decide_freshness` hit plus a
    /// `SymbolFuzzy` runner-up. The runner-up (400 base vs 700) shrinks the
    /// margin enough to drop confidence to 88 — below `early_exit_confidence`
    /// (90) — while the exact match itself stays unambiguous. Exactly the
    /// case QW-2 exists for.
    fn lone_exact_with_fuzzy_runner_up() -> MockMemoryBackend {
        graph_memory(vec![
            finding("crates/x/src/freshness.rs", 12, "decide_freshness"),
            finding("crates/x/src/other.rs", 5, "decide_freshness_v2"),
        ])
    }

    fn two_file_repo(test: &str) -> PathBuf {
        crate::test_support::temp_repo_with(
            "agent_run",
            test,
            &[
                ("crates/x/src/freshness.rs", &numbered_lines(12)),
                ("crates/x/src/other.rs", &numbered_lines(8)),
                // The path `finish_call`'s payload names, so a run that does
                // reach the LLM can finish at Stage 4 instead of escalating.
                ("src/lib.rs", "a\nb\nc\n"),
            ],
        )
    }

    /// Like `agent_with_probe_and_ttl`, but with caller-chosen `AgentSettings`
    /// (the QW-2 escape hatch is the only knob any caller varies).
    fn agent_with_settings(
        memory: MockMemoryBackend,
        provider: MockLlmProvider,
        settings: AgentSettings,
    ) -> AgentLoop<
        MockMemoryBackend,
        MockSearchBackend,
        MockLlmProvider,
        MockRepoStateProbe,
        FakeClock,
    > {
        let router = ProviderRouter::with_clock(
            vec![("primary".to_string(), vec![("m".to_string(), provider)])],
            60,
            FakeClock::new(),
        );
        AgentLoop::new(
            memory,
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new(),
            settings,
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        )
    }

    #[tokio::test]
    async fn sub_threshold_unique_symbol_skips_verification() {
        // The win: one unambiguous exact symbol match, confidence dragged
        // below early_exit_confidence by a fuzzy runner-up. Before QW-2 this
        // paid a full verify round-trip. The provider has no responses queued,
        // so any LLM call would fail the run outright.
        let agent = agent_with_probe_and_ttl(
            lone_exact_with_fuzzy_runner_up(),
            MockLlmProvider::new(),
            MockRepoStateProbe::new(),
            TEST_INDEX_TRUST_TTL,
        );
        let dir = two_file_repo("qw2_unique_symbol");
        let got = agent.run(&dir, &q("decide_freshness")).await.unwrap();
        assert!(
            got.result
                .summary
                .contains("Resolved deterministically by the retrieval pre-stage"),
            "expected the pre-stage answer, got: {}",
            got.result.summary
        );
        // The only end-to-end check that `symbols_for`'s join actually fires
        // against the REAL candidate and finding pipelines (its unit tests
        // build both sides by hand, so they cannot catch a normalization
        // mismatch that empties the field on every run).
        let mut symbols: Vec<&str> = got.symbols.iter().map(|(_, s)| s.as_str()).collect();
        symbols.sort_unstable();
        assert_eq!(symbols, vec!["decide_freshness", "decide_freshness_v2"]);

        let m = taped_metrics().pop().expect("a record must be emitted");
        assert_eq!(m.path, "early-exit");
        assert_eq!(m.early_exit_route, "unique-symbol");
        assert_eq!(m.llm_calls, 0, "the whole point: no verification call");
        assert!(
            m.confidence.unwrap() < AgentSettings::default().early_exit_confidence,
            "fixture must stay below the confidence route's threshold, was {:?}",
            m.confidence
        );
        // The route returns the whole ranked list, like the confidence route.
        assert_eq!(m.candidate_count, Some(2));
        assert_eq!(m.findings_count, 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn unique_symbol_route_verifies_when_its_own_candidate_is_dropped() {
        // The route is authorized by ONE candidate. Here `freshness.rs` was
        // truncated since it was indexed, so line 12 is past EOF and disk
        // verification drops exactly that candidate — leaving only the
        // unrelated `decide_freshness_v2` fuzzy match. Returning that as the
        // answer would be an unverified wrong answer, so the run must fall
        // through and pay for verification after all.
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]);
        let agent = agent_with_probe_and_ttl(
            lone_exact_with_fuzzy_runner_up(),
            provider,
            MockRepoStateProbe::new(),
            TEST_INDEX_TRUST_TTL,
        );
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "qw2_authorizer_dropped",
            &[
                ("crates/x/src/freshness.rs", &numbered_lines(3)),
                ("crates/x/src/other.rs", &numbered_lines(8)),
                ("src/lib.rs", "a\nb\nc\n"),
            ],
        );
        agent.run(&dir, &q("decide_freshness")).await.unwrap();

        let m = taped_metrics().pop().expect("a record must be emitted");
        assert_eq!(
            m.path, "verify",
            "the candidate that authorized the early exit did not survive"
        );
        assert_eq!(m.early_exit_route, "none");
        assert_eq!(m.llm_calls, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn two_exact_matches_in_different_files_still_verify() {
        // Same symbol in two files stays two candidates -> ambiguous -> the
        // LLM still has to pick.
        let memory = graph_memory(vec![
            finding("crates/x/src/freshness.rs", 12, "a::decide_freshness"),
            finding("crates/x/src/other.rs", 5, "b::decide_freshness"),
        ]);
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]);
        let agent = agent_with_probe_and_ttl(
            memory,
            provider,
            MockRepoStateProbe::new(),
            TEST_INDEX_TRUST_TTL,
        );
        let dir = two_file_repo("qw2_two_exact");
        agent.run(&dir, &q("decide_freshness")).await.unwrap();

        let m = taped_metrics().pop().expect("a record must be emitted");
        assert_eq!(m.path, "verify");
        assert_eq!(m.early_exit_route, "none");
        assert_eq!(m.llm_calls, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn skip_verify_on_exact_symbol_false_restores_verification() {
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]);
        let agent = agent_with_settings(
            lone_exact_with_fuzzy_runner_up(),
            provider,
            AgentSettings {
                skip_verify_on_exact_symbol: false,
                ..AgentSettings::default()
            },
        );
        let dir = two_file_repo("qw2_escape_hatch");
        agent.run(&dir, &q("decide_freshness")).await.unwrap();

        let m = taped_metrics().pop().expect("a record must be emitted");
        assert_eq!(m.path, "verify");
        assert_eq!(m.early_exit_route, "none");
        assert_eq!(m.llm_calls, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn metrics_emitted_on_verify_path() {
        // Same shape as `symbol_free_query_is_vetoed_from_early_exit`: high
        // confidence, but the exact match is only a path fragment, so Stage 3
        // is vetoed and the LLM's `finish` lands on the verify exit.
        let memory = graph_memory(vec![finding(
            "crates/repo-explorer-agent/src/verify.rs",
            35,
            "crates",
        )]);
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]);
        let agent = agent_with_probe_and_ttl(
            memory,
            provider,
            MockRepoStateProbe::new(),
            TEST_INDEX_TRUST_TTL,
        );
        let dir = temp_repo("metrics_verify");
        let got = agent
            .run(
                &dir,
                &q("crates/repo-explorer-agent/src/verify.rs:35 what does this constant do"),
            )
            .await
            .unwrap();

        let taped = taped_metrics();
        let m = taped.last().expect("the verify path must emit a record");
        assert_eq!(m.path, "verify");
        assert_eq!(got.stage_exit, StageExit::Verify);
        assert_eq!(got.stage_exit.as_str(), m.path);
        assert_eq!(Some(got.retrieval_confidence), m.confidence);
        assert_eq!(
            m.cited_candidate_ids,
            Some(0),
            "the mock finish cites no candidate_id"
        );
        assert_eq!(
            (m.brief_tokens, m.orientation_calls_in_loop),
            (None, None),
            "Stage 5 never ran, so neither was measured"
        );
        assert_eq!(m.early_exit_route, "none", "Stage 3 was vetoed");
        assert_eq!(m.llm_calls, 1);
        assert_eq!(m.findings_count, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn metrics_emitted_on_fallback_path() {
        let provider = MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]);
        let agent = agent_with(provider);
        let dir = temp_repo("metrics_fallback");
        let got = agent.run(&dir, &q("where is main")).await.unwrap();

        let taped = taped_metrics();
        let m = taped.last().expect("the fallback path must emit a record");
        assert_eq!(m.path, "fallback");
        assert_eq!(got.stage_exit, StageExit::Fallback);
        assert_eq!(got.stage_exit.as_str(), m.path);
        assert_eq!(Some(got.retrieval_confidence), m.confidence);
        assert_eq!(m.cited_candidate_ids, Some(0));
        assert_eq!(m.early_exit_route, "none");
        assert_eq!(m.llm_calls, 1);
        assert!(!m.forced_finish);
        assert_eq!(m.findings_count, 1);
        assert_eq!(m.index_status, "UpToDate");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn orientation_calls_and_brief_tokens_are_emitted_on_the_fallback_path() {
        // Live `get_architecture` payload shape (see the memory crate's
        // decoder tests).
        const ARCH: &str = "\
packages: 1  (cols: name nodes fan_in fan_out)\n  repo-explorer-core 256 0 0\n\
entry_points: 1  (cols: qn file)\n  repo.src.main.main src/main.rs\n";
        let provider = MockLlmProvider::new().with_responses(vec![
            // Batched — a single-call turn is rejected, never executed, so it
            // must not be counted either.
            tool_calls(vec![
                ToolCall {
                    id: "o1".to_string(),
                    name: "get_architecture".to_string(),
                    arguments_json: "{}".to_string(),
                    thought_signatures: None,
                },
                ToolCall {
                    id: "o2".to_string(),
                    name: "grep".to_string(),
                    arguments_json: r#"{"pattern":"main"}"#.to_string(),
                    thought_signatures: None,
                },
            ]),
            tool_calls(vec![finish_call()]),
        ]);
        let memory =
            MockMemoryBackend::new().with_get_architecture_text_result(Ok(ARCH.to_string()));
        let agent = agent_with_settings(memory, provider, AgentSettings::default());
        let dir = temp_repo("metrics_orientation");
        agent.run(&dir, &q("where is main")).await.unwrap();

        let taped = taped_metrics();
        let m = taped.last().expect("the fallback path must emit a record");
        assert_eq!(m.path, "fallback");
        assert!(
            m.brief_tokens.is_some_and(|t| t > 0),
            "an injected brief must be measured: {:?}",
            m.brief_tokens
        );
        assert_eq!(
            m.orientation_calls_in_loop,
            Some(1),
            "the in-loop get_architecture call must be counted"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn metrics_emitted_on_provider_error_path() {
        // Empty provider list -> the fallback loop's first turn is a hard
        // RouterError, the one `run` exit that returns Err.
        let router: ProviderRouter<MockLlmProvider, FakeClock> =
            ProviderRouter::with_clock(vec![], 60, FakeClock::new());
        let agent = AgentLoop::new(
            MockMemoryBackend::new(),
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new(),
            AgentSettings::default(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let got = agent.run(&PathBuf::from("/repo"), &q("x")).await;
        assert!(matches!(got, Err(AgentLoopError::Provider(_))));

        let taped = taped_metrics();
        let m = taped.last().expect("the error path must emit a record");
        assert_eq!(m.path, "error");
        assert_eq!(m.findings_count, 0, "no result exists on the error exit");
        assert_eq!(m.repo_path, "/repo");
    }

    #[test]
    fn query_metrics_json_is_one_line_and_complete() {
        // The record rides a tracing field and a JSONL sink, so it must
        // serialize to a single line and keep every contract key.
        let mut m = QueryMetrics::new(
            Path::new("/repo"),
            &ExplorationQuery {
                text: "a\nb".to_string(),
                scope_hint: Some(PathBuf::from("crates")),
                max_results: Some(3),
                detailed_snippets: false,
            },
            Instant::now(),
        );
        m.record_budget(&TokenBudget::new(0, 0), true);
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains('\n'), "must stay a single line: {json}");
        for key in [
            "ts_unix_ms",
            "repo_path",
            "query",
            "scope_hint",
            "max_results",
            "path",
            "index_status",
            "confidence",
            "candidate_count",
            "early_exit_route",
            "tokens",
            "llm_calls",
            "cache_read_tokens",
            "cache_write_tokens",
            "forced_finish",
            "findings_count",
            "summary_len",
            "git_probe_ms",
            "total_ms",
        ] {
            assert!(
                json.contains(&format!("\"{key}\"")),
                "missing {key}: {json}"
            );
        }
        assert!(!json.contains("started"), "the Instant must not serialize");
    }

    #[test]
    fn metrics_sink_creates_then_appends() {
        // The JSONL sink must extend the file across writes (and across
        // server restarts), never truncate it.
        let dir = crate::test_support::temp_repo_with("agent_run", "metrics_sink", &[]);
        let path = dir.join("metrics.jsonl");
        append_json_line(&path, r#"{"n":1}"#).unwrap();
        append_json_line(&path, r#"{"n":2}"#).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"n\":1}\n{\"n\":2}\n"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sink_path_reads_the_documented_env_var() {
        // README.md documents `REPO_EXPLORER_METRICS`; renaming it here is a
        // silently dead sink, and an empty value (how an MCP `env` map
        // disables a variable) must read as unset, not as the path "".
        let set = |k: &str| (k == "REPO_EXPLORER_METRICS").then(|| "/tmp/m.jsonl".into());
        assert_eq!(sink_path(set), Some(PathBuf::from("/tmp/m.jsonl")));
        assert_eq!(sink_path(|_| None), None);
        assert_eq!(sink_path(|_| Some(std::ffi::OsString::new())), None);
    }

    /// QW-3 settings: a deliberately tiny prompt cap next to a larger
    /// response-only cap, so the two can never be confused for each other.
    fn qw3_settings() -> AgentSettings {
        AgentSettings {
            snippet_max_chars: 10,
            snippet_max_chars_detailed: 50,
            ..AgentSettings::default()
        }
    }

    fn detailed_q(text: &str) -> ExplorationQuery {
        ExplorationQuery {
            detailed_snippets: true,
            ..q(text)
        }
    }

    /// Number of leading `x`s before the truncation marker — the cap that was
    /// actually applied.
    fn kept_chars(snippet: &str) -> usize {
        snippet.chars().take_while(|c| *c == 'x').count()
    }

    #[tokio::test]
    async fn detailed_snippets_raise_the_response_cap_only() {
        // QW-3: `response_format: "detailed"` widens the FINAL findings'
        // snippet cap. `self.caps` — the cap handed to verify/dispatch/render
        // for LLM PROMPT rendering — must stay at `snippet_max_chars` for a
        // detailed request too, or the feature would inflate exactly the
        // prompt cost it exists to cut.
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "qw3_response_cap",
            &[("f.rs", &numbered_lines(5))],
        );
        let candidates = vec![Candidate {
            location: FileLocation {
                path: PathBuf::from("f.rs"),
                line_start: 2,
                line_end: 3,
            },
            symbol: Some("x".to_string()),
            kind: CandidateKind::SymbolExact,
            score: 900,
            snippet: Some("x".repeat(1000)),
        }];
        let agent = agent_with_settings(
            MockMemoryBackend::new(),
            MockLlmProvider::new(),
            qw3_settings(),
        );

        let concise = agent
            .result_from_candidates(&candidates, &q("x"), 100, None, &dir)
            .await;
        let detailed = agent
            .result_from_candidates(&candidates, &detailed_q("x"), 100, None, &dir)
            .await;

        assert_eq!(
            kept_chars(concise.findings[0].snippet.as_deref().unwrap()),
            10,
            "concise must keep today's agent.snippet_max_chars behavior"
        );
        assert_eq!(
            kept_chars(detailed.findings[0].snippet.as_deref().unwrap()),
            50,
            "detailed must use agent.snippet_max_chars_detailed"
        );
        assert_eq!(
            agent.caps.snippet_max_chars, 10,
            "the prompt-side cap must be untouched by a detailed request"
        );
        assert_eq!(agent.response_caps(&detailed_q("x")).snippet_max_chars, 50);
        assert_eq!(agent.response_caps(&q("x")).snippet_max_chars, 10);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn detailed_response_cap_is_a_floor_never_a_ceiling() {
        // `snippet_max_chars_detailed` widens the response cap; it must never
        // narrow it. A config whose concise cap already exceeds the detailed
        // one (or disables the cap with 0) would otherwise make "detailed"
        // truncate harder than "concise", contradicting the tool description.
        let wider_concise = agent_with_settings(
            MockMemoryBackend::new(),
            MockLlmProvider::new(),
            AgentSettings {
                snippet_max_chars: 2000,
                snippet_max_chars_detailed: 1500,
                ..AgentSettings::default()
            },
        );
        assert_eq!(
            wider_concise
                .response_caps(&detailed_q("x"))
                .snippet_max_chars,
            2000,
            "detailed must not return less than concise"
        );
        let uncapped = agent_with_settings(
            MockMemoryBackend::new(),
            MockLlmProvider::new(),
            AgentSettings {
                snippet_max_chars: 0,
                ..AgentSettings::default()
            },
        );
        assert_eq!(
            uncapped.response_caps(&detailed_q("x")).snippet_max_chars,
            0,
            "0 means uncapped on both knobs"
        );
    }

    #[tokio::test]
    async fn fallback_tool_findings_stay_at_the_prompt_cap_even_when_detailed() {
        // Pins the documented limitation on `ExplorationQuery::detailed_snippets`:
        // tool-dispatch findings are capped at the PROMPT cap the moment they
        // are dispatched, long before any response cap applies. The fallback
        // loop's budget-exhausted-without-finish exit returns exactly those,
        // so `response_format: "detailed"` cannot widen them there.
        let dir = crate::test_support::temp_repo_with(
            "agent_run",
            "qw3_dispatch_cap",
            &[("f.rs", &numbered_lines(3))],
        );
        let search = MockSearchBackend::new().with_search_result(Ok(vec![ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from("f.rs"),
                line_start: 1,
                line_end: 1,
            },
            snippet: Some("x".repeat(1000)),
            note: None,
        }]));
        let router = ProviderRouter::with_clock(
            vec![(
                "primary".to_string(),
                vec![("m".to_string(), MockLlmProvider::new())],
            )],
            60,
            FakeClock::new(),
        );
        let agent = AgentLoop::new(
            MockMemoryBackend::new(),
            search,
            router,
            MockRepoStateProbe::new(),
            qw3_settings(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let call = ToolCall {
            id: "c1".to_string(),
            name: "grep".to_string(),
            arguments_json: r#"{"pattern":"x"}"#.to_string(),
            thought_signatures: None,
        };
        let (_message, findings) = agent.cached_dispatch(&dir, &call, None).await;
        assert_eq!(
            kept_chars(findings[0].snippet.as_deref().unwrap()),
            10,
            "dispatch caps at snippet_max_chars, whatever the response format"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn detailed_request_is_not_served_the_cached_concise_snippets() {
        // QW-3 cache hazard: the query cache stores already-capped findings,
        // so a detailed call landing on a concise entry would silently get
        // 10-char snippets. `detailed_snippets` is part of the query key.
        let long = "x".repeat(1000);
        let memory = graph_memory(vec![
            ExplorationFinding {
                snippet: Some(long),
                ..finding("crates/x/src/freshness.rs", 12, "decide_freshness")
            },
            finding("crates/x/src/other.rs", 5, "decide_freshness_v2"),
        ]);
        // A stable fingerprint is what makes the query cache live at all —
        // without it `store_query_cache` is a no-op and this test would pass
        // even with the format missing from the key. No provider responses
        // queued: any LLM call would fail the run, so both calls must stay on
        // the deterministic early-exit path.
        let router = ProviderRouter::with_clock(
            vec![(
                "primary".to_string(),
                vec![("m".to_string(), MockLlmProvider::new())],
            )],
            60,
            FakeClock::new(),
        );
        let agent = AgentLoop::new(
            memory,
            MockSearchBackend::new(),
            router,
            MockRepoStateProbe::new().with_fingerprint(Some(fp("abc"))),
            qw3_settings(),
            CacheSettings::default(),
            TEST_INDEX_TRUST_TTL,
        );
        let dir = two_file_repo("qw3_cache_format");

        let concise = agent.run(&dir, &q("decide_freshness")).await.unwrap();
        let replay = agent.run(&dir, &q("decide_freshness")).await.unwrap();
        assert_eq!(
            replay.result, concise.result,
            "sanity: the repeated concise call must be served from the query cache"
        );
        // A hit replays the stored run's confidence/symbols and rewrites only
        // the exit path.
        assert_eq!(replay.stage_exit, StageExit::Cache);
        assert_eq!(concise.stage_exit, StageExit::EarlyExit);
        assert_eq!(replay.retrieval_confidence, concise.retrieval_confidence);
        assert_eq!(replay.symbols, concise.symbols);
        let detailed = agent
            .run(&dir, &detailed_q("decide_freshness"))
            .await
            .unwrap();

        assert_eq!(
            kept_chars(concise.result.findings[0].snippet.as_deref().unwrap()),
            10
        );
        assert_eq!(
            kept_chars(detailed.result.findings[0].snippet.as_deref().unwrap()),
            50,
            "the detailed call must not be served the cached concise entry"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- M-1: the on-disk L2 and the two validation modes ---

    /// A fresh cache directory, following the crate's no-`tempfile` convention
    /// (`test_support::temp_repo_with`).
    fn temp_cache_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("agent_l2_{test}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cache_at(dir: &Path, key_mode: CacheKeyMode) -> CacheSettings {
        CacheSettings {
            dir: dir.display().to_string(),
            key_mode,
            ..CacheSettings::default()
        }
    }

    /// A `finish` with a summary but no findings — the summary-only entry that
    /// has no dependency stamps to validate.
    fn finish_call_empty() -> ToolCall {
        ToolCall {
            id: "c1".to_string(),
            name: "finish".to_string(),
            arguments_json: r#"{"findings":[],"summary":"nothing found"}"#.to_string(),
            thought_signatures: None,
        }
    }

    fn last_metrics() -> QueryMetrics {
        taped_metrics()
            .pop()
            .expect("every run exit emits a metrics record")
    }

    /// Build a loop that behaves like a fresh process over `cache_dir`: its own
    /// empty L1, the same store on disk.
    fn l2_agent(
        cache_dir: &Path,
        key_mode: CacheKeyMode,
        probe: MockRepoStateProbe,
    ) -> AgentLoop<
        MockMemoryBackend,
        MockSearchBackend,
        MockLlmProvider,
        MockRepoStateProbe,
        FakeClock,
    > {
        agent_with_probe_ttl_and_cache(
            MockMemoryBackend::new(),
            MockLlmProvider::new().with_responses(vec![
                tool_calls(vec![finish_call()]),
                tool_calls(vec![finish_call()]),
            ]),
            probe,
            TEST_INDEX_TRUST_TTL,
            cache_at(cache_dir, key_mode),
        )
    }

    #[tokio::test]
    async fn l2_hit_promotes_into_l1_and_reports_the_layer() {
        // The cross-session acceptance criterion: a new process on an unchanged
        // repo must be served from disk, and must report what that hit saved.
        let cache_dir = temp_cache_dir("promote");
        let repo = temp_repo("l2_promote");
        let query = q("where is main");
        let probe = || MockRepoStateProbe::new().with_fingerprint(Some(fp("abc")));

        let first = l2_agent(&cache_dir, CacheKeyMode::Strict, probe());
        let produced = first.run(&repo, &query).await.unwrap();
        let cold = last_metrics();
        assert_eq!(cold.cache_layer, None, "the producing run is not a hit");
        assert!(cold.llm_calls > 0);
        drop(first);

        let second = l2_agent(&cache_dir, CacheKeyMode::Strict, probe());
        let served = second.run(&repo, &query).await.unwrap();
        assert_eq!(served.result, produced.result);
        assert_eq!(served.retrieval_confidence, produced.retrieval_confidence);
        assert_eq!(served.stage_exit, StageExit::Cache);
        let hit = last_metrics();
        assert_eq!(hit.cache_layer, Some("l2"));
        assert_eq!(hit.turns_saved_by_cache, Some(cold.llm_calls));
        assert_eq!(hit.tokens_saved_by_cache, Some(cold.tokens));
        assert_eq!(hit.tokens, 0, "`tokens` keeps meaning spend");

        // The promote put it in this loop's L1, so the next run never touches
        // the disk again.
        second.run(&repo, &query).await.unwrap();
        assert_eq!(last_metrics().cache_layer, Some("l1"));

        std::fs::remove_dir_all(&repo).ok();
        std::fs::remove_dir_all(&cache_dir).ok();
    }

    #[tokio::test]
    async fn disk_cache_is_off_when_dir_is_empty() {
        // The default `CacheSettings` (dir = "") must behave exactly as before
        // M-1: L1 only, so a new loop starts cold.
        let repo = temp_repo("l2_no_dir");
        let query = q("where is main");
        let probe = || MockRepoStateProbe::new().with_fingerprint(Some(fp("abc")));
        let make = || {
            agent_with_probe_ttl_and_cache(
                MockMemoryBackend::new(),
                MockLlmProvider::new().with_responses(vec![tool_calls(vec![finish_call()])]),
                probe(),
                TEST_INDEX_TRUST_TTL,
                CacheSettings::default(),
            )
        };
        make().run(&repo, &query).await.unwrap();
        let _ = taped_metrics();
        make().run(&repo, &query).await.unwrap();
        assert_eq!(
            last_metrics().cache_layer,
            None,
            "without a cache dir there is no layer behind L1"
        );
        std::fs::remove_dir_all(&repo).ok();
    }

    #[tokio::test]
    async fn a_disk_write_failure_never_fails_a_query() {
        // `dir` names a regular file, so the store can never be opened. The
        // query must still succeed, on L1 alone.
        let repo = temp_repo("l2_unwritable");
        let holder = temp_cache_dir("unwritable");
        let file = holder.join("not-a-dir");
        std::fs::write(&file, b"x").unwrap();
        let agent = l2_agent(
            &file,
            CacheKeyMode::Strict,
            MockRepoStateProbe::new().with_fingerprint(Some(fp("abc"))),
        );
        let query = q("where is main");
        let result = agent.run(&repo, &query).await.unwrap();
        assert_eq!(result.result.findings.len(), 1);
        // And L1 still works.
        agent.run(&repo, &query).await.unwrap();
        assert_eq!(last_metrics().cache_layer, Some("l1"));
        std::fs::remove_dir_all(&repo).ok();
        std::fs::remove_dir_all(&holder).ok();
    }

    #[tokio::test]
    async fn paths_mode_recomputes_when_a_referenced_file_changed() {
        // MANDATORY staleness guard: the answer names `src/lib.rs`, so a
        // rewrite of that file must miss, whatever the mode's tolerance for
        // unrelated edits.
        let repo = temp_repo("paths_referenced");
        let probe = MockRepoStateProbe::new()
            .with_fingerprint(Some(fp("aaa")))
            .with_changed_paths(None);
        let handle = probe.clone();
        let agent = l2_agent(
            &temp_cache_dir("paths_referenced"),
            CacheKeyMode::Paths,
            probe,
        );
        let query = q("where is main");
        agent.run(&repo, &query).await.unwrap();
        let _ = taped_metrics();

        std::fs::write(repo.join("src/lib.rs"), "a\nb\nc\nd\ne\n").unwrap();
        handle.set_fingerprint(Some(fp("bbb")));
        agent.run(&repo, &query).await.unwrap();
        assert_eq!(
            last_metrics().cache_layer,
            None,
            "a changed referenced file must recompute"
        );
        std::fs::remove_dir_all(&repo).ok();
    }

    #[tokio::test]
    async fn paths_mode_serves_when_only_an_unreferenced_file_changed() {
        // The hit the mode exists for: `src/other.rs` is not in the answer.
        // `changed_paths: None` means strict would have invalidated.
        let repo = temp_repo("paths_unreferenced");
        let probe = MockRepoStateProbe::new()
            .with_fingerprint(Some(fp("aaa")))
            .with_changed_paths(None);
        let handle = probe.clone();
        let agent = l2_agent(
            &temp_cache_dir("paths_unreferenced"),
            CacheKeyMode::Paths,
            probe,
        );
        let query = q("where is main");
        let first = agent.run(&repo, &query).await.unwrap();
        let _ = taped_metrics();

        std::fs::write(repo.join("src/other.rs"), "d\ne\nf\ng\n").unwrap();
        handle.set_fingerprint(Some(fp("bbb")));
        let second = agent.run(&repo, &query).await.unwrap();
        assert_eq!(second.result, first.result);
        assert_eq!(second.stage_exit, StageExit::Cache);
        assert_eq!(last_metrics().cache_layer, Some("l1"));
        std::fs::remove_dir_all(&repo).ok();
    }

    #[tokio::test]
    async fn paths_mode_falls_back_to_strict_for_an_entry_with_no_deps() {
        // A summary-only answer stamps no files, so `paths` has nothing to
        // check — without the fallback it would be served forever.
        let repo = temp_repo("paths_no_deps");
        let probe = MockRepoStateProbe::new()
            .with_fingerprint(Some(fp("aaa")))
            .with_changed_paths(None);
        let handle = probe.clone();
        let agent = agent_with_probe_ttl_and_cache(
            MockMemoryBackend::new(),
            MockLlmProvider::new().with_responses(vec![
                tool_calls(vec![finish_call_empty()]),
                tool_calls(vec![finish_call_empty()]),
            ]),
            probe,
            TEST_INDEX_TRUST_TTL,
            cache_at(&temp_cache_dir("paths_no_deps"), CacheKeyMode::Paths),
        );
        let query = q("where is main");
        agent.run(&repo, &query).await.unwrap();
        let _ = taped_metrics();

        handle.set_fingerprint(Some(fp("bbb")));
        agent.run(&repo, &query).await.unwrap();
        assert_eq!(
            last_metrics().cache_layer,
            None,
            "a dep-less entry must fall back to strict validation"
        );
        std::fs::remove_dir_all(&repo).ok();
    }
}
