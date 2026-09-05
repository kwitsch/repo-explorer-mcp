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

use repo_explorer_core::config::{AgentSettings, CacheSettings};
use repo_explorer_core::domain::{
    Candidate, ExplorationFinding, ExplorationQuery, ExplorationResult, FileLocation,
};
use repo_explorer_core::fingerprint::{RepoFingerprint, RepoStateProbe};
use repo_explorer_core::llm::{
    CallOptions, Clock, LlmProvider, Message, ProviderResponse, ProviderRouter, SystemClock,
    TokenUsage, ToolCall,
};
use repo_explorer_core::memory::{IndexStatus, MemoryBackend, MemoryError};
use repo_explorer_core::retrieval::{finding_from_candidate, is_unknown_location};
use repo_explorer_core::search::SearchBackend;
use std::collections::HashSet;
use std::path::Path;

use crate::cache::{QueryEntry, ResultCache};
use crate::dispatch::dispatch_inner;
use crate::pipeline;
use crate::render::{RenderCaps, dedupe_key, tidy_findings};
use crate::tools::{finish_only_catalog, parse_finish, resolve_finish, tool_catalog};
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
}

impl TokenBudget {
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            limit,
            spent: 0,
            llm_calls: 0,
        }
    }

    pub(crate) fn add(&mut self, usage: Option<TokenUsage>) {
        self.llm_calls += 1;
        if let Some(usage) = usage {
            self.spent = self.spent.saturating_add(usage.total());
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
}

/// Consecutive single-call turns rejected before one is executed anyway (the
/// 2-strike batching rule; the escape hatch keeps weak models from
/// deadlocking).
const MAX_SINGLE_CALL_REJECTIONS: u32 = 2;

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
    ) -> Self {
        let cache = cache_settings
            .enabled
            .then(|| ResultCache::new(cache_settings.max_entries));
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
        }
    }

    /// Deterministic per-query cache key, exposed so a caller (the MCP
    /// server's `explore` observability span) can derive a short
    /// request-correlation id from the same normalization the query cache
    /// uses internally.
    pub fn query_cache_key(query: &ExplorationQuery) -> String {
        ResultCache::query_key(query)
    }

    pub async fn run(
        &self,
        repo_root: &Path,
        query: &ExplorationQuery,
    ) -> Result<ExplorationResult, AgentLoopError> {
        // Stage 0: query cache.
        let git_probe_start = std::time::Instant::now();
        let fingerprint = match &self.cache {
            Some(_) => self.probe.fingerprint(repo_root).await,
            None => None,
        };
        let git_probe_ms = git_probe_start.elapsed().as_millis() as u64;
        let query_key = ResultCache::query_key(query);
        if let Some(hit) = self
            .query_cache_lookup(repo_root, &query_key, &fingerprint)
            .await
        {
            tracing::info!(
                path = "cache",
                tokens = 0u64,
                git_probe_ms,
                "exploration served from query cache"
            );
            return Ok(hit);
        }

        // Stage 1: ensure a fresh index (once). Failures are non-fatal notes.
        let index_result = self.memory.ensure_fresh_index(repo_root).await;
        let index_status = index_status_label(&index_result);
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

        let mut budget = TokenBudget::new(self.settings.token_budget);

        // Stage 3: early exit — the pre-stage already answered.
        if outcome.confidence >= self.settings.early_exit_confidence
            && !outcome.candidates.is_empty()
        {
            let result = self.result_from_candidates(
                outcome.candidates,
                query,
                outcome.confidence,
                index_note.as_deref(),
            );
            return Ok(self.complete_run(
                "early-exit",
                0,
                0,
                false,
                index_status,
                git_probe_ms,
                &query_key,
                fingerprint,
                result,
            ));
        }

        // Stage 4: LLM verification over the candidates.
        if outcome.confidence >= self.settings.fallback_confidence && !outcome.candidates.is_empty()
        {
            if let VerifyOutcome::Finished(result) = verify(
                &self.memory,
                &self.router,
                repo_root,
                query,
                index_note.as_deref(),
                &outcome.candidates,
                self.settings.max_verify_iterations,
                &mut budget,
                &self.caps,
            )
            .await
            {
                return Ok(self.finalize_and_complete(
                    "verify",
                    result,
                    query.max_results,
                    &budget,
                    false,
                    index_status,
                    git_probe_ms,
                    &query_key,
                    fingerprint,
                ));
            }
            tracing::info!("verification escalated to the fallback loop");
        }

        // Stage 5: explorative fallback loop.
        let (result, forced_finish) = self
            .fallback_loop(
                repo_root,
                query,
                index_note.as_deref(),
                outcome.candidates,
                fingerprint.as_ref(),
                &mut budget,
            )
            .await?;
        Ok(self.finalize_and_complete(
            "fallback",
            result,
            query.max_results,
            &budget,
            forced_finish,
            index_status,
            git_probe_ms,
            &query_key,
            fingerprint,
        ))
    }

    /// The shared dedupe-then-truncate contract: dedupe first so a run of
    /// colliding-sentinel findings can't push a legitimate distinct finding
    /// out of the `max_results` cap. Used by both `finalize` (verify/fallback
    /// results) and `result_from_candidates` (the early-exit path) so the two
    /// can't silently diverge.
    fn tidy_and_truncate(
        &self,
        findings: Vec<ExplorationFinding>,
        max_results: Option<u32>,
    ) -> Vec<ExplorationFinding> {
        let mut findings = tidy_findings(findings, &self.caps);
        if let Some(max) = max_results {
            findings.truncate(max as usize);
        }
        findings
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
        max_results: Option<u32>,
    ) -> ExplorationResult {
        result.findings = self.tidy_and_truncate(result.findings, max_results);
        result
    }

    /// The shared tail of every `run()` branch: log completion, persist to
    /// the query cache, and hand back the result for the caller to wrap in
    /// `Ok`.
    #[allow(clippy::too_many_arguments)]
    fn complete_run(
        &self,
        path: &'static str,
        tokens: u64,
        llm_calls: u32,
        forced_finish: bool,
        index_status: &'static str,
        git_probe_ms: u64,
        query_key: &str,
        fingerprint: Option<RepoFingerprint>,
        result: ExplorationResult,
    ) -> ExplorationResult {
        tracing::info!(
            path,
            tokens,
            llm_calls,
            forced_finish,
            index_status,
            git_probe_ms,
            "exploration complete"
        );
        self.store_query_cache(query_key, fingerprint, &result);
        result
    }

    /// The shared tail of the verify/fallback branches: finalize the result,
    /// then run it through `complete_run` with the tokens spent so far.
    #[allow(clippy::too_many_arguments)]
    fn finalize_and_complete(
        &self,
        stage: &'static str,
        result: ExplorationResult,
        max_results: Option<u32>,
        budget: &TokenBudget,
        forced_finish: bool,
        index_status: &'static str,
        git_probe_ms: u64,
        query_key: &str,
        fingerprint: Option<RepoFingerprint>,
    ) -> ExplorationResult {
        let result = self.finalize(result, max_results);
        self.complete_run(
            stage,
            budget.spent(),
            budget.llm_calls(),
            forced_finish,
            index_status,
            git_probe_ms,
            query_key,
            fingerprint,
            result,
        )
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

    /// Serve from the query cache when the entry is still valid: same
    /// fingerprint, or a fingerprint change that provably changed nothing at
    /// all. Checking the diff against only the *entry's own* contributing
    /// paths is unsound — retrieval scans the whole repo, so a path outside
    /// those paths (not least a newly added file) can still turn into a
    /// better match that the stale entry never saw — so any actual diff
    /// invalidates the entry.
    async fn query_cache_lookup(
        &self,
        repo_root: &Path,
        query_key: &str,
        fingerprint: &Option<RepoFingerprint>,
    ) -> Option<ExplorationResult> {
        let (cache, fp) = self.cache_for(fingerprint.as_ref())?;
        let entry = cache.get_query(query_key)?;
        if entry.fingerprint == *fp {
            return Some(entry.result);
        }
        match self
            .probe
            .changed_paths(repo_root, &entry.fingerprint, fp)
            .await
        {
            Some(changed) if changed.is_empty() => {
                cache.refresh_query_fingerprint(query_key, &entry.fingerprint, fp.clone());
                Some(entry.result)
            }
            _ => {
                cache.remove_query(query_key, &entry.fingerprint);
                None
            }
        }
    }

    fn store_query_cache(
        &self,
        query_key: &str,
        fingerprint: Option<RepoFingerprint>,
        result: &ExplorationResult,
    ) {
        let (Some(cache), Some(fingerprint)) = (&self.cache, fingerprint) else {
            return;
        };
        cache.put_query(
            query_key.to_string(),
            QueryEntry {
                fingerprint,
                result: result.clone(),
            },
        );
    }

    /// Build the early-exit result straight from the ranked candidates, via
    /// the same dedupe-then-truncate contract as `finalize()`. `index_note`
    /// (e.g. a failed reindex) is appended to the summary — this is the only
    /// stage that doesn't already thread it into an LLM prompt, so it must be
    /// surfaced here or a confident answer from a stale index would carry no
    /// warning at all.
    fn result_from_candidates(
        &self,
        candidates: Vec<Candidate>,
        query: &ExplorationQuery,
        confidence: u32,
        index_note: Option<&str>,
    ) -> ExplorationResult {
        let findings = self.tidy_and_truncate(
            candidates.into_iter().map(finding_from_candidate).collect(),
            query.max_results,
        );
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
    async fn fallback_loop(
        &self,
        repo_root: &Path,
        query: &ExplorationQuery,
        index_note: Option<&str>,
        candidates: Vec<Candidate>,
        fingerprint: Option<&RepoFingerprint>,
        budget: &mut TokenBudget,
    ) -> Result<(ExplorationResult, bool), AgentLoopError> {
        let tools = tool_catalog();
        let mut messages: Vec<Message> = vec![
            Message::system(FALLBACK_SYSTEM_PROMPT),
            Message::user(user_prompt(query, index_note, &candidates)),
        ];

        let mut findings: Vec<ExplorationFinding> = Vec::new();
        let mut seen: HashSet<(FileLocation, Option<String>)> = HashSet::new();
        let mut single_call_rejections = 0u32;
        let mut turn_limit_hit = true;

        for turn in 0..self.settings.max_fallback_iterations {
            if budget.exhausted() {
                turn_limit_hit = false;
                break;
            }
            match self
                .router
                .complete_with_tools(&messages, tools, &CallOptions::default())
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
                            let mut turn_messages: Vec<Message> = match resolve_finish(&calls) {
                                Ok(result) => return Ok((result, false)),
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
        if let Some(result) = self.forced_finish(&mut messages, budget).await {
            return Ok((result, true));
        }
        for candidate in candidates {
            accumulate(&mut findings, &mut seen, finding_from_candidate(candidate));
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
        ))
    }

    /// One last call offering only `finish`, with the tool choice forced.
    async fn forced_finish(
        &self,
        messages: &mut Vec<Message>,
        budget: &mut TokenBudget,
    ) -> Option<ExplorationResult> {
        messages.push(Message::user(
            "The exploration budget is exhausted. Call finish NOW with the best findings gathered so far.",
        ));
        match self
            .router
            .complete_with_tools(messages, finish_only_catalog(), &force_finish_options())
            .await
        {
            Ok(completion) => {
                budget.add(completion.usage);
                if let ProviderResponse::ToolCalls(calls) = completion.response {
                    for call in &calls {
                        if call.name == "finish"
                            && let Ok(result) = parse_finish(&call.arguments_json)
                        {
                            return Some(result);
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
                ResultCache::tool_key(fp, &call.name, &call.arguments_json),
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
When you have gathered enough information, you MUST conclude by calling the finish tool with your findings and a summary.";

/// How many retrieval candidates are listed as starting points in the
/// fallback prompt.
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
pub(crate) fn force_finish_options() -> CallOptions {
    CallOptions {
        force_tool: Some("finish".to_string()),
        max_tokens: None,
    }
}

/// The 4-part preamble shared by the fallback loop's and the verification
/// stage's user prompts: query text, scope hint, max_results, index note.
pub(crate) fn query_preamble(query: &ExplorationQuery, index_note: Option<&str>) -> String {
    let mut s = format!("Exploration query: {}", query.text);
    if let Some(scope) = &query.scope_hint {
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
    index_note: Option<&str>,
    candidates: &[Candidate],
) -> String {
    let mut s = query_preamble(query, index_note);
    if !candidates.is_empty() {
        s.push_str(
            "\nStarting points found by the deterministic retrieval pre-stage (ranked, may be incomplete):",
        );
        for c in candidates.iter().take(SEED_CANDIDATES) {
            let symbol = c
                .symbol
                .as_deref()
                .map(|sym| format!(" `{sym}`"))
                .unwrap_or_default();
            if is_unknown_location(&c.location) {
                s.push_str(&format!(
                    "\n- {} (location unknown){symbol}",
                    c.location.path.display(),
                ));
            } else {
                s.push_str(&format!(
                    "\n- {}:{}-{}{symbol}",
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
    use repo_explorer_core::config::AgentSettings;
    use repo_explorer_core::domain::CandidateKind;
    use repo_explorer_core::fingerprint::RepoFingerprint;
    use repo_explorer_core::fingerprint::mock::MockRepoStateProbe;
    use repo_explorer_core::llm::mock::{FakeClock, MockLlmProvider};
    use repo_explorer_core::llm::{Completion, ToolCall};
    use repo_explorer_core::memory::mock::MockMemoryBackend;
    use repo_explorer_core::search::SearchError;
    use repo_explorer_core::search::mock::MockSearchBackend;
    use std::path::PathBuf;

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
        };
        let got = agent.run(&PathBuf::from("/repo"), &query).await.unwrap();
        assert_eq!(got.summary, "done");
        assert_eq!(got.findings.len(), 1);
        assert_eq!(got.findings[0].location.line_start, 1);
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
        };
        let got = agent.run(&PathBuf::from("/repo"), &query).await.unwrap();
        assert_eq!(got.findings.len(), 1, "capped to max_results");
        assert_eq!(got.findings[0].location.path, PathBuf::from("src/lib.rs"));
    }

    #[test]
    fn early_exit_dedupes_before_truncating_to_max_results() {
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
        let agent = agent_with(MockLlmProvider::new());
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: Some(3),
        };
        let result = agent.result_from_candidates(candidates, &query, 100, None);
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
    }

    #[tokio::test]
    async fn early_exit_surfaces_stale_index_note_in_summary() {
        // Regression: Stage 3's early exit must not silently drop Stage 1's
        // index-freshness note — a confident answer served from a stale
        // index (reindex failed, memory backend still answers from the
        // previous index) must still say so.
        let memory = MockMemoryBackend::new()
            .with_ensure_fresh_index_result(Ok(IndexStatus::IndexingFailed {
                reason: "boom".to_string(),
            }))
            .with_search_graph_result(Ok(ExplorationResult {
                findings: vec![ExplorationFinding {
                    location: FileLocation {
                        path: PathBuf::from("crates/x/src/freshness.rs"),
                        line_start: 12,
                        line_end: 12,
                    },
                    snippet: None,
                    note: Some("decide_freshness".to_string()),
                }],
                summary: "1 row".to_string(),
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
        );
        let query = ExplorationQuery {
            text: "decide_freshness".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let got = agent.run(&PathBuf::from("/repo"), &query).await.unwrap();
        assert!(
            got.summary.contains("memory index could not be refreshed"),
            "early-exit summary must surface the index-freshness note: {}",
            got.summary
        );
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
        );
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: None,
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

        let key = ResultCache::tool_key(&fp, &call.name, &call.arguments_json);
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
        let mut b = TokenBudget::new(0);
        b.add(Some(TokenUsage {
            prompt_tokens: u64::MAX,
            completion_tokens: 1,
        }));
        assert!(!b.exhausted(), "0 means unlimited");

        let mut b = TokenBudget::new(10);
        assert!(!b.exhausted());
        b.add(None);
        assert!(!b.exhausted());
        b.add(Some(TokenUsage {
            prompt_tokens: 7,
            completion_tokens: 3,
        }));
        assert!(b.exhausted(), "exactly at the limit counts as exhausted");
    }
}
