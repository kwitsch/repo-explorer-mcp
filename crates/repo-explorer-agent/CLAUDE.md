# repo-explorer-agent (exploration orchestrator)

`AgentLoop` drives `explore_repository`'s search, plus the tool
catalog/dispatch, compressed rendering, and fingerprint-keyed result caches.
Owns `serde_json` — core stays free of it.

`AgentLoop::run` returns core's `ExplorationOutcome`: the `ExplorationResult`
plus `retrieval_confidence` (the pre-stage's 0-100 candidate-set score, _not_
an answer confidence), `stage_exit` (`early-exit`/`verify`/`fallback`/`cache`,
the same four literals `QueryMetrics::path` logs) and a sparse
`symbols: Vec<(FileLocation, String)>`. It is built in exactly one place,
`complete_run`, and the symbols come from `render::symbols_for` — a pure
post-hoc join against the ranked candidates that emits a pair only when
**exactly one** candidate overlaps a finding's range (one shared line is
enough), so an ambiguous file yields nothing rather than a guess. A cache hit returns the stored outcome
with `stage_exit` rewritten to `cache` and everything else replayed. Full pipeline design:
`docs/project-plan/8-retrieval_pipeline.md`.

## Pipeline stages

- **Retrieval pre-stage** — concurrent symbol/BM25/semantic/grep/file fanout;
  exits with zero LLM calls when it finds a confident match. The BM25 leg
  (`search_graph{query}`) is the only one that forwards the query text as
  written — it ranks symbols by name relevance; its hits are `SemanticHit`.
  Verification skeletons come from `MemoryBackend::file_outline`
  (`get_file_outline`: exact path, source order, no container rows). Two routes into that
  Stage-3 exit, both requiring a trusted exact symbol match (F-16; a match on a
  plain-word name like `server`/`score` is trusted only in a query of at most
  `PLAIN_WORD_TRUST_MAX_IDENTIFIERS` identifier tokens — F-23/F-24), reported
  as `early_exit_route` in `QueryMetrics`: `confidence` (score clears
  `agent.early_exit_confidence`) and `unique-symbol` (exactly one trusted
  exact match at a known location — unambiguous by construction, so the
  score, which a strong `SymbolFuzzy` runner-up deflates, is not consulted;
  disable with `agent.skip_verify_on_exact_symbol = false`). Uniqueness is
  counted **pre-merge** (`merge_and_rank` folds two overlapping symbols in one
  file into one candidate and drops the loser's name), and the exit falls
  through to verification when that one authorizing candidate does not survive
  disk verification or the response caps.
- **Stage-1 index refresh** — before retrieval, `run` ensures a fresh memory
  index. Repeat calls for the same `repo_path` skip this upstream round-trip
  when the local git `RepoFingerprint` is unchanged since the last refresh and
  still within the trust window (`codebase_memory.staleness_seconds`); any git
  change (commit, checkout, working-tree edit) or an elapsed window forces the
  full flow. No fingerprint (not a git repo / probe failure) never skips. A
  skip candidate is still safety-netted by `MemoryBackend::probe_index_ready`
  (a cheap existence-only check, no `detect_changes`) — if the upstream index
  was lost/invalidated for a reason the fingerprint can't see, the full flow
  runs instead. `index_refresh_seen` (the per-repo mark map) is bounded/FIFO
  by `cache_settings.max_entries`, like the sibling caches in `cache.rs`.
- **LLM verification stage** — runs over the top-k candidate skeletons the
  pre-stage produced.
- **`finish` takes an optional `candidate_id`** — the 1-based `[n]` of the
  numbered candidate a finding came from. Both stages number their lists the
  same way and resolve ids against exactly the list they numbered
  (`verify::candidates_block` gets the whole ranked slice; Stage 5 numbers and
  resolves the `SEED_CANDIDATES` prefix, so an id past it can never resolve).
  When it resolves to a candidate **in the same file** with a known range that
  the finding's own range overlaps without sitting strictly inside it
  (`tools::snaps_to_candidate` — a mis-transcription of the candidate, not a
  different site that merely came from it), that range (re-clamped against the
  file, since Stage-4/5 candidates are not disk-verified) replaces the model's,
  and the snippet is re-derived for it. It is never a rejection reason: an
  absent, zero, out-of-range, other-file, unknown-location, disjoint, nested or
  past-EOF id simply leaves the finding's own verified location in place, so
  the loop can never become less able to report a genuine grep/`read_file`
  find. `QueryMetrics::cited_candidate_ids` counts the findings where it
  actually applied, at parse time (before the dedupe and `max_results` cap, so
  it can exceed `findings_count`) and only on the two legs where a `finish`
  call parsed — a run of zeroes across an eval means the field is dead weight
  and should be reverted.
- **Explorative fallback loop** — the hardened path when verification isn't
  confident: enforces a token budget, batches tool calls, and forces a final
  finish (via the Stage-4 `ProviderRouter`) rather than looping indefinitely.
  It is also the only stage that gets a **repo brief**: one deterministic
  `get_architecture_text` call, rendered by `brief.rs` into a token-budgeted
  markdown block (`[agent.repo_brief] max_tokens`, chars/4 estimate, over
  budget drops the smallest modules) and injected as a _second_
  `Message::system` — every system message carries its own provider cache
  breakpoint, so the static prefix stays byte-stable (see
  `fallback_cache_prefix_is_byte_stable`) and the "do NOT call
  get_architecture" hardening line ships inside the brief, never in
  `FALLBACK_SYSTEM_PROMPT`. Memoized in `cache.rs`'s `briefs` map, keyed on
  the repo fingerprint's HEAD alone or HEAD+dirty per `[agent.repo_brief]
key`. Every failure (disabled, backend error, unusable payload) degrades to
  the pre-M-2 single-system-message prompt. Verification (Stage 4) gets no
  brief — it is the common path, the brief is for the blind one.

## Per-query metrics

`QueryMetrics` (agent.rs) is emitted from **all five** `run` exit paths —
cache hit, early exit, verify, fallback, provider error (`path = "error"`) —
via `emit_metrics`. Two sinks: the headline fields as flat tracing fields on
that path's INFO log line (what `eval/run.py` parses), and, when
`REPO_EXPLORER_METRICS` names a non-empty path, the whole record as one
appended JSONL line — the only lossless copy (it also carries `repo_path`,
`query`, `findings_count`, `summary_len`). Never stdout — that is the MCP
JSON-RPC channel.

Stage-5-only fields: `brief_tokens` (estimated size of the injected repo
brief) and `orientation_calls_in_loop` (in-loop `get_architecture` calls
actually executed). Both are `Option<u32>`, seeded on Stage-5 entry — absent
means "Stage 5 never ran", never a fabricated `0` (tracing emits nothing for
`None`, the same `confidence`/`candidate_count` convention `eval/run.py`
relies on).

## Result cache: L1 in memory, L2 on disk

`cache.rs` holds the four in-memory maps (tools, legs, queries, briefs). Only
the **query** map has a second layer: `disk_cache.rs`, one JSON file per
result under `<[cache] dir>/results/v<SCHEMA_VERSION>/<fnv1a64(key)>.json`
(unix 0700/0600; on Windows the per-user `%LOCALAPPDATA%` ACL). Reached solely
through `ResultCache::{get_query_l2, put_query_l2}`, which do their I/O in
`spawn_blocking` and never hold the mutex across it.

- **No database, no locking.** Every entry is independent, idempotent and
  recomputable, so concurrent MCP processes need only an atomic replace
  (temp file + `fs::rename`). Two writers racing on one key both write a
  correct answer for it. Upgrade to sqlite only if `cache stats` ever has to
  report cross-session aggregates the metrics stream cannot derive.
- **`disk_cache::SCHEMA_VERSION` is `2`** (M-3 moved the stored payload from
  `ExplorationResult` to `ExplorationOutcome`). It is the single versioning
  point for the
  stored shape — directory segment _and_ `v` field — so a bump makes every old
  entry unreachable and the sweep deletes strictly _older_ version directories
  (never a newer one: a mixed-version window would otherwise leave both
  binaries permanently cold). Bump it (and nothing else) when the stored value
  changes. The shape spans two crates — the envelope in `disk_cache.rs`, the
  payload in core's `ExplorationOutcome` — so `stored_shape_is_pinned_to_the_
schema_version` pins the serialized JSON as a literal: any move on either
  side fails that test instead of silently changing the on-disk format.
- **The key is the query key plus the response cap.** L2 reuses
  `ResultCache::query_key` and `AgentLoop::run_query_key` appends the
  `snippet_max_chars` this loop would apply — a persisted entry is already
  truncated and outlives the config file that set the cap. `query_cache_key`
  (the MCP server's `req_id` source) stays on the bare query key: it is a
  correlation id, not a cache lookup. No fingerprint in the key either:
  invalidation stays in the _value_, which is what keeps the empty-diff
  relabel working.
- **One validity policy, `AgentLoop::entry_still_valid`, shared by both
  layers.** `[cache] key_mode = "strict"` (default) is the pre-M-1 behaviour:
  identical fingerprint, or a fingerprint change with a provably empty diff.
  `"paths"` additionally serves after an unrelated edit, by re-`stat`ing the
  `(len, mtime)` stamps of the files the answer references; an entry without
  stamps falls back to strict. A `"head"` mode is deliberately absent — it is
  the one policy that serves snippets an uncommitted edit already invalidated.
- **Every failure degrades to L1-only, never to a failed query**: an unusable
  directory makes `DiskCache::open` return `None`, the first write error
  logs and is dropped (never latched off — the next completed exploration
  retries, so a transient ENOSPC or sharing violation is recoverable), and a
  corrupt entry file is a miss and is deleted.
- **No L2 for tools/legs/briefs.** They are keyed on the full fingerprint, so
  they die on every working-tree save — near-zero cross-session value for a
  fraction of a run's cost.
- Cache-hit runs report `cache_layer` (`l1`/`l2`), `turns_saved_by_cache` and
  `tokens_saved_by_cache` in `QueryMetrics`; all three are `Option`, absent on
  every non-cache path.

## Judge input (Stage 10)

`judge_input::render_judge_state` renders the plain-text judge state for ONE
retrieval candidate (`JUDGE_STATE_VERSION = 1`, format pinned in
`repo_explorer_core::judge`). The 10a datagen generator and the 10b runtime both
call it, so training and serving see byte-identical input — **train/serve parity
by construction**. `render_judge_states` batches the per-candidate reads
(outline once per file, body window `[start-3, end+5]`), returning `None` for
unknown-location candidates (never judged). **Bump `JUDGE_STATE_VERSION`
(in core) on any change to the rendering or the D1 constants** — a checkpoint is
valid only for the version it was trained on.

`snapshot::retrieval_snapshot` runs the deterministic pre-stage alone (no index
refresh, no cache, no LLM) and classifies the result as `EarlyExit`/`Verify`/
`Fallback` via the shared `early_exit_route`. The datagen generator labels rows
only from `Verify`-stage snapshots (the only case 10b calls the judge). The disk
authorization of the real early exit is not simulated — an accepted approximation.

## Judge verification (Stage 4, `judge.mode`)

`judge_verify.rs`'s `judge_verify` is the Stage-4 judge path, dispatched from
`run` under the same `outcome.confidence >= fallback_confidence &&
!candidates.is_empty()` gate the LLM path uses. It renders judge states with
`judge_input::render_judge_states` (never anything else — train/serve
parity), calls the loop's `J: CandidateJudge`, and selects the candidates at
`score >= select_threshold * 10` permille (ordered by score descending, then
index ascending), disk-verifying the selection through the shared
`verified_candidate_findings` (below) exactly like every other verification
path.

- **`off`** (default) — unchanged: the LLM `verify` path runs, byte-identical
  to before this feature existed.
- **`laya`** — the judge path runs instead of the LLM. A selection that
  survives disk verification finishes the query at `stage_exit = "verify"`
  with zero LLM calls; an empty selection (`Escalate`) falls through to
  Stage 5; a judge error (`Failed`) degrades to the LLM `verify` path when
  `ProviderRouter::has_providers()` is true, otherwise falls through to
  Stage 5 with no judge scores. A judge failure **never fails a query**.
- **`shadow`** — both the LLM `verify` and `judge_verify` run concurrently
  (`futures_util::future::join`); the LLM's answer is always the one
  returned, and the judge's outcome is only recorded
  (`QueryMetrics.judge_outcome`, `.shadow_agreement`). The judge's scores are
  never passed on to Stage 5.

`QueryMetrics` gains `judge_mode`/`fallback_mode` (set on every exit path,
including cache) and, only on paths where the judge ran,
`judge_outcome`/`judge_ms`/`judge_candidates`/`judge_selected`/`judge_max_p`/
`shadow_agreement` — all `Option`, `None` (never a fabricated value) when the
judge did not run.

## Offline fallback (`agent.fallback = "off"`)

When Stage 4 falls through to Stage 5 with `agent.fallback = "off"`,
`AgentLoop::offline_fallback` replaces the LLM explorative fallback loop with
deterministic synthesis: order the candidates by judge score descending
(`None`/not-judged last, then by retrieval rank), disk-verify them in that
order through `verified_candidate_findings`, and keep the first
`OFFLINE_FALLBACK_MAX_FINDINGS = 5` survivors. Every finding is labelled
**unverified** in its note (`"; unverified candidate; judge p=0.31"` or
`"; unverified candidate; retrieval rank 2"` when no judge score exists), and
the summary says so too. It exits through `finalize_and_complete` with
`stage_exit = "fallback"`; no repo brief is fetched, so `brief_tokens` and
`orientation_calls_in_loop` stay `None`. `agent.fallback = "llm"` (default)
leaves Stage 5 exactly as it was before this feature.

`verified_candidate_findings` (extracted from
`AgentLoop::result_from_candidates`) is the single disk-verification path
shared by the early exit, the judge path above, and offline fallback: one
place enforces "never report a location that does not resolve on disk".

## Judge query-cache key

`AgentLoop::run_query_key` appends `judge=laya:<select_threshold>` when
`judge.mode == "laya"`, and `fallback=off` when `agent.fallback == "off"`.
`off`- and `shadow`-mode keys are byte-identical to before this feature, so
existing L1/L2 entries stay valid; `disk_cache::SCHEMA_VERSION` is unchanged
because the stored value shape did not change.
