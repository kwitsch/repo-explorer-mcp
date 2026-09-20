# explore_repository real-world eval — pilot scaffold

Implements the **pilot scope** (Phases 0-1, `self` + `requests` only) of
[`docs/eval/real-world-test-plan.md`](../docs/eval/real-world-test-plan.md). The rest of the
plan (Tier-2 repos, config sweep, robustness matrix, Claude-in-the-loop) is written but not yet
built out here — see the plan's §11 phase table.

## Status (2026-09-05)

- `repos.toml`, `queries/self.yaml`, `queries/requests.yaml` (16 ground-truthed queries each),
  `config/default.toml`, `run.py`, `score.py` exist and are harness-validated end to end,
  **including Mode A** — the §3.1 observability patch was built, independently re-verified,
  merged to `main`, released as **v0.5.3**, and installed (`repo-explorer-mcp --version` →
  `0.5.3`); it is now the installed SUT. `run.py`/`score.py` were then rewritten to consume its
  new structured lines (see "Mode A parser" below) and re-validated with real live calls against
  the installed 0.5.3 binary.
- Bugs found and fixed by live smoke calls, before any full pass ran:
  - `run.py` read `result.isError` (the MCP wire field name); the Python SDK's
    `CallToolResult` exposes it as `is_error`. Every error response was silently scored as a
    success until this was caught.
  - Every hand-authored ground-truth `span` for a symbol **definition** was originally just the
    `def`/`fn` line, not the full body — the tool correctly returns whole functions, and
    `score.py`'s `range_hit` (span overlap within a 3×/40-line cap, per plan §7.1) rightly
    flagged the too-narrow spans as misses. All symbol-definition spans in both YAML files were
    recomputed from the actual pinned checkouts (brace-matching for Rust, a
    signature-aware indentation walk for Python — a naive indent walk breaks on a multi-line
    `def foo(\n  ...\n) -> T:` signature whose closing paren sits back at the `def`'s own
    indent) and corrected. Call-site references (P4, P2) and single-line literals (P3) were
    correctly single-line already and were left alone.
  - **New, from the Mode-A re-validation**: a whole-file `SemanticHit` candidate (no specific
    symbol, CBM matched somewhere in the file) consistently reports `line_end` = the file's
    actual line count **+ 1** (`config.rs`: 980 real lines, reported `line_end: 981`;
    `update.rs`: 1292 real lines, reported `line_end: 1293`) — `score.py`'s `range_outside_file`
    hallucination check catches this. Traced to `repo-explorer-memory/src/backend.rs`
    `parse_line_range`, which parses whatever `"start-end"` string codebase-memory-mcp's
    `search_code` sends verbatim with no clamp against the file's real length — likely an
    inclusive/exclusive line-range convention mismatch with the upstream CBM tool, not a bug
    introduced by the observability patch. Filed as **F-13**, and since fixed in `07b6602`
    (`correct_module_end` decrements a `Module` row's `line_end` by 1).
  - Also observed live: the pinned `~/repos/eval-corpus/self` clone was not yet indexed by
    codebase-memory-mcp on its first call (`retrieval leg failed leg="symbol"`/`"semantic"`,
    "project not found or not indexed"), confirming F-08 (every new server process reindexes on
    its first call) in the strongest form — the _first_ call's own memory legs failed outright,
    and only the _second_ call (13s later) saw the newly-indexed project. This is exactly why
    the warm-up call exists (§8.1 step 3) and why it must never be scored.
  - Also observed live: an early smoke call's fallback-loop LLM attempt failed with
    `llm provider error: all configured LLM providers are exhausted or cooling down` — real
    evidence for the plan's own risk that this machine's Gemini quota is shared and can be
    exhausted by ordinary background activity (§13). Confirms why the _full_ Phase 1 run
    (2 passes × 32 queries) should happen with Claude Code closed, per the accepted decision
    below — it was not run in this session for that reason; every live call made while
    developing/validating the harness was a single deterministic (0-LLM) query.
- **Mode A parser** (`run.py`): every call is wrapped server-side in an
  `explore{req_id=<hash>-<n>}` tracing span; `n` is a per-process 0-based counter this script
  mirrors exactly (`_ReqCounter`), so stderr lines are associated with their call by that counter
  rather than a line-count/timestamp guess. A generic `extract_fields()` tokenizer handles every
  quoting convention tracing_subscriber actually uses (verified against live captured output, not
  just the source diff): bare fields are quoted, `%field` (Display) fields are not, bracketed
  fields are JSON/Debug-parsed. `score.py` now computes `cand_recall@top_k`/`cand_rank` (was the
  primary target ever in the pre-stage's own ranked list, independent of the LLM stage's
  decision), a full §7.4 failure-attribution class per failed query, provider call outcomes and
  model-drift detection against the run's pinned model, per-leg latency, forced-finish rows, and
  index-status distribution — replacing the old "NOT YET AVAILABLE" placeholder section. A row
  from an older (pre-0.5.3) binary still scores fine; those sections just report zero rows.
- `mcp.json`, `empty-mcp.json`, `claude-profile/settings.json`, `fixtures/make_r18.sh` are
  scaffolded for later phases (4, 3) but not yet exercised.
- `baseline.py` (Layer B), `gen_synthetic.py`, `claude_loop.sh`, `judge_prompt.md` — **not yet
  built**; out of pilot scope (Phase 2+).
- **Phase 1 run for real** (`results/20260905T145436`, installed v0.5.4, full production
  6-model Gemini chain, PR #30's 404-failover live): 254/303 provider calls `ok`, 39 `quota`,
  10 `rate_limited`, 0 hard failures — first pass with quota no longer dominating the result.
  `cand_recall@top_k = 0.38` (95% CI [0.26, 0.51]); negative queries (`N`) score
  `negative_ok = 1.00` (a scoring bug had this misreported as `file_hit@3 = 0.00`, a false
  "complete miss" — `score.py` never printed `negative_ok` and lumped `N` into the file-hit
  table where 0 is actually the correct outcome; both are now fixed, see below).
  `index_status` is `IndexingFailed` on all 60 calls that reported one — filed as **F-14** in
  the plan (`index_repository` fails against the installed codebase-memory-mcp with reason
  `"repo_path is required"`; likely contributor to the low `cand_recall`).
- `score.py` hallucination scoring had two more bugs, found and fixed against this run
  (commit `0eec259` + a follow-up): (1) `snippet_found_at` flattened multi-line snippets and
  compared them against single physical source lines, so a correctly-quoted multi-line snippet
  (struct/function bodies) could never match and was flagged `fabricated_snippet`; whole-file
  `Module` hits were also flagged `misaligned_snippet` since the "near `line_start`" check
  assumed a narrow span. Now matches per-chunk (splitting on `"..."`/`"…[truncated]"` omission
  markers) against contiguous file-line runs, and exempts whole-file spans from the alignment
  check — dropped the flagged-hallucination count from ~50 to 34 on the same data (still not
  confirmed zero; a manual spot-check found at least one of the 34 to be a real LLM digest
  rather than a literal quote). (2) `negative_ok` was computed per-row but never printed, and
  the `file_hit@3` table included negative queries where a 0 is the _correct_ outcome — added a
  dedicated negative-queries section and excluded `N` from the file-hit table.
- `score.py`'s `snippet_found_at` "near"-alignment window ignored `line_end` — filed
  as **F-17** in the plan (widened to the full claimed span, commit `4ca0a18`;
  rescoring `results/20260905T220843` drops hallucinations from **32** to **11**,
  though an escalated-review follow-up further tightened the function afterwards
  and that count needs reconfirming; see the F-17 row in
  `docs/eval/real-world-test-plan.md` for the full breakdown and residual
  follow-ups).

## Coverage gaps closed 2026-09-20

Found by auditing the harness against the 0.10.1 pipeline after the F-08/#56 memory outage
(`results/2026091*`: `symbol`/`semantic` failed on 100% of calls for two weeks and no report line
said so — the per-leg latency table only lists legs that _succeeded_):

- `score.py` prints a **Retrieval leg health** section (runs / failed / with_hits per leg, from
  `leg_timings` + `leg_failures`) and a `WARNING: memory legs ... dead` line when no
  `symbol`/`semantic`/`bm25` leg returned a candidate in the whole run; `cand_recall` gains
  `hit delivered by candidate kind` (Symbol*/SemanticHit = memory, ContentHit/FileNameHit =
  native search). Rescoring `results/20260916T230101` now prints the warning; the 0.8.0 and
  0.10.1 runs do not.
- The warm-up call was `queries[0]`, which seeded the process L1 cache — `P1-01` was a
  `stage=cache` row in **every scored pass of every run** and never measured. `repos.toml` now
  carries a dedicated `warmup_query` (a bare unique symbol not in the corpus).
- `Confident-wrong` lines print `early_exit_route` (F-23 is a `unique-symbol` route defect).
- `P1`/`P1-DE` queries carry `stage: early-exit`, so an early-exit regression shows up in the
  stage-mismatch section instead of nowhere; one `sub: near` negative per repo (`self-N-02`,
  `requests-N-02`) — the plan's N category had only `far` negatives.
- `run.py --binary PATH` drives a build other than the installed one; `manifest.json` records
  `memory_version` (`codebase-memory-mcp --version`) next to `binary_version`.
- `eval/test_score.py` runs in CI (`test-eval` job, gated).

Still unmeasured, by design of a scripted client: `outer_followup_rate`, the manual rubrics
(§7.2), Layer B baselines, `pre_stage_identical`/`variance` (F-03), `result_bytes`.

## Decisions in force for this pilot (accepted 2026-09-05)

- Logging patch: built, independently re-verified, merged, released as v0.5.3, and installed —
  done.
- No interactive Claude Code session while the harness makes real MCP calls against the pinned
  repos (§3.2) — Phases 1-3 and 5 run with Claude Code closed. Every live call made so far was a
  single deterministic (0-LLM, early-exit or cache-hit) query used to validate the harness, not a
  full pass, made before and after this constraint's practical cost (shared CBM daemon, shared
  Gemini quota) was confirmed live by one of those very calls.
- Scope: pilot first — Phase 0 (this scaffold) + Phase 1 (harness validation on `self` +
  `requests`, 2 passes) only. Extending to the full plan is a separate decision.

## Running Phase 1 (harness validation)

Requires Claude Code closed on this machine (daemon/quota sharing, see above) and
`GOOGLE_API_KEY` set in the environment:

```bash
cd /home/kwitsch/repos/repo-explorer-mcp
uv run --with mcp --with pyyaml eval/run.py --repos self requests --passes 2
uv run --with pyyaml eval/score.py results/<run-id>
```

`run.py` writes `results/<run-id>/manifest.json` plus, per repo per pass, a `.jsonl` row file and
a `.stderr.log`. `score.py` prints the §7.1 report — pass@1 per query, per-category `file_hit@3`
with Wilson CIs, hallucinations, confident-wrong cases, stage mismatches, §7.4 failure
attribution, candidate recall@top_k, provider call outcomes and model drift, forced-finish rows,
per-leg latency, and index-status distribution — and can dump the full scored data as JSON with
`--json-out`.

## QW-0 efficiency metrics (added 2026-09-15)

`score.py` prints a **QW-0 efficiency** block at the end of the report, over the same scored rows
as every other section (the `-warmup` row is excluded):

- `llm_turns_per_query` — mean and p95 of `llm_calls`. A cache hit counts as 0 turns by
  construction, even in an older run whose cache line predates the field.
- `stage_exit` — count and share per stage (`verify` / `early-exit` / `cache` / `fallback` /
  `error`), with early-exit split by `early_exit_route` (`confidence` / `unique-symbol` / `none`).
- `tokens_per_query` — mean/p50/p95, reported **both ways**: over all rows, and over
  LLM-touching rows only. `tokens` is 0 by construction on cache and early-exit rows, so the
  all-rows mean is a per-query figure and the LLM-rows mean is a per-LLM-query figure; either
  one alone reads as the other.
- `cached_ratio` — `cache_read_tokens / prompt_tokens` across the run. Reported as
  `n/a - provider reports no cache activity` when either side is absent, never as a silent 0%.
- `cost_per_query` — USD, from the dated `PRICES` table in `score.py`
  (`cost = Σ prompt × p_in + (completion + reasoning) × p_out`, plan §7.1). A model with no price
  entry makes the run's cost `n/a` and names the unpriced models; it is never costed at 0. A
  failed attempt that reports no token usage costs 0 and is not counted as unpriced.
- `latency_per_query` / `total_ms` — mean and p95.
- `brief_tokens` / `orientation_calls_in_loop` (M-2, added 2026-09-15) — **denominated over the
  rows that carry `orientation_calls_in_loop`**, the same restriction `early_exit_route` gets.
  The server emits both exclusively on the Stage-5 path (the deterministic repo brief is
  prefetched on entry to the explorative fallback loop and nowhere else), so a
  verify/early-exit/cache row has no value to contribute and must not pad the denominator. That
  is the field's presence, not `stage == "fallback"`: a Stage-5 run that ends in a provider
  error reports `stage == "error"` and still carries a truthful orientation count, so filtering
  on the stage name would silently drop it from the acceptance gate. `brief_tokens` is the server's own
  chars/4 estimate of the injected brief; `orientation_calls_in_loop` counts the
  `get_architecture` calls the model still made _inside_ the loop — the M-2 acceptance gate is
  `< 0.2` per Stage-5 query, which is only readable against a Stage-5 denominator. Note the
  corpus currently produces very few fallback rows (2 of 64 in the committed baseline, both the
  same query), so `n` on these two lines is small; read them alongside the per-query CSV
  filtered on `stage == fallback`, not as run-wide means.
- `cited_candidate_ids` (M-3, added 2026-09-16) — **denominated over the rows that carry the
  field**, i.e. the runs whose `finish` call actually parsed (verify and fallback; the cache,
  early-exit and no-finish-synthesis legs emit nothing and must not pad the denominator with
  zeroes). It counts the findings whose location came from the numbered candidate registry
  instead of the model's own transcription, counted at parse time and therefore before the
  dedupe and `max_results` cap — it can exceed that row's finding count, and is a
  "do the models use the field at all" signal, not a rate. A run of zeroes means
  `candidate_id` is dead weight and should be reverted.
- `cache_hit_rate` / `cache_hit_l1` / `cache_hit_l2` (M-1, added 2026-09-16) — **denominated
  over every scored row**, and the report line says so, because a hit rate only means anything
  against the set of rows that could have hit. Here that is all of them: each query is issued
  once per pass, and every pass runs a fresh server process, so a pass-2 row repeating a pass-1
  query is exactly the cross-session repeat the persistent L2 cache exists for. (The warm-up
  call is excluded from the denominator as everywhere else — it is also what seeds L1 for
  `queries[0]`, so the in-pass L1 hits in an older run are a harness artifact, not caller
  behaviour.) **A default run reports `cache_hit_l2` as ~0 on purpose**: `config/default.toml`
  pins `[cache] persistent = false`, because an L2 that survives the process would turn every
  pass >= 2 into a replay of pass 1 and collapse `llm_calls`, `tokens`, `cost_usd`,
  `confidence`, `candidate_count` and the pass-to-pass variance into one sample copied N times.
  Measure the cross-session rate deliberately instead:

  ```bash
  uv run --with mcp --with pyyaml eval/run.py --repos self --passes 1 --cache-dir /tmp/rex-eval-l2
  uv run --with mcp --with pyyaml eval/run.py --repos self --passes 1 --cache-dir /tmp/rex-eval-l2
  ```

  The second run is the measurement (the first only fills the store); `run.py` writes the
  derived config into that run's `out_dir` as `config.cache.toml` and records the directory in
  `manifest.json`. Start from an empty `--cache-dir` — a store carried over from an earlier run
  makes even the first invocation warm, and the acceptance number unreproducible. The layer split gets its own presence test rather than riding that denominator: a
  pre-M-1 binary emits `stage == "cache"` rows with no `cache_layer` field, so the total rate
  stays a real measurement while `cache_hit_l1` / `cache_hit_l2` report absent. A run with no
  cache row at all reports a real `0.0%` — `stage` is always emitted, so that is a measurement,
  not an absence.

- `turns_saved_by_cache` / `tokens_saved_by_cache` (M-1) — **denominated over the cache rows
  only**, like the M-2 pair above. They are what the _producing_ run spent, i.e. what the hit
  avoided; a non-hit row has nothing to contribute and would only drag the mean toward zero.
  They are deliberately not folded into `tokens`, which keeps meaning spend — `tokens` stays 0
  on a cache row, so the cost aggregate is unaffected.
- `schema_valid_rate` / `stage_exit_log_agreement` (M-3, added 2026-09-16) — read out of the
  answer itself, not the log. The server returns `Json<ExplorationResultDto>` and rmcp fills
  both `structuredContent` and a text block holding the compact JSON of that same object, so
  `run.py` needs no new field: `score.py` parses `stage_exit` and `retrieval_confidence` out of
  the `row["response"]` it already loads. An answer is schema-valid when `stage_exit` is one of
  `early-exit` / `verify` / `fallback` / `cache` and `retrieval_confidence` is an integer in
  0-100. **Denominated over the answers that carried either key**, and the line prints that
  denominator (`N of M answers carried the M-3 keys`) so a run where only some answers have
  them is visible rather than rounding to 100%. An error row carries no structured payload and
  is excluded, not counted as a violation; a pre-M-3 run reports both lines as absent.
  `stage_exit_log_agreement` cross-checks the response's `stage_exit` against the same run's
  log-parsed `stage` — the two are the same four literals by construction
  (`StageExit::as_str`), so anything below 100% is a real defect on one side or the other.
- `hallucinated_path_rate` (M-3) — fabricated-path findings over **every returned finding**: a
  path in the answer that does not exist in the repo under test. It reads the filesystem, not a
  server field, so it is computable on old and new runs alike, and reports `n/a` only when the
  run returned no findings at all. Read it as a regression alarm, not as an M-3 win: every
  `finish` location and every early-exit location goes through an on-disk check
  (`verify_location`), so it normally reads 0 — but the budget/turn-limit synthesis
  (`agent.rs`, no `finish` call parsed) returns ranked candidates straight from the index
  without one, so a non-zero reading there is real (a stale index naming a renamed/deleted
  file), not a scorer artifact. The committed baseline reads `0.00% (0 of 191)`.
- `location hallucinations by stage` (M-3) — the `range_outside_file` and `misaligned_snippet`
  counts split by the stage that produced them. This is the one accuracy number M-3 can
  actually move: a real path with a wrong _range_ is the residual error that citing an
  inspected candidate id fixes. See the `outer_followup_rate` gap below for why this proxy, and
  not the plan's original acceptance metric.

Every metric degrades to `n/a (field absent in this run)` on a results/ dir written before the
field existed, so old runs rescore without crashing and without reporting a fabricated zero.

`--csv-out PATH` writes one CSV row per scored query (run id, repo, pass, query_id, cat, stage
plus the QW-0 columns) and a sibling `PATH.aggregate.csv` of `metric,value` pairs. stdlib `csv`,
no pandas.

Prices in `score.py`'s `PRICES` are USD per 1M tokens, captured **2026-09-15** from
<https://ai.google.dev/pricing>, covering the six models in `config/default.toml`'s failover
chain. They are the published **tier** rates (Flash / Flash-Lite) applied to every model in that
tier — the individual 3.x per-model rates were not re-verified on the capture date. Treat
`cost_per_query` as a signal comparable _between runs of this harness_, not as an invoice, and
re-check the table before quoting a dollar figure anywhere else.

### Known gap: `outer_followup_rate` is not measurable here

The plan's `outer_followup_rate` — how often the _calling_ agent has to issue a follow-up
`explore_repository` call (or fall back to its own Grep/Read) after one answer — cannot be
produced by this harness at all. `run.py` is a scripted MCP client: it issues exactly one call
per query and never decides it needs another, so the metric is structurally always 0 here.
Measuring it needs a real outer-agent arm (Layer B/C of the plan): Claude Code sessions driven
against the pinned repos with the isolated profile in `claude-profile/`, their transcripts parsed
for follow-up tool calls, and a judge pass to separate "the answer was incomplete" from "the
agent asked a genuinely new question". None of that scaffolding exists yet —
`eval/baseline.py`, `claude_loop.sh` and `judge_prompt.md` are named in the plan but absent from
this directory. Until that arm is built, treat `outer_followup_rate` as unmeasured, not as zero.

This is still true after M-3, and M-3's acceptance criterion "`outer_followup_rate` -50% vs
Stufe 1" therefore cannot be evaluated in this repo. It was struck rather than faked. What M-3
is read on here instead, all printed in the QW-0 block:

- `schema_valid_rate` — the acceptance gate that _is_ measurable (target 100%), plus
  `stage_exit_log_agreement` as its free cross-check against the run's own log.
- the `range_outside_file` + `misaligned_snippet` counts split by stage — the only accuracy
  failure class a registry-sourced location can fix, and so the honest proxy for "the outer
  agent had to go re-read the file itself".
- `hallucinated_path_rate` — reported for completeness and as a regression alarm. It is
  expected to read 0.00% **before and after** M-3, because a nonexistent path is already
  impossible by construction, and must not be quoted as an M-3 improvement.

Everything else M-3 changes is a response-shape change, which no retrieval metric can move.

### Fixed while adding the above

`run.py`'s `provider_call` matcher was `provider call\b`, which also matched
`repo-explorer-core`'s router commentary (`provider call succeeded`, `provider call failed, no
failover`) and `verify.rs`'s `verification stage provider call failed`. Every successful attempt
was therefore recorded **twice** — once real, once as a fieldless ghost with `outcome=None` — so
`Provider calls (N total)` and `provider_events` were roughly double the truth (224 vs 121 in
`results/20260907T091929`) and no per-call cost or token aggregate over them was usable. The
matcher now excludes the commentary lines. Rows already stored in `results/` still carry the
ghosts; `score.py` tolerates them (a call with no reported token usage costs 0 and needs no
price), but the provider-call counts printed for those older runs stay inflated.
