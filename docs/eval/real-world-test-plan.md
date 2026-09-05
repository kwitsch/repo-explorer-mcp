# Real-world test plan: `explore_repository` against diverse repositories

Status: **v3** (2026-09-05). v1 → v2 after three independent reviews (code
fidelity, evaluation methodology, corpus/ground truth); v2 → v3 after a
second-pass fidelity review and an executability review. Review log and
adoption decisions: §17. Effort estimate: §15.

System under test (SUT): the **installed** binary
`~/.local/bin/repo-explorer-mcp` as registered in Claude Code
(`~/.claude.json` → `mcpServers.repo-explorer-mcp`, stdio, no args), with
the per-user config `~/.config/repo-explorer/repo-explorer.toml`. On
2026-09-05 that config is: one Gemini provider entry (`GOOGLE_API_KEY`), an
**11-model failover chain** (`gemini-flash-latest`, `gemini-3.8-flash` …
`gemini-2.5-flash-lite`), `cooldown_seconds = 60`, `[codebase_memory]
command = "codebase-memory-mcp"` (bare name) with `args = ["--stdio"]`,
`staleness_seconds = 3600`, `search.timeout_seconds = 30`, an inert
`prefer_rtk = true` (not a config field — silently accepted, see F-12),
`[agent]`/`[cache]` absent (defaults), `logging.level = "info"`. The manifest
(§8.1) records the config verbatim (key env var **name** only) at run time.
Today the binary is **0.5.2**; §3.1 explains why the run needs a
logging-only patch release installed first, and what Mode B measures if that
is declined.

## 1. Goal, decision, non-goals

Goal: measure how well the tool locates code for **real exploration
requests** — precise and vague, several languages, English and German — and
turn every failure into a backlog item that names the owning code and the
metric it is expected to move.

The decision this run must support: _what to improve first_. That requires
knowing, per failed query, **which stage lost the answer** (retrieval never
produced the candidate / ranked it below top-k / the verify LLM rejected a
correct candidate / the fallback loop missed or ran out of budget / the LLM
fabricated a location). §7.4 defines that attribution; §3.1 lists the log
lines it needs.

Pre-registered primary metrics (fixed before Phase 1; any later change is
logged with a reason in the report):

1. `range_hit@3` and `file_hit@3` per stratum (Precise / Relational /
   Imprecise / Negative), reported as pass@1 across passes with Wilson CIs.
2. **Confident-wrong rate**: `stage = early-exit ∧ miss`. Target ≤ 2 %
   (Wilson upper bound ≤ 5 %) on the hand-written + synthetic precise corpus.
3. **Hallucination count**: fabricated path / range / snippet. Target 0.
4. **LLM lift**: `file_hit@3(tool) − file_hit@3(B2 deterministic-only)` on
   the Imprecise stratum. Target ≥ 0.15 — the number that justifies the
   verify/fallback stages' cost.
5. Cost: tokens and USD per correct answer, per stratum.

Non-goals: unit/integration coverage (327 tests in the workspace);
installer/updater/wizard (`docs/smoke-test.md`); Windows (no host — gap);
provider portability (all token numbers are Gemini-specific).

A fact that frames the whole exercise (found while checking prerequisites):
the user's Claude Code transcripts under `~/.claude/projects/` contain
**zero** real `explore_repository` tool calls, and the `--install` Explore
agent file (`~/.claude/agents/`) is absent. The tool has so far been
installed but not used in anger. Layer C (§8.3) is therefore the only source
of "real usage" data, and §7.4's usage weighting starts from uniform
weights.

## 2. System under test: what actually happens per call

Derived from `crates/repo-explorer-agent/src/{agent,pipeline,verify,tools,
dispatch,cache}.rs`, `crates/repo-explorer-core/src/retrieval.rs`,
`crates/repo-explorer-search/src/{backend,parser,git_probe,process}.rs`,
`crates/repo-explorer-memory/src/{backend,freshness}.rs`, verified line by
line in two review rounds.

```text
tool call {query, scope_hint?, max_results?}
  Stage 0  fingerprint = HEAD sha + sha256(git status --porcelain, git diff HEAD)
           (3 git subprocesses via tokio::join!; any failure → None → all caches off)
           query cache key = trim+lowercase(query) + scope_hint + max_results
           hit → INFO "exploration served from query cache" path="cache" (no tokens field)
  Stage 1  ensure_fresh_index on codebase-memory-mcp (CBM).
           last_reindexed_at is per process → the FIRST call of every server process
           runs index_repository even on an unchanged, indexed repo (F-08).
           IndexingFailed / backend error → index_note in prompts + summary, never fatal.
  Stage 2  derive_patterns(query):
             literals    = content of "…" '…' `…`  ('…' only when not preceded by a word char)
             identifiers = tokens ≥3 chars, not in STOPWORDS (36 EN + 8 DE:
                           der die das und wird wie welche wo), not all-digit, having
                           _ / digit / mixed case — or plain tokens ≥4 chars.
                           Prose words like "defined", "definiert", "raised", "3xx"
                           are identifiers too.
             path_tokens = tokens containing '/' or with a file extension
                           (":line[:col]" suffix stripped) — `Session.request`,
                           `res.json`, `java.util.Date` are path tokens AND get split
             grep_patterns = first 6 of literals+identifiers, regex-escaped;
                           unquoted metacharacters are tokenized away, not escaped
           fanout (concurrent, soft-fail → DEBUG "retrieval leg failed" leg= ctx= error=):
             symbol legs   ≤4 identifiers → CBM search_graph{name_pattern}
                           exact iff last_segment(symbol) == token (case-sensitive)
             semantic legs ≤4 (literals first) → CBM search_code, ONE TOKEN per leg,
                           literal substring search; max_results forwarded as `limit`
             grep legs     ≤6 → `rtk rg -H -n -S -- <pattern> <scope|.>`, 20 hits kept
             file legs     ≤3 path_tokens → `rtk rg -H -n -S -g <glob> -- . .`;
                           glob = basename, or *name* when it has no extension;
                           a glob matching no file → rg exit 1 → leg succeeds with 0 hits
           merge_and_rank: base SymbolExact 700 / SymbolFuzzy 400 / FileNameHit 300 /
             SemanticHit 260 / ContentHit 150; +30 per distinct query token found in
             symbol|path|snippet (cap 120); +15 per additional distinct
             (non-overlapping) hit in the same file (cap 60); overlapping ranges are
             merged, not boosted; total cap 1000; twins in different paths never merge
           confidence = top/10 + (top − runner_up)/20, cap 100; no candidates → 0
           INFO "retrieval pre-stage complete" candidates=N confidence=C
  Stage 3  confidence ≥ 90 ∧ candidates ≠ ∅ → early-exit, 0 LLM calls,
           summary "Resolved deterministically by the retrieval pre-stage (confidence
           N/100, no LLM involved): K location(s) matching "…"." [+ index_note]
  Stage 4  30 ≤ confidence < 90 ∧ candidates ≠ ∅ → verify: 1 LLM turn over top-12
           candidates rendered per file as a skeleton (≤30 symbols from search_graph,
           one extra CBM call per unique file) OR, when no skeleton, the ≤400-char
           snippet; tools expand|finish; turn 2 (or budget exhausted) forces finish.
           Any provider error → WARN "verification stage provider call failed;
           escalating" → INFO "verification escalated to the fallback loop".
  Stage 5  otherwise (incl. candidates = ∅ at ANY confidence setting) → fallback loop:
           ≤12 turns, 60k token budget checked BETWEEN turns, 2-strike single-call
           rejection (fed back to the model, NOT logged), tools search_code/
           search_graph/query_graph/trace_path/get_architecture/get_code_snippet/
           grep/find/read_file/finish; budget end → one forced finish call → else
           deterministic synthesis "Exploration stopped after reaching the token
           budget (N) …". A RouterError here is a HARD failure: MCP result
           isError=true, text "llm provider error: …", NO log line.
  finalize dedupe per location, truncate to max_results (also 0), cache result,
           INFO "exploration complete" path=<early-exit|verify|fallback> tokens=T
```

Definitions used throughout: **first call** = first `tools/call` in a server
process (always reindexes, F-08); **warm** = any later call in the same
process (index refreshed once, leg cache may hit); **cached** = identical
normalized query in the same process with an unchanged fingerprint.

### 2.1 Arithmetic that drives expected stages

- Lone exact symbol hit, one-token query: 700 + 30 coverage + 0 density =
  730 → 73 + 730/20 = 109 → **100**. Early-exit.
- Early-exit condition `top/10 + (top − r)/20 ≥ 90`: at top = 730 the
  runner-up must be **≤ 390** — a single `SymbolFuzzy` sibling (≥ 400)
  already blocks it. With 4 distinct same-file grep hits (top 790) the
  runner-up may be ≤ 570. Early-exit therefore depends on the grep leg
  surviving (F-02).
- **Only one identifier may resolve to an exact symbol.** Prose identifiers
  without a symbol hit (`defined`, `where`-free words) do not block; a
  second `SymbolExact` (class `Session` + method `request`; `uv_run` in
  `unix/` and `win/`; `Hono` in two files) collapses the margin → ≈ 73–78 →
  verify.
- `SymbolFuzzy` max 580 → 87; `FileNameHit` max 480 → 72. **A P2 query with
  no symbol-bearing token cannot early-exit.** P2 queries that mention a
  symbol (`prepare_url`, `ShellCompDirective`, `JsonReader`) can early-exit
  on the correct file through the symbol leg; `confident_wrong` (§7.1)
  decides, not the category.
- `max_results` is applied after ranking, but is also forwarded to CBM as
  `limit` on the semantic legs and printed to the LLM as "Desired maximum
  results: N".

### 2.2 Hypotheses and candidate defects (pre-registered)

Hypotheses about behaviour the corpus must be able to falsify:

- **H1** Early-exit is rare outside single-definition bare-symbol queries;
  every vague query pays for an LLM call.
- **H2** Semantic legs never contribute: CBM `search_code` is a literal
  substring search fed one token, i.e. a second grep.
- **H3** Vague prose does not dilute ranking with noisy grep hits — it
  **loses** grep evidence, because common tokens trip rtk's per-file cap and
  the whole leg is dropped (F-02); such queries reach fallback with few or
  no candidates.
- **H4** Non-code repositories (Markdown, bats, shell) are grep-only: CBM
  yields no symbols, so P1/P4 expectations there reduce to text matching.

Candidate defects found while reading the code for this plan. They are
**hypotheses until Phase 1 confirms them** on the installed binary; each has
a dedicated probe.

| id   | Candidate defect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Evidence                                                                                                                                                                                       | Probe                                                                                                                            |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| F-01 | **Fixed** (c5a4097, obsoleted by the rtk→rg migration). Was: rtk emits a `+N more in <file>` footer once a file exceeds its per-file line budget (≈ 25 match lines; fewer for the `.` pattern the file leg uses); `parse_rtk` turns that footer into `SearchError::Decode`, so the leg soft-fails for any file with more than a couple dozen non-empty lines. Machine-consumed search now shells out to `rg` directly (`search/backend.rs:111` `backend: "rg"`), which has no truncation footer — the file-name leg is live again.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `search/backend.rs:111`; confirmed live in `results/20260905T220843` (`file` leg hits present, no `Decode` errors)                                                                             | every P2 query; `retrieval leg failed leg="file"` count                                                                          |
| F-02 | **Fixed** (c5a4097, same migration as F-01). Was: a pattern with more than ≈ 25 matches in any single file produced the rtk footer → the whole grep leg failed. Direct `rg` has no such cap.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | confirmed live in `results/20260905T220843` (`grep` leg hits up to 20+ per query, no `Decode` errors)                                                                                          | `leg="grep"` failures per query vs `candidates=`                                                                                 |
| F-03 | **Pre-stage is not deterministic.** `rg` runs without `--sort`; with > 20 hits the surviving 20 depend on traversal order → ranking, confidence and even the route can differ between runs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `search/backend.rs` arg list                                                                                                                                                                   | every early-exit query in 2 processes; `pre_stage_identical`                                                                     |
| F-04 | **Empty query runs the full LLM loop.** No non-empty validation; `derive_patterns("")` → 0 legs → confidence 0 → fallback on prompt "Exploration query: ".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `server.rs:30-41`, `agent.rs:136,199`                                                                                                                                                          | R-01                                                                                                                             |
| F-05 | **No path validation on LLM `finish`.** `parse_finish` checks non-empty path + normalizes lines only; fabricated paths reach the client.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `tools.rs:352-374`                                                                                                                                                                             | `hallucinated`                                                                                                                   |
| F-06 | **`scope_hint` escaping (`..`) or absolute → silently dropped** for the legs (no log), still shown to the LLM as prose, and part of the cache key.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `pipeline.rs:56-59`, `dispatch.rs:193-198`, `agent.rs:643-645`, `cache.rs:166`                                                                                                                 | R-08, §6.4                                                                                                                       |
| F-07 | **Cache blind spot**: edits inside an already-untracked file leave `status --porcelain` and `diff HEAD` unchanged → stale cache hit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `git_probe.rs:54` comment                                                                                                                                                                      | §6.5 untracked-edit variant                                                                                                      |
| F-08 | **Every new server process reindexes on its first call.**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `backend.rs:59`, `freshness.rs:53-58`                                                                                                                                                          | `index_ms` on first vs later calls                                                                                               |
| F-09 | `max_results: 0` empties findings while an LLM-written summary may still describe them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `agent.rs:258-260`                                                                                                                                                                             | R-10                                                                                                                             |
| F-10 | **No provider observability**: zero `tracing` in the llm crate and core router; failover/cooldown silent; hard failures log nothing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | grep over four crates                                                                                                                                                                          | §3.1                                                                                                                             |
| F-11 | Failover only on `RateLimited`/`QuotaExceeded` (429/503/529); invalid model → `InvalidRequest` → immediate hard failure in fallback, escalation in verify. The `cooldown_seconds` comment in README (line 99, "after a provider errors") overstates it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `llm.rs:176-179,361`, `llm/src/lib.rs:53-63`                                                                                                                                                   | R-16a                                                                                                                            |
| F-12 | **Unknown config keys are silently accepted** (`Config` is not `deny_unknown_fields` by design); the live config carries an inert `prefer_rtk = true` that the user presumably believes does something.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `config.rs` `Config` derive; user config                                                                                                                                                       | `config test` on the live config — expect no warning                                                                             |
| F-13 | **Fixed** (07b6602). A whole-file `SemanticHit` candidate's `line_end` was consistently `actual_line_count + 1` (`config.rs`: 980 lines, reported 981; `update.rs`: 1292 lines, reported 1293) — `parse_line_range` parsed codebase-memory-mcp's `"start-end"` string verbatim, an inclusive/exclusive convention mismatch. `correct_module_end` now decrements a `Module` row's `line_end` by 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `repo-explorer-memory/src/backend.rs`; live `results/20260905T042837`                                                                                                                          | `range_outside_file` on whole-file hits                                                                                          |
| F-14 | **Fixed** (cb79efa, "Send the CBM 0.10.8 argument keys the tool schemas require"). Was: `index_repository` failed against the installed codebase-memory-mcp on every call (`IndexingFailed`, 60/60 scored calls in `results/20260905T145436`, reason `"repo_path is required"`) because `run_index` sent the arg key `path` instead of the installed binary's actual schema key. `results/20260905T220843` confirms the fix: `index_status=UpToDate` on all 60 calls that reported one, zero `IndexingFailed`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `backend.rs:197-216`; confirmed live in `results/20260905T220843`                                                                                                                              | check installed `codebase-memory-mcp` version/schema for `index_repository`'s actual required key                                |
| F-15 | **No camelCase↔snake_case normalization in identifier legs.** `derive_patterns` classifies `resolveRedirects` as an identifier but emits it verbatim as the grep/symbol pattern; it never generates the `resolve_redirects` variant. Any typo/case-mismatch query against a differently-cased real symbol produces 0 hits on all three legs (grep/semantic/symbol) — `candidates=0` — even though a case-insensitive or cross-convention match would find it. Directly caused `requests-I3-02` (query `resolveRedirects` vs. real symbol `resolve_redirects`) to reach the fallback loop with zero candidates.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `crates/repo-explorer-core/src/retrieval.rs` `derive_patterns`; live `results/20260905T220843` `requests-I3-02` (`retrieval_patterns.identifiers: ["resolveRedirects"]`, all 3 legs `hits: 0`) | add a case-folded/snake↔camel variant to the identifier pattern set; rerun `requests-I3-02`/`self-I3-02` (both I3-typo category) |
| F-16 | **Early-exit can fire confidently-wrong on file:line-pointer queries.** `self-P2-01` (`crates/.../verify.rs:35 what does this constant do`) is explicitly annotated "no symbol token → cannot early-exit", yet pass 2 took the `early-exit` route with `confidence=90` and the wrong top candidate (`server.rs:102-106`, matched on the generic word "agent" in the query's path segment) — score.py flags it `CONFIDENT_WRONG`. Pass 1 (same query, same pin) scored the same top candidate at 865 vs. pass 2's 880 and stayed on `verify`, i.e. the two runs sit on opposite sides of the early-exit confidence threshold for the _same_ underlying match — this is F-03 (pre-stage nondeterminism) manifesting as a false positive with zero LLM oversight, not just a ranking wobble.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | live `results/20260905T220843` `self-P2-01` pass1 (stage=verify, score=865) vs pass2 (stage=early-exit, score=880, confidence=90)                                                              | add a query-shape guard (no identifier token in query ⇒ never early-exit) alongside the F-03 determinism fix                     |
| F-17 | **Fixed** (4ca0a18, "Widen the snippet-alignment window to the full claimed [line_start, line_end] span"). Rescored `results/20260905T220843`: hallucinations 32 → 11 (25 → 4 `misaligned_snippet`; 7 `fabricated_snippet` unchanged; 0 `fabricated_path`/`range_outside_file`). The 4 survivors are three separate pre-existing defects (duplicate-match `find_chunk`, single-character chunks, content genuinely >4 lines off), tracked as F-17 follow-ups, so the §7.5 `hallucinated = 0` gate still fails. Escalated review follow-up in this same PR further tightened `snippet_found_at`: a `line_start=None` finding (location genuinely unknown) is no longer automatically `misaligned`, and the `line_end=None` fallback window no longer silently widens by one line past the legacy `line_start+3` bound — the 32 → 11 count above predates this follow-up and must be reconfirmed with a fresh `score.py` rescore of `results/20260905T220843` in the main checkout before being trusted as final. Was: **Scorer bug (eval-only): snippet-alignment check only searches near `line_start`, not across the claimed `[line_start, line_end]` span.** `snippet_found_at` in `eval/score.py` builds its "near" window as `line_start±~4` and only exempts spans covering nearly the _whole file_; a legitimate representative-line snippet from a large-but-not-whole-file span (e.g. a 16–40-line test/function body cited by a semantic-search hit) that happens to sit more than ~4 lines past `line_start` is flagged `misaligned_snippet` even though it is real content inside the claimed range. Inflates the current hallucination count (e.g. 7 of the flagged instances on `requests-P1-01` alone) and must be fixed before trusting today's hallucination numbers. | `eval/score.py:137-172`; live `results/20260905T220843` `requests-P1-01` (multiple `tests/test_requests.py` findings with real, in-range snippets flagged `misaligned_snippet`)                | widen `near` to `range(line_start - 4, line_end + 4)` when the span isn't whole-file; rescore and recount hallucinations         |

## 3. Prerequisites

### 3.1 Observability patch (logging-only release, e.g. 0.5.3)

Without these lines the run cannot attribute failures (§7.4), cannot tell
which model served a call, and cannot separate rtk from CBM failures.
Install it through the normal update path so the SUT stays "the installed
binary". The explored `self` tree stays pinned at `91eebde` regardless of
the binary version (the patch edits exactly the lines that are `self`
ground truth).

**Where each line goes** (the crates that already depend on `tracing` are
`repo-explorer-agent`, `-mcp`, `-memory`, `-search`, `-llm`; **core has no
`tracing` dependency** and adding one is out of scope):

| Line (level)                                   | Fields                                                                                                                                                                                                                                   | Emitted from                                                                                                                                                                                                                                                   | Consumer                                                        |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| span `explore{req_id}`                         | `req_id = "<sha256(query_key)[..8]>-<per-process AtomicU64>"`                                                                                                                                                                            | `server.rs`: `agent.run(..).instrument(info_span!("explore", req_id = %id))`. The fmt layer then prefixes every line inside the span with `explore{req_id=…}:`; leg fanout uses `join!`, so it inherits the span; the only `tokio::spawn` is the CBM transport | harness correlation (replaces timestamp slicing); R-17          |
| `retrieval patterns` (DEBUG)                   | literals, identifiers, path_tokens, grep_patterns                                                                                                                                                                                        | `pipeline.rs` `retrieve`                                                                                                                                                                                                                                       | H3 quantification                                               |
| `retrieval leg done` (DEBUG)                   | leg, ctx, hits, duration_ms (keep `retrieval leg failed`)                                                                                                                                                                                | `pipeline.rs` `soft_leg`                                                                                                                                                                                                                                       | per-leg latency, F-01/F-02, CBM coverage                        |
| `retrieval candidates` (DEBUG, one JSON array) | rank, kind, score, path, line_start, line_end, symbol                                                                                                                                                                                    | `agent.rs` after `retrieve`                                                                                                                                                                                                                                    | `cand_recall@top_k`, `cand_rank`, offline threshold sweep (§10) |
| `index status` (INFO)                          | outcome (Reindexed / UpToDate / IndexingFailed / Unavailable), duration_ms, **commit** indexed                                                                                                                                           | `memory/src/backend.rs` `ensure_fresh_index`                                                                                                                                                                                                                   | F-08, dirty-tree variants, stale-index exclusion                |
| `provider call` (INFO)                         | provider, `model_requested`, `model_served` (genai `ChatResponse.provider_model_iden.model_name`), attempt, outcome (ok / rate_limited / quota / invalid / auth / other), latency_ms, prompt_tokens, completion_tokens, reasoning_tokens | `llm/src/lib.rs` `GenaiProvider::complete_with_tools` (attempts, incl. failures)                                                                                                                                                                               | model pinning check, `provider_events`, cost split              |
| `provider failover` (WARN)                     | from, to, cooldown_s                                                                                                                                                                                                                     | router lives in core (no tracing) → emit from the agent side when the returned `Completion` names a different provider/model than the previous call, or accept "derived from consecutive `provider call` lines"                                                | R-16b, §13                                                      |
| `verify action` (DEBUG)                        | turn, action (expand / finish / forced-finish / nudge)                                                                                                                                                                                   | `verify.rs`                                                                                                                                                                                                                                                    | attribution `verify-*`                                          |
| `fallback turn` (DEBUG)                        | turn, tools called, rejected_single_call, strikes, budget_spent                                                                                                                                                                          | `agent.rs` `fallback_loop`                                                                                                                                                                                                                                     | attribution `fallback-*`                                        |
| `exploration complete` (existing INFO)         | add `llm_calls` (= number of `Completion`s returned to the agent, counted in `TokenBudget::add` via a new `calls: u32` field), `forced_finish: bool`, `index_status`, `git_probe_ms`                                                     | `agent.rs` `complete_run`                                                                                                                                                                                                                                      | headline metrics                                                |
| `exploration failed` (WARN)                    | error class, message (no key material)                                                                                                                                                                                                   | `server.rs` where `Err` becomes `isError`                                                                                                                                                                                                                      | `stage = error`                                                 |
| cache line (existing INFO)                     | add `tokens = 0`, `git_probe_ms`                                                                                                                                                                                                         | `agent.rs:151`                                                                                                                                                                                                                                                 | cache latency threshold                                         |

Also in the patch: `EnvFilter::new(format!("{level},rmcp=warn,hyper=warn,
reqwest=warn"))` so `logging.level = "debug"` does not flood stderr with
transport noise (`init_tracing` currently uses `with_max_level` globally).
Colour: the harness sets `NO_COLOR=1` in the server environment
(tracing-subscriber honours it, no isatty check) — works in Mode B too;
`.with_ansi(false)` in the patch is optional. JSON log output would need the
`json` feature of `tracing-subscriber` (+3 deps, a Cargo change) — not
required; `req_id` on plain lines is sufficient.

**Logging-only verification** (Phase-0 exit criterion):

1. `git diff v0.5.2 v0.5.3 --stat` touches only: `agent.rs`, `pipeline.rs`,
   `verify.rs`, `tools.rs` (agent crate); `server.rs`, `main.rs` (mcp
   crate); `llm/src/lib.rs`; `memory/src/backend.rs`; version pins.
2. Every hunk that is not a `tracing::` / `span` / `.instrument()` line is
   one of: a counter field (`calls`, `forced_finish`), an `Instant::now()`,
   a `NO_COLOR`/filter line.
3. Parity run: 50 synthetic P1 queries (§6.1) on 0.5.2 and on the patch,
   same repo, fresh process each; findings JSON byte-identical apart from
   F-03 ordering.

**Mode B (patch declined):** run 0.5.2 as-is. Computable: everything in §7.1
that comes from the response and the filesystem; `stage` from the two
existing INFO lines (`error` from `isError`, `timeout` from the harness);
`tokens`; `candidates`; `confidence`; latency; cache behaviour; F-01/F-02
only indirectly (`retrieval leg failed` at DEBUG has `leg`/`ctx`/`error`).
Not computable: attribution (§7.4), model served, per-leg timing,
`provider_events`, `llm_calls`, forced-finish flag, indexed commit. Rate-limit
back-off (§8.1) then triggers on `isError` text matching
`/rate limited|exhausted or cooling down|429|RESOURCE_EXHAUSTED/i`. The
report stops at "which stratum fails", not "which stage".

### 3.2 Environment decisions

- **CBM daemon ownership.** Only one CBM build may run per user; a running
  daemon admits only clients launched from its exact executable path; the
  server reuses a running daemon via a `/proc` scan (5 s grace, Linux only,
  only when `command` is set) and otherwise spawns `[codebase_memory]
command`. Decision: **Phases 1–3 and 5 run with no interactive Claude Code
  session open**; the standalone `~/.local/bin/codebase-memory-mcp` then
  owns the daemon for the whole run and the manifest records its path and
  version. Phase 4 (Layer C) is the only phase with Claude Code running.
  Never call `codebase-memory-mcp cli …` while a daemon runs. The manifest
  records `pgrep -a repo-explorer-mcp` and `pgrep -a codebase-memory-mcp` at
  start of every phase.
- **Interactive sessions share the Gemini key and the daemon** with the
  eval; the rule above also protects the rate-limit budget.
- **Model pinning.** Main runs use **one dated model id, no failover chain**.
  List the catalog with the key taken from the environment only
  (`curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" | jq '.models[].name'`),
  pick the id the alias `gemini-flash-latest` resolves to on the run date,
  record it in the manifest. The 11-model chain is used only in R-16b.
  Temperature is unset by `GenaiProvider` → record "unset → Gemini default
  on <date>".
- `eval/config/default.toml` = the live config **verbatim** minus the chain
  (→ one pinned id), plus `logging.level = "debug"`; keep the bare
  `command`; keep `prefer_rtk` so F-12 is observed as-is.
- **Corpus contamination.** `docs/eval/` and `eval/queries/*.yaml` contain
  every `self` query, expected path and negative term verbatim, and `rg`
  searches untracked files. Every `self` query runs with `scope_hint:
crates`; the report says so.
- **rtk configuration** is part of the SUT environment: record
  `~/.config/rtk/config.toml` and `filters.toml` in the manifest. rtk's
  default filters skip `target/`, `node_modules/`, `vendor/` (the ignored-dir
  variant in §6.4 measures an rtk artefact), and its `max_width` can
  truncate snippet lines → the hallucination check compares a snippet
  **prefix ≤ 100 chars**.
- Add `/results` and `eval/claude-profile/.credentials.json`,
  `eval/claude-profile/.claude.json` to `.gitignore` before the first run;
  add a pre-commit grep for API-key-shaped strings under `eval/`.
- Python deps via `uv run --with mcp --with pyyaml --with rank-bm25`.

## 4. Test dimensions

| Dimension            | Values                                                                                                                                                                                                                                         |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language / ecosystem | Rust (2 repos), Python, JavaScript, TypeScript, Go, Java, C#, C, C++ (header-only), Ruby, PHP (optional), polyglot docs/shell                                                                                                                  |
| Repo shape           | single crate; multi-module Maven; monorepo (gems / packages); duplicated sources (`single_include` vs `include`); platform twins (`unix/` vs `win/`; 7 `uv__io_poll` definitions); test-file siblings (`*.test.ts` next to source); docs-heavy |
| Query precision      | P → I → N; "anchored" imprecise (contains an identifier) vs identifier-free; see §6.1                                                                                                                                                          |
| Query language       | EN, DE (incl. compound nouns), mixed with code formatting, typos (edit and casing)                                                                                                                                                             |
| Parameters           | none; `scope_hint` dir / file / glob / `./` / abs / `../` / missing / ignored dir / Windows-style; `max_results` 0 / 1 / 3 / 50 / invalid                                                                                                      |
| Repo state           | first call (reindex), warm, cached, dirty tracked file, new untracked file, edited untracked file, no-`.git` copy, cwd in a subdirectory, shallow clone                                                                                        |
| Operational          | CBM unavailable at start vs after connect; rtk timeout; provider 429 (mocked); invalid model; tiny budget; concurrency                                                                                                                         |
| Consumer             | Layer A direct MCP driver; Layer B baselines B0–B3; Layer C Claude Code headless, three arms                                                                                                                                                   |

## 5. Test repositories

Clones live in `~/repos/eval-corpus/<name>`, outside the workspace.
`eval/repos.toml`:

```toml
[[repo]]
id = "requests"
url = "https://github.com/psf/requests"        # or file:///home/kwitsch/repos/claude-plugins
branch = "main"                                # default branch verified 2026-09-04
pin = "<sha>"                                  # git rev-parse HEAD at clone time
shallow = false                                # true only for `json`; then pin MUST equal the
                                               # fetched tip and re-pinning means re-cloning
path = "~/repos/eval-corpus/requests"
tier = 1
```

Every run starts with `git checkout --detach <pin>` and a clean-tree
assertion. Facts below are from the GitHub API on 2026-09-04.

### Tier 1

| id         | Repo                                       | Lang                          | Facts                             | Why                                                                                    |
| ---------- | ------------------------------------------ | ----------------------------- | --------------------------------- | -------------------------------------------------------------------------------------- |
| `self`     | `kwitsch/repo-explorer-mcp` @ `91eebde`    | Rust                          | 76 files                          | dogfood; exact ground truth; reported **separately** from external repos (author bias) |
| `requests` | `psf/requests`                             | Python                        | `main`, 13.6 MB, Apache-2.0       | canonical small lib; typed; decorators, hooks                                          |
| `express`  | `expressjs/express`                        | JavaScript                    | **`master` = 5.2.1**, 9.9 MB, MIT | prototype-style, no classes; router in a dependency → natural near-negatives           |
| `cobra`    | `spf13/cobra`                              | Go                            | `main`, 2.1 MB                    | flat package + `doc/`; 5 completion siblings                                           |
| `gson`     | `google/gson`                              | Java                          | `main`, 23.5 MB                   | real multi-module; many same-named `read`/`write`                                      |
| `plugins`  | local `~/repos/claude-plugins` @ `711cc5d` | Markdown, bats, mjs, sh, JSON | 306 files                         | the user's real workload; H4                                                           |

### Tier 2

| id                  | Repo                 | Lang       | Facts                                                                                                        | Why                                                                         |
| ------------------- | -------------------- | ---------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `ripgrep`           | `BurntSushi/ripgrep` | Rust       | pin at clone                                                                                                 | Rust repo the plan author does not know; exercises rtk/rg on its own source |
| `hono`              | `honojs/hono`        | TypeScript | `main`, 9.3 MB                                                                                               | `*.test.ts` sibling per source → test-vs-source ranking                     |
| `humanizer`         | `Humanizr/Humanizer` | C#         | `main`, 15.8 MB; inflection subsystem rewritten (`Inflections/Vocabularies.cs`, `InflectionEngine.cs`)       | extension methods; 11 `*Registry.cs`; 40+ localisation files                |
| `libuv`             | `libuv/libuv`        | C          | **`v1.x`**, 17.8 MB                                                                                          | `uv_run` ×2, `uv__io_poll` ×7                                               |
| `json`              | `nlohmann/json`      | C++        | `develop`, **282 MB** → `git clone --depth 1 --branch develop`; `single_include/nlohmann/json.hpp` = 1.04 MB | duplicated sources; skeleton cap; huge-file ranges                          |
| `sinatra`           | `sinatra/sinatra`    | Ruby       | `main`, 7.9 MB                                                                                               | monorepo (`rack-protection/`, `sinatra-contrib/`); DSL                      |
| `guzzle` (optional) | `guzzle/guzzle`      | PHP        | pin at clone                                                                                                 | closes the PHP gap                                                          |

Deferred: Kotlin/Swift, > 500k-LOC monorepo, Windows.

### Phase-0 probe per repo (all recorded in the manifest)

1. Pin; `git ls-files | wc -l`; LOC per extension (`git ls-files | awk -F.
'{print $NF}' | sort | uniq -c`, LOC via `xargs wc -l`); **git probe
   cost** = median of 5 warm runs of `git status --porcelain && git diff
HEAD` (this feeds the cache-latency threshold).
2. Forced index: the harness warm-up call does it (F-08 guarantees a
   reindex); read the indexed commit from the `index status` line. No CBM
   CLI involvement.
3. `index_first_call_ms` (first call in a fresh process on the indexed repo)
   vs `index_cold_ms` (first call ever) — both reindex; the delta is CBM's
   incremental cost. Warm-up query = a bare-identifier P1 (no LLM variance).
4. CBM coverage: `retrieval leg failed` filtered on `leg="symbol"` /
   `leg="semantic"`; indexed symbol count per language from the
   `gen_synthetic.py` graph query (§6.1). **Zero symbols for a language
   demotes that repo's P1/P4 expectations to "grep-only".**
5. rtk truncation exposure: `leg="grep"` / `leg="file"` failures whose error
   contains the `+N more` footer, per query (F-01/F-02).

## 6. Query taxonomy and corpus

### 6.1 Categories

| Cat   | Name                                                             | Shape                                                                              | Expected stage (from §2.1)                                       | Primary metric                           |
| ----- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------- |
| P1    | exact bare symbol                                                | `where is derive_patterns defined`                                                 | early-exit if unique def and grep leg survives; else verify      | range_hit@1                              |
| P1-DE | same, German                                                     | `wo ist derive_patterns definiert`                                                 | same                                                             | range_hit@1                              |
| P1t   | bare symbol with **twin** definitions                            | `running_memory_binary`, `Hono`, `uv_run`                                          | verify (margin ≈ 0)                                              | range_hit@3, `equivalent` handling       |
| P1d   | dotted symbol                                                    | `Session.request`, `res.json`                                                      | verify (two identifiers; file-leg glob matches nothing)          | range_hit@3                              |
| P2    | path / path:line paste                                           | `crates/…/verify.rs:35 what is this`                                               | verify (no symbol token) or early-exit via a symbol in the query | file_hit@1                               |
| P3    | quoted literal                                                   | `where is "Did you mean this?" generated`                                          | early-exit / verify                                              | range_hit@1 (±5 around the literal line) |
| P4    | symbol + relation                                                | `who calls merge_and_rank`                                                         | verify or fallback (`trace_path`)                                | file_hit@3, precision                    |
| I1    | conceptual, **identifier-free**                                  | `where does the library decide to follow a 3xx and re-send the request`            | fallback (F-02 drops common tokens)                              | file_hit@3                               |
| I1a   | conceptual, **anchored** (contains an identifier or literal key) | `how does trust proxy affect req.ip`                                               | verify                                                           | file_hit@3 — reported separately from I1 |
| I2    | symptom-first / bug-hunt                                         | `my proxy settings from the shell are being ignored`                               | fallback                                                         | file_hit@3, summary rubric               |
| I2x   | two concepts                                                     | `where do cookies from a redirect response get merged into the session jar`        | fallback                                                         | recall over `primary`                    |
| I3    | vague / noisy / German compound / typo                           | `wo ist die fehlerbehandlung für zeitüberschreitungen`, `resolveRedirects`         | fallback                                                         | file_hit@3, no hallucination             |
| I4    | orientation / cross-cutting                                      | `how does an incoming request travel through this codebase from entry to response` | fallback                                                         | precision, summary rubric                |
| M     | multi-target                                                     | `all shell completion generators`                                                  | verify                                                           | recall@max_results                       |
| N     | negative, `sub: near` (exists in docs / a dependency) or `far`   | `where is the SOAP envelope parsed`                                                | verify / fallback                                                | negative_ok, path_valid = 1.0            |

Tier-1 template, 16 per repo: P1, P1-DE, P1t-or-P1d, P2, P3, P4, I1 ×2,
I1a, I2, I2x, I3 ×2 (DE compound, typo), I4, M, N. Tier-2 template, 9 per
repo: P1, P1t-or-P1d, P3, P4, I1, I1a, I3-DE, M, N. **Every Tier-2 set is
filled to the full template in Phase 0** — the anchor lists in §6.3 are
partial. Hand-written total: 6×16 + 7×9 = **159** (6×16 + 6×9 = 150 without
`guzzle`). Plus:

- **Synthetic precise corpus** (`eval/gen_synthetic.py`): uses the Python
  `mcp` SDK to spawn the **same CBM path string** the server uses (from the
  manifest — admission rule), calls `get_graph_schema` once for the label
  vocabulary, then `query_graph` for definitions (`MATCH (n) WHERE n.label IN
[<function/class labels>] RETURN n.name, n.file, n.start_line, n.end_line
LIMIT 5000`, project = the repo root as the memory crate derives it);
  dedupes on `last_segment(name)`, samples 30 with `random.seed(repo_id)`;
  spans come from the returned line range. Plus 10 string literals ≥ 20
  chars per repo via `rg`. Queries `where is <sym> defined` / `where is
"<lit>" produced`. ≈ 40 × 12 code repos (`plugins` excluded — no symbols)
  = 480, mostly early-exit. This is the only way to estimate the
  **confident-wrong rate** with a usable CI.
- **Issue-derived I2 queries** (`requests`, `express`, `cobra`, `gson`, 3
  each): `gh api repos/O/R/issues?state=closed&labels=bug&per_page=100`;
  for each issue N, `gh api search/issues -f q='repo:O/R is:pr is:merged "#N"'`
  and keep PRs whose body matches `/(fix|close|resolve)[sd]? #N/i`; ground
  truth = `gh api repos/O/R/pulls/P/files` minus test paths, **filtered to
  paths that exist at the pin**; take the 3 most recent that survive. Query
  = issue title + first sentence.
- **Transcript-mined I3/I4**: field path `message.content[] where
type == "tool_use" and name == "mcp__repo-explorer-mcp__explore_repository"
→ .input.query` across `~/.claude/projects/**/*.jsonl`. On 2026-09-05 this
  yields **0** rows; the §7.4 weighting therefore uses uniform stratum
  weights, flagged in the report as "no usage data".

### 6.2 Corpus file format (`eval/queries/<repo>.yaml`)

```yaml
- id: requests-I1-01
  cat: I1 # enum: P1 P1-DE P1t P1d P2 P3 P4 I1 I1a I2 I2x I3 I4 M N
  sub: null # optional: de | typo | near | far
  lang: en # en | de | mixed
  query: "where does the library decide to follow a 3xx and re-send the request"
  scope_hint: null # optional
  max_results: null # optional
  expect:
    primary_mode: all # all (default; recall over primary) | any (one suffices)
    primary: # required, ≥1
      - {
          path: src/requests/sessions.py,
          symbol: resolve_redirects,
          span: [186, 320],
        }
    acceptable: [] # optional; credited in precision, not required
    distractor: [] # optional; known-wrong twins → flag twin_confusion (never negates a hit)
    equivalent: [] # optional; groups [[{path, span}, …]]: any member = hit using that
      # member's span; ≥2 members returned = dedupe_defect
    stage: fallback # optional; scorer emits stage_expected / stage_observed / stage_match
  negative: null # for N: {kind: near|far}
  notes: "grep legs: library, decide, follow, 3xx, send, request — 'request' > 25 hits/file → dropped (F-02)"
```

`range_hit` is computed only where a `span` exists (every P*/I1/I1a/I2
entry with a named symbol must carry one); file-only primaries contribute to
`file_hit` alone. Ground truth is **verified at pin time**: `rg -n` the
symbol/literal, record `span` (definition start to end, `rg` + manual end),
fix paths; unverifiable entries are dropped, not guessed. The tables below
are the starting hypothesis, corrected against the default branches on
2026-09-04; line numbers **will drift**.

### 6.3 Query sets

Legend: **(v)** verified on the default branch in review; **(pin)** verify
at pin time.

**`self` (Rust; all queries `scope_hint: crates`)**

| id            | cat     | query                                                                            | primary / notes                                                                                                           |
| ------------- | ------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| self-P1-01    | P1      | `where is derive_patterns defined`                                               | `core/src/retrieval.rs:150` (v); 21 grep hits in one file → leg survives; early-exit expected                             |
| self-P1-DE-01 | P1-DE   | `wo ist wait_for_running_memory_binary definiert`                                | `mcp/src/main.rs:261` (v)                                                                                                 |
| self-P1t-01   | P1t     | `running_memory_binary`                                                          | `cfg`-gated twins `main.rs:279`, `:300` (v) → `equivalent` group; verify expected                                         |
| self-P2-01    | P2      | `crates/repo-explorer-agent/src/verify.rs:35 what does this constant do`         | `VERIFY_SYSTEM_PROMPT` span 35–43 (v); no symbol token → cannot early-exit                                                |
| self-P3-01    | P3      | `where is the error "run \`repo-explorer-mcp --update\` to provision it" raised` | `primary_mode: any` over `main.rs:144`, `setup.rs:343`, `setup.rs:386` (v)                                                |
| self-P4-01    | P4      | `who calls merge_and_rank`                                                       | `agent/src/pipeline.rs:183` (v); acceptable `retrieval.rs` tests                                                          |
| self-I1-01    | I1      | `why does a second identical call come back instantly without hitting the model` | `agent.rs:142-152`, `cache.rs` (v)                                                                                        |
| self-I1-02    | I1      | `what stops the tool from searching outside the checkout`                        | `dispatch.rs:193` (`escapes_repo_root`), `pipeline.rs:56` (v)                                                             |
| self-I1a-01   | I1a     | `the thing that strips additionalProperties for gemini`                          | `llm/src/lib.rs:444` `strip_additional_properties` (v); distractor `agent/src/tools.rs` (14 literal hits → density bonus) |
| self-I2-01    | I2      | `Claude Code says the server exited instead of starting the setup wizard`        | `main.rs:93` (`is_terminal`) (v); acceptable `setup.rs`                                                                   |
| self-I2x-01   | I2x     | `how does the scope hint interact with the query cache key`                      | `cache.rs:166`, `pipeline.rs:56` (v)                                                                                      |
| self-I3-01    | I3-de   | `wo wird das tokenbudget geprüft und was passiert wenn es aufgebraucht ist`      | `agent.rs:52` (`TokenBudget`), `:527` (`forced_finish`) (v); compound `tokenbudget` intended                              |
| self-I3-02    | I3-typo | `derivePatterns`                                                                 | `retrieval.rs:150`                                                                                                        |
| self-I4-01    | I4      | `how are errors surfaced to the MCP client`                                      | `mcp/src/server.rs:128-144`, `main.rs` (v)                                                                                |
| self-M-01     | M       | `all places where a tracing::info! line reports the exploration path`            | exactly `agent.rs:151`, `:290` at pin `91eebde` (v) → recall ∈ {0, .5, 1}                                                 |
| self-N-01     | N-far   | `where is the SQLite schema migration`                                           | none under `crates/` (v)                                                                                                  |

**`requests` (Python)**

| id                | cat     | query                                                                                 | primary / notes                                                                  |
| ----------------- | ------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| requests-P1-01    | P1      | `where is resolve_redirects defined`                                                  | `src/requests/sessions.py:186` (v); early-exit expected                          |
| requests-P1-DE-01 | P1-DE   | `wo ist HTTPBasicAuth definiert`                                                      | `auth.py:85` (v)                                                                 |
| requests-P1d-01   | P1d     | `Session.request`                                                                     | `sessions.py:557` (v); `request` also in `api.py` → verify                       |
| requests-P2-01    | P2      | `src/requests/models.py what does prepare_url do`                                     | `models.py:483` (v); contains a symbol → early-exit on the correct file possible |
| requests-P3-01    | P3      | `where does the error text "Invalid URL" come from`                                   | `models.py:517`, `:522` (v)                                                      |
| requests-P4-01    | P4      | `who calls dispatch_hook`                                                             | `sessions.py:791` (v); acceptable def `hooks.py:32`; **not** `models.py`         |
| requests-I1-01    | I1      | `where does the library decide to follow a 3xx and re-send the request`               | `sessions.py:186` (v)                                                            |
| requests-I1-02    | I1      | `something removes the Authorization header when a redirect hops to another host`     | `sessions.py:155` `rebuild_auth` (v)                                             |
| requests-I1a-01   | I1a     | `how does basic auth get attached to a request`                                       | `auth.py:85`, `models.py:670` `prepare_auth` (v)                                 |
| requests-I2-01    | I2      | `my proxy settings from the shell are being ignored`                                  | `sessions.py:330/353` (`trust_env`), `:831`, `utils.py:873` (v)                  |
| requests-I2x-01   | I2x     | `where do cookies from a redirect response get merged into the session jar`           | `sessions.py` (`resolve_redirects`), `cookies.py` (v)                            |
| requests-I3-01    | I3-de   | `wo ist die fehlerbehandlung für zeitüberschreitungen`                                | `adapters.py` (`ConnectTimeout`/`ReadTimeout`, `:132`) (v)                       |
| requests-I3-02    | I3-typo | `resolveRedirects`                                                                    | `sessions.py:186`                                                                |
| requests-I4-01    | I4      | `where do I start reading if I want to follow a requests.get call down to the socket` | `api.py` → `sessions.py:557` → `adapters.py:634` (v)                             |
| requests-M-01     | M       | `all exception classes raised for connection problems`                                | `exceptions.py:70,74,78,91` (v); acceptable `:82`                                |
| requests-N-01     | N-far   | `where is HTTP/2 stream multiplexing implemented`                                     | none in `src/` (v)                                                               |

**`express` (JavaScript, `master` = 5.2.1)**

| id               | cat     | query                                                                                     | primary / notes                                                                   |
| ---------------- | ------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| express-P1-01    | P1      | `where is createApplication defined`                                                      | `lib/express.js:36` (v)                                                           |
| express-P1-DE-01 | P1-DE   | `wo ist compileETag definiert`                                                            | `lib/utils.js:130` (v)                                                            |
| express-P1d-01   | P1d     | `res.json`                                                                                | `lib/response.js:234` (v); tokenizer keeps only `json` → verify                   |
| express-P2-01    | P2      | `lib/view.js how are view engines resolved`                                               | `lib/view.js` (v); no symbol token → cannot early-exit                            |
| express-P3-01    | P3      | `where is "No default engine was specified and no extension was provided." thrown`        | `lib/view.js:61` (v)                                                              |
| express-P4-01    | P4      | `where is compileETag used`                                                               | `application.js:21,365` (v); acceptable def `utils.js:130`                        |
| express-I1-01    | I1      | `where does the framework work out the client's real address when behind a load balancer` | `request.js:340`, `utils.js:194` (v)                                              |
| express-I1-02    | I1      | `what happens when the template engine cannot be figured out from the file name`          | `view.js:61` (v)                                                                  |
| express-I1a-01   | I1a     | `how does trust proxy affect req.ip`                                                      | `request.js:340`, `utils.js:194` (v)                                              |
| express-I2-01    | I2      | `my app answers 304 to requests I never cached`                                           | `request.js:469` (`fresh`), `response.js` (`send`) (v)                            |
| express-I2x-01   | I2x     | `how does the etag setting interact with the 304 freshness check`                         | `application.js:365`, `utils.js:130`, `response.js`, `request.js:469` (v)         |
| express-I3-01    | I3-de   | `wo wird geprüft ob der client noch eine gültige version im cache hat und ein 304 reicht` | `request.js:469` (v)                                                              |
| express-I3-02    | I3-typo | `createAplication`                                                                        | `express.js:36`                                                                   |
| express-I4-01    | I4      | `how does an incoming request travel through this codebase from entry to response`        | `express.js:36`, `application.js` (`handle`), `request.js`, `response.js` (v)     |
| express-M-01     | M       | `every place that reads the compiled 'trust proxy fn' setting`                            | `request.js:301,341,358,419`, `application.js:112-114,371` (v)                    |
| express-N-01     | N-near  | `where is Layer.prototype.handle_request implemented`                                     | none under `lib/` (v); bonus if the summary says it moved to the `router` package |

**`cobra` (Go)**

| id             | cat     | query                                                                         | primary / notes                                          |
| -------------- | ------- | ----------------------------------------------------------------------------- | -------------------------------------------------------- |
| cobra-P1-01    | P1      | `where is MinimumNArgs defined`                                               | `args.go:87` (v)                                         |
| cobra-P1-DE-01 | P1-DE   | `wo ist ExecuteC definiert`                                                   | `command.go:1084` (v); near-name `ExecuteContextC:1078`  |
| cobra-P1d-01   | P1d     | `Command.Find`                                                                | `command.go` (pin)                                       |
| cobra-P2-01    | P2      | `completions.go how are ShellCompDirective values combined`                   | `completions.go:45` (v); contains a symbol               |
| cobra-P3-01    | P3      | `where is the text "Did you mean this?" generated`                            | `command.go:790` (v)                                     |
| cobra-P4-01    | P4      | `where is EnableTraverseRunHooks consulted`                                   | `cobra.go:66` def, `command.go:974-1038` uses (v)        |
| cobra-I1-01    | I1      | `when a user mistypes a subcommand, where does the suggestion list come from` | `command.go:790` (v)                                     |
| cobra-I1-02    | I1      | `where does the library complain about too few positional arguments`          | `args.go:87` (v)                                         |
| cobra-I1a-01   | I1a     | `how are markdown docs generated from commands`                               | `doc/md_docs.go` (v)                                     |
| cobra-I2-01    | I2      | `a flag I defined on the root command is unknown when I run a subcommand`     | `command.go:1898` `mergePersistentFlags`, `:1775` (v)    |
| cobra-I2x-01   | I2x     | `how does a parent's before-run hook end up running for a child command`      | `command.go:905-1038`, `cobra.go:66` (v)                 |
| cobra-I3-01    | I3-de   | `wo wird die hilfe ausgegeben wenn keine args kommen`                         | `command.go:520`, `:478`, `:1263` (v)                    |
| cobra-I3-02    | I3-typo | `ExcecuteC`                                                                   | `command.go:1084`                                        |
| cobra-I4-01    | I4      | `where does the library print the error a user sees when a command fails`     | `command.go:1084ff` (v)                                  |
| cobra-M-01     | M       | `all shell completion generators`                                             | 5 files (v); ≥ 4 of 5; acceptable `shell_completions.go` |
| cobra-N-01     | N-far   | `where is the interactive TUI prompt implemented`                             | none (v)                                                 |

**`gson` (Java, multi-module)**

| id            | cat     | query                                                                                                        | primary / notes                                                                                                   |
| ------------- | ------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| gson-P1-01    | P1      | `where is ReflectiveTypeAdapterFactory defined`                                                              | `gson/src/main/java/com/google/gson/internal/bind/ReflectiveTypeAdapterFactory.java` (v)                          |
| gson-P1-DE-01 | P1-DE   | `wo ist MapTypeAdapterFactory definiert`                                                                     | `internal/bind/MapTypeAdapterFactory.java` (v)                                                                    |
| gson-P1d-01   | P1d     | `GsonBuilder.setFieldNamingPolicy`                                                                           | `GsonBuilder.java:413` (v)                                                                                        |
| gson-P2-01    | P2      | `com/google/gson/stream/JsonReader.java how is lenient mode handled`                                         | `JsonReader.java` (v); `JsonReader` fills a symbol slot                                                           |
| gson-P3-01    | P3      | `where is "Use JsonReader.setStrictness(Strictness.LENIENT) to accept malformed JSON" appended to the error` | `JsonReader.java:1713` (v)                                                                                        |
| gson-P4-01    | P4      | `who uses the Excluder`                                                                                      | `Gson.java`, `GsonBuilder.java`, `ReflectiveTypeAdapterFactory.java` (v); acceptable def `internal/Excluder.java` |
| gson-I1-01    | I1      | `how does an annotated field get a different name on the wire`                                               | `ReflectiveTypeAdapterFactory.java`, `annotations/SerializedName.java` (v)                                        |
| gson-I1-02    | I1      | `where does the parser refuse to escape a line break when running strict`                                    | `JsonReader.java:1901` (v)                                                                                        |
| gson-I1a-01   | I1a     | `where are java.util.Date values parsed`                                                                     | `internal/bind/DefaultDateTypeAdapter.java` (v); acceptable `JavaTimeTypeAdapters.java`                           |
| gson-I2-01    | I2      | `my long fields silently turn into doubles`                                                                  | `internal/bind/ObjectTypeAdapter.java`, `ToNumberPolicy.java`, `TypeAdapters.java` (v)                            |
| gson-I2x-01   | I2x     | `how does strictness change what the reader accepts and which exception it throws`                           | `Strictness.java`, `JsonReader.java`, `stream/MalformedJsonException.java` (pin)                                  |
| gson-I3-01    | I3-de   | `wo werden maps serialisiert`                                                                                | `internal/bind/MapTypeAdapterFactory.java` (v)                                                                    |
| gson-I3-02    | I3-typo | `ReflectiveTypeAdaptorFactory`                                                                               | `internal/bind/ReflectiveTypeAdapterFactory.java`                                                                 |
| gson-I4-01    | I4      | `what is the top-level entry point for turning an object into a JSON string`                                 | `Gson.java` (`toJson`) (v)                                                                                        |
| gson-M-01     | M       | `all TypeAdapterFactory implementations in internal/bind`                                                    | 5 named impls (v); ≥ 4 of 5; anonymous `FACTORY` fields acceptable                                                |
| gson-N-01     | N-far   | `where is YAML parsing implemented`                                                                          | none (v)                                                                                                          |

**`plugins` (polyglot, local)** — authored in Phase 0 by grepping the
checkout, full 16-slot template. Required: P1 = a bats test name and an mjs
export; P3 = a literal hook message; I1 = a skill's behaviour in prose
(identifier-free); I3-de with a compound noun; M = all `SKILL.md` mentioning
a term; N-far. Snippets from this repo never enter the report (private).

**Tier-2 anchors** (filled to the 9-slot template in Phase 0; missing slots
noted):

- `ripgrep` — authored in Phase 0, ideally not by the plan author. Anchors:
  binary `main`, the `--sort` flag definition (P3 on its help text), glob
  overrides, "wo werden die farben für treffer gesetzt" (I3-de), all
  printer types (M), Windows registry (N-far). Needs P4, I1, I1a.
- `hono` — `Hono` class in `src/hono-base.ts:98` **and** `src/hono.ts` (v) →
  P1t; `RegExpRouter` (v); `HTTPException` (`src/http-exception.ts`) (v);
  `cors` middleware (v); "how is the request path extracted"
  (`src/utils/url.ts`) (v) → I1; "alle middleware für authentifizierung"
  (`basic-auth`, `bearer-auth`, `jwt`, `jwk`) (v) → I3-de; all router
  implementations (5, v) → M; N-far `where is the SOAP envelope parsed`
  (GraphQL is N-near: docs/lockfile hits). Needs P3, P4.
- `humanizer` — `Humanize` (`StringHumanizeExtensions.cs`) (v) → P1;
  `ToWords` (v); `TimeSpanHumanizeExtensions` (v); "wo werden die
  pluralregeln für englische wörter definiert" → `Inflections/Vocabularies.cs`
  / `InflectionEngine.cs` (pin) → I3-de; all `*Formatter` classes
  (`Localisation/Formatters/` 5 + `CollectionFormatters/` 5) (v) → M;
  Markdown rendering (N, pin). Needs P3, P4, I1, I1a.
- `libuv` (`v1.x`) — `uv_timer_start` (`src/timer.c:67`) (v) → P1; `uv_run`
  ×2 (`src/unix/core.c:427`, `src/win/core.c:699`) (v) → P1t; `uv__io_poll`
  ×7 (v) → P1t with a 7-member `equivalent` group; `uv_spawn` run **twice**:
  unscoped (M, expect both `process.c`) and `scope_hint: src/unix` (P1,
  expect one); "where is the thread pool" (`src/threadpool.c`) (v) → I1; "wo
  werden fehlercodes in strings übersetzt" (`src/uv-common.c` `uv_strerror`)
  (v) → I3-de; Bluetooth (N-far, v). Needs P3, P4, I1a.
- `json` (`develop`, shallow) — `parse_error::create` (two overloads,
  `include/nlohmann/detail/exceptions.hpp:179,187`) (v) → P1d; `json_pointer`
  (v) → P1; `binary_reader` CBOR (v) → I1a; "where is the lexer" →
  `equivalent` {`include/nlohmann/detail/input/lexer.hpp`,
  `single_include/nlohmann/json.hpp`} — ≥ 2 returned = dedupe finding; P
  queries additionally with `scope_hint: include`; XML output (N-far, v).
  Needs P3, P4, M, I3-de.
- `sinatra` — `Sinatra::Base#route` (`lib/sinatra/base.rb:1776`; near-name
  `route!:1064`) (v) → P1d; `halt` (`:1028`) (v) → P1; `IndifferentHash`
  (`lib/sinatra/indifferent_hash.rb:41`) (v); P3 `"Sinatra doesn't know this
ditty"` (pin); "wie werden templates gerendert" (`base.rb` Templates) (v) →
  I3-de; all rack-protection middlewares (17 files, v) with `max_results:
50` and `scope_hint: rack-protection` → M; N-far `where is the LDAP bind
implemented` (WebSocket is N-near). Needs P4, I1, I1a.

### 6.4 Parameter variants (one P1, one I1, the N query per repo; pass 1)

Variant vocabulary (the `variant` column): `mr=<v>`, `scope=<kind>:<value>`,
`cache=<casing|ws|scope|mr>`, `state=<…>` (§6.5), always with
`base_query_id`.

- `max_results`: 0, 1, 50, and on the M query 1 (recall must be 1/n, not
  0); invalid `-1`, `"3"`, `3.5` (expect invalid-params), `4294967295`.
- `scope_hint`: correct dir; wrong-but-existing dir (expect 0 candidates →
  fallback; the LLM may legitimately search globally — score whether the
  summary mentions the scope); non-existent dir; `../` and an **absolute
  path inside the repo** (both silently dropped, F-06 — search runs
  unscoped, hint still shown to the LLM); `./src`; `src/../src` (rejected
  as `ParentDir` although inside); a **file** instead of a directory (`rg`
  accepts; check CBM `file_pattern`); a **glob** `src/**/*.py` (grep/file
  legs fail on the non-existent target; CBM legs get `file_pattern =
"src/**/*.py/**"` and return empty without an error); an ignored dir
  (`target/`, `node_modules/` — rtk's own filters apply, §3.2); `""`; `~/…`;
  a path with spaces; Windows-style `crates\repo-explorer-core` (one
  component with a backslash → empty legs).
- Cache-key variants: same text with a different `scope_hint`; same text
  with a different `max_results` (both: no cache hit); same text with
  different casing / leading whitespace (expect **cache hit** — key is
  normalized).

### 6.5 Repo-state variants (each in a fresh process)

- `state=cache-probe`: identical repeat immediately after the base P1 →
  `path="cache"`, latency ≤ git probe cost (§5 step 1) + 100 ms.
- `state=dirty-tracked`: append a comment to the expected file →
  fingerprint changes, no cache hit, still correct; `git checkout -- .`;
  repeat.
- `state=untracked-new`: new untracked file containing the answer → found;
  no cache hit.
- `state=untracked-edit`: edit inside an already-untracked file → expect
  (defect, F-07) a stale `path="cache"` hit.
- `state=no-git`: `git archive` copy → fingerprint `None`, caches off,
  still correct; `path="cache"` never appears.
- `state=subdir:<path>`: server `cwd` in a subdirectory
  (`sinatra-contrib/`, `crates/repo-explorer-mcp/`) → document what
  `repo_root` becomes and what a monorepo user sees.
- `state=shallow`: `--depth 1` clone (needed for `json` anyway) →
  fingerprint and `git diff HEAD` still work.

## 7. Metrics and scoring

### 7.1 Automatic metrics (`eval/score.py`)

| Metric                                                                                                           | Definition                                                                                                                                                                                                                                                                                                            | Source                     |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `path_valid`                                                                                                     | fraction of findings whose `path` exists under the repo root                                                                                                                                                                                                                                                          | fs                         |
| `line_valid`                                                                                                     | `1 ≤ line_start ≤ line_end ≤ file_len`; findings without lines counted as `symbol_only` per stage                                                                                                                                                                                                                     | fs                         |
| `hallucinated` (classified)                                                                                      | path missing → **fabricated path** (P0); range outside file → P0; snippet (whitespace-normalized, truncation marker stripped, **prefix ≤ 100 chars**) found elsewhere in the file → **misaligned** (P1); snippet nowhere → **fabricated snippet** (P0). Staleness excluded by construction (indexed commit = pin, §5) | fs                         |
| `file_hit@1/@3/@any`                                                                                             | a `primary` (or `equivalent` member) file at rank 1 / ≤ 3 / anywhere; file rank = rank of its first finding, files deduped; `primary_mode` decides all-of vs any-of                                                                                                                                                   | corpus                     |
| `range_hit@k`                                                                                                    | a returned range overlaps the target `span` (the matched `equivalent` member's span where applicable) **and** returned length ≤ max(3 × span, 40 lines); primary wherever a span exists                                                                                                                               | corpus                     |
| `range_len`                                                                                                      | distribution of returned range lengths per stage (UX)                                                                                                                                                                                                                                                                 | response                   |
| `recall@k`                                                                                                       | fraction of `primary` returned within `max_results` (M, I2x, `primary_mode: all`)                                                                                                                                                                                                                                     | corpus                     |
| `twin_confusion`                                                                                                 | a `distractor` returned (never negates a hit)                                                                                                                                                                                                                                                                         | corpus                     |
| `dedupe_defect`                                                                                                  | ≥ 2 members of one `equivalent` group returned                                                                                                                                                                                                                                                                        | corpus                     |
| graded relevance                                                                                                 | per finding 2 = primary, 1 = acceptable, 0 = other; manual-blind only for I* findings scoring 0 automatically                                                                                                                                                                                                         | corpus + rubric            |
| `precision@3`, `nDCG@3`                                                                                          | from graded relevance, uniform across categories                                                                                                                                                                                                                                                                      | derived                    |
| `negative_ok`                                                                                                    | N: `findings == []` and summary states nothing found → 1; all `path_valid` but off-topic with a hedging summary → 0.5; confident wrong → 0. Blind; `near`/`far` reported separately                                                                                                                                   | rubric                     |
| `confident_wrong`                                                                                                | `stage = early-exit ∧ ¬file_hit@any`                                                                                                                                                                                                                                                                                  | stderr + corpus            |
| `stage_expected`, `stage_observed`, `stage_match`                                                                | `cache` / `early-exit` / `verify` / `fallback` from the two INFO messages; `error` from `isError`; `timeout` from the harness. `error`/`timeout` count as misses in every hit metric                                                                                                                                  | stderr + response + corpus |
| `confidence`, `candidates`                                                                                       | from `retrieval pre-stage complete`                                                                                                                                                                                                                                                                                   | stderr                     |
| `cand_recall@top_k`, `cand_rank`                                                                                 | primary present in the ranked candidate list / its rank (§3.1 candidates line)                                                                                                                                                                                                                                        | stderr                     |
| `tokens`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `llm_calls`, `model_served`, `forced_finish` | §3.1 lines; `tokens = 0` for cache/early-exit                                                                                                                                                                                                                                                                         | stderr                     |
| `provider_events`                                                                                                | rate_limited / quota / invalid / auth / failover counts                                                                                                                                                                                                                                                               | stderr                     |
| `latency_ms`, `git_probe_ms`, `index_ms`                                                                         | wall time; §3.1 lines                                                                                                                                                                                                                                                                                                 | harness / stderr           |
| `result_bytes`                                                                                                   | UTF-8 size of the tool result JSON                                                                                                                                                                                                                                                                                    | harness                    |
| `pre_stage_identical`                                                                                            | same candidate list across two processes (F-03)                                                                                                                                                                                                                                                                       | stderr                     |
| `variance`                                                                                                       | across passes: distinct top-1 files, Jaccard of finding sets, stage agreement                                                                                                                                                                                                                                         | harness                    |
| `usd_per_correct`, `tokens_per_correct`                                                                          | cost = Σ prompt × p_in + (completion + reasoning) × p_out; price table in the manifest: `{source: "https://ai.google.dev/pricing", captured: <date>, model, usd_per_1M_input, usd_per_1M_output, tier}` with the page saved to `results/`                                                                             | derived                    |

### 7.2 Manual rubrics (blind)

`score.py export-blind --seed 1` → `blind.csv (row_id, finding_idx, query,
findings_json, summary)` with stage, arm, pass, config and repo hidden and
rows shuffled; `key.csv` kept apart. Graders fill `grade_summary` (0–2: 0
wrong or fabricated, 1 correct but generic, 2 correct, specific, honest
about uncertainty), `negative_ok`, and per-finding relevance where the
automatic rule scored 0. Files live in `results/<run>/grades/<rater>.csv`,
keyed `(row_id, finding_idx)`. A second rater (colleague, or an LLM judge
from a different vendor using `eval/judge_prompt.md`, written before Phase 1)
grades all I* and N rows; `score.py import-grades r1.csv r2.csv` reports
**unweighted Cohen's κ** per rubric and writes `adjudicate.csv` for
disagreements, resolved before scoring.

### 7.3 Baselines (Layer B)

| id  | Baseline                      | Definition                                                                                                                                                                                                                                                                                                                           | Purpose                                                      |
| --- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| B0  | naive `rg`                    | `rg -i -F -n` of the whole query and of each whitespace token; files ranked by hit count                                                                                                                                                                                                                                             | what a user does by hand                                     |
| B1  | BM25                          | document = file; tokens `[A-Za-z_][A-Za-z0-9_]{2,}` + camel/snake split; no stopwords; `rank_bm25.BM25Okapi`; top-10 files                                                                                                                                                                                                           | classic IR floor                                             |
| B2  | **deterministic-only SUT**    | config `b2.toml`: `early_exit_confidence = 0` **and** the provider `base_url` pointing at an unreachable local port, so any query with zero candidates (which would otherwise enter the fallback loop and call the LLM) fails immediately → `isError` → scored as a miss. Cross-checked offline from the `retrieval candidates` line | the ablation that isolates the LLM stages (primary metric 4) |
| B3  | CBM `search_code`, full query | same MCP stdio client and CBM path as `gen_synthetic.py`; `project` derived from the repo root as the memory crate does. Expected ≈ 0 hits on I-queries — that **is** the H2 result, not a harness bug                                                                                                                               | tests H2 directly                                            |

File rank is defined identically for all (§7.1).

### 7.4 Failure attribution (one class per failed query)

`retrieval-miss` (primary not in candidates) → `rank>k` (in candidates,
below top-k) → `verify-rejected-correct` (in top-k shown, LLM finished
without it) → `verify-hallucinated` (LLM returned a wrong location not among
the candidates) → `fallback-miss` → `fallback-budget` (forced finish or
synthesis) → `fabricated-path` → `error` / `timeout`. Backlog order = count
per class × stratum weight (uniform until usage data exists, §6.1).

### 7.5 Pre-registered thresholds and statistics

- Pooled strata: **Precise** (P1, P1-DE, P1t, P1d, P2, P3), **Relational**
  (P4, M), **Imprecise** (I1, I1a, I2, I2x, I3, I4), **Negative**. Sizes
  with 7 Tier-2 repos ≈ 51 / 26 / 69 / 13; without `guzzle` 48 / 24 / 66 /
  12; plus synthetic (Precise) and 12 issue-derived (Imprecise). Categories
  reported descriptively with Wilson 95 % CIs.
- Headline = **pass@1** = mean success over passes (a user calls once). Per
  query: deterministic failure (all passes), flaky, pass. Both failure kinds
  enter the backlog (flaky → prompt/robustness owner). "2 of 3" is only the
  bar for "reproducible enough to debug by hand".
- Relative thresholds: Imprecise `file_hit@3(tool) − file_hit@3(B2) ≥
0.15`; Precise `range_hit@1(tool) ≥ file_hit@1(B0)`; `confident_wrong ≤
2 %` (upper CI ≤ 5 %); `hallucinated = 0`; Negative `negative_ok ≥ 0.80`.
- Unit of analysis = query; passes are clustered → cluster bootstrap CIs;
  paired comparisons tool vs baseline on the same queries (McNemar / paired
  bootstrap). Minimum detectable effect at n ≈ 70, 3 passes ≈ 0.12 on a
  proportion near 0.6 — stated in the report.
- Efficiency expectations (descriptive): early-exit p50 < 3 s warm; verify
  < 20 s and < 15k tokens; fallback < 60 s and < 45k tokens.

## 8. Harness

### 8.1 Layer A — direct MCP driver (`eval/run.py`)

**Three full passes.** Each pass spawns a fresh server per repo and iterates
the corpus in a seeded shuffle (seed = pass number). In one process an
identical repeat is a cache hit (§2 Stage 0), so attempts must live in
separate processes.

Per repo per pass:

1. `git -C <clone> checkout --detach <pin>`; assert clean tree.
2. Spawn `~/.local/bin/repo-explorer-mcp` with `cwd = <clone>`, env =
   current env + `REPO_EXPLORER_CONFIG=<abs>/eval/config/<variant>.toml` +
   `NO_COLOR=1`. Capture stderr to
   `results/<run>/<repo>/pass<n>.stderr.log`. `mcp` SDK stdio client:
   `initialize` → `tools/list` (assert `explore_repository`) → `tools/call`.
   Per-call timeout 300 s; warm-up call 900 s.
3. Order: warm-up P1 (`index_first_call_ms`) → shuffled corpus → (pass 1
   only) §6.4 variants → §6.5 state variants, each in its own process.
   Sequential, 2 s pause. On a `rate_limited` provider line (Mode B: the
   `isError` regex in §3.1) pause 60 s and queue the query for a re-run at
   the end of the pass; keep both rows (`superseded = true` on the original,
   `attempt = 2, rerun_of = <row_id>` on the re-run); the scorer uses the
   highest attempt; `provider_events` counts both.
4. JSONL row per call: `row_id, run, pass, repo, pin, binary_version,
config_variant, query_id, variant, base_query_id, attempt, rerun_of,
process_seq, call_seq, request, response, is_error, timeout, ts_start,
ts_end, latency_ms, result_bytes, req_id, stderr_events[]` plus the
   materialized `stage, confidence, candidates, tokens, prompt_tokens,
completion_tokens, reasoning_tokens, llm_calls, model_served,
forced_finish, index_status, index_ms, git_probe_ms`. Events are matched
   by `req_id` (Mode B: the sequential slice between call start and
   response).
5. R-17 uses `run.py --concurrent 2 --queries a,b` (asyncio gather on one
   session).
6. Synthetic corpus: 2 passes, same mechanics. B2: 1 pass with `b2.toml`.

`results/<run>/manifest.json`: binary sha256 + `--version`; CBM binary
path/version and indexed commit per repo; rtk, rg, git, uv, Python + `mcp`
SDK versions; rtk config files; resolved config content (key env var name
only); pinned model id and how it was chosen; temperature note; harness git
sha; corpus file hashes; git probe cost per repo; price table; `pgrep`
snapshots; per-call timestamps.

Config variants (`eval/config/`): `default.toml` (§3.2); `b2.toml` (§7.3);
sweep variants (§10); `r12.toml` (`command = "/nonexistent/codebase-memory-mcp"`);
`r13.toml` (`search.timeout_seconds = 1` — also shortens the git probe and
may disable caching); `r14.toml` (`token_budget = 2000`); `r15.toml`
(`max_fallback_iterations = 1`); `r16a.toml` (`models =
["gemini-does-not-exist"]`); `r16b.toml` (first entry `base_url =
"http://127.0.0.1:<port>"` served by a 15-line `http.server` mock returning
429 `{"error":{"status":"RESOURCE_EXHAUSTED"}}` with a dummy key env var,
second entry the real pinned Gemini model — deterministic, no quota burnt,
no second real key needed).

### 8.2 Layer B — baselines (`eval/baseline.py`)

B0/B1 run offline against the pinned checkouts; B3 through the MCP client
described in §7.3; B2 is a Layer-A run.

### 8.3 Layer C — Claude Code in the loop (`eval/claude_loop.sh`)

Purpose: measure the tool as the user meets it. Verified against `claude
--help` 2.1.261: `--strict-mcp-config`, `--mcp-config`, `--allowedTools`,
`--append-system-prompt`, `--output-format stream-json`, `--model`,
`--max-budget-usd` exist; **`--max-turns` is not listed and unknown flags
are silently accepted** → the pilot session runs with `--max-turns 1` and
asserts `num_turns == 1` in the `result` event; if that fails, cap with
`--max-budget-usd` and report turns descriptively. Add `--verbose`
(historically required with `stream-json` in `-p` mode).

Isolation:

- `CLAUDE_CONFIG_DIR=$(mktemp -d)`; copy `~/.claude/.credentials.json` in
  at run time (`chmod 700`), write `settings.json` = `{}` and `.claude.json`
  = `{"hasCompletedOnboarding": true}`; delete the dir at exit. Only
  `eval/claude-profile/settings.json` is committed as a template; the two
  copied files are git-ignored (§3.2). Smoke criterion: one 1-turn session
  returns a `result` event.
- Pin `--model` to the full id shown in the pilot's `system/init` event;
  record `claude --version`.
- `eval/mcp.json` = `{"mcpServers":{"repo-explorer-mcp":{"command":
"/home/kwitsch/.local/bin/repo-explorer-mcp","args":[],"env":
{"REPO_EXPLORER_CONFIG":"<abs>/eval/config/default.toml","NO_COLOR":"1"}}}}`
  so Layer C uses the pinned model too, not the live 11-model chain.
- Repo CLAUDE.md is a controlled factor: external repos have none; `self`
  runs from a `git worktree add --detach` copy with `CLAUDE.md` deleted (the
  pinned clone stays clean), stated in the report.
- Cost of the with-tool arms includes the tool's Gemini tokens: Claude Code
  copies server stderr into
  `~/.cache/claude-cli-nodejs/<cwd with "/"→"-">/mcp-logs-repo-explorer-mcp/<ISO-ts>.jsonl`
  (one file per server start; records `{"error":"Server stderr: <lines
joined by \n>", "timestamp", "sessionId", "cwd"}`); join on `sessionId` =
  `session_id` in the stream-json `system/init` event; sum `tokens=` from
  `exploration complete`.

Three arms per task:

| Arm            | MCP config                                                                        | allowedTools                                                | Question         |
| -------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------- |
| native         | `--strict-mcp-config --mcp-config eval/empty-mcp.json`                            | `Read,Grep,Glob`                                            | floor            |
| tool-available | `--strict-mcp-config --mcp-config eval/mcp.json`                                  | `mcp__repo-explorer-mcp__explore_repository,Read,Grep,Glob` | uptake × quality |
| tool-mandated  | same + `--append-system-prompt "Use explore_repository before any Grep or Glob."` | same                                                        | quality alone    |

Tasks: per Tier-1 repo 4 (P1, P3, I1, I3), phrased as a developer would:
`Find where redirect handling lives. Answer with path:line only. Do not
modify files.` 6 × 4 × 3 arms × 2 reps = 144 sessions, sequential,
overnight. Pilot rule: after the first repo (24 sessions), if arm-level
success differs by < 0.05 between reps, stay at 2 reps; else 3.

Extraction from stream-json: success = any `([\w./-]+\.[A-Za-z0-9]+):(\d+)`
match in the `result` event's text whose path suffix-matches a `primary` ∪
`equivalent` file and whose line lies within `span` ± 5; no match → miss.
Tool uptake = any `assistant` event with `content[].type == "tool_use" &&
name == "mcp__repo-explorer-mcp__explore_repository"`. **Summary use** =
answer path ∈ paths parsed from that tool's `tool_result` JSON (vs found via
later Grep/Read). Also `num_turns`, `duration_ms`, usage/cost, Grep/Read
calls after the tool call. Analysis: cluster bootstrap by task; labelled
indicative unless extended to all Tier-1 queries.

## 9. Robustness and failure-injection cases (Phase 3)

Against `self` and `requests`, one server process per case, Claude Code
closed. "Expected today" is what the code does (review-verified);
graceful / degraded / defect is assigned afterwards.

| id       | Case (setup)                                                                                                                                                                                                                                                                             | Expected today                                                                                                                                                                                                            |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R-01     | `query: ""`                                                                                                                                                                                                                                                                              | **no validation** → 0 legs → fallback loop with the LLM on an empty prompt, ≤ 12 turns (F-04)                                                                                                                             |
| R-02     | 5 000-char paste: a Python traceback from `requests` tests, padded                                                                                                                                                                                                                       | ≤ 6 grep legs; several dropped by F-02; completes                                                                                                                                                                         |
| R-03     | `Session.request(url, **kwargs)`                                                                                                                                                                                                                                                         | metachars tokenized away; identifiers `Session`, `request`, `kwargs`; path token `Session.request(url` → empty file leg                                                                                                   |
| R-04     | `where is "Invalid URL`                                                                                                                                                                                                                                                                  | rest treated as plain text                                                                                                                                                                                                |
| R-05     | `sessions.py`                                                                                                                                                                                                                                                                            | file leg fails for any real file (F-01) → grep `sessions` → verify at best                                                                                                                                                |
| R-06     | `main` on `self`, `init` on `cobra`                                                                                                                                                                                                                                                      | many exact hits, margin ≈ 0 → verify; early-exit on a wrong file = `confident_wrong`                                                                                                                                      |
| R-07     | `deriv_paterns`                                                                                                                                                                                                                                                                          | 0 candidates → fallback; summary should admit uncertainty                                                                                                                                                                 |
| R-08     | `scope_hint: "../../"`                                                                                                                                                                                                                                                                   | silently dropped for legs (no log); shown to the LLM as `Scope hint: ../../`; in the cache key (F-06)                                                                                                                     |
| R-09     | `scope_hint` non-existent                                                                                                                                                                                                                                                                | rg exit 2 → grep/file legs fail (DEBUG) → 0 candidates → fallback; LLM tools default unscoped                                                                                                                             |
| R-10     | `max_results: 0`                                                                                                                                                                                                                                                                         | findings `[]`; LLM summary may describe dropped findings (F-09)                                                                                                                                                           |
| R-11     | unknown extra argument                                                                                                                                                                                                                                                                   | `deny_unknown_fields` → JSON-RPC invalid params                                                                                                                                                                           |
| R-12a    | CBM unavailable at startup: `r12.toml` (`command = "/nonexistent/…"`), no daemon running. (With the bare name and `~/.local/bin` on PATH the server would simply spawn a new daemon — that is the live behaviour, and it flips daemon ownership away from the plugin's copy afterwards.) | 5 s `/proc` grace wait (Linux, `command` set), then spawn error → exit 1 "failed to connect to codebase-memory-mcp (…)". The `--update` hint appears only when `command` equals the managed path and that file is missing |
| R-12b-i  | kill our server's own `--stdio` CBM child (`pgrep -P <server pid>`) after the first call                                                                                                                                                                                                 | `index_note` "Note: the memory backend is unavailable (…); rely on the grep/find/read_file search tools."; grep-only legs                                                                                                 |
| R-12b-ii | stop the daemon itself (`<daemon path> daemon stop`, or kill its pid) after connect                                                                                                                                                                                                      | same note; afterwards the next server start re-creates a daemon                                                                                                                                                           |
| R-13     | `r13.toml` on `gson`                                                                                                                                                                                                                                                                     | legs fail softly; git probe may time out → fingerprint `None` → caching off for that call                                                                                                                                 |
| R-14     | `r14.toml` on an I3 query **with 0 candidates** (confidence < 30; a verify-routed query would force finish inside verify instead)                                                                                                                                                        | one regular turn + one forced-finish call re-sending the whole context → `tokens` ≈ 2 × prompt size, unbounded relative to the budget; synthesis summary if the forced call fails                                         |
| R-15     | `r15.toml`                                                                                                                                                                                                                                                                               | one turn then forced finish                                                                                                                                                                                               |
| R-16a    | `r16a.toml` (invalid model)                                                                                                                                                                                                                                                              | no failover (`InvalidRequest`, or whatever class Gemini's 404 maps to — record it): verify escalates (WARN), fallback fails hard → `isError` "llm provider error: … request invalid" (F-11)                               |
| R-16b    | `r16b.toml` (mock 429 first, real second)                                                                                                                                                                                                                                                | failover with cooldown; result returned; `provider failover` line (§3.1). A blocked key (401/403) would be `Authentication` → hard failure, not failover                                                                  |
| R-17     | two concurrent **different** queries (`--concurrent 2`)                                                                                                                                                                                                                                  | both complete; no cross-talk in leg/tool caches; lines attributable via `req_id`                                                                                                                                          |
| R-18     | `eval/fixtures/make_r18.sh`: `git init; git submodule add ../requests sub; ln -s . loop; head -c 50M /dev/urandom > blob.bin; git add -A; git commit`; query `where is resolve_redirects defined` with `scope_hint: sub`                                                                 | completes < 300 s; no finding path under `blob.bin`                                                                                                                                                                       |
| R-19     | scratch repo with `printf 'fn x() {}\n\xff\xfe\n' > $(printf 'bad\xff.rs')`                                                                                                                                                                                                              | lossy path, no panic                                                                                                                                                                                                      |
| R-20     | `wo ist der einstiegspunkt`                                                                                                                                                                                                                                                              | `wo`, `der` stopwords; `ist` < 4 chars dropped; `einstiegspunkt` a useless grep leg → fallback                                                                                                                            |

## 10. Configuration sweep (Phase 5)

- `early_exit_confidence` and `fallback_confidence` are thresholds on a
  **logged number**: sweep them **offline** from the `retrieval candidates`
  - `confidence` lines of the main run (exact re-routing per query), then
    run the LLM only for queries whose route changes.
- Online, paired on the 159 hand-written queries, 3 passes, pinned model,
  paired bootstrap / McNemar: `top_k` {6, 12, 20}; `snippet_max_chars`
  {200, 400, 800}; model {pinned flash, pinned flash-lite}. Interaction
  `top_k × snippet_max_chars` at the 2×2 extremes if budget allows;
  everything else out of scope.
- Output per knob: `range_hit@3`, stage distribution, tokens median, latency
  p50, and a recommended default only when the paired CI excludes 0.

## 11. Execution phases

| Phase                | Content                                                                                                                                                                                                                                                                                                                                                                                                                | Exit criterion                                                                                                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 Setup              | §3 prerequisites (patch release + install + logging-only verification incl. parity run; `.gitignore`; profile template); clone + pin; Phase-0 probes; corpus: `rg`-verify every entry, fill `span`, author `plugins`/`ripgrep`, fill Tier-2 templates, generate synthetic corpus, pick 12 issues, transcript check; build `run.py`, `score.py`, `baseline.py`, `gen_synthetic.py`, `claude_loop.sh`, `judge_prompt.md` | `run.py` executes the `self` corpus end to end and `score.py` emits a table; manifest complete; parity run passed                                                                                    |
| 1 Harness validation | `self` + `requests`, full template + all variants, 2 passes                                                                                                                                                                                                                                                                                                                                                            | **harness** criteria only: stage parsed for 100 % of calls; scorer agrees with 10 hand-checked rows; `cand_recall` computed; F-01/F-02/F-03 probes evaluated. SUT findings are recorded, never block |
| 2 Tier-1 matrix      | 6 repos × 16 × 3 passes; synthetic × 2; issue-derived × 3; variants (pass 1); Layer B                                                                                                                                                                                                                                                                                                                                  | complete JSONL + manifest; blind grading done; κ reported                                                                                                                                            |
| 3 Robustness         | R-01 … R-20                                                                                                                                                                                                                                                                                                                                                                                                            | each classified graceful / degraded / defect with evidence                                                                                                                                           |
| 4 Claude in the loop | Layer C (the only phase with Claude Code open)                                                                                                                                                                                                                                                                                                                                                                         | three-arm comparison table                                                                                                                                                                           |
| 5 Tier-2 + sweep     | 7 repos × 9 × 3 passes; offline threshold sweep; online paired sweep                                                                                                                                                                                                                                                                                                                                                   | tables per knob with CIs                                                                                                                                                                             |
| 6 Analysis & report  | `docs/eval/report-<date>.md` + per-query CSV appendix                                                                                                                                                                                                                                                                                                                                                                  | reviewed report; backlog ordered by §7.4                                                                                                                                                             |

Stopping rule: after Phase 1, if < 10 % of LLM-path queries are flaky across
the 2 passes, Phase 2 may use 2 passes; otherwise 3.

## 12. From findings to backlog: mapping table

| Observed (attribution class)              | Owner                                                                                                     | First thing to inspect                                                         |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `retrieval-miss` on vague queries         | `core/src/retrieval.rs` `derive_patterns`, STOPWORDS (36 EN + 8 DE)                                       | `retrieval patterns` line; F-02 drops                                          |
| `retrieval-miss` on P2 / file names       | `search/parser.rs:410-415` + rtk per-file footer (F-01)                                                   | parse the footer instead of failing; or bypass rtk for the file leg            |
| `rank>k`                                  | `merge_and_rank`, base scores, coverage cap 120, density cap 60                                           | ContentHit 150 vs unrelated SymbolFuzzy 400                                    |
| unique symbol but stage = verify          | `confidence()` margin; CBM returning decl + impl + tests                                                  | candidate list; runner-up ≤ 390 rule                                           |
| `confident_wrong`                         | `symbol_candidates` exact rule (case-sensitive `last_segment`); symbol tokens inside P2 queries           | early-exit gate ignoring candidate kind                                        |
| semantic legs never contribute (H2)       | `pipeline.rs` `sanitized_query`; CBM `search_code` literal substring                                      | whether CBM offers non-literal search; otherwise drop the legs                 |
| `twin_confusion` / `dedupe_defect`        | `merge_and_rank` dedupe per (path, range) only                                                            | path-aware near-duplicate collapse; scope suggestion in summary                |
| non-code repos empty (H4)                 | CBM no symbols; grep legs 6 × 20 and F-02                                                                 | raise grep cap when symbol legs are empty; file leg for `SKILL.md`-style names |
| `verify-rejected-correct`                 | `verify.rs` prompt, skeleton cap 30, snippet 400, skeleton-or-snippet rule                                | `verify action` line; show snippet alongside skeleton                          |
| `verify-hallucinated` / `fabricated-path` | `tools.rs` `parse_finish` (F-05)                                                                          | validate `finish` findings against the fs before returning                     |
| `fallback-budget`                         | `agent.rs` `fallback_loop`, batch enforcement, `SEED_CANDIDATES = 8`, budget checked between turns (R-14) | `fallback turn` line                                                           |
| `fallback-miss`                           | `FALLBACK_SYSTEM_PROMPT`, tool catalog                                                                    | which tools the model used; was `trace_path` ever called                       |
| stale cache after change                  | `agent.rs` `query_cache_lookup` (decision), `git_probe.rs` digest (F-07), `cache.rs` CAS helpers          | untracked-edit variant                                                         |
| high warm latency                         | 3 git subprocesses per call; up to 17 legs; 1 `search_graph` per unique candidate file in verify; F-08    | `retrieval leg done` durations, `git_probe_ms`, `index_ms`                     |
| `pre_stage_identical` < 1                 | `search/backend.rs` — `rg` without `--sort` (F-03)                                                        | `--sort path` before truncation                                                |
| German underperformance                   | STOPWORDS (8 DE), no compound splitting, `ist` dropped by length                                          | `retrieval patterns` for DE queries                                            |
| silent scope drop (F-06)                  | `pipeline.rs:56-59`, `dispatch.rs:193-198`                                                                | log + mention in summary                                                       |
| empty query runs LLM (F-04)               | `server.rs` request validation                                                                            | reject blank queries at the MCP boundary                                       |
| inert config keys (F-12)                  | `config.rs` `Config` (no `deny_unknown_fields`)                                                           | warn on unknown keys in `config test`                                          |

## 13. Risks

| Risk                                                    | Mitigation                                                                                                                                                 |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gemini rate limits change the served model mid-run      | single pinned model, no chain (§3.2); `provider call` records the served model; rows with model ≠ pinned are flagged and re-run                            |
| CBM daemon conflicts / ownership flips                  | §3.2: Claude Code closed for Phases 1–3, 5; standalone binary owns the daemon; `pgrep` in the manifest; R-12 explicit                                      |
| Cold indexing of `gson`/`libuv`/`json` exceeds timeouts | 900 s warm-up; shallow clone for `json`; index forced in Phase 0                                                                                           |
| Ground truth drift                                      | pin-time `rg` verification; `span` recorded; unverifiable dropped                                                                                          |
| LLM non-determinism                                     | 3 independent passes; pass@1 headline; flaky vs deterministic reported separately                                                                          |
| Author bias (queries and grading)                       | identifier-free I-queries; issue-derived I2; synthetic P1/P3; `ripgrep` authored by someone else; blind grading + second rater; `self` reported separately |
| Corpus contamination of `self`                          | `scope_hint: crates` on all `self` queries                                                                                                                 |
| Layer C confounders                                     | isolated profile, pinned model and config, `--max-turns` verified in the pilot, third arm, cost incl. tool tokens                                          |
| Layer C cost/time                                       | overnight, sequential, pilot stopping rule                                                                                                                 |
| Private content in `plugins`                            | metrics only in the report                                                                                                                                 |
| Observability patch alters behaviour                    | logging-only verification procedure incl. parity run (§3.1)                                                                                                |
| Time-of-day latency drift                               | timestamp every call; report latency per phase window                                                                                                      |
| `--max-turns` silently ignored                          | pilot assertion; `--max-budget-usd` fallback                                                                                                               |

## 14. Deliverables

1. `eval/` (committed): `repos.toml`, `queries/*.yaml`, `config/*.toml`,
   `run.py`, `score.py`, `baseline.py`, `gen_synthetic.py`,
   `claude_loop.sh`, `judge_prompt.md`, `mcp.json`, `empty-mcp.json`,
   `claude-profile/settings.json`, `fixtures/make_r18.sh`.
2. `results/<run-id>/` (git-ignored): JSONL, stderr logs, manifest, scored
   CSV, blind-grading sheets, price-table capture.
3. `docs/eval/report-<date>.md`: primary metrics with CIs per stratum and
   per language; stage/token/latency distributions; Δ vs B0–B3; Layer C
   three-arm table; robustness classification; sweep tables; **failure
   attribution histogram**; the ordered backlog (P0 hallucination/crash, P1
   reproducible wrong answers, P2 efficiency, P3 UX/summary), each item →
   §12 owner + metric it should move + corpus subset that will show it;
   per-query CSV appendix.
4. Confirmation or refutation of F-01 … F-12 with evidence.

## 15. Effort estimate

Assumptions: Flash-class ≈ $0.30/M input, $2.50/M output; Flash-Lite ≈
$0.10/$0.40; ~90 % of tool tokens are prompt → blended ≈ $0.50/M (Flash).
Stage cost: early-exit 0, verify ≈ 10k, fallback ≈ 35k tokens. Claude Code
sessions on a subscription cost $0 marginal (rate-limited); on API ≈
$0.60/session.

| Phase                | Calls / sessions                                                                                                                     | Machine wall clock           | Gemini tokens                       | USD                                                  | Human hours                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | ----------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 0 Setup              | patch + release; 13 clones; ~160 ground-truth verifications; ~30 authored queries; 12 issues; 480 synthetic; ~1 000 lines of scripts | 2–4 h (cold indexing)        | < 0.5M                              | < $1                                                 | **35–50** (patch 4–6, scripts 12–16, ground truth 8–13, authoring 4–6, issues 2–3, probes 2–3, synthetic client 3–4) |
| 1 Harness validation | ≈ 190 calls                                                                                                                          | ≈ 1.2 h                      | ≈ 2.5M                              | ≈ $1.3                                               | 6–8                                                                                                                  |
| 2 Tier-1 matrix      | 288 corpus + ~390 variants + ~42 state + 960 synthetic + 36 issue + ~600 B2                                                          | ≈ 5.5 h                      | ≈ 15M                               | ≈ $7.5                                               | grading ≈ 11 per rater (+ LLM judge 1–2 h, $2–5) + 2 adjudication → **13–25**                                        |
| 3 Robustness         | ≈ 45 calls                                                                                                                           | 2–3 h hands-on               | ≈ 1M                                | ≈ $0.5                                               | 6–8                                                                                                                  |
| 4 Claude in the loop | 144 (+72 if escalated) sessions                                                                                                      | 7–11 h overnight             | ≈ 2M (+ ~30M Claude, mostly cached) | Gemini ≈ $1; Claude $0 (subscription) or ≈ $90 (API) | 4–6                                                                                                                  |
| 5 Tier-2 + sweep     | 189 Tier-2 + ≈ 3 340 sweep-eligible calls                                                                                            | ≈ 14–16 h                    | ≈ 33M                               | ≈ $14                                                | 4                                                                                                                    |
| 6 Analysis & report  | —                                                                                                                                    | —                            | —                                   | —                                                    | 12–16                                                                                                                |
| **Total**            | ≈ 5 600 calls + 144 sessions                                                                                                         | ≈ 32–38 h, mostly unattended | ≈ 52M                               | ≈ **$25** (+ $0–90 Claude)                           | **≈ 85–120**                                                                                                         |

Disproportionate parts, flagged for the plan owner (not cut — thoroughness
was requested): the online sweep is ≈ 55 % of all LLM tokens for effects
that at n = 159 will rarely clear the CI; the §6.4 scope variants on all 13
repos exercise code paths, not repo properties. If effort ever has to be
halved, cut in this order: sweep interaction cell → `snippet_max_chars`
online sweep → scope/max_results variants on repos beyond `self`/`requests`
→ blind grading of variant rows → Layer C escalation and `tool-mandated`
on repos 4–6 → synthetic second pass (keep a 50-query determinism probe) →
Tier-2 third pass. Never cut: three Tier-1 passes, the synthetic corpus,
B2, the observability patch, issue-derived queries.

## 16. Review checklist for this plan

- [x] Every hypothesis (H1–H4) and candidate defect (F-01–F-12) has a probe that can falsify it.
- [x] Every metric names a data source that exists today or is listed in §3.1.
- [x] Expected stages follow from §2.1 arithmetic.
- [x] Corpus covers every precision × language cell at least once; I4 has queries; German appears in P1 as well as I3.
- [x] No step reads the API key or edits the user's live config; variants go through `REPO_EXPLORER_CONFIG`.
- [x] The installed binary is the SUT; the observability patch is a release with a verification procedure; Mode B is defined.
- [x] Attempts are independent (fresh process per pass); B2 cannot call the LLM.
- [x] Every R-case names its config variant and setup; every §3.1 line names its emitting crate.
- [ ] Ground truth verified at pin (Phase 0).
- [ ] Executed by someone who has not read the codebase (Phase 1 dry run).

## 17. Review log

### v1 → v2 (three reviews: code fidelity, methodology, corpus)

Adopted — blockers: attempts in one process are cache hits → three passes
with fresh processes; no provider/leg/candidate logging → §3.1 prerequisite
and Mode B; failure attribution needs the candidate list (§7.4); Layer C
loaded plugins/hooks/rules/CLAUDE.md in both arms → isolated profile, third
arm, higher turn cap; R-16 invalid model does not fail over → split; **file
leg dead and common grep patterns dropped by rtk truncation** → F-01/F-02,
H3 reframed. Majors: corrected early-exit arithmetic; `cache` line has no
tokens, `error` only via `isError`; R-01/08/09/10/12/13/14 rewritten;
"warm" redefined (F-08); pooled strata, pass@1, relative pre-registered
thresholds, confident-wrong, hallucination classes, `range_hit`, graded
truth sets, nDCG@3, blind grading + second rater, baselines B0–B3,
issue-derived I2, synthetic corpus, transcript mining, offline threshold
sweep, manifest, cost per correct answer, statistics plan, Phase-1 exit =
harness validation, `.gitignore`, corpus contamination. Corpus: express
`master` = 5.2.1 (P3 literal removed → replaced), libuv `v1.x`, json 282 MB
→ shallow, gson P3 literal assembled at runtime → replaced, requests-P4 /
express-I2 / express-M / cobra-P4 fixed, polluted N queries → far negatives,
dotted P1 → P1d, anchored I → I1a, I4 added, German calque replaced,
compound-noun DE and casing typos added, `ripgrep` added, `guzzle`
optional, `uv__io_poll` ×7, `Hono` ×2, §6.4/§6.5 variants.

Declined: dropping `self` (kept, reported separately); `cache.enabled =
false` for independent attempts (changes latency and leg memoization); full
factorial sweep; JSON log format as a requirement.

### v2 → v3 (second-pass fidelity + executability)

Adopted — fidelity: P2 "can never early-exit" narrowed to symbol-free P2
queries (3 of 5 P2 rows carry a symbol); **B2 with `early_exit_confidence =
0` still called the LLM for zero-candidate queries** → unreachable provider
in `b2.toml`; R-12a with the bare name on this machine would spawn a new
daemon rather than fail → `r12.toml` with a non-existent path; H1–H4
definitions restored; "only one identifier may resolve to an exact symbol";
SUT config quoted as the 11-model chain it now is; `json` feature note;
diff-review file list extended to the memory/llm/server/main files;
`git_probe_ms` on every call; `r12`/`r16a`/`r16b` variants; Tier-2 template
fill-in and both stratum counts; P1t label for bare twins; "warm" defined;
Mode B back-off trigger; §10 cross-references; density wording; rtk footer
trigger described precisely; `NO_COLOR=1` instead of ANSI stripping; P1d
file-leg note; small corrections (verify.rs 35–43, 14 literal hits, 12 code
repos, `3xx`, R-16b rate-limited not blocked, F-11 README line 99, R-14
zero-candidate precondition, glob variant CBM behaviour).

Adopted — executability: span placement and emitted line shape; `req_id`
format; `llm_calls` counted in `TokenBudget` (core has no `tracing`);
provider lines from `GenaiProvider`; `model_served` field; `EnvFilter`
target filter; three-step logging-only verification incl. parity run; **CBM
ownership decision** (Claude Code closed for Phases 1–3, 5); `default.toml`
= live config verbatim minus chain; F-12 (inert `prefer_rtk`); isolated
Claude profile with runtime credential copy and git-ignore; `--max-turns`
absent from `claude --help` → pilot assertion + `--max-budget-usd` fallback;
`--verbose`; pinned `--model`; `eval/mcp.json` with env; stream-json
extraction rules; mcp-logs slug/record shape and `sessionId` join; `self`
CLAUDE.md via worktree copy; model catalog listing; temperature note;
`repos.toml` schema; shallow-pin rule; git probe cost measurement; forced
index via warm-up + `commit` field; `gen_synthetic.py` via `query_graph`
with the daemon's own path; issue-derived procedure with `gh api`;
transcript field path and the **zero real usage** finding; schema
completion (`primary_mode`, enums, optional fields, `equivalent` spans);
`self` pin fixed at `91eebde`; rtk config in the manifest and 100-char
snippet prefix; `uv --with` list; JSONL row fields; grades files; token
split and price-table provenance; blind-grading commands and unweighted κ;
variant vocabulary and re-run rows; B1/B3 specifications; R-02/06/12b/
16b/17/18/19 setups (R-16b via a local 429 mock on `base_url`); sweep query
set; effort estimate (§15) with the ranked cut list kept as information.

Declined: adding `tracing` to core (out of scope for a logging-only patch —
failover is derived from consecutive `provider call` lines instead); JSON
log output (Cargo change).
