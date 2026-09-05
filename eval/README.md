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
    introduced by the observability patch. Not yet filed as a plan candidate defect (would be
    F-13); repo-explorer-mcp's own code was not changed to fix it (out of scope for this edit).
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
