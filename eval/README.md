# explore_repository real-world eval — pilot scaffold

Implements the **pilot scope** (Phases 0-1, `self` + `requests` only) of
[`docs/eval/real-world-test-plan.md`](../docs/eval/real-world-test-plan.md). The rest of the
plan (Tier-2 repos, config sweep, robustness matrix, Claude-in-the-loop) is written but not yet
built out here — see the plan's §11 phase table.

## Status (2026-09-05)

- `repos.toml`, `queries/self.yaml`, `queries/requests.yaml` (16 ground-truthed queries each),
  `config/default.toml`, `run.py`, `score.py` exist and are harness-validated: a real smoke call
  against the installed 0.5.2 binary (`self-P1-01`, "where is derive_patterns defined") was run,
  and its response was scored end-to-end. Two bugs found and fixed by that one smoke call:
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
  - Also observed live: the pinned `~/repos/eval-corpus/self` clone was not yet indexed by
    codebase-memory-mcp on its first call (`retrieval leg failed leg="symbol"`/`"semantic"`,
    "project not found or not indexed"), confirming F-08 (every new server process reindexes on
    its first call) in the strongest form — the _first_ call's own memory legs failed outright,
    and only the _second_ call (13s later) saw the newly-indexed project. This is exactly why
    the warm-up call exists (§8.1 step 3) and why it must never be scored.
  - Also observed live: the warm-up call's own fallback-loop LLM attempt failed with
    `llm provider error: all configured LLM providers are exhausted or cooling down` — real
    evidence for the plan's own risk that this machine's Gemini quota is shared and can be
    exhausted by ordinary background activity (§13). Confirms why the _full_ Phase 1 run
    (2 passes × 32 queries) should happen with Claude Code closed, per the accepted decision
    below — it was not run in this session for that reason.
- `mcp.json`, `empty-mcp.json`, `claude-profile/settings.json`, `fixtures/make_r18.sh` are
  scaffolded for later phases (4, 3) but not yet exercised.
- `baseline.py` (Layer B), `gen_synthetic.py`, `claude_loop.sh`, `judge_prompt.md` — **not yet
  built**; out of pilot scope (Phase 2+).
- The §3.1 observability patch is being implemented in a separate background session
  (`claude attach 893c9b71` / `claude logs 893c9b71`); `run.py`/`score.py` run in Mode B until
  it lands (see the plan's §3.1 "Mode B" and the `NOT YET AVAILABLE` block `score.py` prints).

## Decisions in force for this pilot (accepted 2026-09-05)

- Logging patch: build first, install once verified (§3.1) — in progress.
- No interactive Claude Code session while the harness makes real MCP calls against the pinned
  repos (§3.2) — Phases 1-3 and 5 run with Claude Code closed. The smoke call above was a single
  deterministic (0-LLM, early-exit) query, not a full pass, and was run before this constraint's
  practical cost (shared CBM daemon, shared Gemini quota) was confirmed live.
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
a `.stderr.log`. `score.py` prints the §7.1 report (pass@1 per query, per-category `file_hit@3`
with Wilson CIs, hallucinations, confident-wrong cases, stage mismatches) and can dump the full
scored data as JSON with `--json-out`.
