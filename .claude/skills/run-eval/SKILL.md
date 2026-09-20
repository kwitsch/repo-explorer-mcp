---
name: run-eval
description: Run the real-world eval harness (eval/run.py + eval/score.py) against the installed explore_repository binary and report cand_recall/hallucination/index-status metrics. Use when asked to run the eval, check candidate recall, or validate a fix against docs/eval/real-world-test-plan.md.
---

# Run eval

Prerequisites (from the plan's constraints):

- Close any other Claude Code session so the standalone `repo-explorer-mcp` binary owns the `codebase-memory-mcp` daemon.
- Confirm which binary is installed: `repo-explorer-mcp --version`. Metrics are only comparable across runs against the same version. To evaluate a build without replacing the installed binary (an open Claude Code session holds it busy), pass `--binary <path>` to `run.py`; `manifest.json` records both `binary_version` and `memory_version` (the `codebase-memory-mcp` in use).
- Attempts against the same query in one server process are cache hits — independent measurement needs fresh processes (the harness already re-launches per pass).

Run:

```bash
uv run --with mcp --with pyyaml eval/run.py --repos self requests --passes 2
uv run --with pyyaml eval/score.py results/<run-id>
```

`<run-id>` is the timestamp directory `run.py` just created under `results/`.

Read the score report for:

- `cand_recall@top_k` (with its 95% CI) — the primary accuracy signal
- provider call outcomes (`ok`/`quota`/`rate_limited`) — a run dominated by `quota` is not a valid accuracy signal, rerun later
- `index_status` per call — `IndexingFailed` on every call points at a harness/index bug, not the retrieval logic
- `Retrieval leg health` — a `WARNING: memory legs ... dead` line means every `symbol`/`semantic`/`bm25` leg failed or returned nothing; the run measured grep alone and its `cand_recall` is not comparable to any run with a live memory backend (this is how the 2026-09-07..#56 outage hid). Also check `hit delivered by candidate kind`: all-`ContentHit` is the same symptom in a softer form
- `Stage mismatches` — P1/P1-DE now carry `stage: early-exit`; a P1 that lands in `verify` without an `early_exit_fallthrough` cause is a QW-2/confidence regression
- hallucination flags (`fabricated_snippet`, `misaligned_snippet`) — cross-check against known `score.py` false positives before treating them as product bugs

Before trusting numbers from an older `results/` run, rescore it with the _current_ `eval/score.py` — the scorer itself has had bugs (multi-line snippet matching, alignment-window width, `equivalent` shape) that materially changed prior counts.

If a run surfaces a new reproducible defect, record it as a new F-NN row in `docs/eval/real-world-test-plan.md`'s candidate-defect table rather than only noting it in passing.
