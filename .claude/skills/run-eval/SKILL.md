---
name: run-eval
description: Run the real-world eval harness (eval/run.py + eval/score.py) against the installed explore_repository binary and report cand_recall/hallucination/index-status metrics. Use when asked to run the eval, check candidate recall, or validate a fix against docs/eval/real-world-test-plan.md.
---

# Run eval

Prerequisites (from the plan's constraints):

- Close any other Claude Code session so the standalone `repo-explorer-mcp` binary owns the `codebase-memory-mcp` daemon.
- Confirm which binary is installed: `repo-explorer-mcp --version`. Metrics are only comparable across runs against the same version.
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
- hallucination flags (`fabricated_snippet`, `misaligned_snippet`) — cross-check against known `score.py` false positives before treating them as product bugs

Before trusting numbers from an older `results/` run, rescore it with the _current_ `eval/score.py` — the scorer itself has had bugs (multi-line snippet matching, alignment-window width, `equivalent` shape) that materially changed prior counts.

If a run surfaces a new reproducible defect, record it as a new F-NN row in `docs/eval/real-world-test-plan.md`'s candidate-defect table rather than only noting it in passing.
