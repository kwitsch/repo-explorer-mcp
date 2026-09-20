# eval/ (real-world accuracy harness)

Python harness (`run.py`, `score.py`) that drives the installed `explore_repository`
binary against pinned real repos (`repos.toml`) and ground-truthed queries
(`queries/*.yaml`), implementing the pilot scope of
`docs/eval/real-world-test-plan.md`. Full status/history: `eval/README.md`.
How to run it: the `run-eval` skill (`.claude/skills/run-eval/SKILL.md`).

## Gotchas before editing or running

- Run with Claude Code closed. The installed standalone binary and any interactive
  Claude Code session share the same per-user `codebase-memory-mcp` daemon and the
  same LLM provider quota — running both concurrently starves the harness.
- Metrics are only comparable across runs against the same installed
  `repo-explorer-mcp` version (`repo-explorer-mcp --version`).
- A fresh server process's _first_ call always fails its memory legs
  ("project not found or not indexed") — the project only becomes indexed
  partway through that first call. This is why a warm-up call exists before
  the scored passes and must never itself be scored. The warm-up query is
  `warmup_query` in `repos.toml` and must not be a corpus query: it seeds the
  process's L1 cache, so a corpus query used here is a `stage=cache` row in
  every scored pass and never measured.
- `score.py`'s `Retrieval leg health` section is the memory-outage alarm: a
  memory leg that fails never appears in `leg_timings`, so before it existed a
  100%-failing `codebase-memory` backend just vanished from the per-leg table
  (2026-09-07 .. #56, two weeks of grep-only `cand_recall`). Never compare
  `cand_recall` across a run that prints its `WARNING` line.
- Always rescore an older `results/<run-id>` with the _current_ `score.py`
  before trusting its numbers — past scoring bugs have materially changed
  historical counts (see `git log -- eval/score.py` for specifics).
- Ground-truth span authoring rules for `queries/*.yaml` live in
  `eval/queries/CLAUDE.md`.
- `mcp.json`, `empty-mcp.json`, `claude-profile/settings.json`, and
  `fixtures/make_r18.sh` are scaffolded for later plan phases, not yet
  exercised by anything in this directory — don't assume they're dead weight.
