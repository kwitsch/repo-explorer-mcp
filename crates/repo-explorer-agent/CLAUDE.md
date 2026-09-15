# repo-explorer-agent (exploration orchestrator)

`AgentLoop` drives `explore_repository`'s search, plus the tool
catalog/dispatch, compressed rendering, and fingerprint-keyed result caches.
Owns `serde_json` — core stays free of it. Full pipeline design:
`docs/project-plan/8-retrieval_pipeline.md`.

## Pipeline stages

- **Retrieval pre-stage** — concurrent symbol/grep/file fanout; exits with
  zero LLM calls when it finds a confident match. Two routes into that
  Stage-3 exit, both requiring a trusted exact symbol match (F-16), reported
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
- **Explorative fallback loop** — the hardened path when verification isn't
  confident: enforces a token budget, batches tool calls, and forces a final
  finish (via the Stage-4 `ProviderRouter`) rather than looping indefinitely.

## Per-query metrics

`QueryMetrics` (agent.rs) is emitted from **all five** `run` exit paths —
cache hit, early exit, verify, fallback, provider error (`path = "error"`) —
via `emit_metrics`. Two sinks: the headline fields as flat tracing fields on
that path's INFO log line (what `eval/run.py` parses), and, when
`REPO_EXPLORER_METRICS` names a non-empty path, the whole record as one
appended JSONL line — the only lossless copy (it also carries `repo_path`,
`query`, `findings_count`, `summary_len`). Never stdout — that is the MCP
JSON-RPC channel.
