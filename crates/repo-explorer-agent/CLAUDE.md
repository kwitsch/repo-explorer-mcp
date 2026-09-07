# repo-explorer-agent (exploration orchestrator)

`AgentLoop` drives `explore_repository`'s search, plus the tool
catalog/dispatch, compressed rendering, and fingerprint-keyed result caches.
Owns `serde_json` — core stays free of it. Full pipeline design:
`docs/project-plan/8-retrieval_pipeline.md`.

## Pipeline stages

- **Retrieval pre-stage** — concurrent symbol/grep/file fanout; exits with
  zero LLM calls when it finds a confident match.
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
