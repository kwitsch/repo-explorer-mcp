# repo-explorer-agent (exploration orchestrator)

`AgentLoop` drives `explore_repository`'s search, plus the tool
catalog/dispatch, compressed rendering, and fingerprint-keyed result caches.
Owns `serde_json` — core stays free of it. Full pipeline design:
`docs/project-plan/8-retrieval_pipeline.md`.

## Pipeline stages

- **Retrieval pre-stage** — concurrent symbol/grep/file fanout; exits with
  zero LLM calls when it finds a confident match.
- **LLM verification stage** — runs over the top-k candidate skeletons the
  pre-stage produced.
- **Explorative fallback loop** — the hardened path when verification isn't
  confident: enforces a token budget, batches tool calls, and forces a final
  finish (via the Stage-4 `ProviderRouter`) rather than looping indefinitely.
