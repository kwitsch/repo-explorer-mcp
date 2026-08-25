# CLAUDE.md

Rust MCP server, shipped for Linux and Windows. Exposes the `explore_repository` tool over an rmcp stdio transport.

## Layout

- `crates/repo-explorer-core/` — domain logic (lib), no MCP/transport concerns; includes the pure retrieval logic (`retrieval`: pattern derivation, candidate ranking, confidence) and the `RepoStateProbe` fingerprint trait.
- `crates/repo-explorer-mcp/` — server binary: hosts the `explore_repository` MCP tool and the `#[tokio::main]` bootstrap that wires the other five crates (core, memory, search, llm, agent) over rmcp stdio; owns the `rmcp` server dependency, the serde/schemars DTOs, and the interactive setup wizard. Details in `crates/repo-explorer-mcp/CLAUDE.md`.
- `crates/repo-explorer-memory/` — `MemoryBackend` implementation backed by an `rmcp` client to `codebase-memory-mcp`; owns the `rmcp` dependency (core does not).
- `crates/repo-explorer-llm/` — `GenaiProvider` (the sole `LlmProvider` impl) backed by the `genai` crate; owns the `genai` dependency (core does not) and provides `build_router(&LlmConfig)`. The `genai` SDK and every `genai::*` reference are confined to this crate.
- `crates/repo-explorer-search/` — `CliSearchBackend`: subprocess-driven text search over `rtk rg` / `rg --json`, plus `GitStateProbe` (git-based repo fingerprinting for the caches); owns `tokio`, `serde_json`, `sha2`, and `which` (core stays free of subprocess concerns).
- `crates/repo-explorer-agent/` — `AgentLoop`: the exploration orchestrator — deterministic retrieval pre-stage (concurrent symbol/grep/file fanout, early exit with zero LLM calls), LLM verification stage over top-k candidate skeletons, and the hardened explorative fallback loop (token budget, batch enforcement, forced final finish) via the Stage-4 `ProviderRouter`; plus tool catalog/dispatch, compressed rendering, and fingerprint-keyed result caches; owns `serde_json` (core stays free of it). Pipeline details in `docs/project-plan/8-retrieval_pipeline.md`.
- `.claude/rules/` — path-scoped rules Claude Code loads automatically when editing matching files.

## Build & test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Conventions

- Conventions for Rust code live in `.claude/rules/rust-conventions.md`.
- CLI subcommands (`config test`, `setup`, `--update`), config-path resolution, and setup-wizard behavior live in `crates/repo-explorer-mcp/CLAUDE.md`.
