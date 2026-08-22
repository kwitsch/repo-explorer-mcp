# CLAUDE.md

Rust MCP server, shipped for Linux and Windows. Exposes the `explore_repository` tool over an rmcp stdio transport.

## Layout

- `crates/repo-explorer-core/` — domain logic (lib), no MCP/transport concerns.
- `crates/repo-explorer-mcp/` — server binary: hosts the `explore_repository` MCP tool and the `#[tokio::main]` bootstrap that wires the other five crates (core, memory, search, llm, agent) over rmcp stdio; owns the `rmcp` server dependency plus the serde/schemars DTOs.
- `crates/repo-explorer-memory/` — `MemoryBackend` implementation backed by an `rmcp` client to `codebase-memory-mcp`; owns the `rmcp` dependency (core does not).
- `crates/repo-explorer-llm/` — `GenaiProvider` (the sole `LlmProvider` impl) backed by the `genai` crate; owns the `genai` dependency (core does not) and provides `build_router(&LlmConfig)`. The `genai` SDK and every `genai::*` reference are confined to this crate.
- `crates/repo-explorer-search/` — `CliSearchBackend`: subprocess-driven text search over `rtk rg` / `rg --json`; owns `tokio`, `serde_json`, and `which` (core stays free of subprocess concerns).
- `crates/repo-explorer-agent/` — `AgentLoop`: the internal LLM-driven exploration loop over `MemoryBackend`/`SearchBackend` via the Stage-4 `ProviderRouter`, plus the internal tool catalog and dispatch; owns `serde_json` (core stays free of it).
- `.claude/rules/` — path-scoped rules Claude Code loads automatically when editing matching files.

## Build & test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Conventions

- Conventions for Rust code (including the error-handling split between `repo-explorer-core` and `repo-explorer-mcp`) live in `.claude/rules/rust-conventions.md`.
- `.mcp.json` registers `repo-explorer-mcp` (launched via `cargo run --release --quiet`); it reads config from `./repo-explorer.toml` (override with `--config <path>` or the `REPO_EXPLORER_CONFIG` env var) and requires a reachable `codebase-memory-mcp`.
