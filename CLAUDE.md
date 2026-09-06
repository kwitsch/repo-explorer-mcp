# CLAUDE.md

Rust MCP server, shipped for Linux and Windows. Exposes the `explore_repository` tool over an rmcp stdio transport.

## Layout

- `crates/repo-explorer-core/` — domain logic (lib), no MCP/transport concerns. Details in `crates/repo-explorer-core/CLAUDE.md`.
- `crates/repo-explorer-mcp/` — server binary: hosts the `explore_repository` MCP tool and the `#[tokio::main]` bootstrap that wires the other five crates (core, memory, search, llm, agent) over rmcp stdio; owns the `rmcp` server dependency, the serde/schemars DTOs, the interactive setup wizard, and the headless `--install`/`--uninstall` Claude Code integration (`src/install.rs`). Details in `crates/repo-explorer-mcp/CLAUDE.md`.
- `crates/repo-explorer-memory/` — `MemoryBackend` implementation backed by an `rmcp` client to `codebase-memory-mcp`. Details in `crates/repo-explorer-memory/CLAUDE.md`.
- `crates/repo-explorer-llm/` — `GenaiProvider` (the sole `LlmProvider` impl) backed by the `genai` crate. Details in `crates/repo-explorer-llm/CLAUDE.md`.
- `crates/repo-explorer-search/` — `CliSearchBackend`: subprocess-driven text search over `rg` (ripgrep), plus `GitStateProbe`. Details in `crates/repo-explorer-search/CLAUDE.md`.
- `crates/repo-explorer-agent/` — `AgentLoop`: the exploration orchestrator; owns `serde_json` (core stays free of it). Details in `crates/repo-explorer-agent/CLAUDE.md`.
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
- The retrieval pipeline (pre-stage, LLM verification, fallback loop) lives in `crates/repo-explorer-agent/CLAUDE.md`.
- Domain-logic crate boundaries live in `crates/repo-explorer-core/CLAUDE.md`.
- Memory-backend, LLM-provider, and search-backend crate details live in each crate's own `CLAUDE.md` (`crates/repo-explorer-memory/`, `crates/repo-explorer-llm/`, `crates/repo-explorer-search/`).
