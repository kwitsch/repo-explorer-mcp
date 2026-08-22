# CLAUDE.md

Rust MCP server, shipped for Linux and Windows. Exposes the `explore_repository` tool over an rmcp stdio transport.

## Layout

- `crates/repo-explorer-core/` — domain logic (lib), no MCP/transport concerns.
- `crates/repo-explorer-mcp/` — server binary: hosts the `explore_repository` MCP tool and the `#[tokio::main]` bootstrap that wires the other five crates (core, memory, search, llm, agent) over rmcp stdio; owns the `rmcp` server dependency plus the serde/schemars DTOs. Also owns the interactive first-run setup wizard (`src/setup.rs`): all interactive IO, env-var detection, the free-tier model catalog, and TOML file writing live here at the binary boundary (`anyhow`), never in `core`.
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

- Conventions for Rust code live in `.claude/rules/rust-conventions.md`.
- `.mcp.json` registers `repo-explorer-mcp` (launched via `cargo run --release --quiet`); config path precedence is `--config <path>` CLI arg -> `REPO_EXPLORER_CONFIG` env var -> XDG default (`$XDG_CONFIG_HOME/repo-explorer/repo-explorer.toml` on Linux, `%APPDATA%\repo-explorer\repo-explorer.toml` on Windows) -> `./repo-explorer.toml` fallback, and it requires a reachable `codebase-memory-mcp`. `repo-explorer-mcp config test` (or `--config-test`) validates the resolved config only — parse + semantic checks, no server/memory/LLM/search connections — printing a structured JSON report to stdout and exiting non-zero on failure. `repo-explorer-mcp` owns the `dirs` dependency used for XDG resolution; core and the other crates stay free of it.
- The binary supports a `setup` subcommand (mirroring `config test`) that runs the interactive wizard, and auto-runs it when the resolved config is missing **only if stdin is a TTY**; a non-interactive launch with no config prints `run `repo-explorer-mcp setup`` guidance to stderr and exits non-zero (never blocking or writing to stdout). The wizard writes to `xdg_default_config_path()` and self-verifies via `repo_explorer_core::config::load`. Serialization lives in `core` (`config::to_toml_string`); the binary adds no `toml` dependency.
