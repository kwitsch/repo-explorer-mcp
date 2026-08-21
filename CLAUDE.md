# CLAUDE.md

Rust MCP server, shipped for Linux and Windows. Currently scaffolding only — no server logic implemented yet.

## Layout

- `crates/repo-explorer-core/` — domain logic (lib), no MCP/transport concerns.
- `crates/repo-explorer-mcp/` — server binary, depends on `repo-explorer-core` and `rmcp` (the official MCP Rust SDK).
- `crates/repo-explorer-memory/` — `MemoryBackend` implementation backed by an `rmcp` client to `codebase-memory-mcp`; owns the `rmcp` dependency (core does not).
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
- `.mcp.json` is intentionally absent until the server is actually runnable — add it once there's something for Claude Code to connect to.
