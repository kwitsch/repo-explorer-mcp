# repo-explorer-mcp

A Rust MCP server that exposes the `explore_repository` tool over an rmcp stdio
transport, shipped for Linux (`x86_64-unknown-linux-gnu`) and Windows
(`x86_64-pc-windows-msvc`). It drives an internal LLM exploration loop over a
managed `codebase-memory-mcp` backend and in-process text search (over ripgrep's
`ignore` + `grep` library crates, no external binary) — the memory backend is
provisioned by `repo-explorer-mcp --update` into a shared per-user bin dir
(`$XDG_BIN_HOME` if set to an absolute path, else `~/.local/bin`, on Linux;
`%LOCALAPPDATA%\repo-explorer-mcp` on Windows), never a global PATH install.

## Documentation

- [Response shape](docs/response-shape.md) — the `explore_repository` output schema and how to parse it from a client
- [Installation](docs/installation.md) — build from source, download a release, or verify the binary
- [Configuration](docs/configuration.md) — the TOML config, the setup wizard, and the on-disk result cache
- [`.mcp.json`](docs/mcp-json.md) — wiring the server into Claude Code
- [Troubleshooting](docs/troubleshooting.md) — config-load errors and their fixes

## Build & test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
