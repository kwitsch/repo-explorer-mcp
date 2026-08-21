# Rust conventions

- Format with `cargo fmt` (default rustfmt settings, no `rustfmt.toml` overrides).
- `cargo clippy --all-targets -- -D warnings` must be clean — clippy warnings are treated as errors.
- `crates/repo-explorer-core` holds domain logic and must stay free of MCP/transport concerns (no `rmcp` dependency there); `crates/repo-explorer-mcp` wires that logic to the MCP protocol.
- Error handling: `repo-explorer-core` uses `thiserror` for typed errors (`ConfigError`, `ValidationError`) and must not depend on `anyhow`; `repo-explorer-mcp` uses `anyhow` at the binary boundary (added when the binary first does fallible work) to consume core's typed errors via `?`/`.context(...)`.
