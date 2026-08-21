# Rust conventions

- Format with `cargo fmt` (default rustfmt settings, no `rustfmt.toml` overrides).
- `cargo clippy --all-targets -- -D warnings` must be clean — clippy warnings are treated as errors.
- `crates/repo-explorer-core` holds domain logic and must stay free of MCP/transport concerns (no `rmcp` dependency there); `crates/repo-explorer-mcp` wires that logic to the MCP protocol.
- Error handling convention (thiserror vs anyhow) is not decided yet — pick one when the first fallible logic lands and stay consistent afterwards.
