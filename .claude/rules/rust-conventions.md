---
paths: ["**/*.rs", "**/Cargo.toml"]
---

# Rust conventions

- Format with `cargo fmt` (default rustfmt settings, no `rustfmt.toml` overrides).
- `cargo clippy --all-targets -- -D warnings` must be clean — clippy warnings are treated as errors.
- `repo-explorer-core` must not add an `rmcp` dependency — see root `CLAUDE.md`'s Layout section for the crate-boundary rationale.
- `repo-explorer-core` must not add an `anyhow` dependency — use `thiserror` typed errors (`ConfigError`, `ValidationError`) instead; see `crates/repo-explorer-mcp/CLAUDE.md` for how the binary boundary consumes them.
