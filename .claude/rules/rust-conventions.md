---
paths: ["**/*.rs", "**/Cargo.toml"]
---

# Rust conventions

- Format with `cargo fmt` (default rustfmt settings, no `rustfmt.toml` overrides).
- `cargo clippy --all-targets -- -D warnings` must be clean — clippy warnings are treated as errors.
- `repo-explorer-core`'s crate-boundary and error-handling rules live in `crates/repo-explorer-core/CLAUDE.md`.
