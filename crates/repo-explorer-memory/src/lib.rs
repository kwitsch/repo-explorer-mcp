//! `MemoryBackend` implementation backed by an `rmcp` client to
//! `codebase-memory-mcp`. This crate owns the `rmcp` dependency; the trait and
//! its value types live in `repo-explorer-core::memory`.

mod freshness;
