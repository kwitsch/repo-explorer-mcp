# repo-explorer-core (domain logic)

Domain logic (lib), no MCP/transport concerns: the pure retrieval logic
(`retrieval`: pattern derivation, candidate ranking, confidence) and the
`RepoStateProbe` fingerprint trait.

## Crate-boundary rules

- Must not add an `rmcp` dependency — `crates/repo-explorer-mcp` wires this
  crate's logic to the MCP protocol instead.
- Must not add an `anyhow` dependency — use `thiserror` typed errors
  (`ConfigError`, `ValidationError`) instead; see
  `crates/repo-explorer-mcp/CLAUDE.md` for how the binary boundary consumes
  them via `?`/`.context(...)`.
