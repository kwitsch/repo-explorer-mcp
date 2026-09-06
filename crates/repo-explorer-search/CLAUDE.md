# repo-explorer-search (CliSearchBackend)

`CliSearchBackend`: subprocess-driven text search over `rg` (ripgrep), plus
`GitStateProbe` (git-based repo fingerprinting for the caches). Owns
`tokio`, `sha2`, `hex`, and `which` — core stays free of subprocess concerns.
