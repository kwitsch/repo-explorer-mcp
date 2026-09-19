# repo-explorer-search (NativeSearchBackend)

`CliSearchBackend`: subprocess-driven text search over `rg` (ripgrep), plus
`GitStateProbe` (git2-backed, in-process libgit2 repo fingerprinting for the
caches). Owns `tokio`, `sha2`, `hex`, `which`, and `git2` — core stays free of
subprocess and git concerns.
