# repo-explorer-search (NativeSearchBackend)

`NativeSearchBackend`: in-process text/filename search over ripgrep's `ignore`

- `grep` library crates (no external `rg` binary), plus `GitStateProbe`
  (git-based repo fingerprinting for the caches). Owns `tokio` (with `process`
  feature for git probe), `sha2`, `hex`, `ignore`, and `grep` — core stays free
  of subprocess/search concerns. `which` is a dev-dependency (git-probe tests).
  Traversal uses rg's defaults (`.gitignore`/`.ignore`/hidden/binary skips, no
  symlink follow, single-threaded); determinism (F-03) comes from an explicit
  `(path, line)` sort before `max_results` truncation.
