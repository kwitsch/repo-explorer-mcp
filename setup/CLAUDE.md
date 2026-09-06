# setup/ (npx installer)

Node.js installer (`index.mjs`, ES module) for the `repo-explorer-mcp` binary:
resolves the platform/version, downloads the release archive, verifies its
`.sha256`, extracts it, then provisions managed dependencies via the freshly
installed binary's `--update` and prints an `.mcp.json` snippet. Invoked via
`npx github:kwitsch/repo-explorer-mcp`. Tests: `npm test` (`node --test
index.test.mjs`, pure/exported helpers only). Lint: `npm run lint` (`eslint .`).

## Gotchas before editing

- `installDir()` reimplements the Rust binary's own `dirs::executable_dir()` /
  `dirs::data_local_dir()` resolution by hand (see
  `crates/repo-explorer-mcp/CLAUDE.md`'s Config path resolution section) — same
  shared-bin-dir contract, independently implemented, not shared code. Keep
  both in sync if either side's resolution logic changes.
- `package.json`'s version is not bumped in lockstep with the Rust workspace's
  release version — don't "fix" that silently, it's unconfirmed whether it's
  deliberate.
- `provisionManagedTools`'s log line, `reprobeRtk`, and `printSummary`'s
  `rtkVer` still treat `rtk` as a tool `repo-explorer-mcp --update` provisions.
  Per that binary's own `crates/repo-explorer-mcp/CLAUDE.md` (Self-update
  section), `rtk` is no longer provisioned there — these probes now only ever
  report a stale leftover binary or `missing`. Flag before touching this area;
  unclear whether the mismatch is deliberate.
- `ensureRipgrep` tries an interactive Linux package-manager (apt/dnf/pacman)
  or `winget` install before falling back to the managed copy `--update`
  provisions. On Windows, a successful `winget` install can't be re-probed by
  the running process (its PATH was inherited at startup) — that branch
  reports "installed (restart your terminal to use)" instead of re-checking.
- The Windows install dir requires `LOCALAPPDATA` and throws early if unset;
  there's no fallback chain like Linux's `$XDG_BIN_HOME` / `~/.local/bin`.
- `isMainModule()` compares realpaths, not raw paths: npm's generated bin is a
  symlink to this file, and Node resolves symlinks for `import.meta.url` but
  not for `process.argv[1]` — don't simplify that comparison.
- The "impure helpers" (spawnSync-based probes, download, extraction) are
  intentionally untested by `index.test.mjs`; they're exercised by the manual
  smoke checklist in `docs/smoke-test.md` (§6, the idempotent-re-run step).
