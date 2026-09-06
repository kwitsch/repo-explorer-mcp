---
name: release
description: Cut a repo-explorer-mcp release on request ("erstelle neues release auf Basis von origin/main") - direct version-bump commit on main, vX.Y.Z tag, then watch the Release workflow. Use only when the user explicitly asks to cut/ship a release.
disable-model-invocation: true
---

# Release

Direct-to-main procedure, never a PR:

1. `git checkout main && git pull`
2. Bump the workspace version everywhere it's pinned (7 files) — miss one and `cargo build` fails, not the workflow:
   - root `Cargo.toml` (`[workspace.package] version`)
   - `crates/repo-explorer-agent/Cargo.toml` (two core pins)
   - `crates/repo-explorer-mcp/Cargo.toml` (five pins)
   - `crates/repo-explorer-llm/Cargo.toml`
   - `crates/repo-explorer-memory/Cargo.toml`
   - `crates/repo-explorer-search/Cargo.toml`
   - `Cargo.lock` — run `cargo build --workspace` to refresh it
3. Do NOT bump `package.json` (the npx installer) — it deliberately stays independent of the workspace version.
4. Commit directly on `main`: `Bump workspace version to X.Y.Z`, push.
5. Tag: `git tag vX.Y.Z` (lightweight, on the bump commit) and push the tag. `.github/workflows/release.yml` refuses the release if the tag version doesn't match `cargo pkgid -p repo-explorer-mcp` — bump must land before the tag.
6. Watch it: `gh run list --workflow Release`, `gh run view <id>`. Report the asset list (linux tar.gz + windows zip, each with a `.sha256`).

Before a **bugfix** release: verify the bug still reproduces against the current released version first. If the fix already shipped, stop — don't cut a no-op release.

If the workspace version already has a matching `vX.Y.Z` tag/release, any bugfix branch/PR touching shipped code must bump at least the patch level as part of that branch (same 7-file mechanics, `cargo build` to refresh the lock) — but don't tag/push a release yourself unless separately asked; that stays a distinct, explicit request. Bugfixes go on a `fix/<slug>` branch + PR, never straight to `main` — this `release` skill's direct-to-main bump only applies to the version-bump-and-tag step itself, not to landing the underlying fix.

If the tag push produces no workflow run, check githubstatus.com before digging into repo config.
