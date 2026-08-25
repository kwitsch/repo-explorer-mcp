# repo-explorer-mcp (server binary)

The MCP boundary: hosts the `explore_repository` tool, owns the `rmcp` server
dependency plus the serde/schemars DTOs, and uses `anyhow` here (core's typed
errors are consumed via `?`/`.context(...)`).

## Setup wizard (`src/setup.rs`)

- All interactive IO, the env-var scan, the free-tier model catalog, and TOML file writing live here at the binary boundary.
- Which env var belongs to which provider kind is core's (`config::default_api_key_env`), as is what counts as "set" (`config::env_var_is_set`).
- The wizard derives its candidate table from those rather than restating them.
- Serialization lives in core (`config::to_toml_string`); the binary adds no `toml` dependency.
- The wizard writes to the _resolved_ config path (the XDG default unless `--config`/`REPO_EXPLORER_CONFIG` overrides it).
- It self-verifies the written file via `repo_explorer_core::config::load`.
- The `[search]`, `[agent]`, `[cache]`, and `[logging]` sections are left at their (fully defaulted) core values — not prompted. `main.rs::run` plumbs `config.agent`/`config.cache` into `AgentLoop::new` together with a `GitStateProbe` built from `config.search.timeout_seconds`.

## Config path resolution

- `.mcp.json` registers `repo-explorer-mcp`, launched via `cargo run --release --quiet`.
- Precedence: `--config <path>` CLI arg -> `REPO_EXPLORER_CONFIG` env var -> XDG default **if it exists** -> `./repo-explorer.toml` **if it exists** -> the XDG default again, as the wizard's write target.
- The XDG default is `$XDG_CONFIG_HOME/repo-explorer/repo-explorer.toml` on Linux, `%APPDATA%\repo-explorer\repo-explorer.toml` on Windows.
- The two `exists` gates are load-bearing: the XDG default resolves on essentially every machine, so returning it unconditionally would make the `./repo-explorer.toml` fallback dead code and silently ignore an in-repo config — including `.mcp.json`'s own no-`--config` launch.
- This crate owns the `dirs` dependency used for XDG resolution; core and the other crates stay free of it.
- A reachable `codebase-memory-mcp` is required at run time.

## Subcommands

- `config test` (or `--config-test`) validates the resolved config only — parse + semantic checks, no server/memory/LLM/search connections.
- It prints a structured JSON report to stdout and exits non-zero on failure.
- `setup` (mirroring `config test`) runs the interactive wizard.
- The wizard also auto-runs when the resolved config is missing, but **only if stdin is a TTY**.
- A non-interactive launch with no config prints guidance to stderr naming the `setup` subcommand, then exits non-zero — never blocking, never writing to stdout.
- "Missing" means `ConfigError::is_not_found`, not a bare `Path::exists` probe, so an unreadable or malformed config reports its real error instead of "no config".
- `--update` checks this binary and its runtime dependency binaries against their latest GitHub release, installs anything newer, and prints a structured JSON report to stdout; non-zero exit if any component errors.
- Subcommand/flag detection runs over `args_without_config_value(argv)`, never raw `argv`: the value of a `--config <path>` pair must never be read as a subcommand (`--config setup` names a file, not the wizard).

## Self-update (`src/update.rs`)

- Tracked components: `repo-explorer-mcp` (`kwitsch/repo-explorer-mcp`), `rtk` (`rtk-ai/rtk`), `rg`/ripgrep (`BurntSushi/ripgrep`), `codebase-memory-mcp` (`DeusData/codebase-memory-mcp`).
- Runs instead of the MCP server loop, dispatched before config resolution and before the `setup` dispatch/auto-run.
- Downloads the release asset matching the current OS/arch and verifies it against a `<asset>.sha256` sidecar when the release publishes one.
- Extracts the binary when the asset is a `.tar.gz`/`.zip` archive.
- Runs the extracted file with `--version` to confirm it executes **before** replacing anything already installed — self via `self_replace`, dependency binaries via atomic rename next to their resolved `which` path.
- A dependency binary whose installed version can't be determined, or that isn't on `PATH`, is skipped rather than blindly overwritten.
- This crate owns the `reqwest`/`semver`/`sha2`/`hex`/`flate2`/`tar`/`zip`/`self-replace` dependencies; core stays free of them.
- It also uses `which`, already owned by `repo-explorer-search` — the only dependency this crate shares with another non-core crate rather than owning outright.
