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
- The `[search]` section receives `rtk_path` — the resolved absolute path of the managed `rtk` binary in the shared bin dir (not interactively prompted); `[agent]`, `[cache]`, and `[logging]` are left at their (fully defaulted) core values. `main.rs::run` plumbs `config.agent`/`config.cache` into `AgentLoop::new` together with a `GitStateProbe` built from `config.search.timeout_seconds`.

## Config path resolution

- Precedence: `--config <path>` CLI arg -> `REPO_EXPLORER_CONFIG` env var -> XDG default **if it exists** -> `./repo-explorer.toml` **if it exists** -> the XDG default again, as the wizard's write target.
- The XDG default is `$XDG_CONFIG_HOME/repo-explorer/repo-explorer.toml` on Linux, `%APPDATA%\repo-explorer\repo-explorer.toml` on Windows.
- The two `exists` gates are load-bearing: the XDG default resolves on essentially every machine, so returning it unconditionally would make the `./repo-explorer.toml` fallback dead code and silently ignore an in-repo config.
- This crate owns the `dirs` dependency used for XDG resolution; core and the other crates stay free of it.
- `codebase-memory-mcp` and `rtk` are managed per-user copies in the shared bin dir (`~/.local/bin` on Linux via `dirs::executable_dir()`, `%LOCALAPPDATA%\repo-explorer-mcp` on Windows via `dirs::data_local_dir().join("repo-explorer-mcp")` — the same dir the npx installer uses for the main binary), provisioned/updated only by `--update` and launched by absolute path — never resolved via PATH/`which`. `setup` writes those absolute paths into `[codebase_memory] command` and `[search] rtk_path`; `run()` fails fast with a `--update` hint if either is missing and never downloads.

## Subcommands

- `config test` (or `--config-test`) validates the resolved config only — parse + semantic checks, no server/memory/LLM/search connections.
- It prints a structured JSON report to stdout and exits non-zero on failure.
- `setup` (mirroring `config test`) runs the interactive wizard.
- The wizard also auto-runs when the resolved config is missing, but **only if stdin is a TTY**.
- A non-interactive launch with no config prints guidance to stderr naming the `setup` subcommand, then exits non-zero — never blocking, never writing to stdout.
- "Missing" means `ConfigError::is_not_found`, not a bare `Path::exists` probe, so an unreadable or malformed config reports its real error instead of "no config".
- `--update` checks this binary and its runtime dependency binaries against their latest GitHub release, installs anything newer, and prints a structured JSON report to stdout; non-zero exit if any component errors.
- Subcommand/flag detection runs over `args_without_config_value(argv)`, never raw `argv`: the value of a `--config <path>` pair must never be read as a subcommand (`--config setup` names a file, not the wizard).
- `--install` registers this binary with Claude Code and is fully headless: it shells out to `claude mcp add repo-explorer-mcp --scope user -- <current_exe>` (idempotent remove-then-add, never hand-editing `~/.claude.json`) and, only once that registration succeeds, writes a Haiku subagent to `<home>/.claude/agents/explore.md` with `name: Explore` (capital E, deliberately matching Claude Code's built-in `Explore` agent's `agentType` exactly, since override/shadowing a built-in is a literal, case-sensitive name match — a lowercase `explore` would just add a second, separate agent instead of replacing the built-in one); a failed registration reports the agent-file step `skipped` instead, so a broken/unregistered server is never left shadowing the built-in agent. Install also leaves a pre-existing file at that path alone (`skipped`) if its contents don't match what install would write, symmetric with uninstall's guard below. It fails fast with a non-zero exit and `claude_code_detected: false` when `claude` is not on PATH. `--uninstall` reverses those two changes idempotently, tolerating an absent `claude` and an already-deleted agent file (both reported as `skipped`, not errors); it only deletes the agent file if its contents still match what `--install` wrote — a hand-edited or replaced file at that path is left in place (`skipped`), never silently destroyed. Both print a per-step (`mcp-server`, `agent-file`) JSON report to stdout and exit non-zero only when a step errors, mirroring `--update`. Dispatched before config resolution, so neither loads or creates `repo-explorer.toml`; `--update` (checked first) takes precedence over both, and when `--install`/`--uninstall` are both passed without `--update`, `--install` wins (checked first).

## Self-update (`src/update.rs`)

- Tracked components: `repo-explorer-mcp` (`kwitsch/repo-explorer-mcp`) and `rg`/ripgrep (`BurntSushi/ripgrep`) — `rg` resolved on PATH via `which`, skip-if-absent — plus two managed install-if-absent / update-if-stale copies in the shared bin dir (`~/.local/bin` on Linux, `%LOCALAPPDATA%\repo-explorer-mcp` on Windows): `rtk` (`rtk-ai/rtk`) via `provision_or_update_rtk_binary` and `codebase-memory-mcp` (`DeusData/codebase-memory-mcp`) via `provision_or_update_memory_binary`.
- Runs instead of the MCP server loop, dispatched before config resolution and before the `setup` dispatch/auto-run.
- The `which`-resolved `rg` dependency, when its installed version can't be determined or it isn't on `PATH`, is skipped rather than blindly overwritten. The managed `rtk` and `codebase-memory-mcp` copies are instead installed when absent and updated when stale (`action` `installed`/`updated`/`up-to-date`).
- This crate owns the `reqwest`/`semver`/`sha2`/`hex`/`flate2`/`tar`/`zip`/`self-replace` dependencies; core stays free of them.
- It also uses `which`, already owned by `repo-explorer-search` — the only dependency this crate shares with another non-core crate rather than owning outright.

## Install/uninstall (`src/install.rs`)

- Sibling module to `setup.rs`/`update.rs`, dispatched from `main()` after the `--update` check and before `resolve_config_path`. Synchronous `run_install`/`run_uninstall` returning `ExitCode` (no async runtime — all work is `std::fs` writes and bounded blocking `claude` subprocess calls).
- `wants_install`/`wants_uninstall` reuse `crate::has_flag` over `args_without_config_value` output, exactly like `update.rs`.
- The MCP server name (`repo-explorer-mcp`, `--scope user`) and the agent's `tools: mcp__repo-explorer-mcp` line both derive from a single `SERVER_NAME` constant. The registered command is `std::env::current_exe()`.
- Reuses `update.rs`'s `pub(crate)` `run_with_timeout` + `SUBPROCESS_TIMEOUT` (10s) to bound the `claude` subprocess (stdin is explicitly `/dev/null`-equivalent so a prompting CLI can't hang it); on failure the step `detail` surfaces the CLI's own trimmed stdout+stderr combined, falling back to the exit status if both streams were empty. On Windows, a `claude` that resolves via PATHEXT to a `.cmd`/`.bat` shim (an `npm install -g` install) is routed through `cmd.exe /C` since `Command::new` can't exec a script file directly.
- No new crate dependencies; `repo-explorer-core` is untouched. This install/uninstall feature does not touch the Node.js installer (`setup/index.mjs`) — but that file is not otherwise "unchanged": it independently provisions the private `codebase-memory-mcp` copy (see `## Self-update` above), so don't assume it's untouched by memory-binary work when changing `update.rs`.
