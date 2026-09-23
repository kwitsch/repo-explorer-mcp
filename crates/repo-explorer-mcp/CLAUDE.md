# repo-explorer-mcp (server binary)

The MCP boundary: hosts the `explore_repository` tool, owns the `rmcp` server
dependency plus the serde/schemars DTOs, and uses `anyhow` here (core's typed
errors are consumed via `?`/`.context(...)`).

## Response DTO (`src/server.rs`)

- `ExplorationResultDto` maps from `repo_explorer_core::domain::ExplorationOutcome` (not `ExplorationResult`): `findings` + `summary` come from `outcome.result`, plus `retrieval_confidence: u32` and `stage_exit: String` (`StageExit::as_str()` — the same four literals `QueryMetrics::path` logs).
- `ExplorationFindingDto.symbol` is filled by joining `outcome.symbols` (a sparse `Vec<(FileLocation, String)>`) on the finding's location, in one `HashMap` pass per response. It is `skip_serializing_if = "Option::is_none"`, like `line_start`/`line_end`/`snippet`/`note` — an absent key means "unknown", never "none exists".
- The three fields above are the only client-visible additions; every pre-existing key, its order and its omit-when-`None` behavior are unchanged.
- The output schema is **derived by rmcp** from the tool's `Result<Json<ExplorationResultDto>, String>` return type, and `content[0].text` is rmcp's compact JSON of the same value. Do not hand-build a `CallToolResult`: that drops the derived schema (pinned by `tool_description_and_annotations_are_advertised`) and would break `eval/score.py`, which `json.loads` the text block.
- Doc comments on the DTO fields are the schema's descriptions — schemars picks them up, so they are client-facing text.

## Setup wizard (`src/setup.rs`)

- All interactive IO, the env-var scan, the free-tier model catalog, and TOML file writing live here at the binary boundary.
- Which env var belongs to which provider kind is core's (`config::default_api_key_env`), as is what counts as "set" (`config::env_var_is_set`).
- The wizard derives its candidate table from those rather than restating them.
- Serialization lives in core (`config::to_toml_string`); the binary adds no `toml` dependency.
- The wizard writes to the _resolved_ config path (the XDG default unless `--config`/`REPO_EXPLORER_CONFIG` overrides it).
- It self-verifies the written file via `repo_explorer_core::config::load`.
- The `[search]` section is left at core defaults because search is in-process (no external `rg` binary); `[agent]`, `[cache]`, and `[logging]` are likewise left at their (fully defaulted) core values. `main.rs::run` plumbs `config.agent`/`config.cache` into `AgentLoop::new` together with a `GitStateProbe` built from `config.search.timeout_seconds`.

## Config path resolution

- Precedence: `--config <path>` CLI arg -> `REPO_EXPLORER_CONFIG` env var -> XDG default **if it exists** -> `./repo-explorer.toml` **if it exists** -> the XDG default again, as the wizard's write target.
- The XDG default is `$XDG_CONFIG_HOME/repo-explorer/repo-explorer.toml` on Linux, `%APPDATA%\repo-explorer\repo-explorer.toml` on Windows.
- The two `exists` gates are load-bearing: the XDG default resolves on essentially every machine, so returning it unconditionally would make the `./repo-explorer.toml` fallback dead code and silently ignore an in-repo config.
- This crate owns the `dirs` dependency used for XDG resolution; core and the other crates stay free of it.
- The on-disk result cache follows the same rule: `xdg_default_cache_dir()` (`dirs::cache_dir()/repo-explorer`; `$XDG_CACHE_HOME` on Linux, `%LOCALAPPDATA%` on Windows) and `resolve_cache_dir` — `[cache] dir` verbatim when set, else the XDG default, `None` when neither resolves. `run()` writes the result back into `config.cache.dir` before `AgentLoop::new`, so the agent crate receives an already-resolved (or deliberately empty = no on-disk layer) string and never touches XDG. `cache stats`/`cache clear` reuse the same two functions.
- `codebase-memory-mcp` managed copy: lives in the shared bin dir (`~/.local/bin` on Linux via `dirs::executable_dir()`, `%LOCALAPPDATA%\repo-explorer-mcp` on Windows via `dirs::data_local_dir().join("repo-explorer-mcp")` — the same dir the npx installer uses for the main binary); provisioned/updated only by `--update` and launched by absolute path — never resolved via PATH/`which`.
- `setup` writes the absolute `codebase-memory-mcp` path into `[codebase_memory] command`.
- `run()` fails fast with a `--update` hint if the memory binary is missing, and never downloads.
- The Node installer's `installDir()` mirrors this same resolution (`$XDG_BIN_HOME` if absolute, else `$HOME/.local/bin`; `%LOCALAPPDATA%\repo-explorer-mcp` on Windows) so both installers place binaries in one shared dir.
- `codebase-memory-mcp` runs one per-user daemon that admits only clients launched from the _exact same executable path string_ (a same-build copy/symlink/hardlink elsewhere hangs 30s then fails; a different build is refused outright).
- `run()` scans `/proc` (Linux only, `running_memory_binary`) for an already-running `codebase-memory-mcp` owned by this user — e.g. the Claude Code plugin's — and spawns that path instead of the configured command.
- It waits up to 5s (`wait_for_running_memory_binary`) so a plugin CBM starting concurrently still wins the daemon; only with none running does it fall back to `[codebase_memory] command`.

## Subcommands

- `config test` (or `--config-test`) validates the resolved config only — parse + semantic checks, no server/memory/LLM/search connections.
- It prints a human-readable report to stdout by default, or JSON with `--json`, and exits non-zero on failure.
- `cache stats` / `cache clear` report or wipe the on-disk result cache (the agent crate's L2), printing a human-readable report to stdout by default (or JSON with `--json`) and exiting non-zero on an io error or when no cache directory resolves.
- They read `[cache]` through `config::cache_settings`, **not** `config::load`: validation is skipped, because it fails whenever the provider's `api_key_env` is not exported in the calling shell — and a fallback there would retarget `cache clear` from the configured `dir` to the per-user one while still reporting `ok`. A missing or unparseable file still falls back to `CacheSettings::default()` (reporting a broken config is `config test`'s job), announced on **stderr**, separate from the report on stdout (text by default, JSON with `--json`).
- Numbers come from `repo_explorer_agent::disk_cache::{stats, clear}`; the binary only adds `status` and `max_bytes` and never re-derives them.
- `setup` (mirroring `config test`) runs the interactive wizard.
- The wizard also auto-runs when the resolved config is missing, but **only if stdin is a TTY**.
- A non-interactive launch with no config prints guidance to stderr naming the `setup` subcommand, then exits non-zero — never blocking, never writing to stdout.
- "Missing" means `ConfigError::is_not_found`, not a bare `Path::exists` probe, so an unreadable or malformed config reports its real error instead of "no config".
- `--update` checks this binary and its runtime dependency binaries against their latest GitHub release, installs anything newer, and prints a human-readable report to stdout by default (or JSON with `--json`); non-zero exit if any component errors.
- Subcommand/flag detection runs over `args_without_config_value(argv)`, never raw `argv`: the value of a `--config <path>` pair must never be read as a subcommand (`--config setup` names a file, not the wizard). A single `--json` flag, detected once in `main()` via `has_flag` over `args_without_config_value` (position-independent), is threaded into every one-shot dispatch function and selects JSON instead of the default human-readable output in `print_report`.
- `--install` registers this binary with Claude Code and is fully headless: it shells out to `claude mcp add repo-explorer-mcp --scope user -- <current_exe>` (idempotent remove-then-add, never hand-editing `~/.claude.json`); only once that registration succeeds does it write a Haiku subagent to `<home>/.claude/agents/explore.md`.
- The agent file uses `name: Explore` (capital E, deliberately matching Claude Code's built-in `Explore` agent's `agentType` exactly, since overriding/shadowing a built-in is a literal, case-sensitive name match — a lowercase `explore` would just add a second, separate agent instead of replacing the built-in one).
- Skip-vs-error semantics (shared by install and uninstall): a failed MCP registration reports the agent-file step `skipped` instead, so a broken/unregistered server is never left shadowing the built-in agent.
- Install leaves a pre-existing file at that path alone (`skipped`) if its contents don't match what install would write.
- `--uninstall` tolerates an absent `claude` and an already-deleted agent file (both reported as `skipped`, not errors).
- `--uninstall` only deletes the agent file if its contents still match what `--install` wrote — a hand-edited or replaced file at that path is left in place (`skipped`), never silently destroyed.
- `--install` fails fast with a non-zero exit and `claude_code_detected: false` when `claude` is not on PATH.
- Both print a per-step (`mcp-server`, `agent-file`) human-readable report to stdout by default (or JSON with `--json`) and exit non-zero only when a step errors, mirroring `--update`.
- Both are dispatched before config resolution, so neither loads or creates `repo-explorer.toml`.
- Dispatch precedence: `--update` (checked first) takes precedence over both; when `--install`/`--uninstall` are both passed without `--update`, `--install` wins (checked first).
- Full dispatch order in `main()`: `--version` -> `--help` -> `--update` -> `--install` -> `--uninstall` -> `resolve_config_path` -> `config test` -> `cache stats|clear` -> `setup` -> load config -> `run()`. Everything from `config test` down needs the resolved config path; everything above it must not create or read a config.

## Self-update (`src/update.rs`)

- Tracked components: `repo-explorer-mcp` (`kwitsch/repo-explorer-mcp`) plus one managed install-if-absent / update-if-stale copy in the shared bin dir (`$XDG_BIN_HOME` or `~/.local/bin` on Linux, `%LOCALAPPDATA%\repo-explorer-mcp` on Windows): `codebase-memory-mcp` (`DeusData/codebase-memory-mcp`) via `provision_or_update_memory_binary`.
- Runs instead of the MCP server loop, dispatched before config resolution and before the `setup` dispatch/auto-run.
- `codebase-memory-mcp` is always managed (installed when absent, updated when stale — `installed`/`updated`/`up-to-date`). Search is in-process (`NativeSearchBackend`), so no external `rg` binary is provisioned.
- This crate owns the `reqwest`/`semver`/`sha2`/`hex`/`flate2`/`tar`/`zip`/`self-replace` dependencies; core stays free of them.

## Install/uninstall (`src/install.rs`)

- Sibling module to `setup.rs`/`update.rs`, dispatched from `main()` after the `--update` check and before `resolve_config_path`. Synchronous `run_install`/`run_uninstall` returning `ExitCode` (no async runtime — all work is `std::fs` writes and bounded blocking `claude` subprocess calls).
- `wants_install`/`wants_uninstall` reuse `crate::has_flag` over `args_without_config_value` output, exactly like `update.rs`.
- The MCP server name (`repo-explorer-mcp`, `--scope user`) and the agent's `tools: mcp__repo-explorer-mcp` line both derive from a single `SERVER_NAME` constant. The registered command is `std::env::current_exe()`.
- Reuses `update.rs`'s `pub(crate)` `run_with_timeout` + `SUBPROCESS_TIMEOUT` (10s) to bound the `claude` subprocess (stdin is explicitly `/dev/null`-equivalent so a prompting CLI can't hang it); on failure the step `detail` surfaces the CLI's own trimmed stdout+stderr combined, falling back to the exit status if both streams were empty.
- On Windows, a `claude` that resolves via PATHEXT to a `.cmd`/`.bat` shim (an `npm install -g` install) is routed through `cmd.exe /C`, since `Command::new` can't exec a script file directly.
- No new crate dependencies; `repo-explorer-core` is untouched. This install/uninstall feature does not touch the Node.js installer (`setup/index.mjs`) — but that file is not otherwise "unchanged": it independently provisions the private `codebase-memory-mcp` copy (see `## Self-update` above), so don't assume it's untouched by memory-binary work when changing `update.rs`.
