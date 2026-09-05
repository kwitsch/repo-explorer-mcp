# repo-explorer-mcp

A Rust MCP server that exposes the `explore_repository` tool over an rmcp stdio
transport, shipped for Linux (`x86_64-unknown-linux-gnu`) and Windows
(`x86_64-pc-windows-msvc`). It drives an internal LLM exploration loop over a
managed `codebase-memory-mcp` backend and ripgrep-based text search — both
provisioned by `repo-explorer-mcp --update` into a shared per-user bin dir
(`$XDG_BIN_HOME` if set to an absolute path, else `~/.local/bin`, on Linux;
`%LOCALAPPDATA%\repo-explorer-mcp` on Windows), never a global PATH install.

## Install (recommended: npx setup script)

The setup script detects your OS/arch, checks/installs runtime dependencies,
downloads and checksum-verifies the matching release archive, installs the
binary, reports PATH status, and prints a ready-to-use `.mcp.json` snippet.

```bash
npx github:kwitsch/repo-explorer-mcp
```

Flags:

- `-y`, `--yes` — non-interactive; auto-approve dependency installs.
- `--force` — reinstall even if an up-to-date binary is already present.
- `--version <x.y.z>` — install a specific release (default: latest).
- `-h`, `--help` — usage.

The script installs into `$XDG_BIN_HOME` (when set to an absolute path) or
`~/.local/bin` (Linux), or `%LOCALAPPDATA%\repo-explorer-mcp` (Windows) — the
same directory `repo-explorer-mcp --update` provisions the managed helpers
into. It never edits your shell profile or system PATH — it only reports
whether the install directory is on PATH. `ripgrep` is installed via
`apt`/`dnf`/`pacman` on Linux or `winget` on Windows when available; when no
package manager is found, `rg` is instead provisioned on demand (latest GitHub
release) into the shared bin dir by `repo-explorer-mcp --update`. A system `rg`
already on PATH is always preferred and left untouched — the managed copy is a
fallback, created only when none is present.
`codebase-memory-mcp` is not taken from PATH: the installer invokes `repo-explorer-mcp --update` to install it as a managed per-user copy in the shared bin dir above (best-effort; if that step fails, run `repo-explorer-mcp --update` later). Search uses `rg`: a system `rg` on PATH is preferred, and a managed `rg` copy is provisioned into that shared bin dir only when none is present. The server fails fast if no `rg` can be resolved, pointing you at `--update`.

## Install (manual fallback)

Build from source:

```bash
cargo build --release --workspace
# binary at target/release/repo-explorer-mcp
```

Or download a release archive and its checksum, verify, and place the binary on
PATH (see `docs/smoke-test.md` for the verification commands). Release assets
follow the frozen naming contract:

```text
repo-explorer-mcp-<version>-x86_64-unknown-linux-gnu.tar.gz (+ .sha256)
repo-explorer-mcp-<version>-x86_64-pc-windows-msvc.zip       (+ .sha256)
```

from `https://github.com/kwitsch/repo-explorer-mcp/releases`.

Verify the version:

```bash
repo-explorer-mcp --version # prints: repo-explorer-mcp 0.1.0
```

## Configuration

The server reads a TOML config. Path precedence:

1. `--config <path>` / `--config=<path>`
2. the `REPO_EXPLORER_CONFIG` env var
3. the per-user config file —
   `$XDG_CONFIG_HOME/repo-explorer/repo-explorer.toml` (Linux, defaulting to
   `~/.config`) or `%APPDATA%\repo-explorer\repo-explorer.toml` (Windows) —
   when it exists
4. `./repo-explorer.toml` in the launch directory, when it exists

With no config anywhere, the per-user path is where `repo-explorer-mcp setup`
writes. The launch working directory is treated as the repository root to
explore.

### First run

```bash
repo-explorer-mcp setup # interactive wizard: detects provider API-key
# env vars and writes the per-user config
repo-explorer-mcp config test # validate the resolved config (JSON report on
# stdout, non-zero exit on failure)
```

The wizard runs automatically when no config is found **and** stdin is a TTY.
Launched non-interactively (as an MCP server is) with no config, the binary
prints setup guidance to stderr and exits non-zero rather than blocking.

Example `repo-explorer.toml`:

```toml
[llm]
# Failover cooldown after a provider errors, in seconds.
cooldown_seconds = 90
# Optional HTTPS proxy for all model upstream requests; omit for none.
# https_proxy = "https://proxy.example.com:8443"

# Providers are tried in file order (= failover order).
[[llm.providers]]
name = "primary"
kind = "anthropic"                  # anthropic | openai | gemini | google
api_key_env = "ANTHROPIC_API_KEY"   # names an env var; never the key itself
                                    # omit it to use the kind's default var
# Ordered model list: the first is tried first; on a usage-limit error the
# router advances to the next model, then to the next provider entry.
models = ["claude-sonnet-4", "claude-haiku-4"]

[[llm.providers]]
name = "secondary"
kind = "openai"
api_key_env = "OPENAI_API_KEY"
models = ["gpt-4o"]

# Exactly one of `command`+`args` (stdio) XOR `endpoint` (network).
# `setup` writes the absolute path of the managed codebase-memory-mcp copy
# (provisioned by `repo-explorer-mcp --update` into the shared bin dir) into
# `command`; the path below is illustrative and machine-specific.
[codebase_memory]
command = "/home/you/.local/bin/codebase-memory-mcp"
args = ["--stdio"]

[search]
timeout_seconds = 45
# rg_path may be set explicitly to an existing rg binary; omitted => runtime
# resolution — a system `rg` on PATH (via `which`) is preferred, and the
# managed `rg` copy provisioned by `--update` is used only as a fallback.

# Exploration pipeline knobs (all optional; shown with their defaults).
[agent]
max_fallback_iterations = 12  # turn limit for the explorative fallback loop
max_verify_iterations = 2     # turns for the LLM verification stage
token_budget = 60000          # total tokens per exploration; 0 = unlimited
top_k = 12                    # candidates handed from retrieval to the LLM
early_exit_confidence = 90    # >= this (0-100): answer without any LLM call
fallback_confidence = 30      # < this: skip verification, run the full loop
snippet_max_chars = 400       # snippet cap in prompts and tool results

# In-memory result caching, keyed by git state (HEAD + dirty digest).
[cache]
enabled = true
max_entries = 256

[logging]
level = "info"            # trace | debug | info | warn | error
```

The env var named by each `api_key_env` must actually be set in the environment,
or config loading fails with `MissingEnvVar`.

## `.mcp.json`

Installed-binary form (what the setup script prints):

```json
{
  "mcpServers": {
    "repo-explorer": {
      "command": "/home/you/.local/bin/repo-explorer-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

On Windows the `command` is
`%LOCALAPPDATA%\\repo-explorer-mcp\\repo-explorer-mcp.exe`. With no `--config`
in `args`, the config is resolved by the precedence above (per-user file first,
then a `repo-explorer.toml` in the launch directory); add
`"--config", "<path>"` to `args` to point at a specific file. The in-repo
development form instead launches via
`cargo run --release --quiet -p repo-explorer-mcp --`.

## Build & test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Troubleshooting

Config loading fails fast with a named error. `repo-explorer-mcp config test`
prints the error plus the offending TOML key path as JSON.

| Error                                 | Cause                                                | Fix                                        |
| ------------------------------------- | ---------------------------------------------------- | ------------------------------------------ |
| `EmptyProviderList`                   | `llm.providers` is empty                             | Add at least one `[[llm.providers]]`.      |
| `DuplicateProviderName`               | Two providers share a `name`                         | Make each provider `name` unique.          |
| `EmptyModelsList`                     | A provider's `models` list is empty                  | List at least one model ID.                |
| `UnknownProviderKind`                 | `kind` is not `anthropic`/`openai`/`gemini`/`google` | Use one of the supported kinds.            |
| `MissingEnvVar`                       | An `api_key_env` names an unset or blank variable    | `export` the named variable before launch. |
| `MissingCodebaseMemoryConnection`     | Neither `command` nor `endpoint` set                 | Set exactly one under `[codebase_memory]`. |
| `ConflictingCodebaseMemoryConnection` | Both `command` and `endpoint` set                    | Keep exactly one.                          |

See `docs/smoke-test.md` for verifying a downloaded release artifact end to end.
