# repo-explorer-mcp

A Rust MCP server that exposes the `explore_repository` tool over an rmcp stdio
transport, shipped for Linux (`x86_64-unknown-linux-gnu`) and Windows
(`x86_64-pc-windows-msvc`). It drives an internal LLM exploration loop over a
`codebase-memory-mcp` backend and ripgrep/rtk text search.

## Install (recommended: npx setup script)

The setup script detects your OS/arch, checks/installs runtime dependencies,
downloads and checksum-verifies the matching release archive, installs the
binary, reports PATH status, and prints a ready-to-use `.mcp.json` snippet.

```bash
npx github:kwitsch/repo-explorer-mcp
# After the package is published to npm:
# npx repo-explorer-mcp-setup
```

Flags:

- `-y`, `--yes` — non-interactive; auto-approve dependency installs.
- `--force` — reinstall even if an up-to-date binary is already present.
- `--version <x.y.z>` — install a specific release (default: latest).
- `-h`, `--help` — usage.

The script installs into `~/.local/bin` (Linux) or
`%LOCALAPPDATA%\repo-explorer-mcp` (Windows). It never edits your shell profile
or system PATH — it only reports whether the install directory is on PATH.
`ripgrep` is installed automatically via `apt`/`dnf`/`pacman` on Linux or
`winget` on Windows; if none of those is available, the script warns and
points at ripgrep's own install docs instead of installing it itself. `rtk` and
`codebase-memory-mcp` are report-only: if missing, the script tells you and
points at their upstream install docs (`rtk` is optional; search falls back to
plain ripgrep).

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

The server reads a TOML config. Path precedence: `--config <path>` →
`REPO_EXPLORER_CONFIG` env var → `./repo-explorer.toml`. The launch working
directory is treated as the repository root to explore.

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
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"   # names an env var; never the key itself
model = "claude-sonnet-4"

[[llm.providers]]
name = "secondary"
kind = "openai"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o"

# Exactly one of `command`+`args` (stdio) XOR `endpoint` (network).
[codebase_memory]
command = "codebase-memory-mcp"
args = ["--stdio"]

[search]
prefer_rtk = false        # true prefers `rtk rg`; false uses plain ripgrep
timeout_seconds = 45
# rtk_path / ripgrep_path may be set explicitly; omitted => auto-detected on PATH.

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
`%LOCALAPPDATA%\\repo-explorer-mcp\\repo-explorer-mcp.exe`. Add
`"--config", "<path>"` to `args` if your `repo-explorer.toml` is not at the
launch cwd. The in-repo development form instead launches via
`cargo run --release --quiet -p repo-explorer-mcp --`.

## Build & test

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Troubleshooting

Config loading fails fast with a named error:

| Error                                 | Cause                                    | Fix                                        |
| ------------------------------------- | ---------------------------------------- | ------------------------------------------ |
| `EmptyProviderList`                   | `llm.providers` is empty                 | Add at least one `[[llm.providers]]`.      |
| `DuplicateProviderName`               | Two providers share a `name`             | Make each provider `name` unique.          |
| `MissingEnvVar`                       | An `api_key_env` names an unset variable | `export` the named variable before launch. |
| `MissingCodebaseMemoryConnection`     | Neither `command` nor `endpoint` set     | Set exactly one under `[codebase_memory]`. |
| `ConflictingCodebaseMemoryConnection` | Both `command` and `endpoint` set        | Keep exactly one.                          |

See `docs/smoke-test.md` for verifying a downloaded release artifact end to end.
