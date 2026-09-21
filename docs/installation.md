# Installation

## Recommended: npx setup script

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
whether the install directory is on PATH.
`codebase-memory-mcp` is not taken from PATH: the installer invokes `repo-explorer-mcp --update` to install it as a managed per-user copy in the shared bin dir above (best-effort; if that step fails, run `repo-explorer-mcp --update` later). Search is in-process, so no external binary is required.

## Manual fallback

Build from source:

```bash
cargo build --release --workspace
# binary at target/release/repo-explorer-mcp
```

Or download a release archive and its checksum, verify, and place the binary on
PATH (see [`smoke-test.md`](smoke-test.md) for the verification commands).
Release assets follow the frozen naming contract:

```text
repo-explorer-mcp-<version>-x86_64-unknown-linux-gnu.tar.gz (+ .sha256)
repo-explorer-mcp-<version>-x86_64-pc-windows-msvc.zip       (+ .sha256)
```

from `https://github.com/kwitsch/repo-explorer-mcp/releases`.

Verify the version:

```bash
repo-explorer-mcp --version # prints: repo-explorer-mcp 0.1.0
```
