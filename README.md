# repo-explorer-mcp

A Rust MCP server that exposes the `explore_repository` tool over an rmcp stdio
transport, shipped for Linux (`x86_64-unknown-linux-gnu`) and Windows
(`x86_64-pc-windows-msvc`). It drives an internal LLM exploration loop over a
managed `codebase-memory-mcp` backend and ripgrep-based text search — both
provisioned by `repo-explorer-mcp --update` into a shared per-user bin dir
(`$XDG_BIN_HOME` if set to an absolute path, else `~/.local/bin`, on Linux;
`%LOCALAPPDATA%\repo-explorer-mcp` on Windows), never a global PATH install.

## Response shape

`explore_repository` returns its answer as MCP `structuredContent` (the schema
is advertised as the tool's `outputSchema`, derived from the response type):

```json
{
  "findings": [
    {
      "location": {
        "path": "crates/repo-explorer-core/src/domain.rs",
        "line_start": 27,
        "line_end": 31
      },
      "snippet": "pub struct FileLocation {\n    pub path: PathBuf,\n    pub line_start: u32,\n    pub line_end: u32,\n}",
      "note": "exact symbol match: `FileLocation`",
      "symbol": "repo_explorer_core.domain.FileLocation"
    },
    {
      "location": { "path": "crates/repo-explorer-agent/src/render.rs" },
      "note": "graph row with no resolvable line"
    }
  ],
  "summary": "FileLocation is defined in core's domain module and re-used by every backend.",
  "retrieval_confidence": 92,
  "stage_exit": "early-exit"
}
```

- `retrieval_confidence` (0-100) scores the deterministic pre-stage's
  **candidate set**, not the answer: a low value means the answer came from the
  explorative fallback loop, not that it is wrong.
- `stage_exit` is one of `"early-exit"`, `"verify"`, `"fallback"`, `"cache"` —
  the same literal the server logs as `path` for that call.
- `line_start`/`line_end`, `snippet`, `note` and `symbol` are **omitted** (never
  `null`) when unknown. A missing `line_start` means the backend had no
  resolvable line; a missing `symbol` means no single ranked candidate
  overlapped that location.

### Parsing this from a client

`content[0].text` carries the same object as compact JSON, so a client without
`structuredContent` support loses nothing:

```python
payload = result.structuredContent or json.loads(result.content[0].text)
for f in payload["findings"]:
    loc = f["location"]
    print(loc["path"], loc.get("line_start"), f.get("symbol"), sep=":")
print(payload["summary"], payload["stage_exit"], payload["retrieval_confidence"])
```

Nothing in this response shape is configurable — there is no config key for it.

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
# Failover cooldown after a provider hits a rate limit, quota, or
# unavailable-model error, in seconds. Other errors (e.g. an invalid model
# name, bad credentials) fail immediately instead of triggering cooldown.
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
# Exception: a gemini/google entry instead rotates which model it starts at
# on every request (still failing over through the rest of the list on a
# usage-limit error), since Gemini's per-model rate limits are unusually
# tight and spreading requests across the configured models up front reduces
# the error rate.
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
snippet_max_chars_detailed = 1500  # response-only cap for response_format = "detailed"
                              # (a floor: never narrower than snippet_max_chars)
skip_verify_on_exact_symbol = true  # one exact symbol hit? answer without the LLM

# Deterministic repository brief prefetched once on entry to the explorative fallback
# loop (Stage 5 only) and injected as its own system message, so the loop does not spend
# its first turns re-deriving the module layout. A nested table must come AFTER every
# bare `[agent]` key above — in TOML, everything below a `[agent.repo_brief]` header
# belongs to that sub-table, not to `[agent]`.
[agent.repo_brief]
enabled = true    # false = Stage 5 behaves exactly as before, no prefetch at all
max_tokens = 3000 # budget for the rendered brief; over budget, the smallest modules are dropped
key = "head"      # cache key: "head" survives a dirty working tree, "full" re-builds on every save

# Result caching, keyed by git state (HEAD + dirty digest). The in-memory maps
# are the L1; `persistent` adds a per-user on-disk L2 for the query->result
# cache, so a repeated query in a new process does not pay the full loop again.
[cache]
enabled = true                       # false disables both layers
max_entries = 256                    # entry cap per in-memory map (L1 only)
persistent = true                    # false = in-memory only (the privacy opt-out)
persistent_max_bytes = 268435456     # 256 MiB budget for the on-disk layer; 0 disables it
key_mode = "strict"                  # strict | paths (see below)
dir = ""                             # "" = the per-user cache dir, resolved by the binary

[logging]
level = "info"            # trace | debug | info | warn | error
```

The env var named by each `api_key_env` must actually be set in the environment,
or config loading fails with `MissingEnvVar`.

### On-disk result cache

With `persistent = true` a completed exploration is written to disk so a later
process (a new Claude Code session, a second editor) can serve it without
re-running the loop. `dir = ""` resolves to the per-user cache directory —
`$XDG_CACHE_HOME/repo-explorer` (falling back to `~/.cache/repo-explorer`) on
Linux, `%LOCALAPPDATA%\repo-explorer` on Windows; a non-empty value overrides
it verbatim. If no directory resolves at all, the on-disk layer is simply off
and caching stays in memory.

A stored entry contains **verbatim source snippets and absolute repository
paths**. On Unix the cache directory is created `0700` and its files `0600`; on
Windows it relies on the per-user ACL `%LOCALAPPDATA%` already carries. Set
`persistent = false` (or `persistent_max_bytes = 0`) to keep everything in
memory, and use `repo-explorer-mcp cache clear` to delete what is already
there (that command reads `[cache]` alone, so it works without the LLM
section validating). The store is swept back under `persistent_max_bytes` once
per process, on the tail of the first exploration that writes to it, evicting
least-recently-used entries first.

`key_mode` decides when a cached answer may still be served after the
repository fingerprint moved:

- `strict` (default) — serve only for a repo state proven unchanged: an
  identical fingerprint, or a fingerprint change with a provably empty diff.
  This is the pre-existing behavior.
- `paths` — additionally serve after an *unrelated* edit, by re-stat'ing only
  the files the cached answer references. Much higher hit rate while you are
  editing, with a known ceiling: a newly added file that would have been the
  better match is missed. Entries with no referenced files fall back to
  `strict`.

There is deliberately no `head` mode. Keying on HEAD alone would serve snippets
an uncommitted edit has already invalidated — the one failure this cache must
never produce.

Cache inspection, both printing a JSON report to stdout and exiting non-zero on
failure (neither starts the server, connects to anything, or prompts):

```bash
repo-explorer-mcp cache stats   # dir, schema_version, entries, bytes, max_bytes, oldest/newest
repo-explorer-mcp cache clear   # delete every persisted result; reports what was removed
```

Setting `REPO_EXPLORER_METRICS=<path>` appends one JSON line of per-query
metrics (exit path, tokens, cache read/write tokens, `cache_layer` —
`l1`/`l2` on a cache hit — plus `turns_saved_by_cache`/`tokens_saved_by_cache`,
confidence, timings) to
that file; an empty value counts as unset. Without it, the headline fields are
still logged — on `exploration complete` for a normal run and on `exploration
served from query cache` for a cache hit, so grepping only the first message
silently drops every 0-token row. A sink write failure is logged and never
fails a query.

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
