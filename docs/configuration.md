# Configuration

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

## First run

```bash
repo-explorer-mcp setup # interactive wizard: detects provider API-key
# env vars and writes the per-user config
repo-explorer-mcp config test # validate the resolved config (human-readable
# report by default, or `--json` for JSON; non-zero exit on failure)
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
# rg_path is deprecated/ignored: search is in-process (no external rg binary).
# The field is kept for configuration back-compatibility.

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

## On-disk result cache

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
- `paths` — additionally serve after an _unrelated_ edit, by re-stat'ing only
  the files the cached answer references. Much higher hit rate while you are
  editing, with a known ceiling: a newly added file that would have been the
  better match is missed. Entries with no referenced files fall back to
  `strict`.

There is deliberately no `head` mode. Keying on HEAD alone would serve snippets
an uncommitted edit has already invalidated — the one failure this cache must
never produce.

Cache inspection, both printing a human-readable report to stdout by default
(`--json` for JSON) and exiting non-zero on failure (neither starts the
server, connects to anything, or prompts):

```bash
repo-explorer-mcp cache stats # dir, schema_version, entries, bytes, max_bytes, oldest/newest
repo-explorer-mcp cache clear # delete every persisted result; reports what was removed
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
