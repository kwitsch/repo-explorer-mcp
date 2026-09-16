# Changelog

Notable changes per release. Dates are release dates; an unreleased section
collects what is on `main` but not yet tagged.

## [Unreleased] — 0.9.0

### Added

- **Persistent cross-session result cache.** Results now survive a process
  restart: a second layer stores one JSON entry per query under
  `<cache dir>/<repo-id>/results/v1/`, behind the existing in-memory cache.
  New `[cache]` keys: `persistent`, `persistent_max_bytes` (256 MiB default),
  `key_mode`, `dir`. `key_mode` defaults to `strict`, which validates a hit
  exactly as before; `paths` is an opt-in that serves a hit as long as every
  file the answer references is unchanged. Every I/O error degrades to
  in-memory-only — the cache never fails a query.
- **`cache stats` and `cache clear` subcommands**, reporting entry count,
  bytes, age range and the resolved cache directory as JSON.
- **Deterministic repository brief for the exploration loop.** When a query
  reaches the explorative fallback stage, one `get_architecture` call is
  rendered into a token-budgeted brief (graph vocabulary, entry points,
  modules by size) and injected as a second system message, so the model no
  longer spends its first turns orienting itself. Configured under
  `[agent.repo_brief]` (`enabled`, `max_tokens`, `key`); `max_tokens = 0`
  opts out before the call is made.
- **Richer `explore_repository` result.** Responses gain
  `retrieval_confidence` (0–100, scoring the deterministic pre-stage's
  candidate set, not the answer), `stage_exit` (`early-exit` | `verify` |
  `fallback` | `cache`), and an optional per-finding `symbol`. All existing
  keys keep their name, order and omit-when-absent behaviour.
- **Cited candidates.** The exploration loop numbers the candidates it shows
  the model, and `finish` accepts a `candidate_id`, so a citation resolves
  against a location the model was actually shown. Findings discovered
  outside that list (via grep or a file read) are still reported as before.
- **Mandatory `repo_path` on `explore_repository` requests**, so one server
  process serves several repositories without cross-contamination.
- **Three example prompts** advertised via the MCP prompts capability.
- **Efficiency metrics** per query — LLM turns, tokens, cost, cache layer,
  turns saved by the cache, brief size, in-loop orientation calls — on the
  log line and, when `REPO_EXPLORER_METRICS` is set, as a JSONL record. The
  eval harness aggregates them and exports a per-query CSV.

### Changed

- **Stufe-1 token reduction (QW-0…QW-3):** a stable cached prompt prefix,
  batched tool calls, compressed tool-result rendering and tighter response
  caps.
- The memory index refresh is skipped when the repository fingerprint is
  unchanged and still inside its trust window, saving an upstream round trip
  on repeat calls.
- The codebase-memory project name is derived from a canonical-path hash, so
  two checkouts with the same directory name stay separate.
- Every user-facing surface states that queries are English-only.

### Fixed

- Result-cache keys fold in the repository root, so one process serving
  several repositories can no longer cross-serve.
- Trusted-symbol deduplication no longer depends on the order retrieval legs
  complete in.
- `repo_path`'s directory check runs off the async reactor.

## Earlier releases

See the [GitHub releases](https://github.com/kwitsch/repo-explorer-mcp/releases)
for 0.8.0 and earlier.
