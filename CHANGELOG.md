# Changelog

Notable changes per release. Dates are release dates; an unreleased section
collects what is on `main` but not yet tagged.

## [Unreleased] — 0.9.0

### Changed

- **Breaking: `judge.mode = "laya"` is now `judge.mode = "on"`.** The old
  spelling is rejected at config parse time; the `judge_mode` metric reports
  `"on"`. Cache entries written
  under the old judge key are invalidated once.
- **Breaking: new judge defaults.** `judge.base_url` defaults to
  `http://127.0.0.1:8741` and `judge.model` to `h2oai/h2o-lightning-4b`;
  configs that relied on the old defaults must set both explicitly. The
  removed keys `checkpoint_dir`, `device` and `backend = "candle"` make the
  config fail to load (`backend = "http"` is still accepted).

### Removed

- **In-process candle judge backend** and the `candle-judge` / `cuda-judge`
  build features. The judge is now HTTP-only (`/v1/systemone`).

### Added

- **`judge.escalate_below` confidence cascade.** An integer permille cutoff
  (default `0` = off); a query whose judged candidates include an uncertain
  one degrades to LLM verification (`judge_outcome = "escalated-uncertain"`).
- **`judge_latency_ms` metric** (server-reported `usage.latency_ms`),
  reported by `eval/score.py` as p50/p95. HTTP 503 now counts as "judge
  unavailable", and warm-up probes `GET /v1/models` plus one request.
  `docs/laya-judge.md` is renamed to `docs/judge.md`; the setup wizard asks
  for the judge mode and base URL.

- **Training-data generator (`repo-explorer-datagen`, dev-only) and Laya
  fine-tune tooling (`train/laya/`)** for the upcoming local candidate judge.
- **Local candidate judge (`[judge]`, `repo-explorer-judge`).** Stage-4
  verification can now run against a local HTTP judge instead of an
  external LLM. `judge.mode` (`off` default / `on` / `shadow`) and
  `agent.fallback` (`llm` default / `off`) combine into five configurations,
  including a fully LLM-free one (`on` + `off`, `[llm]` optional); a judge
  failure degrades to LLM verification and never fails a query. `shadow`
  mode keeps returning the LLM answer while measuring judge agreement
  (`QueryMetrics.shadow_agreement`) as a rollout safety net. New
  `judge_mode`, `fallback_mode`, `judge_outcome`, `judge_ms`,
  `judge_candidates`, `judge_selected`, `judge_max_p` and `shadow_agreement`
  metrics fields. The `judge-serve/` Python launcher serves a fine-tuned
  checkpoint through upstream `laya-serve`. See `docs/judge.md`.
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

- The six one-shot CLI reports (`config test`, `cache stats`, `cache clear`,
  `--update`, `--install`, `--uninstall`) now print human-readable text by
  default; pass `--json` for the previous pretty-JSON output. The flag is
  position-independent (before or after the subcommand).
- **Stufe-1 token reduction (QW-0…QW-3):** a stable cached prompt prefix,
  batched tool calls, compressed tool-result rendering and tighter response
  caps.
- The memory index refresh is skipped when the repository fingerprint is
  unchanged and still inside its trust window, saving an upstream round trip
  on repeat calls.
- The codebase-memory project name is derived from a canonical-path hash, so
  two checkouts with the same directory name stay separate.
- Every user-facing surface states that queries are English-only.
- `repo-explorer-datagen`'s query templates are English-only: the German
  `words-de`/`define-de` templates are removed.

### Fixed

- Result-cache keys fold in the repository root, so one process serving
  several repositories can no longer cross-serve.
- Trusted-symbol deduplication no longer depends on the order retrieval legs
  complete in.
- `repo_path`'s directory check runs off the async reactor.

## Earlier releases

See the [GitHub releases](https://github.com/kwitsch/repo-explorer-mcp/releases)
for 0.8.0 and earlier.
