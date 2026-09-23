# CLAUDE.md — repo-explorer-judge

The local candidate judge over HTTP. The only crate that owns `reqwest` for the
judge and the only crate that knows the upstream `laya-serve` wire format.

## What it does

- `LayaHttpJudge` speaks `POST {base_url}/v1/systemone`. One request per
  candidate state, built from `repo_explorer_core::judge` constants (the D2 body
  with `model` from `JudgeSettings`).
- Per-candidate requests run concurrently, bounded by `Semaphore(max_concurrency)`;
  the whole `judge()` call is wrapped in `tokio::time::timeout(timeout_ms)` and
  maps elapse to `JudgeError::Timeout`.
- Responses are indexed by state order (via `try_join_all`), not arrival order.
- Response mapping: `answers.relevant.probabilities.A` (a finite `f64` in
  `[0, 1]`) becomes `Judgement.p_relevant_permille`. Anything else → `Protocol`.
  HTTP 401/403 → `Unavailable`; other non-2xx → `Protocol`.
- The HTTP client is built with `no_proxy()` — the judge is a local/LAN service.
- The API key (from `api_key_env`, read once at construction via the injected
  accessor) is sent as a `Bearer` header and is never logged or included in an
  error; transport error strings are `sanitize`d to redact any `Bearer` token.
- `warm_up()` is a best-effort `GET {base_url}/health` probe.
- `ConfiguredJudge` gives the runtime `off`/`laya` choice via static dispatch
  (`Disabled(NoJudge)` / `Laya(LayaHttpJudge)`), no `dyn`.
