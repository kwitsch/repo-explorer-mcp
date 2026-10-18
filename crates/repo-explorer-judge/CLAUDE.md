# CLAUDE.md — repo-explorer-judge

The local candidate judge client over the model-neutral HTTP `/v1/systemone`
protocol. The only crate that owns `reqwest` for the judge and the judge wire
format. Any server speaking that schema works; the default target is the H2O
shim serving `h2oai/h2o-lightning-4b` on `127.0.0.1:8741`.

## `SystemOneHttpJudge`

- Speaks `POST {base_url}/v1/systemone`. One request per candidate state, built from
  `repo_explorer_core::judge` constants (the body carries `model` from `JudgeSettings`).
- Per-candidate requests run concurrently, bounded by `Semaphore(max_concurrency)`;
  the whole `judge()` call is wrapped in `tokio::time::timeout(timeout_ms)` and
  maps elapse to `JudgeError::Timeout`.
- Responses are indexed by state order (via `try_join_all`), not arrival order.
- Response mapping: `answers.relevant.probabilities.A` (a finite `f64` in
  `[0, 1]`) becomes `Judgement.p_relevant_permille`; anything else → `Protocol`.
  The optional `usage.latency_ms` (finite, `>= 0`, rounded) becomes
  `Judgement.latency_ms`; missing or garbled is `None`, never an error.
- HTTP 401/403 and 503 → `Unavailable`; every other non-2xx (422, 500, 502, …) → `Protocol`.
- The HTTP client is built with `no_proxy()` — the judge is a local/LAN service.
- The API key (from `api_key_env`, read once at construction via the injected
  accessor) is sent as a `Bearer` header only when set, is never logged or included
  in an error; transport error strings are `sanitize`d to redact any `Bearer` token.
- `warm_up()`: `GET {base_url}/v1/models` (2 s timeout, bearer when configured),
  then one 1-state probe request through `judge()` (checks the model name, warms the
  prefix cache). Failures are `Unavailable`; the mcp crate logs them as a background
  WARN and never fails startup.

## `ConfiguredJudge`

Runtime choice via enum dispatch (no `dyn`):

- `Disabled(NoJudge)` — `judge.mode = "off"`
- `Http(SystemOneHttpJudge)` — `judge.mode = "on"` or `"shadow"`

`ConfiguredJudge::from_settings(&JudgeSettings) -> Result<Self, JudgeError>`; both
variants implement `CandidateJudge` (`judge`, `warm_up`).

## Tests

`tests/http.rs` uses a hand-rolled tokio TCP mock (`spawn_server`, `ok_json`,
`http_response`) and the closed port `127.0.0.1:9` for unreachable servers; no model
or GPU is needed. The mock does not expose request heads, so the bearer header itself
is not asserted.
