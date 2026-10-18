# The local candidate judge

`explore_repository`'s Stage-4 verification can run against a local judge
instead of an external LLM. The judge scores each retrieval candidate with one
`POST /v1/systemone` request against any server that speaks that wire schema.
The default target is the zero-shot `h2oai/h2o-lightning-4b` model behind the
H2O shim on `127.0.0.1:8741`. `[judge]` defaults to `off`, so nothing changes
until it is configured. Calibration tooling still lives in
`docs/laya-judge-training.md` until a later phase replaces it.

## Mode matrix

`judge.mode` and `agent.fallback` combine into five effective
configurations:

| `judge.mode` | `agent.fallback` | Stage 4                                   | Stage 5            | `[llm]` needed |
| ------------ | ---------------- | ----------------------------------------- | ------------------ | -------------- |
| `off`        | `llm`            | LLM                                       | LLM loop           | yes            |
| `off`        | `off`            | LLM                                       | offline synthesis  | yes            |
| `shadow`     | `llm`/`off`      | LLM (+ judge measured)                    | LLM loop / offline | yes            |
| `on`         | `llm`            | judge (LLM on judge error or uncertainty) | LLM loop           | yes            |
| `on`         | `off`            | judge                                     | offline synthesis  | no             |

Only the `on` + `off` row is fully LLM-free: `[llm]` may be empty or absent,
and no LLM call is attempted on any path.

`shadow` never changes what a query returns — the LLM verification answer is
always the one served. The judge runs concurrently on the same candidate
states and its agreement with the LLM answer is recorded in
`QueryMetrics.shadow_agreement` (`exact`/`overlap`/`disjoint`/
`judge-escalated`/`llm-escalated`/`both-escalated`), purely for measurement.

A judge failure (unreachable, HTTP 503, malformed response, wrong bearer
token, or a timeout) never fails a query: in `on` mode it degrades to LLM
verification when `[llm]` has providers, otherwise it falls through to
Stage 5.

The legacy spelling `judge.mode = "laya"` is rejected at parse time; use
`judge.mode = "on"`.

## Serving the judge

Any server implementing `POST /v1/systemone` works; the recommended one is the
H2O shim serving `h2oai/h2o-lightning-4b`. The repo's own launcher in
`judge-serve/` is replaced in a later phase. Until then the legacy
`judge-serve/serve.py` still works with an explicit
`base_url = "http://127.0.0.1:8765"` and `model = "typed-decisions"`.

Wire contract, per candidate:

- Request: `{"model": <judge.model>, "state": <rendered candidate>, "questions": {"relevant": {"type": "choice", ...}}}`.
- Response: `answers.relevant.probabilities.A` (a finite number in `[0, 1]`)
  is the relevance probability. An optional `usage.latency_ms` is reported
  as `judge_latency_ms`; a missing or garbled value is ignored.

Warm-up runs once in the background at startup: `GET {base_url}/v1/models`
followed by one 1-state probe request (checks the model name and warms the
prefix cache). A server without `/v1/models` (e.g. the legacy launcher) only
produces a startup WARN; queries are unaffected.

## Configuration

```toml
[judge]
mode = "on"
base_url = "http://127.0.0.1:8741"
# api_key_env = "REX_JUDGE_KEY"
model = "h2oai/h2o-lightning-4b"
timeout_ms = 20000
max_concurrency = 4
select_threshold = 50
escalate_below = 0
```

See `docs/configuration.md` for the full `[judge]` key reference
(`mode`, `base_url`, `api_key_env`, `model`, `timeout_ms`, `max_concurrency`,
`select_threshold`, `escalate_below`) and `agent.fallback`. The removed keys
`backend`, `checkpoint_dir` and `device` are ignored with a WARN.

### Confidence cascade (`escalate_below`)

`judge.escalate_below` is an integer permille (`0` = off, otherwise
`501..=999`). Before selection, if any judged candidate's confidence
`max(p, 1000 - p)` is below it, the whole query degrades to LLM verification
(`judge_outcome = "escalated-uncertain"`); without providers it falls through
to Stage 5 with the judge's scores, and the offline summary says the judge was
not confident. Calibrate the value per model; the default `0` changes nothing.

## Rollout procedure

Promote in stages, each one a config change with no code change:

1. **`off`** (default). Nothing changes. Establish a baseline if you have
   not already (`docs/eval/baseline-2026-09-15.md` is the reference one).
2. **`shadow`.** Start the judge server, set `judge.mode = "shadow"`, and run
   real or eval traffic (`eval/config/judge-shadow.toml`). The LLM keeps
   answering every query; the judge's agreement is what you are collecting.
   - **Promotion rule: promote `shadow` → `on` only once
     `shadow_agreement_rate` (the fraction of `exact`/`overlap` outcomes) is
     at least 0.80 and `judge_outcome.error` is 0** across a representative
     run.
3. **`on`.** Set `judge.mode = "on"`, keeping `agent.fallback = "llm"`.
   Stage 4 now runs on the judge; a judge failure or an uncertain judgement
   still degrades to the LLM path, so `[llm]` stays required. Re-run the eval
   gate (`eval/config/judge-on.toml`) and compare pass@1, hallucinations and
   latency against the baseline before relying on it.
4. **`on` + `fallback = "off"` (optional, fully LLM-free).** Stage 5
   synthesizes up to five disk-verified, explicitly _unverified_ candidates
   instead of the LLM loop, and `[llm]` may be dropped. Validate with
   `eval/config/judge-offline.toml`, which must show `llm_calls == 0` on every
   row.

## Metrics

On every query where the judge ran, `QueryMetrics` (tracing line and
`REPO_EXPLORER_METRICS` JSONL) carries:

- `judge_mode`: `off` | `on` | `shadow` (set on every query).
- `judge_outcome`: `selected` | `escalated` | `escalated-uncertain` | `error`.
- `judge_ms`: client-side wall time of the judge call.
- `judge_latency_ms`: max server-reported `usage.latency_ms`; absent when the
  server does not report it.
- `judge_candidates`, `judge_selected`, `judge_max_p` (permille),
  `shadow_agreement`.

`eval/score.py`'s Judge section reports all of them, including
`judge latency (shim)` p50/p95; rows written with the old `laya` mode name are
counted as `on`.

## Troubleshooting

- **Judge down / unreachable during a query.** WARN and degrade (LLM verify if
  `[llm]` has providers, otherwise Stage 5); the query never fails.
  `judge_outcome = "error"` marks every degraded query.
- **HTTP 503.** Treated as "judge unavailable" (model still loading or
  overloaded): same degrade as an unreachable judge.
- **HTTP 422 (or any other non-2xx).** A protocol error, usually a state the
  server rejects (e.g. too long) or a wrong `model` name; check the server log.
- **Judge down at startup.** Warm-up is background and best-effort: a WARN
  ("local judge not ready"), never a blocked startup.
- **WARNs about `judge.backend` / `judge.checkpoint_dir` / `judge.device`.**
  These keys were removed with the in-process backend; delete them.
- **WARN about `judge.mode = "laya"`.** Replace it with `"on"`.
- **Threshold tuning.** `judge.select_threshold` is an integer percent
  (`1..=99`, default 50); a candidate is selected at
  `score >= select_threshold * 10` permille. Too high pushes queries to
  Stage 5 (`judge_outcome = "escalated"`); too low admits weak candidates.
- **`JUDGE_STATE_VERSION`.** The judge state text
  (`repo_explorer_agent::judge_input::render_judge_states`) and the fixed
  question in `repo_explorer_core::judge` are pinned by it; a calibration is
  valid only for the version it was computed on.

## Security

The judge's API key (`judge.api_key_env`, optional) is read once from the
named environment variable, sent only as a `Bearer` header (also on the
warm-up `GET /v1/models`), and never logged or included in error messages or
metrics. The judge HTTP client never uses a proxy — it always talks directly
to `judge.base_url`, which is expected to be a local or LAN address.
