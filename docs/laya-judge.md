# The local candidate judge (Laya)

`explore_repository`'s Stage-4 verification can run against a local judge
instead of an external LLM. This is the runtime half (Stage 10b) of a
three-part track:

- **10a — training.** `repo-explorer-datagen` and `train/laya/` produce and
  fine-tune the checkpoint the judge serves. See
  `docs/laya-judge-training.md`.
- **10b — this document.** Wires a `CandidateJudge` into `AgentLoop`'s
  Stage 4 (and, with `agent.fallback = "off"`, Stage 5 too), gated behind
  `[judge]`. Default is `off`, so today's behaviour is unchanged.
- **10c — in-process inference (future).** An in-process `candle` backend
  as an alternative to the HTTP client below; out of scope here.

## Mode matrix

`judge.mode` and `agent.fallback` combine into five effective
configurations:

| `judge.mode` | `agent.fallback` | Stage 4                    | Stage 5            | `[llm]` needed |
| ------------ | ---------------- | -------------------------- | ------------------ | -------------- |
| `off`        | `llm`            | LLM                        | LLM loop           | yes            |
| `off`        | `off`            | LLM                        | offline synthesis  | yes            |
| `shadow`     | `llm`/`off`      | LLM (+ judge measured)     | LLM loop / offline | yes            |
| `laya`       | `llm`            | judge (LLM on judge error) | LLM loop           | yes            |
| `laya`       | `off`            | judge                      | offline synthesis  | no             |

Only the `laya` + `off` row is fully LLM-free: `[llm]` may be empty or
absent, and no LLM call is attempted on any path.

`shadow` never changes what a query returns — the LLM verification answer is
always the one served. The judge runs concurrently on the same candidate
states and its agreement with the LLM answer is recorded in
`QueryMetrics.shadow_agreement` (`exact`/`overlap`/`disjoint`/
`judge-escalated`/`llm-escalated`/`both-escalated`), purely for measurement.
This makes `shadow` the safe way to collect real-traffic agreement data
before trusting the judge to answer on its own.

A judge failure (unreachable, malformed response, wrong bearer token, or a
timeout) never fails a query: in `laya` mode it degrades to LLM verification
when `[llm]` has providers, otherwise it falls through to Stage 5.
`QueryMetrics.judge_outcome` records the outcome (`selected`/`escalated`/
`error`) on every query where the judge ran.

## Serving the judge: `judge-serve/`

The judge speaks the upstream `laya-serve` wire protocol
(`POST /v1/systemone`). Upstream's own launcher has no way to point at a
locally fine-tuned checkpoint, so `judge-serve/` is a small Python launcher
around `laya.serve.create_app(router)` that maps the `typed-decisions` model
name repo-explorer-mcp sends to a local checkpoint directory. See
`judge-serve/README.md` for the full command line, environment variables and
a systemd user-unit example. In short:

```bash
REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 LAYA_DEVICE=cuda \
  uv run --project judge-serve python judge-serve/serve.py
curl -sf http://127.0.0.1:8765/health
```

`REPO_EXPLORER_LAYA_CHECKPOINT` unset serves the zero-shot upstream
`typed-decisions` checkpoint instead — useful to smoke-test the wiring, but
not calibrated for production judging. The launcher refuses to start against
a checkpoint whose `repo_explorer_judge.json` has a `judge_state_version`
that does not match `repo_explorer_core::judge::JUDGE_STATE_VERSION` (see
"Troubleshooting" below), and prints the calibrated
`judge.select_threshold` recommendation on startup.

**Serving is serial.** Upstream's `/v1/systemone` handler calls the
blocking `router.predict` inline from an `async def`, so one `uvicorn`
worker processes requests **one at a time** — `judge.max_concurrency` only
overlaps HTTP round trips, not model inference. On a CPU-only host, a dozen
serial inferences at roughly 0.3-1 s each can noticeably add to query
latency; `judge.timeout_ms` defaults to 20,000 for this reason. GPU serving
(`LAYA_DEVICE=cuda`) is recommended for anything beyond smoke-testing.

## Backends

The judge can run via HTTP (the default) or as an in-process `candle` backend:

| Aspect          | HTTP (`judge-serve`)       | Candle (in-process)      |
| --------------- | -------------------------- | ------------------------ |
| Dependencies    | Python sidecar + open port | none (Rust binary only)  |
| Per-candidate   | one HTTP request           | one forward pass         |
| Batching        | concurrent HTTP clients    | up to 4 per forward pass |
| Memory (f32)    | ~1.7 GB weights            | ~1.7 GB weights          |
| Transient (GPU) | ~1 GiB per request         | ~1 GiB per batch         |

### HTTP backend (default)

```toml
[judge]
mode = "laya"
backend = "http"
base_url = "http://127.0.0.1:8765"
model = "typed-decisions"
timeout_ms = 20000
max_concurrency = 4
select_threshold = 50
```

Start the judge with `judge-serve/serve.py` (see above).

### Candle backend (in-process)

```toml
[judge]
mode = "laya"
backend = "candle"
checkpoint_dir = "/home/<user>/laya-ckpt/v1"
device = "cpu"
timeout_ms = 20000
select_threshold = 50
```

**Note:** The binary does not expand `~`, so use an absolute path for `checkpoint_dir`.

**Building with CUDA:** By default, the candle backend runs on CPU. To enable GPU inference, install with the `cuda-judge` feature:

```bash
cargo install --path crates/repo-explorer-mcp --features cuda-judge
```

Then set `device = "cuda"` in the config. The checkpoint and its encoder model are the same for both backends; only the runtime differs.

### Parity validation

Before promotion, verify that the in-process candle backend produces the same decisions as the HTTP backend against your checkpoint. The `judge-serve/export_golden.py` utility records golden-standard token sequences and logits from the PyTorch model:

```bash
# With laya==0.3.7 installed and the checkpoint ready:
REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 \
  python judge-serve/export_golden.py \
  --checkpoint ~/laya-ckpt/v1 \
  --data ~/eval-data --n 50

# Run the ignored parity tests (requires the checkpoint and golden.jsonl nearby):
REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 \
  cargo test -p repo-explorer-judge --features candle -- --ignored
```

Expected: all three parity tests pass (`parity_token_ids`, `parity_logits_and_decisions`, `bench_judge_latency`).

For the evaluation harness, copy your HTTP judge config to a local variant:

```bash
cp eval/config/judge-laya.toml eval/config/judge-candle.local.toml
```

Edit `eval/config/judge-candle.local.toml` to set `backend = "candle"`, absolute `checkpoint_dir`, and `device`:

```toml
[judge]
backend = "candle"
checkpoint_dir = "/home/<user>/laya-ckpt/v1"
device = "cpu"
```

Then run the evaluation:

```bash
uv run --with pyyaml eval/run.py --config eval/config/judge-candle.local.toml
```

## Configuration

See `docs/configuration.md` for the full `[judge]` key reference
(`mode`, `base_url`, `api_key_env`, `model`, `timeout_ms`, `max_concurrency`,
`select_threshold`) and `agent.fallback`.

## Rollout procedure

Promote in stages, each one a config change with no code change:

1. **`off`** (default). Nothing changes. Establish a baseline if you have
   not already (`docs/eval/baseline-2026-09-15.md` is the reference one).
2. **`shadow`.** Start `judge-serve`, set `judge.mode = "shadow"`, and run
   real or eval traffic. The LLM keeps answering every query; the judge's
   agreement is what you are collecting. Look at `shadow_agreement` in the
   metrics stream (`REPO_EXPLORER_METRICS`) or via
   `eval/score.py`'s `Judge` section.
   - **Promotion rule: promote `shadow` → `laya` only once `shadow_agreement_rate`
     (the fraction of `exact`/`overlap` outcomes) is at least 0.80 and
     `judge_outcome.error` is 0** across a representative run — the same M2
     gate the delivery plan's eval harness checks with
     `eval/config/judge-shadow.toml`.
3. **`laya`.** Set `judge.mode = "laya"`, keeping `agent.fallback = "llm"`.
   Stage 4 now runs entirely on the judge; a judge failure still degrades to
   the LLM path, so `[llm]` stays required. Re-run the eval gate
   (`eval/config/judge-laya.toml`) and compare pass@1, hallucinations and
   latency against the baseline before relying on it.
4. **`laya` + `fallback = "off"` (optional, fully LLM-free).** Once `laya`
   mode is trusted, `agent.fallback = "off"` removes the LLM fallback loop
   too: Stage 5 synthesizes up to five disk-verified, explicitly
   _unverified_ candidates instead, and `[llm]` may be dropped from the
   config entirely. Validate with `eval/config/judge-offline.toml`, which
   must show `llm_calls == 0` on every row.

`judge.mode` defaults to `off` in code regardless of what any deployment
promotes to — promotion is always a per-deployment config choice, never a
code change.

## Troubleshooting

- **Judge down / unreachable during a query.** The server logs a WARN and
  degrades (LLM verify if `[llm]` has providers, otherwise Stage 5) — the
  query never fails outright. `QueryMetrics.judge_outcome = "error"` marks
  every degraded query, so a spike in that counter (via `eval/score.py` or
  the metrics JSONL) is the signal to check `judge-serve`'s health
  (`curl -sf <base_url>/health`).
- **Judge down at startup.** Warm-up runs once in the background and is
  best-effort: an unreachable judge at process start logs a WARN but never
  blocks the server from starting or answering queries (which will simply
  degrade per the point above until the judge comes up).
- **Threshold tuning.** `judge.select_threshold` is an integer percent
  (`1..=99`, default 50); a judgement selects a candidate at
  `score >= select_threshold * 10` permille. Too high starves Stage 4 and
  pushes queries to Stage 5 (`judge_outcome = "escalated"`); too low
  admits weak candidates. `train/laya/calibrate.py` (10a) recommends a
  value per checkpoint, printed by `judge-serve/serve.py` on startup and
  written to the checkpoint's `repo_explorer_judge.json`.
- **`judge_state_version` mismatch.** The judge state text
  (`repo_explorer_agent::judge_input::render_judge_states`) and the fixed
  question in `repo_explorer_core::judge` are pinned by
  `JUDGE_STATE_VERSION`. A checkpoint trained against a different version is
  refused at launcher startup (`judge-serve/serve.py` exits with a clear
  message) rather than served with silently mismatched input — retrain or
  recalibrate against the current version instead of overriding the check.

## Security

The judge's API key (`judge.api_key_env`, optional — only needed when
`judge-serve` is run with `LAYA_API_KEY` set) is read once from the named
environment variable and is never logged, and never included in error
messages or metrics. The judge HTTP client never uses a proxy — it always
talks directly to `judge.base_url`, which is expected to be a local or LAN
address.
