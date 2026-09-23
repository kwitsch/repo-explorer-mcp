# CLAUDE.md — repo-explorer-judge

The local candidate judge with HTTP and in-process backends. The only crate that
owns `reqwest` for the judge and the upstream `laya-serve` wire format (HTTP backend).
The candle backend (feature `candle`, optional) runs fine-tuned Laya checkpoint inference
in-process via `candle-transformers`.

## What it does

### HTTP backend (`LayaHttpJudge`)

- Speaks `POST {base_url}/v1/systemone`. One request per candidate state, built from
  `repo_explorer_core::judge` constants (the D2 body with `model` from `JudgeSettings`).
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

### Candle backend (`LayaCandleJudge`, feature `candle`)

A faithful in-process port of Laya 0.3.7 inference. The checkpoint, tokenizer and
encoder config are identical across HTTP and candle backends; only the runtime differs.

**Module structure:**

- `candle/mod.rs` — `LoadedModel` (full checkpoint + encoder + head + tokenizer), `LayaCandleJudge` (async wrapper with lazy `OnceCell` loading), `InferenceWorker` (a dedicated OS thread owning the loaded model, serializing inference via an mpsc job queue so a hung call only ties up that one thread, never tokio's shared blocking pool), integration as a `CandidateJudge`.
- `candle/calib.rs` — temperature selection (honoring `temperature_by_options["choice:2"]` if present, else `temperature[0]`, clamped to `[0.5, 5.0]`), numerically stable softmax over two logits, permille rounding.
- `candle/sequence.rs` — exact port of Laya 0.3.7 `build_sequence` (choice branch) with fixed judge question from `repo_explorer_core::judge`. Builds token sequences with option markers.
- `candle/head.rs` — decision head: type embedding, two pre-norm transformer encoder layers (relu, eps 1e-5), and a 2-logit scorer MLP. Inference is f32 on all devices.
- `candle/checkpoint.rs` — file resolution, JSON parsing (agent config, judge metadata, encoder config), encoder-config fix-up (transforms `transformers` library versions), tokenizer loading, device resolution (CPU or CUDA).

**Weight loading and renaming:**

Checkpoint `model.safetensors` key prefix `encoder.` is renamed to `model.` during load (in `LoadedModel::load`, mod.rs) to match `candle-transformers` ModernBert's expected namespace.

**Constraints:**

- Pinned versions: `candle-core`, `candle-nn`, `candle-transformers` all `=0.11.0`; `tokenizers` `0.22` (the version `candle-core` 0.11.0 depends on), with `default-features = false` and `features = ["onig"]`. Exactly one `tokenizers` version in the graph.
- Candle forward pass: at most 4 states per batch (ponytail: `# ponytail: batch limit, per-chunk in raw_logits if throughput matters`).
- State version check: checkpoint must contain `repo_explorer_judge.json` with `judge_state_version == JUDGE_STATE_VERSION` (set at load time in `LoadedModel::load`).
- Load/inference failures never fail a query; they degrade per 10b (Stage 4 falls back to LLM).

**Testing and parity:**

- `tests/candle_parity.rs` (feature-gated, requires real checkpoint + golden.jsonl) validates token-id parity, logit magnitude, and decision equivalence against PyTorch.
- `judge-serve/export_golden.py` exports golden-standard fixtures from the PyTorch model for the parity tests.
- Any change to `sequence.rs` or `head.rs` must keep the parity tests green against a real checkpoint before merge.
- New `candle` or `tokenizers` versions: bump only when a parity run confirms numerics still match (e.g. rounding, library version differences in RoPE or softmax).

### ConfiguredJudge

Gives the runtime `off`/`laya` + `http`/`candle` choice via enum dispatch:

- `Disabled(NoJudge)` — no judge
- `Laya(LayaHttpJudge)` — HTTP backend
- `#[cfg(feature = "candle")] Candle(LayaCandleJudge)` — in-process candle backend

The enum's `from_settings` method branches on `settings.backend` to select the variant.
Both backends implement `CandidateJudge` with the same interface (`judge`, `warm_up`).
