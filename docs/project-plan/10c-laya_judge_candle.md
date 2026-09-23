# Stage 10c — Laya Judge: In-Process candle Backend

Spec 3 of 3 in the Laya judge track (10a → 10b → 10c). Input for
`/taskflow:spec-driven-delivery` (`SPEC_PATH` = this file). Target location in
the repo: `docs/project-plan/10c-laya_judge_candle.md`.

**Depends on 10a and 10b being merged.** It does not depend on Stages 9/9b
(see 10a "Prerequisites"). It consumes:

- `CandidateJudge`, `Judgement` and `JudgeError`, including the `warm_up`
  default method;
- `JudgeSettings` and `ConfiguredJudge`;
- `render_judge_states`;
- `judge_state_version` in `repo_explorer_judge.json`;
- `judge-serve/`.

A fine-tuned checkpoint (10a gate G4) is needed only for the ignored
parity and latency tests and the manual gate.

## Keypoints

- New `judge.backend = "http" | "candle"` (default `http`). The `candle`
  backend runs the fine-tuned Laya checkpoint **inside the Rust binary**, so
  serving needs no Python, no sidecar and no port. One batched forward pass
  runs per query instead of N HTTP requests.
- It is a faithful port of Laya 0.3.7 inference for the one question
  repo-explorer asks:
  - `build_sequence` (choice branch);
  - the ModernBERT encoder, via `candle-transformers` 0.11.0 `modernbert`;
  - the 2-layer pre-norm decision head;
  - the scorer MLP;
  - the clamped temperature.
- **The gate is a parity test against PyTorch.** Golden fixtures are
  exported next to the checkpoint by a Python script and checked by an
  ignored Rust test: token ids exact, |Δlogit| ≤ 1e-3, and identical
  select/reject decisions.
- CPU is always compiled in (the default feature of `repo-explorer-mcp`),
  so release binaries include it. CUDA is an opt-in build feature for source
  builds.
- The checkpoint is local (`judge.checkpoint_dir`). It is loaded lazily on
  first use or background warm-up, and a load failure degrades exactly like
  a 10b judge outage.

## Goal

Remove the Python serving dependency and the per-candidate HTTP overhead
from `judge.mode = "laya"`/`"shadow"`, with numerically equivalent
decisions to the HTTP backend.

## Non-goals

- Training, fine-tuning or calibrating in Rust. 10a stays the only training
  path.
- The act head (`action.act_probability`): it carries no usable signal (Laya
  #185) and is unused.
- fp16/bf16 inference, quantization, Metal/macOS, or mmBERT/multilingual
  checkpoints.
- Auto-downloading checkpoints (`--update`, Hugging Face).
- Removing or changing the HTTP backend or `judge-serve/`.
- A thread-count setting (candle's defaults are used).

## Chosen approach

**Encoder.** Reuse `candle_transformers::models::modernbert::ModernBert`,
pinned to `=0.11.0`. The Laya weights' `encoder.` prefix is renamed to the
`model.` prefix that `ModernBert::load` expects, by loading the safetensors
into a `HashMap` and building `VarBuilder::from_tensors`. The decision head
and scorer are about 150 lines of hand-written candle code.

**Tokenizer.** The `tokenizers` crate on the **same 0.22.x line that
candle-core 0.11.0 already depends on**: `version = "0.22"`,
`default-features = false`, `features = ["onig"]`. candle-core 0.11.0
unconditionally depends on `tokenizers ^0.22.0` with `onig`, so the
Oniguruma C build (`onig_sys` via `cc`) is compiled anyway. Matching its
version and features avoids a second tokenizers copy in the dependency
graph. `esaxx_fast` (C++) stays off.

**Parity is proven, not assumed.** Fixtures are generated from the real
PyTorch model next to the checkpoint; they are not committed, which avoids
committing third-party code excerpts and 1.7 GB weights.

**Loading.** Lazy loading via a `OnceCell`, triggered by 10b's background
`warm_up`, keeps MCP startup instant.

**Inference.** It runs in `spawn_blocking`, one inference at a time behind
a `Mutex`, so concurrent queries never oversubscribe cores.

Alternatives rejected:

| Alternative | Why it lost |
|---|---|
| ONNX export + `ort` | Needs an export step with custom-head tracing, a native ONNX Runtime library per platform, and a second model artifact to keep in sync |
| Own ModernBERT implementation | Duplicates maintained upstream code (RoPE local/global, sliding-window mask) |
| A different `tokenizers` version (for example 0.23 with `fancy-regex`) | candle-core 0.11.0 already pulls in tokenizers 0.22 with `onig`; a second version adds compile time and binary size without removing the C dependency |
| Commit golden fixtures | They would contain third-party source excerpts from the corpus, and they are tied to one checkpoint anyway |
| Eager load at startup | Adds seconds to MCP startup; the background warm-up gives the same result without blocking |
| fp16 on CUDA | Parity budget; f32 everywhere for v1 |

## Detailed design

### D1 — Config (`repo-explorer-core`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JudgeBackend { #[default] Http, Candle }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JudgeDevice { #[default] Cpu, Cuda }
```

`JudgeSettings` gains three fields, with `Default` updated to match:

- `#[serde(default)] pub backend: JudgeBackend`
- `#[serde(default)] pub checkpoint_dir: String` (empty = unset)
- `#[serde(default)] pub device: JudgeDevice`

`KNOWN_SECTIONS` `judge` gains `"backend"`, `"checkpoint_dir"` and
`"device"`. Every existing `JudgeSettings { … }` struct literal from 10b
(tests in core, the agent and the judge crate) must either add the three
fields or end with `..JudgeSettings::default()`.

Validation when `mode != Off`:

- `backend = Http`: exactly 10b's checks.
- `backend = Candle`:
  - `checkpoint_dir.trim()` must be non-empty; otherwise
    `InvalidJudgeSetting { key: "checkpoint_dir", reason: "must be set when judge.backend = \"candle\"" }`;
  - the `select_threshold` and `timeout_ms` checks apply;
  - `base_url`, `api_key_env`, `model` and `max_concurrency` are not
    validated, because they are unused.
- Filesystem existence is **not** checked in validation: core does no I/O.
  It is checked at load time.

The cache key is unchanged: the backend does not alter the semantics, and
parity is gated.

### D2 — Crate features

`crates/repo-explorer-judge/Cargo.toml`:

```toml
[features]
default = []
candle = ["dep:candle-core", "dep:candle-nn", "dep:candle-transformers", "dep:tokenizers"]
cuda = ["candle", "candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]

[dependencies]
candle-core = { version = "=0.11.0", optional = true }
candle-nn = { version = "=0.11.0", optional = true }
candle-transformers = { version = "=0.11.0", optional = true }
tokenizers = { version = "0.22", default-features = false, features = ["onig"], optional = true }

[dev-dependencies]
sha2 = "0.11"   # parity test: checkpoint SHA-256 (already in Cargo.lock)
hex = "0.4"
```

`cargo tree -p repo-explorer-judge --features candle -i tokenizers` must show
exactly one `tokenizers` version.

No tokio feature change is needed: `tokio::task::spawn_blocking` is part of
the `rt` feature, which 10b already enables.

`crates/repo-explorer-mcp/Cargo.toml`:

```toml
[features]
default = ["candle-judge"]
candle-judge = ["repo-explorer-judge/candle"]
cuda-judge = ["candle-judge", "repo-explorer-judge/cuda"]
```

`release.yml` is unchanged: `cargo build --release --package repo-explorer-mcp`
now includes the CPU candle backend through default features.

`ci.yml` `test` job adds, after `cargo test --workspace`:

- `cargo build -p repo-explorer-mcp --no-default-features`, to prove the
  HTTP-only build still works;
- `cargo test -p repo-explorer-judge`, with no features: `-p` builds only
  this package, so the `candle` feature is off. This runs the
  `#[cfg(not(feature = "candle"))]` tests;
- `cargo test -p repo-explorer-judge --features candle`;
- `cargo clippy -p repo-explorer-judge --features candle --all-targets -- -D warnings`.

### D3 — Module layout (`repo-explorer-judge`, behind `#[cfg(feature = "candle")]`)

```
src/candle/mod.rs         LayaCandleJudge (CandidateJudge impl), LoadedModel, loading orchestration
src/candle/checkpoint.rs  CheckpointFiles, AgentConfig parsing, encoder config fix-up, version check
src/candle/sequence.rs    build_sequence port
src/candle/head.rs        DecisionHead (type_emb, 2× pre-norm encoder layer, scorer)
src/candle/calib.rs       temperature selection/clamp, softmax over 2 logits
```

Public surface: `src/lib.rs` declares `#[cfg(feature = "candle")] pub mod candle;`.
The parity tests and the latency bench need these items, so they are
`pub`:

```rust
// repo_explorer_judge::candle
pub use sequence::{build_sequence, Encoded, SpecialIds};
pub struct LoadedModel { /* tokenizer, special ids, agent cfg, encoder, head, device, temperature */ }
impl LoadedModel {
    /// Blocking. Full D4 load incl. version check.
    pub fn load(dir: &Path, device: JudgeDevice) -> Result<LoadedModel, JudgeError>;
    /// D5 for the judge question.
    pub fn encode(&self, state: &str) -> Result<Encoded, JudgeError>;
    /// D6 steps 1–6: pre-temperature logits per sequence, same order.
    pub fn raw_logits(&self, batch: &[Encoded]) -> Result<Vec<[f32; 2]>, JudgeError>;
    /// D6 steps 7–8 over `raw_logits` of `encode(state)`.
    pub fn judge_states(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError>;
    /// The applied (selected + clamped) choice temperature.
    pub fn temperature(&self) -> f32;
    /// `select_threshold` from `repo_explorer_judge.json`.
    pub fn select_threshold(&self) -> u32;
}
pub struct LayaCandleJudge { /* D7 */ }
```

Everything else in `src/candle/` stays `pub(crate)`.

`ConfiguredJudge` gains `#[cfg(feature = "candle")] Candle(LayaCandleJudge)`.
`ConfiguredJudge::from_settings` behaves as follows:

- `mode == Off` → `Disabled`.
- `backend == Http` → `Laya(LayaHttpJudge::new(..)?)`, as in 10b.
- `backend == Candle`:
  - with the feature → `Candle(LayaCandleJudge::new(settings)?)`;
  - without it → `Err(Unavailable { message: "this build has no candle judge backend (built without the candle-judge feature)" })`.

  In the no-feature case the server fails at startup via 10b's
  `.context(...)`, which is intended: it is an explicit misconfiguration.

### D4 — Checkpoint loading (`checkpoint.rs`)

Required files in `checkpoint_dir`. A missing file gives
`Unavailable { message: "checkpoint <dir>: missing <file>" }`:

- `rl_agent_config.json`
- `model.safetensors`
- `tokenizer/tokenizer.json`
- `encoder/config.json`
- `repo_explorer_judge.json`

Checks and parsing:

- **State version.** `repo_explorer_judge.json`'s `judge_state_version`
  must equal `repo_explorer_core::judge::JUDGE_STATE_VERSION`. Otherwise:
  `"checkpoint was calibrated for judge state v{n}, this binary renders v{JUDGE_STATE_VERSION}"`.
- **`rl_agent_config.json`**, parsed from a `serde_json::Value`:

  | Field | Rule | Default |
  |---|---|---|
  | `max_len` | usize | 512 |
  | `head_max_len` | usize | 192 |
  | `head_layers` | usize | 2 |
  | `temperature` | array of 3 numbers; a non-number entry counts as non-finite | `[1, 1, 1]` |
  | `temperature_by_options` | map string → number | empty |

  `max_len > head_max_len + 16` is required.
- **`encoder/config.json`.** Parse to a `serde_json::Value`, apply these
  fix-ups for keys that are absent, then deserialize into
  `candle_transformers::models::modernbert::Config`. transformers 5.x's
  `save_pretrained` drops `global_rope_theta`, `local_rope_theta` and
  `layer_norm_eps`, writing `norm_eps`, `rope_parameters` and `layer_types`
  instead. This was checked with transformers 5.17: `global_attn_every_n_layers`
  and `local_attention` are kept.

  | Missing key | Filled from | If the source is missing too |
  |---|---|---|
  | `layer_norm_eps` | `norm_eps` | error naming both keys |
  | `global_rope_theta` | `rope_parameters.full_attention.rope_theta` | error naming both keys |
  | `local_rope_theta` | `rope_parameters.sliding_attention.rope_theta` | error naming both keys |
  | `global_attn_every_n_layers` | the index of the first `"full_attention"` entry in `layer_types` after index 0 | error naming both keys |

  If `layer_types` is present, it must satisfy
  `layer_types[i] == "full_attention"` ⇔ `i % global_attn_every_n_layers == 0`
  for every `i`, since that is the only pattern candle's `ModernBert`
  implements. Otherwise it is an error naming the first mismatching index.
- **Tokenizer.** `tokenizers::Tokenizer::from_file`, then call
  `with_truncation(None)` and `with_padding(None)`. Python's per-call
  `tok(...)` never applies stored truncation or padding, while the Rust
  tokenizer would apply any stored in `tokenizer.json`. Resolve the ids of
  the special tokens `[CLS]`, `[SEP]`, `[MASK]` and `[PAD]` with
  `token_to_id`; any of them missing is an error.
- **Device.**
  - `Cpu` → `Device::Cpu`.
  - `Cuda` → with the `cuda` feature, `Device::new_cuda(0)?`; without it,
    `Unavailable { message: "judge.device = \"cuda\" requires a build with the cuda-judge feature" }`.
- **Weights.**
  1. `candle_core::safetensors::load(path, &device)`.
  2. Rename every key starting with `encoder.` to `model.` + rest; leave the
     other keys unchanged.
  3. Build `VarBuilder::from_tensors(map, DType::F32, &device)`.
  4. `ModernBert::load(vb.clone(), &enc_cfg)`.
  5. Load the head from the same `vb`, following D6.

  Any candle error becomes
  `Unavailable { message: format!("loading checkpoint failed: {e}") }`.

### D5 — Sequence port (`sequence.rs`)

This is an exact port of laya 0.3.7 `common.build_sequence` for a `choice`
question with identity option order and `truncate_left = false`:

```rust
pub struct Encoded { pub ids: Vec<u32>, pub markers: Vec<usize> }
pub struct SpecialIds { pub cls: u32, pub sep: u32, pub mask: u32, pub pad: u32 }

pub fn build_sequence(
    tok: &Tokenizer, sp: &SpecialIds, state: &str,
    instructions: &str, options: &[String], max_len: usize, head_max_len: usize,
) -> Result<Encoded, JudgeError>
```

In the steps below, `enc(s)` is
`tok.encode(s, false)?.get_ids().to_vec()`, which is the Python
`tok(s, add_special_tokens=False)["input_ids"]`.

1. **Options and instructions.** For the judge question,
   `options = ["A: yes, this location answers the query", "B: no, this location does not answer the query"]`.
   That is `format!("{key}: {criterion}")` from the core constants in A, B
   order, which is Laya's `render_options` for `choice`.
   `ins = instructions.replace("[MASK]", " ")`.
2. **Head ids.** `head_ids = enc(&format!("choice question: {ins}"))`.
3. **Option ids.** For each option `o`:
   `opt_ids = [mask] ++ enc(&(" ".to_string() + &o.replace("[MASK]", " ")))`,
   keeping only the first 48 entries after the mask.
4. **Option budget.** Compute `opt_budget = head_max_len as isize - Σ len(opt_ids)`.
   If `opt_budget < 16`:
   - `per = max(4, (head_max_len - 16) / max(1, n))`, using integer
     division, where `n` is the number of options;
   - truncate every `opt_ids` to `per`;
   - recompute `opt_budget`.
5. **Head truncation.** Keep the first `max(8, opt_budget)` entries of
   `head_ids`, with a negative budget counting as 0 before the `max`.
6. **Assemble the head.** `ids = [cls] ++ head_ids ++ [sep]`. For each
   option: push `ids.len()` to `markers`, then extend `ids` with its
   `opt_ids`. Push `sep`.
7. **State.** `room = max(0, max_len - ids.len() - 1)` and
   `st = enc(&state.replace("[MASK]", " "))`, keeping its first `room`
   entries. `ids = ids ++ st ++ [sep]`.
8. **Final truncation.** Truncate `ids` to `max_len`, and keep only
   markers `< max_len`. If `markers.len() != options.len()`, return
   `Protocol { message: "options exceed head_max_len" }`.

### D6 — Model (`head.rs`, `mod.rs`)

Weights are addressed in PyTorch `state_dict` names. `D` is the encoder's
`hidden_size`, `H = max(1, D / 64)` and `F = 4 × D`.

| Component | Tensors |
|---|---|
| `type_emb` | `type_emb.weight` [3, D]; row 0 = `choice` |
| head layer `i ∈ 0..head_layers` | `head.layers.{i}.self_attn.in_proj_weight` [3D, D], `.self_attn.in_proj_bias` [3D], `.self_attn.out_proj.weight` [D, D], `.self_attn.out_proj.bias` [D], `.linear1.weight` [F, D], `.linear1.bias` [F], `.linear2.weight` [D, F], `.linear2.bias` [D], `.norm1.weight/.bias` [D], `.norm2.weight/.bias` [D] |
| scorer | `scorer.0.weight/.bias` (LayerNorm D), `scorer.1.weight/.bias` (Linear D→D), `scorer.3.weight/.bias` (Linear D→1); index 2 is GELU and has no params |
| ignored | `act_head.*`, `temperature` (buffer) |

Forward, for a batch `B` of encoded sequences:

1. **Pad.** Right-pad `ids` to the batch max length `L` with `pad`. Build
   `attention_mask` [B, L] as u32 1/0, and `marker_pos` [B, 2] as u32.
2. **Encode.** `h = modernbert.forward(&ids, &attention_mask)?` gives
   [B, L, D] in f32.
3. **Type embedding.** `h = h.broadcast_add(type_emb.weight.get(0))`.
4. **Head layers.** For each head layer, in PyTorch
   `TransformerEncoderLayer(norm_first=True, activation=relu, layer_norm_eps=1e-5)`
   semantics, with dropout inactive:
   - `x = h + SA(LN1(h))`, where SA is standard multi-head attention with
     `H` heads:
     - `q, k, v` are the three D-slices of `x·in_proj_weightᵀ + in_proj_bias`;
     - scores are `q·kᵀ / sqrt(D/H)`;
     - key padding mask: add `f32::NEG_INFINITY` where
       `attention_mask == 0`;
     - softmax over keys;
     - `·v`, merge heads, then `out_proj`.
   - `h = x + linear2(relu(linear1(LN2(x))))`.
5. **Gather.** `m` = the rows of `h` at `marker_pos`, giving [B, 2, D].
6. **Score.** `logits = scorer(m)`, giving [B, 2]: LayerNorm(eps 1e-5) →
   Linear → `gelu_erf` → Linear, then squeeze.
7. **Temperature.** Select `t` as follows:
   - take `temperature_by_options["choice:2"]` if present, else
     `temperature[0]`;
   - a non-finite value becomes 1.0;
   - clamp to `[0.5, 5.0]`.

   Then `p = softmax(logits / t)` over the last dim, and
   `p_relevant = p[:, 0]`.
8. **Judgement.** `Judgement { p_relevant_permille: (p_relevant * 1000).round() }`,
   clamped to `0..=1000`.

Batching: process states in chunks of at most 4 per forward to bound
memory. The output order equals the input order.

At L ≈ 1024 with 16 heads, one encoder attention score tensor is
`4 × 16 × 1024 × 1024 × 4 B` = 256 MiB. With the mask-add and softmax
copies, that means up to ~1 GiB transient, on top of ~1.7 GB of f32
weights.

### D7 — `LayaCandleJudge` (`mod.rs`)

```rust
pub struct LayaCandleJudge {
    dir: PathBuf,
    device: JudgeDevice,
    timeout_ms: u64,
    model: tokio::sync::OnceCell<Result<Arc<std::sync::Mutex<LoadedModel>>, String>>,
}
impl LayaCandleJudge {
    /// No I/O here: validates the settings shape only; loading is lazy.
    pub fn new(settings: &JudgeSettings) -> Result<Self, JudgeError>;
}
impl CandidateJudge for LayaCandleJudge {
    async fn judge(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError>;
    /// Triggers (and awaits) the lazy load; Ok when loaded.
    async fn warm_up(&self) -> Result<(), JudgeError>;
}
```

- **`new`.** It uses no I/O. With the `cuda` feature absent and
  `device = Cuda`, it returns `Unavailable` immediately.
- **Loading.** The first `judge` or `warm_up` call runs
  `get_or_init(|| async { spawn_blocking(move || LoadedModel::load(&dir, device)).await })`.
  - The load is **not** covered by `timeout_ms`; only inference is.
    Nothing in the agent cancels a `judge` future, so the init future runs
    to completion and there is only ever one load.
  - The load result is cached for the process lifetime, including a
    failure: the error string is logged once at WARN, and every later call
    returns `Unavailable { message: <cached error> }`.
  - Log the load time at INFO.
- **`judge`.**
  - An empty `states` returns `Ok(vec![])` without loading.
  - Otherwise await the load (above), clone the `Arc`, then `spawn_blocking`
    a closure that locks the `Mutex` and calls
    `LoadedModel::judge_states`.
  - Only that inference step is wrapped in
    `tokio::time::timeout(Duration::from_millis(timeout_ms))`, which gives
    `Timeout` when elapsed. The blocking task is left to finish.
  - A panic or join error in the blocking task gives
    `Unavailable { message: "judge inference task failed" }`.
- **`warm_up` with `device = Cpu`** makes 10b's background warm-up preload
  the ~1.7 GB f32 weights at startup. It does not block MCP handshakes.

### D8 — Golden fixtures and parity (`judge-serve/export_golden.py`)

`export_golden.py --checkpoint DIR --data DATA_DIR [--n 50] [--out DIR/golden.jsonl]`:

1. **Load.** `laya.Agent(DIR, device="cpu")`, in fp32.
2. **Pick states.**
   - the first `n` rows of `DATA_DIR/test.jsonl`, in file order, using
     `json.loads(row["state"])`;
   - plus three synthetic states:
     - a long state: `"query: x\ncandidate: a.rs:1-2 (text match)\ncode:\n" + "  let v = 1;\n" * 400`;
     - a state containing `[MASK]` twice;
     - the state `"query: x"`.
3. **Build and run.** The question is 10a's judge question.
   - `q = agent._to_internal(question)`, then
     `seq, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])`.
   - Run `agent.model(...)` on the single item via `collate_items` under
     `torch.no_grad()`.
   - `logits` is `[:2]` of the raw output (before temperature), as float32.
     `p_relevant = softmax(logits / t)[0]` in float64, where `t` is chosen
     with the D6 step-7 rule from `agent.temperature_by_options` and
     `agent.temperature`. `Agent.predict` is **not** used, because it
     rounds probabilities to 4 decimals.
4. **Write** JSONL:
   - line 1:
     `{"header": {"checkpoint_sha256": <sha256 of model.safetensors>, "laya_version": laya.__version__, "judge_state_version": 1, "n": <rows>}}`;
   - then one line per state: `{"state", "input_ids", "markers", "logits": [a, b], "p_relevant"}`.

Ignored Rust tests in `crates/repo-explorer-judge/tests/candle_parity.rs`
(`#![cfg(feature = "candle")]`) need the env var
`REPO_EXPLORER_LAYA_CHECKPOINT`, and read `$REPO_EXPLORER_LAYA_CHECKPOINT/golden.jsonl`:

- **`parity_token_ids`** — the header SHA-256 equals the checkpoint's
  `model.safetensors` SHA-256 (otherwise fail with "regenerate
  golden.jsonl"). Every row's `input_ids` and `markers` equal the D5
  output exactly.
- **`parity_logits_and_decisions`**:
  - raw logits within 1e-3 absolute of the golden values, per element;
  - `p_relevant` within 1e-3;
  - identical threshold decisions:
    `(permille >= select_threshold * 10) == ((golden_p * 1000).round() >= select_threshold * 10)`
    for every row whose golden permille is **not** within 2 of the
    threshold boundary (`|golden_p * 1000 − select_threshold * 10| ≥ 2`).
    `select_threshold` comes from `repo_explorer_judge.json`. Rows within
    that band are counted and printed, not asserted, because the 1e-3
    logit tolerance can legitimately flip them.
- **`bench_judge_latency`** — loads via `LoadedModel::load`, and uses the
  first 12 golden states per call, 3 warm-up calls and 20 timed calls of
  `judge_states`. It prints `{"p50_ms", "p95_ms", "device"}` as JSON, with no
  assertion.

`judge-serve/bench_http.py --url http://127.0.0.1:8765 --golden DIR/golden.jsonl [--concurrency 4] [--runs 20]`
is the matching HTTP reference, used for the P2 gate:

- it takes the same first 12 golden states per call;
- it issues one `POST /v1/systemone` per state with the 10b request body
  and `model: "typed-decisions"`, using at most `--concurrency` requests in
  flight (mirroring `LayaHttpJudge`);
- it makes 3 warm-up calls and 20 timed calls, and prints
  `{"p50_ms", "p95_ms"}` as JSON;
- it uses only the Python standard library (`urllib.request`,
  `concurrent.futures`), so it adds no dependency.

### D9 — Docs

- **`docs/laya-judge.md`:** a "Backends" section.
  - The http/candle comparison table: dependencies, latency behaviour,
    memory (~1.7 GB f32 weights plus up to ~1 GiB transient per forward
    pass), and batching (4 states per forward pass).
  - The candle config example:
    `backend = "candle"`, `checkpoint_dir = "/home/<user>/laya-ckpt/v1"`,
    `device = "cpu"`.
  - How to build with CUDA:
    `cargo install --path crates/repo-explorer-mcp --features cuda-judge`.
  - The parity procedure (D8 commands).
- **`docs/configuration.md`:** the three new `[judge]` keys.
- **`crates/repo-explorer-judge/CLAUDE.md`:**
  - the candle module map;
  - the weight-renaming rule (`encoder.` → `model.`);
  - "any change to `sequence.rs`/`head.rs` must keep `parity_*` green
    against a real checkpoint";
  - the pinned versions, and the rule "bump candle/tokenizers only
    together with a parity run".
- **`CHANGELOG.md` `[Unreleased]` → `### Added`:** the in-process candle
  judge backend (`judge.backend = "candle"`).

## Error handling

| Situation | Behaviour |
|---|---|
| `backend = candle`, but built without `candle-judge` | Server start fails with the D3 message |
| `device = cuda`, but built without `cuda-judge` | `LayaCandleJudge::new` returns `Unavailable`, so server start fails with context |
| Missing checkpoint file, version mismatch, bad config, CUDA init failure, corrupt weights | The lazy load fails. The error is cached and logged once at WARN, and every judge call returns `Unavailable` → 10b degrade path (LLM verify or Stage 5). Queries never fail |
| Options exceed the head budget | `Protocol`. Unreachable with the fixed question, but kept as a defensive check |
| Inference exceeds `timeout_ms` | `Timeout` → 10b degrade path. The blocking task completes in the background and releases the mutex |
| Inference panic | `Unavailable("judge inference task failed")` → degrade. The mutex may be poisoned: recover with `into_inner()` (same convention as `agent.rs`) |

## Testing strategy

CI (`cargo test --workspace`, with the candle feature enabled through
`repo-explorer-mcp` default features; plus `cargo test -p repo-explorer-judge --features candle`):

- **`sequence.rs`** — tests with a synthetic `tokenizers` WordLevel model
  built in-test from a JSON string. It has a whitespace pre-tokenizer and a
  vocabulary containing `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]` and 30
  words. The expected ids are hand-derived from the D5 rules:
  - (a) a short state fits: the layout is
    `[CLS] head [SEP] [MASK] optA [MASK] optB [SEP] state [SEP]`, with
    markers at the two `[MASK]` positions;
  - (b) a long state is truncated to `room`, and the total equals
    `max_len`;
  - (c) with a tiny `head_max_len` (for example 20), the options are
    truncated to `per = max(4, (20 - 16) / 2) = 4`, and the head is
    truncated to `max(8, budget)`;
  - (d) `[MASK]` inside the state and the instructions becomes a space, so
    no extra mask id appears;
  - (e) `max_len` so small that a marker falls outside → `Protocol`.
- **`calib.rs`** — the bucket `choice:2` beats `temperature[0]`; 0.1 clamps
  to 0.5; 9 clamps to 5.0; NaN becomes 1.0; softmax of `[0, 0]` is
  `[0.5, 0.5]`; the permille rounding of 0.9995 is 1000.
- **`head.rs`** — with random but fixed weights, `D = 64`, `H = 1`,
  `head_layers = 2`:
  - the output shape is [B, 2];
  - **padding invariance**: the logits of a sequence alone equal its logits
    when batched with a longer sequence (padded), within 1e-5;
  - the marker gather picks the right rows (compared with a manual
    `index_select`).
- **`checkpoint.rs`**, using temp dirs with small JSON files and no weights:
  - each missing required file gives an error naming it;
  - `judge_state_version = 2` gives the mismatch message;
  - encoder config fix-ups:
    - a transformers-5-style config (`norm_eps`, `rope_parameters`,
      `layer_types`, without `layer_norm_eps`/`*_rope_theta`) deserializes
      with the expected values;
    - a hub-style config with the old keys passes through unchanged;
    - each missing key whose source is also missing is an error naming
      both keys;
    - a `layer_types` pattern inconsistent with
      `global_attn_every_n_layers` is an error naming the index;
  - `rl_agent_config` defaults apply when keys are absent;
  - the `max_len > head_max_len + 16` check.
- **Config** (core) — `backend = candle` with an empty `checkpoint_dir` →
  `InvalidJudgeSetting { key: "checkpoint_dir", .. }`; `backend = candle`
  with `base_url = "ftp://x"` is valid (not validated); `unknown_key_warnings`
  accepts the 3 new keys.
- **`ConfiguredJudge::from_settings`** — `backend = candle` gives `Candle`
  with the feature (a `#[cfg(feature = "candle")]` test), and the D3 error
  without it (a `#[cfg(not(feature = "candle"))]` test). The no-feature test
  runs in the CI step `cargo test -p repo-explorer-judge`, which is built
  without features; `cargo test --workspace` enables `candle` through
  feature unification.
- **Tokenizer loading** — a `tokenizer.json` fixture with a stored
  truncation (`max_length: 4`) still encodes a 10-token text to 10 ids after
  loading (`with_truncation(None)` applied).
- **`LayaCandleJudge`** — an empty states list gives `Ok(vec![])` without
  loading (a nonexistent `checkpoint_dir` does not error). With a
  nonexistent dir, `warm_up` returns `Unavailable` naming the missing file,
  and a second `judge` call returns the same cached error without
  retrying.

Ignored (manual, real checkpoint): `parity_token_ids`,
`parity_logits_and_decisions` and `bench_judge_latency` (D8).

## Acceptance criteria

Automated (CI):

1. `cargo fmt --check`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo clippy -p repo-explorer-judge --features candle --all-targets -- -D warnings`
4. `cargo build --workspace`
5. `cargo build -p repo-explorer-mcp --no-default-features`
6. `cargo test --workspace`, `cargo test -p repo-explorer-judge` and
   `cargo test -p repo-explorer-judge --features candle`, on ubuntu-latest
   and windows-latest.
7. `python -m py_compile judge-serve/export_golden.py judge-serve/bench_http.py`
   (added to the `test-eval` job next to 10b's `serve.py` step).
8. `cargo tree -p repo-explorer-judge --features candle -i tokenizers`
   lists exactly one `tokenizers` version.

Manual gates (maintainer; checkpoint `~/laya-ckpt/v1` from 10a G4):

- **P1 — parity.**
  ```bash
  uv run --project judge-serve python judge-serve/export_golden.py --checkpoint ~/laya-ckpt/v1 --data ~/judge-data/v1
  REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 cargo test --release -p repo-explorer-judge --features candle -- --ignored parity
  ```
  Both `parity_*` tests pass.
- **P2 — latency (CPU, same host, same 12 states).** First the HTTP
  reference with judge-serve on CPU, then the candle bench:
  ```bash
  REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 LAYA_DEVICE=cpu uv run --project judge-serve python judge-serve/serve.py &
  uv run --project judge-serve python judge-serve/bench_http.py --url http://127.0.0.1:8765 --golden ~/laya-ckpt/v1/golden.jsonl
  kill %1
  REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 cargo test --release -p repo-explorer-judge --features candle -- --ignored bench_judge_latency --nocapture
  ```
  The candle `p50_ms` must be ≤ the `bench_http.py` `p50_ms`.
- **P3 — end-to-end equivalence.**
  - Create the uncommitted file `eval/config/judge-candle.local.toml` as a
    copy of `eval/config/judge-laya.toml`, with `[judge] backend = "candle"`,
    `device = "cpu"`, and `checkpoint_dir` set to the absolute path of the
    local checkpoint. The binary does not expand `~`.
  - This spec adds `eval/config/*.local.toml` to `.gitignore` and documents
    this step in `docs/laya-judge.md`.
  - Run `uv run --with pyyaml eval/run.py --config eval/config/judge-candle.local.toml`
    and score it.
  - Pass@1 must be within ±1 query of the M3 (HTTP) run. Hallucinations and
    confident-wrong must be 0. `judge_selected` must be identical to M3 on
    ≥ 95 % of the rows where both runs have a `judge_outcome`.
- Record P1–P3 in `docs/eval/judge-candle-<date>.md`.

## Risks

- **candle ModernBERT fidelity** (RoPE theta per layer type, the
  sliding-window mask, final norm). Mitigation: P1 parity is the gate;
  a mismatch blocks the backend, not the release, since the default backend
  stays `http`.
- **Weight-name drift** between laya checkpoints and candle's expected
  names. Mitigation: the explicit renaming rule, plus a loader error listing
  the first missing tensor name.
- **Binary size and CI time** go up with candle and tokenizers in the
  default build. Mitigation: accepted, since a single binary is the point;
  `--no-default-features` keeps a slim build possible.
- **Memory:** ~1.7 GB of f32 weights plus up to ~1 GiB transient per
  forward pass when the candle backend is active. Mitigation: only loaded
  when `backend = candle` is configured; chunks of 4; documented.
- **C toolchain for `onig_sys`.** This is inherited from candle-core 0.11.0's
  own tokenizers dependency. Mitigation: `cc` builds it on the stock
  GitHub ubuntu and windows runners; CI on both is an acceptance criterion.
- **transformers-version config drift** (`encoder/config.json` key
  renames). Mitigation: the D4 fix-up table and its tests.
- **Blocking-pool saturation under concurrent queries.** Mitigation: one
  inference at a time (`Mutex`), and the timeout surfaces overload as a
  degrade, not a hang.
- **Pinned `=` versions can go stale.** Mitigation: bump only together with
  P1.

## Global Constraints

- All code, comments, docs and spec text are in English.
- Rust edition 2024, with the workspace version; `cargo fmt --check` and
  `cargo clippy --all-targets -- -D warnings` are clean, with and without
  `--features candle` on `repo-explorer-judge`.
- CI passes on ubuntu-latest and windows-latest, including
  `cargo build -p repo-explorer-mcp --no-default-features`.
- `repo-explorer-core` gains no dependency.
- Pinned versions: `candle-core`, `candle-nn` and `candle-transformers`
  `=0.11.0`; `tokenizers` `0.22` (the line candle-core 0.11.0 depends on)
  with `default-features = false` and `features = ["onig"]`. Exactly one
  `tokenizers` version is in the graph.
- The candle forward pass takes at most 4 states per chunk.
- Inference is f32 on every device.
- The sequence build is an exact port of laya 0.3.7 `build_sequence`
  (choice branch), with the judge question from `repo_explorer_core::judge`.
- The temperature rule is `temperature_by_options["choice:2"]`, else
  `temperature[0]`; non-finite → 1.0; clamp to `[0.5, 5.0]`.
- The checkpoint must contain `repo_explorer_judge.json` with
  `judge_state_version == JUDGE_STATE_VERSION`.
- `judge.backend` defaults to `http`; `judge.mode` defaults to `off`
  (unchanged).
- A judge load or inference failure never fails a query or blocks startup;
  it degrades per 10b. A misconfigured build/backend combination fails
  startup explicitly.
- The default features of `repo-explorer-mcp` include `candle-judge`, CPU
  only; CUDA exists only via `cuda-judge`.
- Golden fixtures are never committed.

## Decisions & assumptions

1. `candle-transformers` `modernbert` is used for the encoder, and the head
   and scorer are hand-written.
2. The `encoder.` → `model.` key rename via `VarBuilder::from_tensors`.
3. `tokenizers` 0.22 with `onig`, the same version and backend
   candle-core 0.11.0 already pulls in, with `with_truncation(None)` and
   `with_padding(None)` applied after loading.
4. Lazy load behind a `OnceCell`, triggered by 10b's background `warm_up`.
   The load is outside `timeout_ms`, and a load failure is cached for the
   process lifetime.
5. One inference at a time (`Mutex`) in `spawn_blocking`, with chunks of at
   most 4 states per forward pass.
6. The act head is not ported.
7. f32 only, on both CPU and CUDA.
8. Golden fixtures are generated next to the checkpoint by
   `judge-serve/export_golden.py` and are not committed. Parity tests are
   ignored and read `REPO_EXPLORER_LAYA_CHECKPOINT`.
9. Parity tolerances: exact ids and markers, |Δlogit| ≤ 1e-3,
   |Δp| ≤ 1e-3, and identical threshold decisions outside a ±2-permille
   band around the threshold. Golden `p` is computed from the raw logits,
   never from `Agent.predict`.
10. The P2 latency gate compares against `judge-serve/bench_http.py` on CPU,
    on the same host with the same 12 states. The P3 gate is equivalence
    with 10b's M3 run.
11. CPU candle is in default features, and therefore in the release
    artifacts; CUDA is opt-in.
12. Assumption: `candle_transformers::models::modernbert::ModernBert::load`
    expects tensors under `model.embeddings.*`, `model.layers.{i}.*` and
    `model.final_norm`, and `forward(&ids, &mask)` returns the final-norm
    hidden states [B, L, D]. This is verified against candle `main`'s
    `modernbert.rs`, which matches 0.11.0.
13. Assumption: laya checkpoints store the encoder under `encoder.*` with
    HF `ModernBertModel` names, the head under
    `head.layers.{i}.{self_attn,linear1,linear2,norm1,norm2}.*`, and the
    scorer as `scorer.{0,1,3}.*`, following `DecisionModel` in laya 0.3.7
    `common.py`.
14. The encoder config is normalized per the D4 fix-up table
    (transformers 4.x and 5.x layouts are both accepted).
15. `repo_explorer_judge::candle` exposes `LoadedModel`
    (`load`/`encode`/`raw_logits`/`judge_states`/`temperature`/`select_threshold`),
    `build_sequence`, `Encoded` and `SpecialIds` as its public API.
