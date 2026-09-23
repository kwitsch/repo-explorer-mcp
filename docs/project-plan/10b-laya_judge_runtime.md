# Stage 10b — Laya Judge: Runtime Integration (Stage 4), Shadow Mode, Offline Fallback

Spec 2 of 3 in the Laya judge track (10a → 10b → 10c). Input for
`/taskflow:spec-driven-delivery` (`SPEC_PATH` = this file). Target location in
the repo: `docs/project-plan/10b-laya_judge_runtime.md`.

**Depends on 10a being merged.** It does not depend on Stages 9/9b, which
are not implemented (see 10a "Prerequisites"). This spec consumes:

- `repo_explorer_core::judge::*` constants;
- `repo_explorer_agent::judge_input::render_judge_states`;
- the `early_exit_route` refactor.

The fine-tuned checkpoint from 10a (gate G4) is needed only for this spec's
manual end-to-end gate, not for implementation or CI.

## Keypoints

- **New `[judge]` config**, with `mode = "off" | "laya" | "shadow"` and
  default `off`, so existing behaviour is unchanged.
- **New `[agent] fallback = "llm" | "off"`**, with default `llm`. `laya` +
  `off` is a fully LLM-free mode, and in that mode `[llm]` may be empty or
  absent.
- **Core trait `CandidateJudge`** plus a new crate `repo-explorer-judge`
  with `LayaHttpJudge`. It speaks the upstream `laya-serve` wire protocol
  (`POST /v1/systemone`): one request per candidate, bounded concurrency,
  bearer auth optional.
- **Stage 4 in `laya` mode:**
  1. render per-candidate states with the 10a renderer, identical to
     training;
  2. judge them;
  3. select `p ≥ select_threshold`;
  4. disk-verify the selection and build a deterministic result
     (`stage_exit = "verify"`, 0 LLM calls).

  With no selection, the run escalates to Stage 5. A judge failure degrades
  to LLM verification when providers exist, otherwise to Stage 5.
- **Stage 5 with `fallback = "off"`**: deterministic offline synthesis. Up to
  5 disk-verified candidates, ranked by judge probability (or by retrieval
  rank), labelled as unverified.
- **`shadow` mode:** LLM verification answers exactly as today, while the
  judge runs concurrently on the same states. The agreement is logged in
  `QueryMetrics`. This is the rollout safety net.
- **`judge-serve/`**: a ~40-line Python launcher. It serves a local
  fine-tuned checkpoint through upstream `laya.serve.create_app`, which has
  no environment variable for a custom checkpoint path.
- **Eval:** three new configs; `run.py` and `score.py` learn the judge
  fields. The end-to-end gate compares against the 2026-09-15 baseline.

## Goal

Let `explore_repository` answer Stage-4 verification with a local Laya
judge instead of an external LLM, observable via `shadow` mode first. Also
allow a fully LLM-free configuration.

## Non-goals

- In-process inference and the `judge.backend` key: those are 10c.
- Training, data or calibration: those are 10a.
- Judging candidates of queries that skip Stage 4 today (early exit, or
  confidence below `fallback_confidence`): the Stage-4 gate is unchanged.
- Changing the LLM verify or fallback prompts, the tool catalogs, or the
  response DTO shape. `stage_exit` keeps its four values.
- Setup-wizard support for `[judge]`: the wizard keeps writing defaults.
- Provisioning Python or Laya via `--update`.
- Proving zero outbound traffic with a network sandbox.

## Chosen approach

**Wire protocol.** The upstream `laya-serve` protocol is used as it is. Its
`create_app(router)` is wrapped by a tiny launcher that maps the
`typed-decisions` checkpoint name to a local directory. Rust sends
`model: "typed-decisions"`, and that is honoured explicitly, so there is no
auto-routing by language.

**One HTTP request per candidate.** Upstream accepts one state per request.
Requests run concurrently up to `max_concurrency`, and any single failure
fails the whole judge call: no partial scoring.

**Selection threshold.** `judge.select_threshold` is an integer percent, in
line with the repo's integer-only confidence convention. Default 50; 10a's
`calibrate.py` recommends a value, written in `repo_explorer_judge.json`.

**Judge generic.** The judge enters `AgentLoop` as a sixth generic, `J`,
with default `NoJudge`, added by a `with_judge` builder. `AgentLoop::new`'s
signature and every existing test stay unchanged.

**Cache keys.** The query-cache key changes only in `laya` mode (judge mode
+ threshold) and with `fallback = "off"`. `off`/`shadow` keys stay
byte-identical, so existing L2 entries remain valid.

Alternatives rejected:

| Alternative | Why it lost |
|---|---|
| Model Laya as an `LlmProvider` | That trait is chat + tool calling; Laya returns probabilities only. It would need fake tool calls and would break budget and metrics semantics |
| Own batch endpoint in a Python sidecar | Couples to Laya internals (`build_sequence`, `collate_items`). 10c's in-process backend gets batching instead |
| Fork `laya-serve` or add an env var upstream | An external dependency on merge timing. The launcher uses only the public `create_app(router)` seam |
| `dyn CandidateJudge` | The crate convention is static dispatch (AFIT, no `async-trait`). An enum `ConfiguredJudge` gives runtime choice |
| Judge replaces verify in `shadow` | Shadow must be risk-free: the LLM answer is returned, and the judge is only measured |

## Detailed design

### D1 — Core: trait, errors, config

**`crates/repo-explorer-core/src/judge.rs`** (extends 10a's constants):

```rust
/// One judged candidate. Per-mille, keeping the domain integer-only / `Eq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Judgement {
    /// P(relevant) × 1000, rounded, clamped to 0..=1000.
    pub p_relevant_permille: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JudgeError {
    #[error("judge unavailable: {message}")]
    Unavailable { message: String },
    #[error("judge protocol error: {message}")]
    Protocol { message: String },
    #[error("judge timed out after {timeout_ms} ms")]
    Timeout { timeout_ms: u64 },
}

#[allow(async_fn_in_trait)]
pub trait CandidateJudge {
    /// Exactly one judgement per state, same order. Any per-state failure
    /// fails the whole call (no partial results).
    async fn judge(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError>;
    /// Best-effort readiness step run once in the background at startup
    /// (HTTP: health probe; 10c candle: model load). Default: nothing.
    async fn warm_up(&self) -> Result<(), JudgeError> {
        Ok(())
    }
}

/// The judge of a loop built without `with_judge`: every call is `Unavailable`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoJudge;
impl CandidateJudge for NoJudge { /* Err(Unavailable { message: "no judge configured" }) */ }

#[cfg(any(test, feature = "test-support"))]
pub mod mock {
    /// Canned responses in order (then `fallback`, default Unavailable);
    /// records every `states` slice it was called with.
    pub struct MockJudge { /* Mutex<VecDeque<Result<Vec<Judgement>, JudgeError>>>, Mutex<Vec<Vec<String>>> */ }
    /// `Default` is implemented (== `new()`), like the other core mocks, so
    /// `clippy::new_without_default` stays clean.
    impl Default for MockJudge { /* Self::new() */ }
    impl MockJudge {
        pub fn new() -> Self;
        pub fn with_responses(self, r: Vec<Result<Vec<Judgement>, JudgeError>>) -> Self;
        /// Score every call by a closure over the state text (for tests that
        /// don't know the call count up front).
        pub fn with_scorer(self, f: impl Fn(&str) -> u32 + Send + Sync + 'static) -> Self;
        pub fn calls(&self) -> Vec<Vec<String>>;
    }
}
```

**`crates/repo-explorer-core/src/llm.rs`**: add
`pub fn has_providers(&self) -> bool` on `ProviderRouter`. It returns `true`
iff at least one provider entry was configured.

**`crates/repo-explorer-core/src/config.rs`**:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum JudgeMode { #[default] Off, Laya, Shadow }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FallbackMode { #[default] Llm, Off }

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct JudgeSettings {
    #[serde(default)] pub mode: JudgeMode,
    #[serde(default = "default_judge_base_url")] pub base_url: String,          // "http://127.0.0.1:8765"
    #[serde(default, skip_serializing_if = "Option::is_none")] pub api_key_env: Option<String>,
    #[serde(default = "default_judge_model")] pub model: String,                // "typed-decisions"
    #[serde(default = "default_judge_timeout_ms")] pub timeout_ms: u64,         // 20_000
    #[serde(default = "default_judge_max_concurrency")] pub max_concurrency: u32, // 4
    #[serde(default = "default_judge_select_threshold")] pub select_threshold: u32, // 50 (percent)
}
impl Default for JudgeSettings { /* hand-written, same values as the serde defaults */ }
```

The other config changes:

- `Config` gains `#[serde(default)] pub judge: JudgeSettings`, placed after
  `cache`. `Config::llm` gains `#[serde(default)]`, with a hand-written
  `impl Default for LlmConfig`: empty `providers`,
  `cooldown_seconds: default_cooldown_seconds()`, `https_proxy: None`.
- `AgentSettings` gains `#[serde(default)] pub fallback: FallbackMode`,
  inserted **before** `repo_brief`, which must stay last. `Default` sets
  `FallbackMode::Llm`.
- `KNOWN_SECTIONS`:
  - `agent` gains `"fallback"`;
  - add `("judge", &["mode", "base_url", "api_key_env", "model", "timeout_ms", "max_concurrency", "select_threshold"])`.
- Every struct literal of `Config` must add `judge: JudgeSettings::default()`:
  `crates/repo-explorer-mcp/src/setup.rs` (`let cfg = Config {`) and
  `config.rs` test helpers. `AgentSettings` literals already use
  `..AgentSettings::default()`.

Validation (`validate_with_env`):

- `let llm_required = !(self.judge.mode == JudgeMode::Laya && self.agent.fallback == FallbackMode::Off);`
- `EmptyProviderList` is returned only when `llm_required &&
  providers.is_empty()`. The per-provider checks run unchanged for every
  configured provider.
- When `judge.mode != Off`, in this order:
  1. `base_url` must satisfy `is_valid_https_proxy_url`; otherwise
     `InvalidJudgeBaseUrl { url }`.
  2. `select_threshold` must be in `1..=99`; otherwise
     `InvalidJudgeSetting { key: "select_threshold", reason: "must be between 1 and 99" }`.
  3. `max_concurrency` must be ≥ 1: `{ key: "max_concurrency", reason: "must be at least 1" }`.
  4. `timeout_ms` must be ≥ 1: `{ key: "timeout_ms", reason: "must be at least 1" }`.
  5. `model` must not be blank after trimming: `{ key: "model", reason: "must not be empty" }`.
  6. If `api_key_env` is `Some(var)`, `env_var_is_set(&get_env, var)` must
     hold; otherwise `MissingJudgeEnvVar { var }`.

New `ValidationError` variants:

```rust
#[error("judge.base_url `{url}` is not a valid http(s):// URL")]
InvalidJudgeBaseUrl { url: String },
#[error("judge.{key} {reason}")]
InvalidJudgeSetting { key: &'static str, reason: String },
#[error("judge.api_key_env references environment variable `{var}`, which is not set")]
MissingJudgeEnvVar { var: String },
```

Their `toml_path` values are `"judge.base_url"`, `format!("judge.{key}")`
and `"judge.api_key_env"`.

### D2 — `repo-explorer-judge` crate

New workspace member `crates/repo-explorer-judge` (lib), with its own
`CLAUDE.md`. It owns `reqwest` usage for the judge and is the only crate
that knows the Laya wire format.

```toml
[dependencies]
repo-explorer-core = { version = "0.10.3", path = "../repo-explorer-core" }
reqwest = { version = "0.13", default-features = false, features = ["json", "rustls"] }
serde_json = "1"
tokio = { version = "1", features = ["rt", "sync", "time"] }
futures-util = { version = "0.3", default-features = false, features = ["std"] }
tracing = "0.1"
[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net", "io-util", "time"] }
```

The path-dependency version string follows the sibling crates' current
form. All dependencies are already in `Cargo.lock`.

```rust
pub struct LayaHttpJudge { /* client, endpoint, health_url, api_key: Option<String>, model, timeout, semaphore: Arc<Semaphore> */ }

impl LayaHttpJudge {
    /// `new_with_env(settings, |v| std::env::var(v).ok())`.
    pub fn new(settings: &JudgeSettings) -> Result<Self, JudgeError>;
    /// Reads the bearer token from `api_key_env` (if set) once, here, via the
    /// injected accessor (the `Config::validate_with_env` convention: tests
    /// never mutate the process environment). A set-but-blank or unset
    /// variable → `Unavailable { message: "judge.api_key_env `<var>` is not set" }`.
    pub fn new_with_env(
        settings: &JudgeSettings,
        get_env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, JudgeError>;
}
impl CandidateJudge for LayaHttpJudge { /* judge + warm_up (= GET /health) */ }

pub enum ConfiguredJudge { Disabled(NoJudge), Laya(LayaHttpJudge) }
impl ConfiguredJudge {
    /// `mode == Off` → `Disabled`; otherwise `Laya(LayaHttpJudge::new(..)?)`.
    pub fn from_settings(settings: &JudgeSettings) -> Result<Self, JudgeError>;
}
impl CandidateJudge for ConfiguredJudge { /* delegate */ }
```

**Client.** `reqwest::Client::builder().no_proxy().build()`: the judge is a
local or LAN service and is never proxied. A build error becomes
`Unavailable`.

**Endpoints.** `endpoint = base_url.trim_end_matches('/') + "/v1/systemone"`
and `health_url = base_url.trim_end_matches('/') + "/health"`.

**Request.** One `POST endpoint` per state, with header
`content-type: application/json` and `authorization: Bearer <key>` when a
key is set. The body is built with `serde_json::json!` from the core
constants:

```json
{"model": "<settings.model>",
 "state": "<state>",
 "questions": {"relevant": {"type": "choice",
                            "instructions": "Does this code location answer the repository search query?",
                            "criteria": {"A": "yes, this location answers the query",
                                         "B": "no, this location does not answer the query"}}}}
```

**Concurrency.** Each request acquires a permit from a
`Semaphore(max_concurrency)`, and all requests are driven with
`futures_util::future::try_join_all` over the states in order. The whole
call is wrapped in `tokio::time::timeout(Duration::from_millis(timeout_ms))`;
elapsing gives `Timeout { timeout_ms }`.

**Response mapping** (per request):

| Response | Result |
|---|---|
| Connect or transport error | `Unavailable { message }` |
| 401 or 403 | `Unavailable { message: "judge rejected the bearer token (HTTP <code>)" }` |
| Any other non-2xx | `Protocol { message: "HTTP <code>: <first 200 chars of body>" }` |
| Body not JSON, or `answers.relevant.probabilities.A` missing or not a finite number in [0, 1] | `Protocol { message }` |
| Otherwise | `Judgement { p_relevant_permille: (p * 1000.0).round() as u32 }`, clamped to 1000 |

**Length check.** Result length must equal `states.len()`; this is
structural, and a mismatch is `Protocol`. An empty `states` returns
`Ok(vec![])` without any request.

**`warm_up`.** `GET health_url` with a 2 s timeout; 200 gives `Ok`, anything
else `Unavailable`.

**Logging.** The API key never appears in logs or errors.

### D3 — Agent integration (`repo-explorer-agent`)

**Generic and builder.** The struct becomes

```rust
pub struct AgentLoop<M, S, P, R, C = SystemClock, J = NoJudge>
where M: MemoryBackend, S: SearchBackend, P: LlmProvider, R: RepoStateProbe, C: Clock, J: CandidateJudge
```

with new fields `judge: J` and `judge_settings: JudgeSettings`.

- `new(..)` keeps its exact signature. It lives in
  `impl<M, S, P, R, C> AgentLoop<M, S, P, R, C, NoJudge>` and sets
  `judge: NoJudge` and `judge_settings: JudgeSettings::default()`.
- All other methods move to `impl<M, S, P, R, C, J> AgentLoop<M, S, P, R, C, J>`.
- New builder:
  `pub fn with_judge<J2: CandidateJudge>(self, judge: J2, settings: JudgeSettings) -> AgentLoop<M, S, P, R, C, J2>`
  moves every field across.
- New `pub async fn warm_judge(&self) -> Result<(), JudgeError>`, delegating
  to `self.judge.warm_up()`.

**New module `judge_verify.rs`:**

```rust
pub(crate) enum JudgeVerifyOutcome {
    Finished { result: ExplorationResult, scores: Vec<Option<u32>>, selected: Vec<usize>, judged: u32, elapsed_ms: u64 },
    Escalate { scores: Vec<Option<u32>>, judged: u32, elapsed_ms: u64 },
    Failed { error: JudgeError, elapsed_ms: u64 },
}

pub(crate) async fn judge_verify<M: MemoryBackend, J: CandidateJudge>(
    memory: &M, judge: &J, repo_root: &Path, query: &ExplorationQuery,
    note: Option<&str>, candidates: &[Candidate], settings: &JudgeSettings,
) -> JudgeVerifyOutcome
```

`scores` is indexed like `candidates` and holds permille, or `None` when
not judged. `selected` holds candidate indices, in the result's order.

The algorithm:

1. `let states = render_judge_states(memory, repo_root, &query.text, candidates).await;`
   Collect `(index, state)` for every `Some`. If there are none: `Escalate`
   with all scores `None` and `judged = 0`.
2. Time the call. `judge.judge(&state_texts)`:
   - `Err(e)` → `Failed`;
   - `Ok(v)` with a length mismatch → `Failed(Protocol)`;
   - otherwise fill `scores`.
3. The selection is the indices with `score >= select_threshold * 10`,
   sorted by score descending, then index ascending.
4. Disk-verify the selection in that order with the helper from D3.1. Each
   surviving finding's `note` becomes
   `format!("{}; judge p={:.2}", finding.note.unwrap_or_default(), score as f64 / 1000.0)`,
   where the base note is `finding_from_candidate`'s provenance note.
5. If nothing survived → `Escalate`. Otherwise `Finished` with
   `result.summary =`
   `format!("Selected by the local judge (no LLM involved): {n} of {judged} candidate(s) for \"{}\".", query.text)`,
   where `n = selected.len()`, the survivors of step 4, counted before
   `finalize`. When `note` is `Some`, append `" " + note`.

**D3.1 — Shared disk verification (refactor).** Extract the
per-candidate verify/clamp loop of `AgentLoop::result_from_candidates` into

```rust
pub(crate) async fn verified_candidate_findings(
    repo_root: &Path,
    candidates: &[&Candidate],
) -> Vec<(usize, ExplorationFinding)>
```

It returns the input position and the finding for each survivor, and keeps
the per-path line-count cache. `result_from_candidates` calls it, with no
behaviour change.

**D3.2 — Stage-4 dispatch in `run`.** The existing guard
`outcome.confidence >= fallback_confidence && !candidates.is_empty()`
stays. Inside it:

- **`Off`:** the existing LLM `verify(..)` path, byte-identical.
- **`Laya`:** `judge_verify(..)`:
  - `Finished` → `metrics.judge_outcome = Some("selected")`, then
    `finalize_and_complete(StageExit::Verify, …)`.
  - `Escalate` → `metrics.judge_outcome = Some("escalated")`, keep `scores`,
    then go to Stage 5.
  - `Failed` → `metrics.judge_outcome = Some("error")`, then
    `tracing::warn!(error = %e, "local judge failed; degrading")`. If
    `self.router.has_providers()`, run the existing LLM `verify(..)` path
    exactly as in `Off`. Otherwise go to Stage 5 with no scores.
- **`Shadow`:** `let (llm, judged) = futures_util::future::join(verify(..), judge_verify(..)).await;`
  - Record the judge metrics from `judged`: `judge_outcome` is
    `"selected"`, `"escalated"` or `"error"`.
  - Record `shadow_agreement` (D3.4).
  - Then continue **exactly** as `Off` would with `llm`. The judge result is
    never returned, and the judge's scores are **not** passed on: if the run
    reaches Stage 5 with `fallback = "off"`, `offline_fallback` gets
    `scores = None`.

**D3.3 — Stage 5 dispatch.**

- `FallbackMode::Llm`: the existing `fallback_loop`, unchanged.
- `FallbackMode::Off`: `self.offline_fallback(repo_root, query, note, candidates, scores)`,
  where `scores: Option<&[Option<u32>]>` is `Some` only when the judge ran
  and returned scores (`Escalate`).

`offline_fallback`:

- Order the candidates by score descending (`None` last), then index
  ascending. Without scores, use index ascending, which is retrieval rank.
- Disk-verify them in that order with `verified_candidate_findings` and keep
  the first `OFFLINE_FALLBACK_MAX_FINDINGS = 5` survivors.
- Each note becomes `"<provenance>; unverified candidate; judge p=0.31"`
  when the candidate has a score. It becomes
  `"<provenance>; unverified candidate; retrieval rank 2"` (1-based rank)
  when `scores` is `None`, or when `scores` is `Some` but this candidate's
  entry is `None` (not judged).
- Summary with scores:
  `format!("No candidate reached the local judge's selection threshold ({}%) and the LLM fallback is disabled (agent.fallback = \"off\"); returning {n} unverified candidate(s) ranked by judge probability.", select_threshold)`.
  Without scores:
  `format!("The LLM fallback is disabled (agent.fallback = \"off\"); returning {n} unverified candidate(s) ranked by retrieval score.")`.
  When `note` is `Some`, append `" " + note`.
- Exits through `finalize_and_complete(StageExit::Fallback, forced_finish = false, …)`.
  The repo brief is not fetched, so `brief_tokens` and
  `orientation_calls_in_loop` stay `None`, and `cited_candidate_ids` stays
  `None`.

**D3.4 — Shadow agreement.**

- `llm_set`: when LLM verify finished, the candidate indices `i` whose
  location overlaps any result finding. Overlap means the same
  `normalize_rel_path` path and
  `c.line_start <= f.line_end && c.line_end.max(c.line_start) >= f.line_start`,
  with both locations known. This is a small private helper in
  `judge_verify.rs`.
- `judge_set`: `selected` when the judge finished.
- The value of `shadow_agreement`:

| LLM verify | Judge | Value |
|---|---|---|
| Finished | Finished, sets equal | `"exact"` |
| Finished | Finished, sets intersect | `"overlap"` |
| Finished | Finished, disjoint | `"disjoint"` |
| Finished | Escalate | `"judge-escalated"` |
| Escalate | Finished | `"llm-escalated"` |
| Escalate | Escalate | `"both-escalated"` |
| any | Failed | `None` |

**D3.5 — Cache key.** In `run_query_key`, after the snippet-cap field:

- if `judge_settings.mode == Laya`:
  `encode_field_into(&mut key, &format!("judge=laya:{}", select_threshold))`;
- if `settings.fallback == Off`: `encode_field_into(&mut key, "fallback=off")`.

Nothing else changes. `disk_cache::SCHEMA_VERSION` is unchanged, because
the value shape is unchanged.

**D3.6 — `QueryMetrics`.** New fields, all serialized into the JSONL
record and all added to the INFO line in `emit_metrics`:

| field | type | value |
|---|---|---|
| `judge_mode` | `&'static str` | `"off"`/`"laya"`/`"shadow"`, set right after `QueryMetrics::new` in `run`, on every path including cache |
| `fallback_mode` | `&'static str` | `"llm"`/`"off"`, set likewise |
| `judge_outcome` | `Option<&'static str>` | `"selected"`/`"escalated"`/`"error"`; `None` when the judge did not run |
| `judge_ms` | `Option<u64>` | judge call wall time, from `elapsed_ms` |
| `judge_candidates` | `Option<u32>` | states sent (`judged`) |
| `judge_selected` | `Option<u32>` | `selected.len()`: selected candidates that survived disk verification, counted **before** `finalize`'s dedupe and `max_results` cap (the same `n` as in the summary). `Some(0)` on `Escalate`, `None` on `Failed` |
| `judge_max_p` | `Option<u32>` | max permille over `scores` (`None` if no score) |
| `shadow_agreement` | `Option<&'static str>` | D3.4 |

`QueryMetrics::new` initializes `judge_mode: "off"`, `fallback_mode: "llm"`,
and every `Option` to `None`.

**D3.7 — Crate docs.** `crates/repo-explorer-agent/CLAUDE.md` gets the
sections "Judge verification (Stage 4, `judge.mode`)" and "Offline fallback
(`agent.fallback = "off"`)". They restate the D3.2–D3.5 rules, and name
`verified_candidate_findings` as the single disk-verification path for the
early-exit, judge and offline paths.

### D4 — Server wiring (`repo-explorer-mcp`)

- `Cargo.toml`: add `repo-explorer-judge = { version = "0.10.3", path = "../repo-explorer-judge" }`.
- `src/server.rs`: change the alias to
  `pub type Agent = AgentLoop<MemoryClientBackend, NativeSearchBackend, GenaiProvider, GitStateProbe, SystemClock, ConfiguredJudge>;`
- `src/main.rs` `run()`: after `build_router` and before `AgentLoop::new`:

```rust
let judge = repo_explorer_judge::ConfiguredJudge::from_settings(&config.judge)
    .context("failed to configure the local judge ([judge])")?;
tracing::info!(mode = ?config.judge.mode, fallback = ?config.agent.fallback,
               base_url = %config.judge.base_url, "local judge configuration");
let agent = Arc::new(AgentLoop::new(/* unchanged args */).with_judge(judge, config.judge.clone()));
if config.judge.mode != JudgeMode::Off {
    let a = Arc::clone(&agent);
    tokio::spawn(async move {
        match a.warm_judge().await {
            Ok(()) => tracing::info!("local judge ready"),
            Err(e) => tracing::warn!(error = %e, "local judge not ready; queries will degrade until it is"),
        }
    });
}
let server = RepoExplorerServer::new(agent);
```

  `config.agent` is moved into `AgentLoop::new`, so read `config.agent.fallback`
  for the log line before that move (copy it into a local).
- `config test` needs no code change: it already runs `validate`. Its report
  now covers `[judge]`.
- `crates/repo-explorer-mcp/CLAUDE.md` gets one line: "The judge is built
  by `ConfiguredJudge::from_settings`; startup never fails on an unreachable
  judge (warm-up is background, best-effort)."

### D5 — `judge-serve/` launcher (Python)

```
judge-serve/
  pyproject.toml   # name="repo-explorer-judge-serve", requires-python=">=3.10",
                   # dependencies=["laya[serve]==0.3.7"]
  serve.py
  README.md
```

`serve.py`:

```python
"""Serve a Laya checkpoint for repo-explorer-mcp's [judge] via laya-serve's /v1/systemone."""
import json, os, sys

EXPECTED_JUDGE_STATE_VERSION = 1  # keep equal to repo_explorer_core::judge::JUDGE_STATE_VERSION

def main() -> None:
    import torch, uvicorn
    from laya.router import Router
    from laya.serve import create_app

    threads = int(os.environ.get("LAYA_THREADS", "0") or "0")
    if threads > 0:
        torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    ckpt = os.environ.get("REPO_EXPLORER_LAYA_CHECKPOINT", "").strip()
    if ckpt:
        meta = os.path.join(ckpt, "repo_explorer_judge.json")
        if not os.path.exists(meta):
            sys.exit(f"{meta} missing: calibrate the checkpoint first (train/laya/calibrate.py)")
        with open(meta) as f:
            info = json.load(f)
        if info.get("judge_state_version") != EXPECTED_JUDGE_STATE_VERSION:
            sys.exit(f"checkpoint judge_state_version {info.get('judge_state_version')} != {EXPECTED_JUDGE_STATE_VERSION}")
        print(f"[judge-serve] checkpoint {ckpt}; recommended judge.select_threshold = {info.get('select_threshold')}", flush=True)
        models = {"typed-decisions": ckpt}
    else:
        print("[judge-serve] WARNING: REPO_EXPLORER_LAYA_CHECKPOINT unset - serving the zero-shot upstream "
              "typed-decisions checkpoint (smoke tests only, not for production)", flush=True)
        models = None

    router = Router(models=models, device=os.environ.get("LAYA_DEVICE") or None, max_loaded=1)
    router.preload(["typed-decisions"])
    uvicorn.run(create_app(router),
                host=os.environ.get("LAYA_HOST", "127.0.0.1"),
                port=int(os.environ.get("LAYA_PORT", "8765")),
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))

if __name__ == "__main__":
    main()
```

`README.md` contains:

- the run command:
  `REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 LAYA_DEVICE=cuda uv run --project judge-serve python judge-serve/serve.py`;
- the environment variables (`LAYA_API_KEY` is honoured by upstream
  `create_app`);
- the matching `[judge]` TOML;
- a systemd user-unit example;
- a note that upstream `laya-serve` and its NixOS module cannot select a
  local checkpoint, which is why this launcher exists.

CI: the `test-eval` job adds `python -m py_compile judge-serve/serve.py`.

### D6 — Eval harness

New config files, each a copy of `eval/config/default.toml` with the
comment header adapted and these differences:

- **`eval/config/judge-shadow.toml`:**
  `[judge]` with `mode = "shadow"`, `base_url = "http://127.0.0.1:8765"` and
  `select_threshold = 50`.
- **`eval/config/judge-laya.toml`:** `[judge]` with `mode = "laya"` and the
  same URL and threshold.
- **`eval/config/judge-offline.toml`:**
  - `[judge]` with `mode = "laya"` and the same URL and threshold;
  - `[agent]` with `fallback = "off"`;
  - **no** `[llm]` section.

The maintainer sets `select_threshold` to the value printed by `serve.py`
before the gate runs; the committed default is 50.

`eval/run.py`:

- add `"judge_mode"`, `"fallback_mode"`, `"judge_outcome"`, `"judge_ms"`,
  `"judge_candidates"`, `"judge_selected"`, `"judge_max_p"`,
  `"shadow_agreement"` to the key tuple in `take_qw0_fields`;
- add them as `None` defaults in `parse_call_lines`'s `out` dict, with a
  comment block explaining `None` semantics: the judge did not run, or the
  binary predates 10b;
- update the docstrings that list QW-0 fields.

`eval/score.py`:

- **Aggregate.** In the QW-0 aggregate function (where `early_exit_route`
  is aggregated), add a `judge` block:
  - `judge_mode`: `Counter` over all rows' non-`None` values;
  - `judge_outcome`: `Counter` over non-`None` values;
  - `judge_ms`, `judge_selected`, `judge_max_p`: `_dist` over the rows that
    carry each field;
  - `shadow_agreement`: `Counter`;
  - `shadow_agreement_rate` = (exact + overlap) / (exact + overlap +
    disjoint), or `None` when the denominator is 0.
- **Printed section.** `Judge`, printed only when some row has a non-`None`
  `judge_outcome`; otherwise print `  judge: n/a — judge not run`.
- **CSV.** Append the eight fields to `CSV_COLUMNS` and fill them in
  `qw0_csv_rows` via `qw0_field`.

`eval/test_score.py`: add tests. Synthetic rows carrying the judge fields
give the expected counters, distributions and agreement rate. Rows without
them give the `n/a` line, and CSV rows contain the eight new columns.

`.claude/skills/run-eval/SKILL.md`: add one paragraph naming the three
judge configs, and the rule "start `judge-serve` first; run shadow before
laya".

### D7 — Docs

- **New `docs/laya-judge.md`:**
  - an overview of 10a/10b/10c;
  - the mode matrix (below);
  - serving with `judge-serve`;
  - the rollout procedure: `off` → `shadow` (collect agreement) → `laya` →
    optional `laya` + `fallback = "off"`;
  - the promotion rule, "promote shadow → laya only when the M2 and M3
    gates below pass";
  - troubleshooting: judge down (the WARN line, then degrade), threshold
    tuning, and `judge_state_version` mismatch.

  The mode matrix:

  | `judge.mode` | `agent.fallback` | Stage 4 | Stage 5 | `[llm]` needed |
  |---|---|---|---|---|
  | off | llm | LLM | LLM loop | yes |
  | off | off | LLM | offline synthesis | yes |
  | shadow | llm/off | LLM (+ judge measured) | LLM loop / offline | yes |
  | laya | llm | judge (LLM on judge error) | LLM loop | yes |
  | laya | off | judge | offline synthesis | no |

- **`docs/configuration.md`:** a `[judge]` section with every key, default
  and validation rule, and `agent.fallback`.
- **Root `CLAUDE.md` Layout:** add `crates/repo-explorer-judge/` ("local
  candidate judge clients; owns the Laya wire format") and `judge-serve/`.
- **`CHANGELOG.md` `[Unreleased]` → `### Added`:** the local candidate judge
  (`[judge]`, modes `off`/`laya`/`shadow`), `agent.fallback = "off"`, the
  `judge-serve` launcher, and the new metrics fields.

## Error handling

| Situation | Behaviour |
|---|---|
| Judge unreachable, 5xx or timeout during a query (`laya`) | WARN log. LLM verify if providers exist, else Stage 5 (offline if `fallback = "off"`). Metric `judge_outcome = "error"`. The query never fails |
| Judge error in `shadow` | Metric `judge_outcome = "error"`, `shadow_agreement = None`. The LLM result is returned as usual |
| Malformed judge response | `Protocol` error, handled as above |
| Wrong bearer token | `Unavailable` ("rejected the bearer token"), handled as above |
| Judge unreachable at startup | Background warm-up logs a WARN. Startup continues |
| Invalid `[judge]` config | `config test` and server start fail with the D1 `ValidationError` naming `judge.<key>` |
| `mode = laya`, `fallback = off`, no `[llm]` | Valid. The router has no providers, and no LLM call is attempted on any path |
| `api_key_env` names an unset variable | `MissingJudgeEnvVar` at validation |
| Every selected candidate fails disk verification | Escalate (Stage 5), `judge_outcome = "escalated"` |

## Testing strategy

**Core (`config.rs` tests):**

- defaults: `mode = off`, `fallback = llm`, `base_url`, `select_threshold = 50`;
- a TOML with `[judge] mode = "laya"` plus `[agent] fallback = "off"` and no
  `[llm]` section loads and validates;
- `mode = "off"` with no providers → `EmptyProviderList`;
- `mode = "shadow"` with no providers → `EmptyProviderList`;
- base URLs `ftp://x` and `http://` → `InvalidJudgeBaseUrl`;
- `select_threshold` 0 and 100 → `InvalidJudgeSetting { key: "select_threshold", .. }`;
- `max_concurrency = 0`, `timeout_ms = 0` and `model = " "`, each giving
  the matching key;
- an unset `api_key_env` → `MissingJudgeEnvVar`, using `validate_with_env`
  with a fake env;
- `unknown_key_warnings` flags `judge.bogus` but not the 7 known keys or
  `agent.fallback`;
- `to_toml_string` round-trips a config with `[judge]`;
- `toml_path` for every new variant.

**Core (`judge.rs`, `llm.rs`):**

- `NoJudge` returns `Unavailable`;
- `MockJudge` replays its responses in order and records its calls;
- `has_providers` is false for an empty router and true otherwise.

**Judge crate** (`tests/http.rs`). An in-test HTTP/1.1 responder over
`tokio::net::TcpListener` reads the request line, headers and
`Content-Length` body, then writes a canned response. No new
mock-server dependency. Cases:

1. Success with 3 states under `max_concurrency = 2`: judgements come back
   in state order, even when the responder answers out of order via
   per-request delays.
2. The request body equals the pinned JSON above; the `model` is honoured.
3. The `authorization: Bearer <key>` header is present only when
   `api_key_env` is set. The judge is built via `new_with_env` with a fake
   accessor; the process environment is never mutated. Also: `api_key_env`
   set but the accessor returns `None` → `new_with_env` returns
   `Unavailable`.
4. 401 → `Unavailable`.
5. 500 → `Protocol` containing `HTTP 500`.
6. Non-JSON body → `Protocol`.
7. Missing `probabilities.A` → `Protocol`.
8. `p = 1.2` → `Protocol`.
9. A response delayed past `timeout_ms = 200` → `Timeout { timeout_ms: 200 }`.
10. Empty states → `Ok(vec![])` with no connection accepted.
11. `warm_up`: 200 → `Ok`, and a closed port → `Unavailable`.
12. `ConfiguredJudge::from_settings` with `mode = off` → `Disabled`.

**Agent.** Tests 1–12 live in `agent.rs`'s existing `#[cfg(test)] mod tests`,
because the helpers they need (`agent_with*`, `taped_metrics()`, the
`EMITTED` tape) are private to that module. They use those helpers plus
`.with_judge(MockJudge…)` and a temp repo. Pure helpers of
`judge_verify.rs` (selection ordering, the overlap helper, the agreement
classification) get their own unit tests in that file.

1. `off`: the judge is never called, and existing verify tests are
   unchanged.
2. `laya`, the judge scores 2 of 5 candidates ≥ threshold:
   - `stage_exit = Verify`, `llm_calls = 0`;
   - findings in descending-p order with the `; judge p=` suffix;
   - the summary template;
   - `judge_outcome = "selected"`, `judge_selected = 2`,
     `judge_candidates = 5`.
3. `laya`, everything below threshold, providers present: the Stage-5 LLM
   loop runs (the mock provider saw calls) and
   `judge_outcome = "escalated"`.
4. `laya` + `fallback = off`, everything below threshold:
   - `stage_exit = Fallback`, `llm_calls = 0`;
   - at most 5 findings, ordered by p, with `unverified candidate; judge p=`
     notes;
   - the summary template with the threshold;
   - `brief_tokens = None`.
5. `laya`, `MockJudge` returns `Unavailable`, providers present: the LLM
   verify path runs and `judge_outcome = "error"`.
6. `laya` + `fallback = off`, no providers (empty router), judge error:
   offline synthesis ordered by retrieval rank, with
   `retrieval rank <n>` notes.
7. `shadow`, where the LLM finishes on candidate 1 and the judge selects
   candidate 1:
   - the returned result equals the `off`-mode result for the same mocks;
   - `shadow_agreement = "exact"`;
   - the judge was called once, with the states `render_judge_states`
     produces.
8. `shadow`, with the LLM on candidate 1 and the judge on candidate 3 →
   `"disjoint"`; with the judge escalating → `"judge-escalated"`.
9. `laya`, where the selected candidate's path does not exist on disk: it
   is dropped; if it was the only one, the run escalates.
10. An unknown-location candidate is never sent to the judge: the
    `MockJudge` calls exclude it.
11. `run_query_key` differs between `off` and `laya`, differs between two
    thresholds, differs with `fallback = off`, and is identical between
    `off` and `shadow`.
12. Every exit path emits `judge_mode` and `fallback_mode`, including the
    cache hit.

**Integration** (`crates/repo-explorer-agent/tests/integration.rs`): one
end-to-end `laya` run over the existing fixture repo with
`MockJudge::with_scorer`, which scores 900 when the state contains the
target symbol and 100 otherwise. It asserts the exact final
`ExplorationOutcome`.

**Eval** (`eval/test_score.py`): the D6 tests.

## Acceptance criteria

Automated (CI):

1. `cargo fmt --check`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo build --workspace`
4. `cargo test --workspace` on ubuntu-latest and windows-latest, including
   every test listed above.
5. `python eval/test_score.py`, `python -m py_compile eval/run.py`,
   `python -m py_compile judge-serve/serve.py`.
6. With a config using `[judge] mode = "laya"`, `[agent] fallback = "off"`,
   `[codebase_memory] command = "codebase-memory-mcp"` and
   `args = ["--stdio"]`, and no `[llm]`,
   `repo-explorer-mcp --config <that file> config test`
   prints `"status": "valid"`. With `select_threshold = 0` it prints
   `"toml_path": "judge.select_threshold"` and exits non-zero. This is an
   integration test in the mcp crate that runs the binary via
   `env!("CARGO_BIN_EXE_repo-explorer-mcp")`.

Manual end-to-end gates (maintainer; GPU host; installed binary at
`~/.local/bin/repo-explorer-mcp`; Claude Code closed; `GOOGLE_API_KEY`
exported where `[llm]` is present). Start the judge first:

```bash
REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 LAYA_DEVICE=cuda uv run --project judge-serve python judge-serve/serve.py &
curl -sf http://127.0.0.1:8765/health
```

Then set `select_threshold` in the three eval configs to the value
`serve.py` printed.

- **M1 — shadow.**
  `uv run --with pyyaml eval/run.py --config eval/config/judge-shadow.toml`,
  then
  `uv run --with pyyaml eval/score.py results/<run-id>`.
  Record `judge_outcome` counts, `judge_ms` p50/p95 and
  `shadow_agreement_rate`.
- **M2 — shadow agreement gate.** `shadow_agreement_rate ≥ 0.80`, and
  `judge_outcome.error == 0`.
- **M3 — laya gate.**
  `uv run --with pyyaml eval/run.py --config eval/config/judge-laya.toml`,
  then score. Compared with `docs/eval/baseline-2026-09-15.md`:
  - pass@1 ≥ 0.72 (at least 23 of 32 queries);
  - hallucinations (P0/P1) = 0;
  - confident-wrong = 0;
  - `tokens_per_query` (all rows) mean ≤ 3,450 (≤ 30 % of 11,500.6);
  - `latency_per_query` p50 ≤ 2,805 ms.
- **M4 — offline run (recorded, no threshold).**
  `uv run --with pyyaml eval/run.py --config eval/config/judge-offline.toml`
  must show `llm_calls == 0` on every row. Record pass@1.
- Record the M1–M4 numbers in `docs/eval/judge-<date>.md`, in the style of
  the baseline doc.
- `[judge] mode` stays `off` by default in code whatever the outcome.
  Promotion is a per-user config choice, documented in `docs/laya-judge.md`.

## Risks

- **Upstream serving is serial.** Upstream `/v1/systemone` is an `async def`
  that calls the blocking `router.predict` inline, so one uvicorn worker
  serves requests **one at a time**. `max_concurrency` only overlaps HTTP
  round trips, not inference. That is why `timeout_ms` defaults to 20,000;
  `docs/laya-judge.md` must state this and recommend GPU serving.
- **Latency on CPU-only hosts.** 12 serial inferences at about 0.3–1 s each
  could exceed today's p50 of 2.8 s. Mitigation: GPU serving and 10c
  in-process batching; M3 gates on
  latency.
- **Threshold miscalibration** shifts precision and recall. Mitigation:
  10a's `calibrate.py` recommendation, and M2/M3 before promotion.
- **Router auto-selection** would bypass the fine-tuned model if `model`
  were not honoured. Mitigation: the client always sends the explicit
  `typed-decisions` name, and the launcher maps exactly that name.
- **Laya upstream API drift** (`create_app`, `Router(models=…)`, response
  shape). Mitigation: `laya[serve]==0.3.7` is pinned, and the HTTP tests pin
  the response fields used.
- **Offline mode returns unverified candidates.** Mitigation: every note
  and the summary say "unverified"; `stage_exit = "fallback"` marks it.
- **Shadow doubles Stage-4 wall time** to max(LLM, judge). Mitigation: it
  is a rollout mode, not a steady state.

## Global Constraints

- All code, comments, docs and spec text are in English.
- Rust edition 2024, with the workspace version; `cargo fmt --check` and
  `cargo clippy --all-targets -- -D warnings` are clean.
- CI passes on ubuntu-latest and windows-latest.
- `repo-explorer-core` gains no dependency (no `serde_json`, `anyhow`,
  `reqwest` or `rmcp`), and `thiserror` stays its only error crate.
- New Rust crates use only dependencies already present in `Cargo.lock`.
- Defaults preserve today's behaviour exactly: `judge.mode = "off"`,
  `agent.fallback = "llm"`, and the query-cache keys of `off`/`shadow` runs
  are unchanged.
- The judge wire format is exactly D2's request body, built from
  `repo_explorer_core::judge` constants, with `model = "typed-decisions"`
  by default.
- The judge state is produced only by
  `repo_explorer_agent::judge_input::render_judge_states` (train/serve
  parity).
- `select_threshold` is an integer percent in `1..=99`, default 50.
  Judgements are per-mille integers.
- A judge failure never fails a query, and a judge outage never blocks
  server startup.
- `stage_exit` keeps exactly four values; the MCP response DTO is
  unchanged.
- The API key is never logged, and never included in errors or metrics.
- The judge HTTP client never uses a proxy (`no_proxy()`).
- The launcher defaults are `127.0.0.1:8765` and `laya[serve]==0.3.7`.

## Decisions & assumptions

1. Upstream `laya-serve` protocol, one request per candidate, bounded
   concurrency (default 4), fail-fast with no partial results.
2. A Python launcher around `laya.serve.create_app(router)` maps
   `typed-decisions` to the local checkpoint. It refuses to start without
   `repo_explorer_judge.json` or on a `judge_state_version` mismatch.
3. The judge is a sixth generic `J = NoJudge` via `with_judge`, and
   `AgentLoop::new` is unchanged.
4. `laya` mode replaces only Stage 4, under today's Stage-4 gate.
5. The selection rule is `score ≥ select_threshold × 10` permille, ordered
   by score, then retrieval rank, followed by the shared disk
   verification.
6. A judge failure degrades to LLM verify iff `router.has_providers()`,
   otherwise to Stage 5.
7. `shadow` returns the LLM result and records `shadow_agreement` per D3.4.
8. `agent.fallback = "off"` replaces Stage 5 with offline synthesis of at
   most 5 unverified candidates.
9. `[llm]` becomes optional, and is required unless `mode = laya` and
   `fallback = off`.
10. The cache-key suffix applies only for `laya` mode and `fallback = off`.
11. Warm-up runs in the background at startup and is best-effort.
12. The M2 and M3 gate thresholds are as listed. Default config stays `off`
    regardless.
13. Assumption: `laya.router.Router(models={"typed-decisions": <local dir>})`
    plus `preload(["typed-decisions"])`, and `laya.serve.create_app(router)`,
    behave as in laya 0.3.7's source (Router maps names to local paths via
    `Agent(path)`; `create_app` honours an explicit known `model` name and
    `LAYA_API_KEY`).
