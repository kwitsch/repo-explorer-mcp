# Stage 10a — Laya Judge: Shared Judge Input, Training-Data Generator, Fine-Tune Tooling

Spec 1 of 3 in the Laya judge track (10a → 10b → 10c). Input for
`/taskflow:spec-driven-delivery` (`SPEC_PATH` = this file). Target location in
the repo: `docs/project-plan/10a-laya_judge_training.md`.

**Prerequisites.** Stage 8 (retrieval pipeline) only, which is already
implemented. Stages 9 and 9b (training or fine-tuning a generative LLM) are
**not** prerequisites and are not implemented; nothing in 10a–10c uses their
code. Stage 10 is an alternative to them for the Stage-4 role only. The
Stage-5 fallback loop keeps using the configured LLM providers; plan 9b
remains the option if that loop should also run on a local model later.

## Keypoints

- A Laya checkpoint can take over the Stage-4 LLM verification step (75 % of
  queries in the 2026-09-15 baseline). But Laya's base checkpoints are close
  to chance zero-shot (README: "a fast base to specialise, not a zero-shot
  decision engine"), so the checkpoint **must be fine-tuned**. This spec
  delivers everything that fine-tuning needs. Runtime integration is 10b.
- **Train/serve parity by construction.** One Rust function renders the
  plain-text judge state. The training-data generator and the 10b runtime
  both call it. Laya passes string states through untouched
  (`serialize_state`), so training, HTTP serving and the 10c candle port see
  byte-identical input.
- **No teacher LLM.** A new dev-only binary, `repo-explorer-datagen`, builds
  labels programmatically:
  - it generates queries by rule from real repository artifacts (symbol
    names, doc comments, error literals);
  - it runs the **real** deterministic pre-stage on each query;
  - it labels every candidate against the known ground-truth location.
- The corpus is pinned: 36 permissively licensed repositories in 6
  languages. The data is split by repository. The eval corpus
  (`eval/repos.toml`) and this repository are excluded.
- Python tooling in `train/laya/` covers bake-off, preprocessing, the
  fine-tune (derived from Laya's own RLCD notebook), temperature calibration,
  threshold selection and an offline acceptance gate.
- GPU training runs are **manual gates** executed by the maintainer. The
  implementer builds the tooling and smoke-tests it on tiny data on CPU.

## Goal

Produce reproducible tooling that turns a pinned public-repo corpus into
labelled judge data. Use it to produce a fine-tuned, calibrated Laya
checkpoint directory that scores one retrieval candidate with P(relevant),
plus the shared judge-input renderer that 10b and 10c reuse.

## Non-goals

- Any change to `explore_repository` behaviour or to the Stage-4/5 runtime:
  that is 10b. This spec adds only refactors with no behaviour change, plus
  new public APIs.
- In-process inference (10c).
- LLM-generated or LLM-labelled data. No provider call anywhere.
- Training anything for the Stage-5 fallback loop.
- Changing retrieval ranking, `top_k`, or early-exit rules.
- Publishing checkpoints or datasets (Hugging Face or elsewhere). Checkpoints
  stay local.
- Mining real user query logs.
- Multilingual (mmBERT) fine-tuning.

## Chosen approach

**Question shape.** The model is asked one 2-option `choice` question per
candidate, with neutral keys `A` (relevant) and `B` (not relevant). Findings
are multi-select, so an independent score per candidate is needed.

**State shape.** The state is a plain-text string in a fixed, versioned
format (`JUDGE_STATE_VERSION = 1`), rendered by one Rust function.

**Data source.** Only queries whose snapshot reaches the Verify stage
produce rows, because that is the only situation in which 10b will call the
judge.

**Base model and recipe.** Fine-tune `convaiinnovations/laya` (repo root,
ModernBERT-large, 421M) with `max_len = 1024` and `head_max_len = 256`. This
is the recipe that produced `laya-typed-decisions`. The training loop is
Laya's own RLCD DDP script from its fine-tuning notebook, adapted to a local
JSONL dataset.

Alternatives rejected:

| Alternative                      | Why it lost                                                                                                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `noul` question                  | Laya issue #156: `noul` can follow its `false:`/`true:` labels instead of the state on the English checkpoint                                                                                                |
| One `choice` over all candidates | Softmax forces a single selection, but findings are multi-select. The head budget (256 tokens, 48 per option) and the state budget (~768 tokens) cannot hold 12 candidates' code                             |
| JSON-object state                | Laya re-serializes objects with Python `json.dumps` (`", "`/`": "` separators). serde_json writes compact JSON, so the candle port would need a json.dumps emulator. Plain text is also more token-efficient |
| Teacher-LLM labels               | Sends queries and snippets to a third-party model (data sovereignty), costs tokens, and is unnecessary: the ground truth can be derived programmatically                                                     |
| Base `laya-typed-decisions`      | Specialised to four business workflows. Rejected in favour of the same recipe from the neutral root checkpoint                                                                                               |
| Base `laya-multilingual`         | Weaker on English and code. Queries are mostly English, and German is measured separately                                                                                                                    |
| Split by query                   | Leaks repo-specific vocabulary between splits, which inflates metrics                                                                                                                                        |

## Detailed design

### D1 — Judge question constants (`repo-explorer-core`)

New file `crates/repo-explorer-core/src/judge.rs`, registered as
`pub mod judge;` in `crates/repo-explorer-core/src/lib.rs`. This spec adds
constants only; 10b adds the trait to the same module. Core keeps no
serde_json: consumers build wire JSON from these strings.

```rust
//! The local candidate judge (Stage 10): the fixed question every judge
//! backend asks about ONE retrieval candidate. Constants only; the
//! `CandidateJudge` trait lives here too once Stage 10b adds it.

/// Version of the plain-text state rendered by
/// `repo_explorer_agent::judge_input::render_judge_state`. Bump on ANY change
/// to that rendering or to the constants below: a checkpoint is valid only
/// for the version it was trained on.
pub const JUDGE_STATE_VERSION: u32 = 1;
pub const QUESTION_ID: &str = "relevant";
pub const QUESTION_TYPE: &str = "choice";
pub const QUESTION_INSTRUCTIONS: &str =
    "Does this code location answer the repository search query?";
pub const POSITIVE_KEY: &str = "A";
pub const POSITIVE_CRITERION: &str = "yes, this location answers the query";
pub const NEGATIVE_KEY: &str = "B";
pub const NEGATIVE_CRITERION: &str = "no, this location does not answer the query";
```

Criteria order is always `A` then `B`. serde_json's default sorted map keeps
that order because "A" < "B". A unit test pins every constant's value
(`judge_constants_are_pinned`).

### D2 — Judge state renderer (`repo-explorer-agent`)

New file `crates/repo-explorer-agent/src/judge_input.rs`, exported as
`pub mod judge_input;` in `lib.rs`.

```rust
pub const BODY_CONTEXT_BEFORE: u32 = 3;
pub const BODY_CONTEXT_AFTER: u32 = 5;
pub const BODY_MAX_LINES: usize = 60;
pub const LINE_MAX_CHARS: usize = 200;
pub const OUTLINE_MAX_LINES: usize = 30;

/// Everything the judge sees for one candidate.
pub struct JudgeCandidateView<'a> {
    pub query: &'a str,
    pub candidate: &'a Candidate,
    /// `skeleton_for` output for the candidate's file, if the graph knows it.
    pub outline: Option<&'a str>,
    /// Raw file lines of the body window (see render_judge_states step 3).
    pub body: Option<&'a str>,
}

/// Pure, deterministic. Format version = core::judge::JUDGE_STATE_VERSION.
pub fn render_judge_state(view: &JudgeCandidateView<'_>) -> String;

/// One entry per candidate, same order. `None` for an unknown-location
/// candidate (`is_unknown_location`): those are never judged or labelled.
pub async fn render_judge_states<M: MemoryBackend>(
    memory: &M,
    repo_root: &Path,
    query: &str,
    candidates: &[Candidate],
) -> Vec<Option<String>>;
```

`render_judge_state` builds these lines in this order and joins them with
`\n`, with no trailing newline:

1. Query line: `query: <q>`. Every `\r` and `\n` in `q` becomes a space,
   then the result is trimmed.
2. Candidate line:
   `candidate: <path>:<line_start>-<line_end> (<kind_label(kind)><sym>)`.
   - `<path>` is the candidate path via `to_string_lossy()`, with every `\`
     replaced by `/`.
   - Start and end come from `normalize_location`, so `line_end >=
line_start`.
   - `<sym>` is `, symbol `<symbol>`` when `symbol` is `Some`, otherwise
     empty.
3. Code section. The source text is `body` if it is `Some` and non-empty,
   otherwise `candidate.snippet` if that is `Some` and non-empty, otherwise
   none.
   - With a source text: a line `code:`, then the first `BODY_MAX_LINES`
     lines of the source. Split on `\n`, strip one trailing `\r` per line,
     and emit each line as `"  " + <first LINE_MAX_CHARS chars of the
line>`. The limit counts `char`s, not bytes, and tabs are kept as they
     are.
   - With no source text: the single line `code: (unavailable)`.
4. Outline section, only when `outline` is `Some` and non-empty: a line
   `outline:`, then the first `OUTLINE_MAX_LINES` lines of `outline`,
   verbatim. `skeleton_for` lines already start with two spaces.

The outline is last on purpose: Laya truncates the state from the right
(`st[:room]`).

`render_judge_states` works like this:

1. Canonicalize the root once with `crate::dispatch::canonical_repo_root`.
   If that fails, every body is `None`.
2. Fetch outlines once per distinct path, concurrently (`join_all` over
   `crate::skeleton::skeleton_for`).
3. For each known-location candidate, let `loc = normalize_location(candidate.location.clone())`,
   `start = loc.line_start.saturating_sub(BODY_CONTEXT_BEFORE).max(1)` and
   `end = loc.line_end.saturating_add(BODY_CONTEXT_AFTER)`. Read the body
   with `crate::dispatch::read_file_canonical(repo_root, &canonical_root,
&path_str, Some(start), Some(end))`, where `path_str` is
   `loc.path.to_string_lossy()`. `Err` becomes `None`.
4. Render each candidate's state with `render_judge_state`.

This intentionally differs from `verify::candidates_block`, which shows an
outline only once per file: every state here must be self-contained.

### D3 — Retrieval snapshot API (`repo-explorer-agent`)

1. **Refactor, no behaviour change.** Extract the Stage-3 route choice in
   `AgentLoop::run` (the `let early_exit_route = if outcome.candidates.is_empty() … else { None };`
   block) into:

   ```rust
   pub(crate) fn early_exit_route(
       outcome: &pipeline::RetrievalOutcome,
       settings: &AgentSettings,
   ) -> Option<&'static str>
   ```

   `run` calls it. The `authorized` check stays in `run`. All existing tests
   must pass unmodified.

2. **New file `crates/repo-explorer-agent/src/snapshot.rs`.** Registered as
   `mod snapshot;` with `pub use snapshot::{RetrievalSnapshot, SnapshotStage, retrieval_snapshot};`
   in `lib.rs`:

   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub enum SnapshotStage { EarlyExit, Verify, Fallback }

   #[derive(Debug, Clone)]
   pub struct RetrievalSnapshot {
       pub candidates: Vec<Candidate>,
       pub confidence: u32,
       pub stage: SnapshotStage,
   }

   /// The deterministic pre-stage alone: no index refresh, no cache, no LLM.
   pub async fn retrieval_snapshot<M: MemoryBackend, S: SearchBackend>(
       memory: &M,
       search: &S,
       repo_root: &Path,
       query: &ExplorationQuery,
       settings: &AgentSettings,
   ) -> RetrievalSnapshot
   ```

   The body calls `pipeline::retrieve(memory, search, repo_root, query, settings.top_k, None)`
   and sets `stage`:

   - `EarlyExit` if `early_exit_route(&outcome, settings).is_some()`.
   - otherwise `Verify` if `outcome.confidence >= settings.fallback_confidence && !outcome.candidates.is_empty()`.
   - otherwise `Fallback`.

   The disk-authorization step of the early exit is not simulated: that is
   an accepted approximation.

### D4 — `repo-explorer-datagen` crate (dev-only binary)

New workspace member `crates/repo-explorer-datagen`, added to the root
`Cargo.toml` `members`. It is never built by `release.yml`, which builds
`--package repo-explorer-mcp` only. The crate also gets its own `CLAUDE.md`.

`Cargo.toml`. The path-dependency version strings follow the sibling
crates' current form (`version = "<workspace version>", path = "…"`). **No
crate outside this list, and nothing absent from `Cargo.lock`:**

```toml
[package]
name = "repo-explorer-datagen"
version.workspace = true
edition.workspace = true
publish = false

[dependencies]
repo-explorer-core   = { version = "0.10.3", path = "../repo-explorer-core" }
repo-explorer-memory = { version = "0.10.3", path = "../repo-explorer-memory" }
repo-explorer-search = { version = "0.10.3", path = "../repo-explorer-search" }
repo-explorer-agent  = { version = "0.10.3", path = "../repo-explorer-agent" }
anyhow = "1"
ignore = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.9"
sha2 = "0.11"
hex = "0.4"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "process"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
repo-explorer-core = { version = "0.10.3", path = "../repo-explorer-core", features = ["test-support"] }
```

The crate is a library plus a thin binary, so that `tests/` can reach the
generate core. `src/lib.rs` declares every module below as `pub mod`, and
`src/main.rs` only parses the CLI and calls into the library.

Modules:

| Module         | Contents                                                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.rs`      | CLI parsing and dispatch (binary target only)                                                                                                                       |
| `corpus.rs`    | Schema and validation                                                                                                                                               |
| `fetch.rs`     | The `fetch` subcommand                                                                                                                                              |
| `generate.rs`  | The `generate` subcommand: `run_generate` (orchestration: git rev check, index refresh, output files, manifest) and the backend-generic core `generate_repo` (D4.4) |
| `symbols.rs`   | File walk, outline filtering, doc and literal extraction                                                                                                            |
| `templates.rs` | Query templates                                                                                                                                                     |
| `label.rs`     | Candidate labelling                                                                                                                                                 |
| `rows.rs`      | Row and manifest types                                                                                                                                              |
| `rng.rs`       | splitmix64 + FNV-1a 64                                                                                                                                              |
| `stats.rs`     | The `stats` subcommand                                                                                                                                              |

Logs go to stderr via tracing-subscriber at `info`. stdout carries only the
JSON printed by `stats` and the final JSON summary of `fetch`/`generate`.

#### D4.1 CLI

Arguments are parsed by hand as `--flag value` pairs; no `clap`.

```
repo-explorer-datagen fetch    --corpus <file> --checkouts <dir>
repo-explorer-datagen generate --corpus <file> --checkouts <dir> --out <dir> --memory-command <path>
                               [--memory-arg <arg>]...            (default: a single "--stdio")
                               [--eval-repos <file>]              (default: eval/repos.toml)
                               [--max-queries-per-repo <n>]       (default: 400)
                               [--max-negatives-per-query <n>]    (default: 4)
                               [--seed <u64>]                     (default: 20260923)
                               [--only <repo-name>]...            (default: all repos)
repo-explorer-datagen stats    --data <dir>
repo-explorer-datagen --help | -h
```

Exit codes:

- `0` — success.
- `1` — invalid corpus or config, or at least one repo failed. Output files
  and the manifest are still written for the repos that succeeded.
- `2` — usage error: unknown subcommand or flag, or a missing required flag.
  Usage text goes to stderr.

#### D4.2 Corpus file

Fields and validation (`corpus.rs`). Every violation is an error naming the
repo and the field:

- `version = 1`
- `[[repo]]` with:
  - `name` — unique, matching `^[a-z0-9._-]+$`
  - `url` — starts with `https://`
  - `rev` — exactly 40 lowercase hex characters
  - `license` — exactly one of `MIT`, `Apache-2.0`, `BSD-2-Clause`,
    `BSD-3-Clause`, `ISC`, `MIT OR Apache-2.0`, `MIT OR Unlicense`
  - `lang` — one of `rust`, `python`, `typescript`, `javascript`, `go`,
    `java`, `kotlin`
  - `split` — one of `train`, `val`, `test`

Eval exclusion, checked by both `fetch` and `generate`:

- Load `--eval-repos`, a TOML file with `[[repo]] url = …`. Normalize every
  URL: lowercase, strip a trailing `/`, strip a trailing `.git`.
- Fail if any corpus URL equals an eval URL after the same normalization.
- Fail if any normalized corpus URL ends with `/repo-explorer-mcp`.
- A missing eval-repos file is an error.

Initial content of `crates/repo-explorer-datagen/corpus.toml`, committed
verbatim. Revs are the default-branch HEADs, and licenses were verified from
each repo's root license file, on 2026-09-23. There are six language groups,
each with 4 train, 1 val and 1 test repo. TypeScript and JavaScript form one
group: `javascript` appears only in two train repos (`express`, `prettier`).

```toml
version = 1

# --- rust ---
[[repo]]
name = "ripgrep"
url = "https://github.com/BurntSushi/ripgrep"
rev = "3fce3b5bb0236da2df6d99672afb8a719642eca7"
license = "MIT OR Unlicense"
lang = "rust"
split = "train"

[[repo]]
name = "serde"
url = "https://github.com/serde-rs/serde"
rev = "6693a89cca77e0151437da1c7f890090b9ebf04c"
license = "MIT OR Apache-2.0"
lang = "rust"
split = "train"

[[repo]]
name = "tracing"
url = "https://github.com/tokio-rs/tracing"
rev = "d9d4c542de10f5d3a711b7a45ffe450fd0666437"
license = "MIT"
lang = "rust"
split = "train"

[[repo]]
name = "regex"
url = "https://github.com/rust-lang/regex"
rev = "72d650cb0a880a01ab6dc2137c0888e8f89740f7"
license = "MIT OR Apache-2.0"
lang = "rust"
split = "train"

[[repo]]
name = "clap"
url = "https://github.com/clap-rs/clap"
rev = "8ab46fe22b4aa5c3bd09cc0e6c3fe9a75fc178a8"
license = "MIT OR Apache-2.0"
lang = "rust"
split = "val"

[[repo]]
name = "axum"
url = "https://github.com/tokio-rs/axum"
rev = "c44f6650aa38d4c154e3cdb7347205e03813f4d5"
license = "MIT"
lang = "rust"
split = "test"

# --- python ---
[[repo]]
name = "flask"
url = "https://github.com/pallets/flask"
rev = "d73fa1cdcbd8b1465c151db8924ba58b1dd14e35"
license = "BSD-3-Clause"
lang = "python"
split = "train"

[[repo]]
name = "fastapi"
url = "https://github.com/fastapi/fastapi"
rev = "50113da16fec53b66b80d75e80a89296de4fa5a5"
license = "MIT"
lang = "python"
split = "train"

[[repo]]
name = "rich"
url = "https://github.com/Textualize/rich"
rev = "9d8f9a372cc5916fd4781fec207ced7ddac2f08f"
license = "MIT"
lang = "python"
split = "train"

[[repo]]
name = "pydantic"
url = "https://github.com/pydantic/pydantic"
rev = "0384c970e37a59b344e75161eb106ea9996378ba"
license = "MIT"
lang = "python"
split = "train"

[[repo]]
name = "httpx"
url = "https://github.com/encode/httpx"
rev = "b5addb64f0161ff6bfe94c124ef76f6a1fba5254"
license = "BSD-3-Clause"
lang = "python"
split = "val"

[[repo]]
name = "click"
url = "https://github.com/pallets/click"
rev = "06b2a678741131fd577ce170e23e5ca0aeba0309"
license = "BSD-3-Clause"
lang = "python"
split = "test"

# --- typescript / javascript ---
[[repo]]
name = "vite"
url = "https://github.com/vitejs/vite"
rev = "f155b62ceabbaaf71abbf59933968787e3961c56"
license = "MIT"
lang = "typescript"
split = "train"

[[repo]]
name = "trpc"
url = "https://github.com/trpc/trpc"
rev = "8b649ad874a8fe97c73962d4617912231b975736"
license = "MIT"
lang = "typescript"
split = "train"

[[repo]]
name = "express"
url = "https://github.com/expressjs/express"
rev = "9a34acf03cb818ff3f8bc40e44176e277a25cbb9"
license = "MIT"
lang = "javascript"
split = "train"

[[repo]]
name = "prettier"
url = "https://github.com/prettier/prettier"
rev = "de9ec361d3e2bb8fbc30da34ceaf402d351b5918"
license = "MIT"
lang = "javascript"
split = "train"

[[repo]]
name = "hono"
url = "https://github.com/honojs/hono"
rev = "6cadf7537385c6e3df9cbd6d05c9b84c82ae0361"
license = "MIT"
lang = "typescript"
split = "val"

[[repo]]
name = "zod"
url = "https://github.com/colinhacks/zod"
rev = "10dda3a3a4bc76ed0bf83e994de194587e489ef8"
license = "MIT"
lang = "typescript"
split = "test"

# --- go ---
[[repo]]
name = "cobra"
url = "https://github.com/spf13/cobra"
rev = "adbc8813901bba65827259daa8e22ff94ec1f30e"
license = "Apache-2.0"
lang = "go"
split = "train"

[[repo]]
name = "gin"
url = "https://github.com/gin-gonic/gin"
rev = "3b08cd7235bd5ad2f055aa9e38135f111f6c5926"
license = "MIT"
lang = "go"
split = "train"

[[repo]]
name = "viper"
url = "https://github.com/spf13/viper"
rev = "528f7416c4b56a4948673984b190bf8713f0c3c4"
license = "MIT"
lang = "go"
split = "train"

[[repo]]
name = "fzf"
url = "https://github.com/junegunn/fzf"
rev = "b1be3a8be1b833ce5b92fbbac11637643d60a046"
license = "MIT"
lang = "go"
split = "train"

[[repo]]
name = "urfave-cli"
url = "https://github.com/urfave/cli"
rev = "4fa44ab9ea85ac65410c91c77efeb72a4e8517bc"
license = "MIT"
lang = "go"
split = "val"

[[repo]]
name = "chi"
url = "https://github.com/go-chi/chi"
rev = "3d1777a1ef8881f7d1da0b02c76ca8f0a29cd2bc"
license = "MIT"
lang = "go"
split = "test"

# --- java ---
[[repo]]
name = "gson"
url = "https://github.com/google/gson"
rev = "854c8255b625cf1e13c701a83ea9ccb4caaa576a"
license = "Apache-2.0"
lang = "java"
split = "train"

[[repo]]
name = "mockito"
url = "https://github.com/mockito/mockito"
rev = "5a2f0f81cdafc9b34ddb4520db7f8866c787dac0"
license = "MIT"
lang = "java"
split = "train"

[[repo]]
name = "commons-lang"
url = "https://github.com/apache/commons-lang"
rev = "6e8ced140e457a1622c8dd031cfa826d962a95ad"
license = "Apache-2.0"
lang = "java"
split = "train"

[[repo]]
name = "javalin"
url = "https://github.com/javalin/javalin"
rev = "20ba71b211b5efa91f5c10ccb467326b86a1e5dc"
license = "Apache-2.0"
lang = "java"
split = "train"

[[repo]]
name = "javapoet"
url = "https://github.com/square/javapoet"
rev = "b9017a9503b76e11b4ad4c1a9f050e2d29112cb0"
license = "Apache-2.0"
lang = "java"
split = "val"

[[repo]]
name = "jsoup"
url = "https://github.com/jhy/jsoup"
rev = "49a15317317970a7ea3f0a5ded303ef319860f4a"
license = "MIT"
lang = "java"
split = "test"

# --- kotlin ---
[[repo]]
name = "ktor"
url = "https://github.com/ktorio/ktor"
rev = "d08fd0382b2acf314b2beb430b66ba148845a892"
license = "Apache-2.0"
lang = "kotlin"
split = "train"

[[repo]]
name = "kotlinx-coroutines"
url = "https://github.com/Kotlin/kotlinx.coroutines"
rev = "bd2e9a1b90400fb7b2fa4f8731e7d8148799ab73"
license = "Apache-2.0"
lang = "kotlin"
split = "train"

[[repo]]
name = "detekt"
url = "https://github.com/detekt/detekt"
rev = "e3c5b3ed68740908fcc2e25afa4fac5de6ab7908"
license = "Apache-2.0"
lang = "kotlin"
split = "train"

[[repo]]
name = "koin"
url = "https://github.com/InsertKoinIO/koin"
rev = "dc86ef8dd8fbe8564fb7453c03f5b738da3450bb"
license = "Apache-2.0"
lang = "kotlin"
split = "train"

[[repo]]
name = "moshi"
url = "https://github.com/square/moshi"
rev = "889013ec2edb8d8034902662a1dc8c4f3b3f8111"
license = "Apache-2.0"
lang = "kotlin"
split = "val"

[[repo]]
name = "okio"
url = "https://github.com/square/okio"
rev = "c7d097b256530744652a17fb467e2e6db49ba371"
license = "Apache-2.0"
lang = "kotlin"
split = "test"
```

#### D4.3 `fetch`

For each repo, in file order, with `dir = <checkouts>/<name>`. Every step
runs `git` via `std::process::Command` with inherited stderr.

1. If `dir` is absent: `git clone --filter=blob:none --no-checkout <url> <dir>`.
2. `git -C <dir> fetch --filter=blob:none origin <rev>`.
3. `git -C <dir> checkout --detach --force <rev>`.
4. Require `git -C <dir> rev-parse HEAD` to equal `rev`.
5. Require at least one root-level file whose lowercase name matches:
   - `license`, `licence`, `copying` or `unlicense`, optionally followed by
     `.` and any extension; or
   - a name starting with `license-`.

A failed step records `{name, error}` and the loop continues with the next
repo. The final stdout JSON is
`{"fetched": [names], "failed": [{"name","error"}]}`, with exit code 1 if
`failed` is non-empty.

#### D4.4 `generate` pipeline

One `MemoryClientBackend::connect(&CodebaseMemoryConfig { command: Some(<memory-command>), args: <memory-args>, endpoint: None, staleness_seconds: 3600 })`
serves the whole run, along with one
`NativeSearchBackend::new(&SearchConfig::default())` and
`AgentSettings::default()`. Repos are processed sequentially. For each repo
selected by `--only` (or every repo), `run_generate` performs steps 1–2 and
then calls the core for steps 3–7:

```rust
pub struct GenerateOptions { pub seed: u64, pub max_queries_per_repo: usize, pub max_negatives_per_query: usize }
pub struct RepoOutput { pub rows: Vec<Row>, pub stats: RepoStats }   // Row/RepoStats from rows.rs (D4.7)

/// Steps 3–7 only: no git, no index refresh, no file output.
pub async fn generate_repo<M: MemoryBackend, S: SearchBackend>(
    memory: &M, search: &S, repo: &CorpusRepo, repo_root: &Path, opts: &GenerateOptions,
) -> RepoOutput
```

1. Require `git -C <checkouts>/<name> rev-parse HEAD == rev`. Otherwise the
   repo fails with "checkout not at pinned rev; run fetch".
2. `memory.ensure_fresh_index(repo_root)` must return
   `Ok(UpToDate | Reindexed)`. `IndexingFailed` or `Err` fails the repo.
3. **File walk.** Use `ignore::WalkBuilder::new(repo_root)` with defaults,
   which honour `.gitignore` and skip hidden files. Keep only regular files
   and collect repo-relative paths with `/` separators, sorted
   lexicographically.
   - Keep extensions `rs`, `py`, `ts`, `tsx`, `js`, `jsx`, `mjs`, `go`,
     `java`, `kt`.
   - Drop the path if any directory component, compared case-insensitively,
     is in {`test`, `tests`, `testdata`, `testing`, `__tests__`, `vendor`,
     `node_modules`, `third_party`, `dist`, `build`, `target`, `examples`,
     `benches`, `fixtures`, `docs`}.
   - Drop the path if its file name ends with `_test.go`, `_test.py`,
     `.min.js` or `.d.ts`; or starts with `test_` and ends with `.py`; or
     contains `.test.` or `.spec.`.
4. **Symbols.** For each file, in order, call
   `memory.file_outline(repo_root, Path::new(rel), Some(200))`. Each finding
   with `note = Some(qualified_name)` is a symbol candidate. Its path is the
   walked `rel` (authoritative, since the outline is per file); only
   `line_start`/`line_end` are taken from the finding's location. Keep it
   only when all of these hold:
   - the location is known;
   - `line_end > line_start`;
   - the short name has at least 4 chars. The short name is the last
     non-empty segment after splitting the qualified name on any of `.`,
     `::`, `/`, `#`, `$`;
   - the lowercase short name is not in {`main`, `init`, `__init__`, `new`,
     `default`, `from`, `into`, `fmt`, `drop`, `clone`, `hash`, `test`,
     `setup`, `run`, `get`, `set`, `len`, `call`, `next`, `build`, `close`,
     `open`, `read`, `write`, `apply`, `invoke`, `equals`, `hashcode`,
     `tostring`, `compareto`, `iter`, `item`, `value`, `values`, `keys`}.

   Dedupe on `(path, line_start, line_end)`, keeping the first occurrence.
   An outline error for one file skips that file only and counts it as
   `outline_errors`.

5. **Sampling.** If the symbol count exceeds `3 × max_queries_per_repo`,
   select exactly `3 × max_queries_per_repo` of them with a partial
   Fisher–Yates shuffle, seeded with
   `splitmix64(seed ^ fnv1a64(repo name bytes))`. Then restore the original
   order of the selection.
6. **Queries.** For each symbol, in order, try to build one query (D4.5).
   Stop after `max_queries_per_repo` accepted queries. A query is _accepted_
   when it is built, whatever stage it reaches.
7. **Snapshot.** For each query, call
   `retrieval_snapshot(&memory, &search, repo_root, &ExplorationQuery { text, scope_hint: None, max_results: None, detailed_snippets: false }, &settings)`.
   - `EarlyExit` increments `filtered_early_exit`; `Fallback` increments
     `filtered_fallback`. Neither produces rows.
   - `Verify`: call `render_judge_states(&memory, repo_root, &text, &snapshot.candidates)`,
     then label (D4.6) and append rows (D4.7).

#### D4.5 Query templates (`templates.rs`)

Definitions:

- **`words(N)`** splits an identifier into words:
  - break on `_`, `-`, lower→upper transitions, and acronym ends
    (`HTTPServer` → `http`, `server`);
  - digits stay attached to the preceding word;
  - lowercase everything and drop empty words.
- **`w`** is `words(N)` joined with single spaces.
- **Per-symbol RNG:** seeded with
  `splitmix64(seed ^ fnv1a64(format!("{repo}\0{path}\0{line_start}\0{qualified_name}")))`.

Templates. `N` is the short name, `D` the doc sentence (D4.5.1), `L` the
error literal (D4.5.2):

| id           | lang | applicable when          | text                                                                                                        |
| ------------ | ---- | ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `define-en`  | en   | always                   | `where is {N} defined`                                                                                      |
| `words-en`   | en   | ≥ 2 words                | one of: `where is the code that handles {w}`, `how does this project {w}`, `where do we {w}` (uniform pick) |
| `doc-en`     | en   | doc sentence `D` exists  | `where is the code that {d}`, where `d` = `D` with its first char lowercased and one trailing `.` removed   |
| `literal-en` | en   | error literal `L` exists | `where is the error "{L}" raised`                                                                           |

Selection: draw one template from the applicable set with integer weights
`define-en` 2, `words-en` 6, `doc-en` 6, `literal-en` 4. Use
`rng.next_u64() % total_weight` walked over the applicable templates in
table order. The same RNG then makes the phrasing pick inside the chosen
template.

**D4.5.1 Doc sentence.** The file is read once and cached per file. Let
`def = line_start` (1-based).

1. **Comment block.** Walk upward from line `def - 1`, one line at a time:
   - a line whose trimmed form starts with `#[`, `#!` or `@` (attributes,
     inner attributes, decorators, annotations) is skipped: never collected,
     never a stop, both before and during collection;
   - otherwise, a line whose trimmed form starts with one of `///`, `//!`,
     `//`, `/**`, `/*`, `*`, or `#` is collected;
   - any other line stops the walk.

   Reverse the collected lines into source order. Strip the leading marker
   (the longest matching of `///`, `//!`, `/**`, `//`, `/*`, `*`, `#`) and
   any trailing `*/`, then trim. Drop empty lines and lines starting with
   `@`.

2. **Python docstring.** Only for `.py` files, and only if step 1 found
   nothing. Take the first non-empty line in `(def, line_end]`. If its
   trimmed form starts with `"""` or `'''`, take the text after the opening
   quotes, up to closing quotes on the same line if present.
3. Join the collected text with single spaces and collapse whitespace. Take
   the text up to and including the first `". "` (keeping the `.`), or the
   whole text if there is none.
4. Accept `D` when it has 4–25 whitespace-separated words, contains neither
   `TODO` nor `FIXME`, and does not start with `@` or `#`.

**D4.5.2 Error literal.**

1. Scan lines `line_start..=line_end` for the first line that contains any
   of: `panic!(`, `bail!(`, `anyhow!(`, `ensure!(`, `Err(`, `raise `,
   `throw `, `errors.New(`, `fmt.Errorf(`, `Exception(`, `Error(`, and also
   contains a `"`.
2. `L` is the text between the first `"` on that line and the next `"` not
   preceded by `\`.
3. Accept `L` when it has 12–120 chars and at least 2 whitespace-separated
   words.

#### D4.6 Labelling (`label.rs`)

The ground truth `G` is the symbol's path and `[line_start, line_end]`.
Only candidates whose state is `Some` are considered.

A candidate `c` is **positive** when both hold:

- `normalize_rel_path(c.path)`, printed with `/` separators, equals `G.path`;
- `c.line_start <= G.line_end && max(c.line_end, c.line_start) >= G.line_start`.

Otherwise it is negative.

Rows emitted per query:

- every positive;
- plus the first `max_negatives_per_query` negatives in retrieval-rank
  order;
- if the query has no positive, at most 2 negatives in rank order, and the
  query counts toward `queries_without_positive`.

#### D4.7 Output

`<out>/train.jsonl`, `<out>/val.jsonl` and `<out>/test.jsonl` (chosen by the
repo's `split`) each hold one JSON object per line. Every file is streamed
to `<name>.tmp` and renamed at the end of the run.

Row keys, in exactly this struct field order:

```json
{
  "id": "ripgrep:000017:03",
  "workflow": "repo_explorer_verify",
  "repo": "ripgrep",
  "lang": "rust",
  "split": "train",
  "query_id": "ripgrep:000017",
  "query": "where is the code that searches a single file",
  "query_lang": "en",
  "template": "doc-en",
  "candidate_rank": 3,
  "candidate_kind": "text match",
  "label": "B",
  "state": "<JSON-encoded string of the rendered state>",
  "questions": "<JSON-encoded {\"relevant\":{\"criteria\":{\"A\":\"yes, this location answers the query\",\"B\":\"no, this location does not answer the query\"},\"instructions\":\"Does this code location answer the repository search query?\",\"type\":\"choice\"}}>",
  "gold": "<JSON-encoded {\"relevant\":{\"label\":\"A\"|\"B\",\"probabilities\":{\"A\":1.0,\"B\":0.0}|{\"A\":0.0,\"B\":1.0}}}>"
}
```

- `state`, `questions` and `gold` are **strings containing JSON**, which is
  the `LocalLLaMA/typed-decisions` schema: the consumer calls `json.loads`
  on each. `json.loads(state)` yields the Python `str` of the rendered
  state.
- `questions` and `gold` are built with `serde_json::json!` from D1's
  constants. `query_id` is `"{repo}:{query_index:06}"`, where `query_index`
  counts accepted queries of the repo from 0. `candidate_rank` is 1-based.
  `id` is `"{query_id}:{candidate_rank:02}"`.
- `<out>/manifest.json` is pretty JSON with no timestamps, containing:
  - `datagen_version` (`env!("CARGO_PKG_VERSION")`), `judge_state_version`
    (1), `seed`;
  - `corpus_sha256` and `eval_repos_sha256` (hex SHA-256 of the file bytes);
  - `settings`: `max_queries_per_repo`, `max_negatives_per_query`, `top_k`
    (12), `fallback_confidence`, `early_exit_confidence`;
  - `repos`: one entry per repo with `name`, `rev`, `split`, `lang`,
    `status` (`ok`|`failed`), `error` (only when failed), `files`,
    `outline_errors`, `symbols_total`, `symbols_sampled`,
    `queries_generated`, `filtered_early_exit`, `filtered_fallback`,
    `queries_with_positive`, `queries_without_positive`, `rows_positive`,
    `rows_negative`, and `templates` (`{template_id: queries}`);
  - `totals`, summing the numeric fields per split.
- **Determinism:** identical inputs (corpus, checkouts, CBM version and seed)
  produce byte-identical `*.jsonl` and `manifest.json`.

#### D4.8 `stats`

`stats` reads the three JSONL files of `--data` and prints JSON to stdout:
`{"<split>": {"rows","positive","negative","queries","queries_with_positive","per_template","per_lang","per_query_lang"}}`.
`rows`, `positive`, `negative`, `queries` (distinct `query_id`) and
`queries_with_positive` are integers. `per_template`, `per_lang` and
`per_query_lang` map each template id, `lang` or `query_lang` value that
occurs to its **row** count.
It validates every row: all keys present, `label ∈ {A, B}`, and `gold`
consistent with `label`. It exits 1 on the first invalid row, naming the
file, the line and the problem.

### D5 — Python training tooling (`train/laya/`)

```
train/laya/
  pyproject.toml          # name="repo-explorer-laya-train", requires-python=">=3.10",
                          # dependencies=["laya==0.3.7", "numpy>=1.26"]  (torch/transformers via laya)
  README.md               # short pointer to the runbook docs/laya-judge-training.md (D6)
  judge_train/__init__.py
  judge_train/constants.py  # MAX_LEN=1024, HEAD_MAX_LEN=256, BASE_MODEL="convaiinnovations/laya",
                            # JUDGE_STATE_VERSION=1, QUESTION_ID="relevant", POSITIVE_KEY="A",
                            # LAYA_VERSION="0.3.7"
  judge_train/data.py     # load_rows(path) -> list[dict] with the D4.8 validation; NO torch/laya import at module level
  judge_train/metrics.py  # numpy only (D5.2)
  judge_train/logits.py   # raw_logits() and resolve_checkpoint() (D5.1); imports torch/laya inside functions only
  bakeoff.py  prepare.py  train_ddp.py  calibrate.py  evaluate.py
  tests/__init__.py  tests/test_metrics.py  tests/test_data.py   # unittest, numpy only
```

`judge_train/logits.py`:

- `resolve_checkpoint(name_or_dir) -> str` — returns the directory itself
  if it exists. Otherwise it calls `huggingface_hub.snapshot_download(name_or_dir, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])`,
  then `laya.agent._fix_tokenizer_config(dir)`, and returns the directory.
  This matches how `laya.Agent` resolves a repo-root checkpoint.
- `raw_logits(agent, states) -> numpy.ndarray[n, 2]` — the pre-temperature
  logits of the judge question for each state.
  - Build `q = agent._to_internal(question)` once from the D1 question.
  - Per state, call
    `seq, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])`.
  - Batch in chunks of 16 with `laya.common.collate_items([items], agent.tok.pad_token_id)`.
  - Run `agent.model(...)` under `torch.no_grad()`, with the same autocast
    rule as `Agent.system_one`: autocast with `agent.dtype` on CUDA only.
  - Return `logits[:, :2]` as float32.

  This avoids `Agent.predict`'s 4-decimal rounding of probabilities, which
  would turn confident rows into `log(0)`.

#### D5.1 Scripts

Every script uses `argparse`, prints a JSON summary to stdout, and imports
`laya` and `torch` inside `main()` only.

- **`bakeoff.py --data DIR --out FILE [--device cpu|cuda] [--limit N]`**
  - For each upstream checkpoint name in (`english`, `multilingual`,
    `typed-decisions`): create `laya.Router(max_loaded=1, device=…)` and
    predict every row of `DIR/test.jsonl` (or its first `N` rows) via
    `router.predict(json.loads(row["state"]), json.loads(row["questions"]), model=name)`.
  - Read `p = answers["relevant"]["probabilities"]["A"]`.
  - Write `{name: metrics(D5.2)}` to `FILE`. No gate.
- **`prepare.py --data DIR --split train|val|test --base BASE_MODEL --out FILE.pt`**
  - Resolve the base with `resolve_checkpoint(base)`, then load the
    tokenizer from `<dir>/tokenizer`, as the notebook does.
  - Build items with
    `build_sequence(tok, state, {"t":"choice","ins":instructions,"crit":criteria}, MAX_LEN, HEAD_MAX_LEN)`.
    It **must** use `MAX_LEN`/`HEAD_MAX_LEN`, not the base config's values:
    the notebook builds sequences at 512/192 and trains at 1024/256, and
    that mismatch is deliberately fixed here.
  - Each item has `target = [p_A, p_B]` from `gold`, `label = argmax`, and
    `qtype = QTYPES["choice"]`.
  - Drop and count rows whose `len(markers) != 2`.
  - Save with `torch.save(items, FILE)` and print
    `{"items": n, "dropped": m}`.
- **`train_ddp.py BASE_MODEL ITEMS_PT OUTPUT_DIR [--epochs 4] [--micro-batch 8] [--max-steps N] [--device cuda|cpu]`**
  - Derived from the `%%writefile /kaggle/working/train_ddp.py` cell of
    `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` at laya tag
    `v0.3.7`. Keep an attribution header: `# Derived from NandhaKishorM/laya
(Apache-2.0), notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb @ v0.3.7`.
  - Changes versus the notebook, and nothing else:
    - argument parsing uses `argparse` with the signature above, replacing
      the notebook's positional `sys.argv`;
    - `BASE_MODEL` goes through `resolve_checkpoint`, so a Hub id or a local
      directory both work; the notebook assumed a local directory;
    - items are read from `ITEMS_PT`;
    - `cfg["max_len"] = MAX_LEN` and `cfg["head_max_len"] = HEAD_MAX_LEN`;
    - `--epochs` replaces the notebook's epoch constant, and `--micro-batch`
      replaces its micro-batch constant (default 8, the notebook's value),
      so the maintainer can lower it on out-of-memory at 1024 tokens;
    - `--device cpu` is for smoke runs:
      - `backend = "gloo"`;
      - before `init_process_group`, set any missing `RANK=0`,
        `WORLD_SIZE=1`, `LOCAL_RANK=0`, `MASTER_ADDR=127.0.0.1` and
        `MASTER_PORT` (a free port from binding a socket to port 0), so the
        script runs without `torchrun`;
      - wrap the model as `DDP(model)` without `device_ids`;
      - no `torch.cuda` calls, no autocast, no `GradScaler`, fp32
        throughout.
    - `--max-steps` stops early after N optimizer steps;
    - temperature fitting is removed, **together with** the notebook's
      calibration hold-out slice. Training uses all items in `ITEMS_PT`,
      because 10a has a separate val split. The saved
      `rl_agent_config.json` has `temperature = [1.0, 1.0, 1.0]` and no
      `temperature_by_options`; `calibrate.py` owns calibration.
  - The output directory must load with `laya.Agent(OUTPUT_DIR)`: it holds
    `rl_agent_config.json` (with `max_len = 1024` and
    `head_max_len = 256`), `model.safetensors`, `tokenizer/` and `encoder/`.
  - Launch with
    `torchrun --standalone --nproc_per_node=<gpus> train/laya/train_ddp.py …`.
- **`calibrate.py --checkpoint DIR --data DATA_DIR [--device cpu|cuda]`**
  1. Require `DATA_DIR/manifest.json` to have `judge_state_version == 1`;
     otherwise exit with an error naming both versions.
  2. Load `laya.Agent(DIR, device=…)` and compute
     `L = raw_logits(agent, states)` for `DATA_DIR/val.jsonl`. Then
     `z = L[:, 0] - L[:, 1]` is the raw logit difference.
  3. Fit T by grid search over `T ∈ {0.50, 0.51, …, 5.00}`, minimizing the
     mean binary NLL of `sigmoid(z/T)` against the labels (`y = 1` for
     label `A`), with `p` clipped to `[1e-7, 1 − 1e-7]`. Ties go to the
     smaller T.
  4. Write `temperature[0] = T` into `DIR/rl_agent_config.json` (index 0 is
     `choice`) and delete the `temperature_by_options` key.
  5. Compute `p = sigmoid(z/T)` on val. Choose `select_threshold` = the
     smallest τ in {30, 35, …, 90} (percent) whose candidate-level precision
     is ≥ 0.90; if none qualifies, use 90.
  6. Write `DIR/repo_explorer_judge.json`:
     `{"judge_state_version": 1, "select_threshold": τ, "temperature_choice": T, "base_model": BASE_MODEL, "laya_version": laya.__version__, "data_manifest_sha256": <sha256 of DATA_DIR/manifest.json>, "val_metrics": M}`.

     `M` has exactly these keys, computed with `p` and τ: `rows`,
     `queries`, `auroc`, `ece`, `query_top1`, `none_rate`, `precision`,
     `recall` and `selected`, as defined in D5.2. `nan` is written as JSON
     `null`.
- **`evaluate.py --checkpoint DIR --data DATA_DIR --out FILE [--device …] [--baseline BAKEOFF_JSON]`**
  - Require `DIR/repo_explorer_judge.json` with `judge_state_version == 1`.
    Take τ from it.
  - Compute `z` with `raw_logits` on `DATA_DIR/test.jsonl`, and
    `p = sigmoid(z / T)`, where T is the checkpoint's clamped
    `temperature[0]`. Compute the D5.2 metrics with the same keys as `M`.
  - Add breakdowns by `template`, `lang` and `query_lang`. Copy the baseline
    numbers in when `--baseline` is given (reported only).
  - Print the gate table and exit 0 if every D5.3 threshold holds,
    otherwise 1.

#### D5.2 Metrics (`judge_train/metrics.py`, numpy only)

- `auroc(p, y)` — rank-based Mann–Whitney U with average ranks for ties;
  `nan` if only one class is present.
- `ece(p, y, bins=15)` — confidence `c = max(p, 1-p)`, correct =
  `(p >= 0.5) == y`, equal-width bins. The first bin includes 0; bins are
  `(lo, hi]`.
- `query_top1(rows, p)` — over queries with at least one positive: the
  fraction whose highest-p row is positive. Ties go to the lower
  `candidate_rank`.
- `none_rate(rows, p, tau)` — over queries without positives: the fraction
  with every `p < tau/100`.
- `select_metrics(p, y, tau)` — `{"precision","recall","selected"}` for
  `p >= tau/100`. Precision is `nan` when nothing is selected.
- `pick_threshold(p, y)` — the rule in `calibrate.py` step 4.

#### D5.3 Offline acceptance gate (enforced by `evaluate.py`, test split)

| Metric                                          | Threshold |
| ----------------------------------------------- | --------- |
| `query_top1`                                    | ≥ 0.90    |
| `auroc`                                         | ≥ 0.90    |
| `ece` (after calibration)                       | ≤ 0.10    |
| candidate-level precision at `select_threshold` | ≥ 0.85    |

### D6 — Docs, CI, housekeeping

- **`docs/laya-judge-training.md`** — the runbook, with the exact commands
  from the Acceptance criteria:
  - prerequisites: close Claude Code, because datagen shares the per-user
    `codebase-memory-mcp` daemon, and pass the managed binary path
    `~/.local/bin/codebase-memory-mcp`;
  - fetch → generate → stats → bakeoff → prepare → train → calibrate →
    evaluate;
  - the Kaggle variant: upload the `datagen` output directory as a Kaggle
    dataset, then run `prepare.py` and `train_ddp.py` in cells on
    `GPU T4 x2`.
- `train/laya/README.md` links to the runbook.
- Root `CLAUDE.md` Layout: add `crates/repo-explorer-datagen/` (dev-only
  training-data generator, not shipped) and `train/laya/` (Python fine-tune
  tooling).
- `crates/repo-explorer-agent/CLAUDE.md`: new section "Judge input (Stage
  10)" documenting `judge_input` and `retrieval_snapshot`, the
  train/serve-parity rule, and the `JUDGE_STATE_VERSION` bump rule.
- `crates/repo-explorer-datagen/CLAUDE.md` covers:
  - purpose, CLI and determinism guarantees;
  - the eval-exclusion rule;
  - the license allowlist;
  - "never add an LLM call".
- `CHANGELOG.md` `[Unreleased]` → `### Added`: "Training-data generator
  (`repo-explorer-datagen`, dev-only) and Laya fine-tune tooling
  (`train/laya/`) for the upcoming local candidate judge."
- `.gitignore`: add `train/laya/.venv/`, `train/laya/**/*.pt` and
  `train/laya/**/__pycache__/`.
- `.github/workflows/ci.yml`, `test-eval` job:
  - change `pip install pyyaml` to `pip install pyyaml numpy`;
  - add `python -m unittest discover -s train/laya/tests -t train/laya`;
  - add `python -m py_compile train/laya/bakeoff.py train/laya/prepare.py train/laya/train_ddp.py train/laya/calibrate.py train/laya/evaluate.py train/laya/judge_train/__init__.py train/laya/judge_train/constants.py train/laya/judge_train/data.py train/laya/judge_train/metrics.py train/laya/judge_train/logits.py`.

## Error handling

- **datagen, per-repo failures** (clone/fetch/checkout, wrong HEAD, index
  failure) mark the repo `failed` with its error in the manifest. The run
  continues, and the exit code is 1 at the end.
- **datagen, per-file failures** (outline error, unreadable file) skip the
  file, increment `outline_errors`, and never fail the repo.
- **datagen, snapshot and render** have no error paths of their own: a
  retrieval leg error degrades inside `pipeline::retrieve`, as it does at
  runtime, and render returns `None` bodies.
- **datagen, invalid corpus or eval file** fails before any work, with exit
  code 1 and a message naming the repo and field.
- **datagen, usage errors** exit 2 and print the usage text to stderr.
- **Python, invalid rows** make `data.load_rows` raise `ValueError` naming
  the file, the line number and the key. The scripts let it propagate
  (non-zero exit).
- **Python, version mismatches.** `calibrate.py` fails with a clear message
  if `DATA_DIR/manifest.json`'s `judge_state_version != 1`. `evaluate.py`
  fails if `DIR/repo_explorer_judge.json` is missing or its
  `judge_state_version != 1`.
- **Python, missing optional pieces.** `evaluate.py` never crashes on an
  empty breakdown bucket: it reports `null`.

## Testing strategy

Rust (CI, `cargo test --workspace`):

- **`core::judge`** — `judge_constants_are_pinned`.
- **`agent::judge_input`** — golden-string tests for `render_judge_state`:
  - known location with symbol, body and outline;
  - no symbol;
  - body `None` falling back to the snippet;
  - neither body nor snippet (`code: (unavailable)`);
  - body over 60 lines (truncated at 60);
  - a line over 200 chars, containing multi-byte chars (cut at 200
    `char`s);
  - a query containing `\n` and `\r\n`;
  - a Windows-style `\` path;
  - `line_end < line_start` (normalized).

  Plus a `render_judge_states` test with `MockMemoryBackend` and a temp repo
  (`test_support::temp_repo_with`): an unknown-location candidate gives
  `None`, the body window is `[start-3, end+5]` clamped at 1, and an outline
  is attached to both candidates of the same file.

- **`agent::snapshot`** — with mocks: early-exit route → `EarlyExit`;
  confidence below `fallback_confidence` → `Fallback`; otherwise `Verify`.
  The existing `early_exit_*` and `symbol_free_query_is_vetoed_from_early_exit`
  tests stay unmodified and green.
- **`datagen`** unit tests:
  - `words()` cases: `parse_finish_lenient`, `HTTPServer`,
    `getHTTPResponse2`, `__init__`, `kebab-case-name`;
  - doc extraction for `///`, `/** … */`, `#`, `//` with a `#[derive]` and
    an `@Override` above, and a Python docstring;
  - literal extraction, including an escaped quote and too-short or
    too-long literals;
  - the stoplist and short-name split for `a::b::Foo`, `pkg.Class$Inner` and
    `mod/fn`;
  - dir and file exclusion rules;
  - corpus validation: each invalid field, duplicate name, eval URL match
    after normalization, the self-repo suffix;
  - the labelling overlap rule, including edge-touching ranges;
  - row serialization, where `state`/`questions`/`gold` are JSON strings
    that `serde_json::from_str` parses back to the expected values;
  - a splitmix64 and FNV-1a known-answer test;
  - template weight selection being deterministic for a fixed seed.
- **datagen** integration test (`tests/generate.rs`), with `MockMemoryBackend`
  (outline + search legs) and `MockSearchBackend`, calling the library's
  `generate_repo` twice on a temp-dir fixture. The fixture is not a git
  repo, since the git check lives in `run_generate`, and it contains
  **exactly one** source file, because `MockMemoryBackend` returns the same
  outline for every path. Assert that the two runs' serialized rows are
  byte-identical, and that `RepoStats` matches the fixture's expected
  counts.

Python (CI, numpy only): `test_metrics.py` covers `auroc` on known vectors
(perfect = 1.0, inverted = 0.0, all ties = 0.5), `ece` on a hand-computed
example, `query_top1` tie-break, `none_rate`, `select_metrics` with an empty
selection, and `pick_threshold` including the fallback to 90. `test_data.py`
covers `load_rows` accepting a valid row and rejecting a missing key, a bad
label, and `gold` inconsistent with `label`.

Manual smoke on CPU (implementer, before the PR). Recorded in the PR body,
not in CI:

```bash
cargo run -p repo-explorer-datagen -- generate --corpus crates/repo-explorer-datagen/corpus.toml \
  --checkouts ~/corpus --out /tmp/judge-data-smoke --memory-command ~/.local/bin/codebase-memory-mcp \
  --only chi --max-queries-per-repo 10
cd train/laya && uv run python prepare.py --data /tmp/judge-data-smoke --split test \
  --base convaiinnovations/laya --out /tmp/items.pt \
  && uv run python train_ddp.py convaiinnovations/laya /tmp/items.pt /tmp/ckpt-smoke --device cpu --max-steps 2 \
  && uv run python -c "import laya; laya.Agent('/tmp/ckpt-smoke', device='cpu')"
```

For the smoke run only, `chi` is fetched first (`fetch --corpus … --checkouts ~/corpus`),
and its `test` split output is used as the `prepare.py` input.

## Acceptance criteria

Automated (must pass in CI):

1. `cargo fmt --check`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo build --workspace`
4. `cargo test --workspace` on ubuntu-latest and windows-latest, including
   every test named in the Testing strategy.
5. `cargo run -p repo-explorer-datagen -- --help` exits 0 and lists the
   three subcommands.
6. `cargo run -p repo-explorer-datagen -- fetch` (no flags) exits 2.
7. `python -m unittest discover -s train/laya/tests -t train/laya` passes
   with only `numpy` installed.
8. All `py_compile` steps from D6 pass.
9. `git diff --stat main...HEAD -- crates/repo-explorer-mcp crates/repo-explorer-llm crates/repo-explorer-memory crates/repo-explorer-search`
   shows no changes: the runtime is untouched apart from the agent-crate
   refactor and the additions in core and agent.

Manual gates (maintainer, a GPU host; commands are exact):

- **G1 — data.**
  ```bash
  cargo run --release -p repo-explorer-datagen -- fetch --corpus crates/repo-explorer-datagen/corpus.toml --checkouts ~/corpus
  cargo run --release -p repo-explorer-datagen -- generate --corpus crates/repo-explorer-datagen/corpus.toml \
    --checkouts ~/corpus --out ~/judge-data/v1 --memory-command ~/.local/bin/codebase-memory-mcp
  cargo run --release -p repo-explorer-datagen -- stats --data ~/judge-data/v1
  ```
  Pass when all of these hold:
  - at least 30 of the 36 repos have status `ok`;
  - train rows ≥ 25,000;
  - the positive share of train rows is in [0.12, 0.50];
  - in the `stats` output (`train.per_template`, row counts), each of
    `words-en`, `doc-en` and `literal-en` accounts for ≥ 3 % of train rows.
    `define-en` is exempt, because exact-symbol queries mostly take the
    early exit and produce few rows by design;
  - `train.per_query_lang` contains only `en`;
  - test `queries_with_positive` ≥ 1,000.
- **G2 — bake-off (recorded, no threshold).**
  `cd train/laya && uv run python bakeoff.py --data ~/judge-data/v1 --out ~/judge-data/v1/bakeoff.json --device cuda`
- **G3 — train and calibrate.**
  ```bash
  cd train/laya
  uv run python prepare.py --data ~/judge-data/v1 --split train --base convaiinnovations/laya --out ~/judge-data/v1/train.pt
  uv run torchrun --standalone --nproc_per_node=1 train_ddp.py convaiinnovations/laya ~/judge-data/v1/train.pt ~/laya-ckpt/v1 --epochs 4
  uv run python calibrate.py --checkpoint ~/laya-ckpt/v1 --data ~/judge-data/v1 --device cuda
  ```
- **G4 — offline gate.**
  `uv run python evaluate.py --checkpoint ~/laya-ckpt/v1 --data ~/judge-data/v1 --out ~/laya-ckpt/v1/eval.json --device cuda --baseline ~/judge-data/v1/bakeoff.json`
  must exit 0 (D5.3). On failure, iterate on the data or training knobs
  (epochs, `--max-negatives-per-query`, template weights). 10b's `laya`
  mode is not enabled by default until G4 passes.

## Risks

- **Template–reality mismatch.** Rule-made queries cover real phrasing only
  partly. Mitigations: four template families; the per-template breakdown;
  the 10b end-to-end eval as the real gate.
- **Label noise.** A candidate may answer the query without overlapping the
  ground truth (for example a caller, or a second implementation).
  Mitigations: accepted, since those count as hard negatives for the "where
  is X defined/handled" semantics; tracked through `query_top1` rather than
  row accuracy.
- **CBM determinism.** Outline or search ordering may change across CBM
  versions. Mitigations: the manifest records hashes and counts; the
  determinism guarantee is scoped to the same CBM version.
- **Large repos dominate.** Mitigations: the `max_queries_per_repo` cap and
  the sampling.
- **Drift of the notebook-derived trainer.** Mitigations: `laya==0.3.7`
  pinned, and the derivation source is recorded in the header.
- **The notebook trainer may not fit in 16 GB at 1024 tokens.** The
  notebook sets `max_len = 1024` in its training script, but builds its
  items at 512/192, so 1024-token memory use at micro-batch 8 is unproven.
  Mitigations: gradient checkpointing (kept from the notebook), and
  `--micro-batch` to lower the per-step batch on out-of-memory.

## Global Constraints

- All code, comments, docs and spec text are in English.
- Rust edition 2024, with the workspace version; `cargo fmt --check` and
  `cargo clippy --all-targets -- -D warnings` are clean.
- CI passes on ubuntu-latest and windows-latest.
- `repo-explorer-core` gains no dependency: no `serde_json`, no `anyhow`, no
  `rmcp`.
- No LLM or provider call exists anywhere in code or tooling added by this
  spec.
- `JUDGE_STATE_VERSION = 1`.
- The judge question is exactly D1's constants: id `relevant`, type
  `choice`, keys `A`/`B`, in that order.
- Laya sequence limits in `prepare.py` and `train_ddp.py`: `max_len = 1024`,
  `head_max_len = 256`. `calibrate.py` and `evaluate.py` use the
  checkpoint's own `cfg`, which `train_ddp.py` saved with those values.
  `bakeoff.py` uses each upstream checkpoint's own config.
- The base model is `convaiinnovations/laya` (repo-root checkpoint), and
  `laya` is pinned to `==0.3.7`.
- Corpus licenses are restricted to `MIT`, `Apache-2.0`, `BSD-2-Clause`,
  `BSD-3-Clause`, `ISC`, `MIT OR Apache-2.0` and `MIT OR Unlicense`.
- Repos listed in `eval/repos.toml`, and this repository, never enter
  generated data.
- Data is split by repository only.
- New Rust crates use only dependencies already present in `Cargo.lock`
  (the D4 list).
- Python CI tests need only `numpy` (no `torch`, no `laya`).
- `explore_repository` behaviour is unchanged, and every pre-existing test
  passes unmodified.

## Decisions & assumptions

1. One 2-option `choice` question per candidate, with neutral keys `A`/`B`.
2. The state is a plain-text string in the D2 v1 format, with the outline
   last because Laya truncates the tail.
3. Only Verify-stage snapshots become training rows, since that is where 10b
   calls the judge.
4. Unknown-location candidates are never labelled and never judged.
5. Hard labels (1.0/0.0). Calibration is a single `choice` temperature
   fitted on val by grid search.
6. Fine-tune from the Laya repo root at 1024/256, the recipe that produced
   `laya-typed-decisions`.
7. The corpus is the 36 pinned repos listed in D4.2: 6 language groups
   (TypeScript and JavaScript form one group) × (4 train, 1 val, 1 test).
8. Query templates are exactly the four English-only ids of D4.5
   (`define-en`, `words-en`, `doc-en`, `literal-en`).
9. One query per symbol, at most 400 queries per repo, and at most 4
   negatives per query (at most 2 when there is no positive).
10. Datagen is a dev-only workspace binary (`publish = false`) and is not a
    release artifact.
11. GPU training is a manual gate. The tooling must be smoke-testable on
    CPU.
12. Checkpoints stay local, with no Hub upload.
13. The offline gate thresholds are those of D5.3. Missing them blocks
    enabling `laya` mode by default, not 10b implementation.
14. Assumption: `codebase-memory-mcp`'s `get_file_outline` returns qualified
    names in `note` for all six languages. A language that yields no outline
    only lowers that repo's counts, which are visible in the manifest.
15. Assumption: the notebook's training script at laya `v0.3.7` saves a
    directory loadable by `laya.Agent` (`rl_agent_config.json`,
    `model.safetensors`, `tokenizer/`, `encoder/`), as the notebook's own
    evaluation cell does with `laya.Agent(OUTPUT_DIR)`.
