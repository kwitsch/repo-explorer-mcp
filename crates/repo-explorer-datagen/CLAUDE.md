# repo-explorer-datagen (dev-only training-data generator)

Builds labelled judge rows for the Stage-10 Laya judge from a pinned public-repo
corpus. A library plus a thin binary; **not shipped** — `release.yml` builds
`--package repo-explorer-mcp` only, and this crate is `publish = false`.

## CLI

- `fetch --corpus <file> --checkouts <dir>` — clone/fetch/checkout each corpus
  repo at its pinned rev; require a root license file.
- `generate --corpus <file> --checkouts <dir> --out <dir> --memory-command <path>`
  `[--memory-arg <arg>]... [--eval-repos <file>] [--max-queries-per-repo <n>]`
  `[--max-negatives-per-query <n>] [--seed <u64>] [--only <name>]...` — emit
  `train/val/test.jsonl` + `manifest.json`.
- `stats --data <dir>` — validate and summarise the JSONL splits.
- Exit codes: `0` ok, `1` invalid corpus/config or ≥1 repo failed (partial
  output still written), `2` usage error.

## Determinism

Identical inputs (corpus, checkouts, CBM version, seed) produce byte-identical
`*.jsonl` and `manifest.json`. All randomness flows through
`rng::{splitmix64, fnv1a64, Rng}` seeded from the run seed; maps that reach
output are `BTreeMap`s; `manifest.json` carries no timestamps.

## Hard rules

- **Never add an LLM or provider call.** Labels are derived programmatically
  from the known ground-truth symbol location; there is no teacher model.
- **Eval exclusion.** Every corpus URL is checked (in both `fetch` and
  `generate`) against `eval/repos.toml` after URL normalization, and any URL
  ending in `/repo-explorer-mcp` (this repository) is rejected.
- **License allowlist.** Only `MIT`, `Apache-2.0`, `BSD-2-Clause`,
  `BSD-3-Clause`, `ISC`, `MIT OR Apache-2.0`, `MIT OR Unlicense`.
- **Train/serve parity.** The judge state is rendered by
  `repo_explorer_agent::judge_input::render_judge_state`; never re-implement it
  here. Bump `repo_explorer_core::judge::JUDGE_STATE_VERSION` on any state change.
