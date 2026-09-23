# repo-explorer-core (domain logic)

Domain logic (lib), no MCP/transport concerns: the pure retrieval logic
(`retrieval`: pattern derivation, candidate ranking, confidence) and the
`RepoStateProbe` fingerprint trait.

## Serde in `domain`

`FileLocation`, `ExplorationFinding`, `ExplorationResult` and the
`ExplorationOutcome`/`StageExit` wrapper around them are the only domain types
with serde derives, and for one reason only: the agent crate's persistent
result cache writes them to disk (`repo_explorer_agent::disk_cache::
SCHEMA_VERSION` versions that shape in exactly one place). `ExplorationOutcome`
is built once, at the end of `AgentLoop::run`; every backend leg keeps passing
the bare `ExplorationResult`, for which `retrieval_confidence`/`stage_exit`/
`symbols` are meaningless. JSON-Schema generation stays at the MCP boundary —
`schemars` is owned by `repo-explorer-mcp`, and this crate has no `serde_json`
(the `StageExit` wire strings are therefore pinned by a test in the agent
crate, `disk_cache::tests::stage_exit_wire_strings_are_pinned`).

## `judge` module (Stage 10)

`judge.rs` holds the fixed local-candidate-judge question as constants only
(`QUESTION_ID`, `QUESTION_INSTRUCTIONS`, `POSITIVE_KEY`/`POSITIVE_CRITERION`,
`NEGATIVE_KEY`/`NEGATIVE_CRITERION`) plus `JUDGE_STATE_VERSION`, which pins the
plain-text state format rendered by `repo_explorer_agent::judge_input::
render_judge_state`. **Bump `JUDGE_STATE_VERSION` on any change to that
rendering or to these constants** — a trained checkpoint is valid only for the
version it was trained on. See `crates/repo-explorer-agent/CLAUDE.md` (Judge
input, Stage 10) for how the datagen and runtime crates consume this module.

## Crate-boundary rules

- Must not add an `rmcp` dependency — `crates/repo-explorer-mcp` wires this
  crate's logic to the MCP protocol instead.
- Must not add an `anyhow` dependency — use `thiserror` typed errors
  (`ConfigError`, `ValidationError`) instead; see
  `crates/repo-explorer-mcp/CLAUDE.md` for how the binary boundary consumes
  them via `?`/`.context(...)`.
