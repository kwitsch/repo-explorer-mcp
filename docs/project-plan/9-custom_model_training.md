# Stage 9 — Training a Specialized LLM as a Self-Hosted Upstream

## Goal

A model trained from scratch (no distillation, no fine-tune of someone
else's weights), small and specialized, that fully takes over both LLM
roles of the pipeline (Stage 8) and runs self-hosted — no query, no
snippet leaves the own infrastructure.

## Requirements profile (derived from Stage 8)

The model does **not** need to chat freely, code, or supply world
knowledge. It has exactly two jobs:

1. **Verification turn** (`agent::verify`): query + top-k skeletons →
   exactly one tool call (`expand` or `finish`, last turn forces
   `finish`). At its core a selection/ranking problem with structured
   output.
2. **Explorative fallback loop** (Stage 5, hardened): multi-turn tool
   calling with batch enforcement over the tool catalog (symbol lookup,
   grep, read), synthesis via `finish` under a token budget.

This implies:

- **Capabilities**: code understanding across common languages,
  query-to-code matching (EN + DE queries), reliable JSON tool calls,
  compact synthesis.
- **Context**: skeletons + snippets + findings → 16k target, 32k
  headroom.
- **Not needed**: dialogue ability, world knowledge, creative writing,
  math. This is exactly what makes a from-scratch training run at small
  scale realistic.

## Architecture

| Decision  | Choice                                                                                                                        | Rationale                                                         |
| --------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Type      | Decoder-only transformer                                                                                                      | Standard, best tooling support                                    |
| Size      | **1.5B** (base) / 3B (quality reserve) / 0.5B (draft experiment)                                                              | Quantized 1.5B runs on a consumer GPU, or is CPU-usable           |
| Attention | GQA (8 KV heads), RoPE, 8k pretrain context → 32k via RoPE scaling during mid-training                                        | Memory/inference cost                                             |
| Norm/Act  | RMSNorm, SwiGLU                                                                                                               | Llama family, supported out of the box by llama.cpp/vLLM          |
| Tokenizer | Own BPE, 64k vocabulary, trained on code + EN + DE; special tokens for the tool-call frame (`<call>`, `<args>`, `<finish>` …) | Code-efficient; native tool tokens instead of a prompt convention |

Important: keep the architecture strictly Llama-compatible so GGUF export
and vLLM serving work without a custom inference implementation.

## Data (without distillation)

Core idea: **the ground truth for this task is programmatically
derivable.** No teacher LLM is needed, because for real repos it can be
determined deterministically which symbol/file is the correct answer.

### Phase A — Pretraining corpus (~200–300B tokens)

- Permissively licensed code: The Stack v2 (deduplicated, license filter
  MIT/Apache/BSD), weighted toward the target languages (Rust, TS/JS,
  Python, Go, Java, C/C++, C#).
- Technical English: Wikipedia excerpt, StackExchange dumps,
  documentation corpora (Rust Book, MDN-style sources with a compatible
  license).
- German: small share (~2–3%) of technical German for DE queries.
- Mix roughly 70% code / 25% EN / 5% repo metadata + DE.

### Phase B — Mid-training (~10–20B tokens, task-adjacent)

Formatted material in exactly the pipeline's format:

- Repo-level packages: file tree + skeleton views (symbol name + line
  range, exactly the `agent::skeleton` rendering) + the associated full
  texts.
- Context extended to 32k during this phase.

### Phase C — SFT: procedurally generated trajectories (~1–5M examples)

Generator harness (a new internal tool that uses this codebase itself):

1. Clone a corpus of ~50–100k public repos (license filter).
2. **Generate queries without an LLM**: rule-based templates over real
   artifacts — doc comments ("Where is X validated?"), symbol names
   (paraphrase rules: snake_case → word sequence), commit messages,
   error-message literals, README sentences. Ground truth = the
   definition/location the query was derived from.
3. Run the **deterministic pre-stage** (Stage 8, `core::retrieval`) on
   every query → real candidate lists including confidence.
4. Build training examples from that:
   - **Verify examples**: candidates contain the ground truth → target is
     `finish` with the correct selection; ground truth only visible after
     `expand` → target is the correct `expand` call; ground truth is
     missing → target is escalation. Negative/distractor candidates come
     for free from the ranking.
   - **Loop trajectories**: deterministically construct optimal tool
     sequences (which grep/read batches minimally cover the ground
     truth), including the batch-enforcement format and forced-final
     `finish`.
   - **Format hardening**: broken/truncated contexts, empty legs, budget
     exhaustion → correct behavior (escalation, synthesis from findings).

### Phase D — RL with verifiable reward (RLVR)

The task is ideal for RL because the reward is computable without humans
and without a judge LLM:

- Environment = the real `AgentLoop` against sandboxed repos.
- Reward: ground-truth symbol/file hit in `finish` findings (+), valid
  tool JSON (+), token/turn budget respected (+), hallucinated paths (−),
  single call instead of a batch (−).
- Algorithm: GRPO (no value model, saves half the memory), ~50–100k
  episodes.

## Compute & cost (order of magnitude)

| Item                                   | 1.5B     | 3B        |
| -------------------------------------- | -------- | --------- |
| Pretraining FLOPs (6·N·D, 300B tokens) | ~2.7e21  | ~5.4e21   |
| 8×H100, ~40% MFU                       | ~10 days | ~20 days  |
| Rental cost (~$2.5/GPU-h)              | ~$5k     | ~$10k     |
| Mid-training + SFT + RL                | +20–30%  | +20–30%   |
| **Total (one iteration)**              | **~$7k** | **~$13k** |

Realistically plan for 2–3 full iterations (tokenizer/data-mix mistakes
only surface late) → budget frame **~$15–40k** in GPU rental. Framework:
torchtitan or Megatron-LM for pretraining, TRL/verl for SFT+GRPO.

## Evaluation

Eval harness = the pipeline itself, run over held-out repos (never seen
during training):

- **Verify accuracy**: correct candidate chosen (target > 95% when the
  ground truth is in the top-k).
- **Tool-call validity**: parseable, catalog-conformant call (target ≈
  100% — see grammar-constrained decoding below).
- **End-to-end exact match**: ground-truth location present in the
  findings, across the full pipeline path including the fallback loop.
- **Budget discipline**: turns/tokens to `finish` vs. the optimal
  trajectory.
- Reference line: measure the same metrics against the current cloud
  provider before switching over.

## Serving & integration

- Export: GGUF (llama.cpp/Ollama) and safetensors (vLLM).
- **Grammar-constrained decoding**: enforce the tool-call JSON via a GBNF
  grammar or vLLM guided decoding — takes almost the entire formatting
  burden off the small model.
- Integration costs **zero Rust code**: `genai` speaks OpenAI-compatible
  endpoints and Ollama; one `llm.providers` entry with `base_url` pointed
  at the local server is enough. Cloud providers can stay as a router
  fallback behind the local entry during the transition — or be dropped
  entirely for strict data sovereignty.
- Inference hardware: 1.5B @ Q8 ≈ 2 GB VRAM + KV cache; runs on any
  current consumer GPU, or on CPU if needed.

## Phase plan

| #   | Milestone                                                              | Duration (calendar)      |
| --- | ---------------------------------------------------------------------- | ------------------------ |
| 1   | Data pipeline: corpus ingest, dedupe, license filter, tokenizer        | 3–4 weeks                |
| 2   | Generator harness for Phase-C data (builds on this codebase)           | 2–3 weeks, parallel to 1 |
| 3   | 0.5B pilot run end-to-end (pretrain→SFT→eval) to validate the pipeline | 2 weeks                  |
| 4   | 1.5B main run + mid-training + SFT                                     | 3–4 weeks                |
| 5   | RLVR + eval gate against the cloud reference                           | 2–3 weeks                |
| 6   | Serving setup, config switch-over, shadow operation, cutover           | 1–2 weeks                |

## Risks

- **From scratch vs. fine-tune**: a deliberate choice for data sovereignty
  and license clarity of the weights; a fine-tune of open weights would
  be ~10–50× cheaper and likely stronger — an accepted trade-off,
  documented here only.
- Small model + free-form synthesis in the fallback loop is the weakest
  point → grammar-constrained decoding, forced-final `finish`, and the
  deterministic synthesis (Stage 8) structurally contain this; early
  exit keeps many queries LLM-free anyway.
- Corpus license diligence (permissive licenses only, dedupe against
  memorization) is a precondition, not a nice-to-have.
- Query templates only partially cover real user phrasing — mitigation:
  once in shadow operation, use real (local!) query logs as SFT
  follow-up material, still without a third-party LLM.

## Dependencies

Stage 8 (the pipeline as data generator and eval harness).

## Acceptance criteria

- Eval gate: the own model's verify accuracy and end-to-end exact match
  ≥ 95% of the cloud reference values on held-out repos.
- Tool-call validity 100% (with grammar-constrained decoding).
- Full pipeline run with zero outbound network traffic (proven via a
  network sandbox).
