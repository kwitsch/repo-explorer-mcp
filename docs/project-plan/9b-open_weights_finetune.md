# Stage 9b — Alternative: Fine-Tuning Open Weights as a Self-Hosted Upstream

## Goal

Same goal as Stage 9 (self-hosted upstream, no data leaves the own
infrastructure), but instead of training from scratch, an open base
model is specialized for the two pipeline roles (verify turn, fallback
loop). Data sovereignty is preserved identically — only the **weights**
come from a third party; queries/snippets still never go out.

## Comparison to Stage 9

|                         | Stage 9 (from scratch)   | Stage 9b (fine-tune)                              |
| ----------------------- | ------------------------ | ------------------------------------------------- |
| Pretraining + tokenizer | own (~10–20 days 8×H100) | not needed                                        |
| Code/language ability   | must be learned          | inherited from the base model (incl. stronger DE) |
| Weight provenance       | fully own                | Apache-2.0 base + own deltas                      |
| Cost                    | ~$15–40k                 | **~$0.5–2k**                                      |
| Calendar time           | ~3–4 months              | **~4–6 weeks**                                    |
| Expected quality        | uncertain, small budget  | higher, thanks to a strong code base              |

Reused unchanged from Stage 9: generator harness (Phase C), RLVR setup
(Phase D), eval harness, serving/integration, acceptance criteria.

## Base-model selection

Hard criteria: **Apache-2.0 license** (no usage restrictions like the
Llama/Gemma terms), self-hostable size (≤ 8B), strong at code, tool-call
pretrained, GGUF/vLLM support.

| Candidate                        | License                 | Note                                                                              |
| -------------------------------- | ----------------------- | --------------------------------------------------------------------------------- |
| **Qwen3-4B** (recommendation)    | Apache-2.0              | current generation, native tool calling, 32k+ context, good DE                    |
| Qwen2.5-Coder-7B                 | Apache-2.0              | strongest code specialization; note: the 3B variant is research-only — do not use |
| Qwen3-1.7B                       | Apache-2.0              | minimal/CPU variant, as a draft candidate                                         |
| IBM Granite-Code / StarCoder2-7B | Apache-2.0 / OpenRAIL-M | fallback; StarCoder2 only after a license review (RAIL restrictions)              |

Selection not by gut feeling but by a **bake-off**: run all candidates
zero-shot through the eval harness (Stage 9, held-out repos) — the
baseline measured before any training decides which model gets tuned.

## Training plan

### Phase 1 — SFT on procedural trajectories

- Data: the Stage 9 Phase-C generator, unchanged (verify examples, loop
  trajectories, format hardening), but rendered into the base model's
  **chat/tool template** instead of into custom special tokens — no
  tokenizer intervention.
- Scope: 200k–1M examples are enough with a strong base (instead of
  1–5M).
- Method: **full-parameter SFT** at ≤ 4B (hours on 8×H100, no LoRA
  compromises needed); LoRA/QLoRA only as a budget fallback at 7B.
- Framework: Axolotl or TRL; 1–3 epochs, early stopping on held-out
  verify accuracy.
- Important: mix in 5–10% generic code/instruction data (Apache-licensed)
  to avoid catastrophic forgetting of the code base.

### Phase 2 — RLVR (identical to Stage 9 Phase D)

- Environment = the real `AgentLoop` against sandboxed repos, reward
  computed programmatically (ground-truth hit, tool-JSON validity, batch
  enforcement, budget), GRPO via TRL/verl.
- With a strong base, optionally: check the eval gate first — if SFT
  alone already reaches the target, RL is dropped entirely.

### Phase 3 — Compression (optional)

- Quantization Q8/Q5 (GGUF) or AWQ (vLLM); repeat the eval gate after
  quantization — acceptance applies to the shipped artifact, not to the
  FP16 weights.

## Compute & cost

| Item                                       | Effort                           |
| ------------------------------------------ | -------------------------------- |
| Bake-off (inference, 4 candidates)         | < $100                           |
| Full SFT Qwen3-4B, ~500k examples          | ~8×H100 for 6–12h ≈ $200–400     |
| RLVR ~50k episodes                         | ~8×H100 for 1–2 days ≈ $500–1000 |
| 2–3 iterations of data mix/hyperparameters | ×2                               |
| **Total**                                  | **~$0.5–2k**                     |

## Serving & integration

Unchanged from Stage 9: GGUF/vLLM, grammar-constrained decoding for the
tool JSON, one `llm.providers` entry with a local `base_url`, zero Rust
changes. Qwen3-4B @ Q8 ≈ 5 GB VRAM + KV cache.

## Phase plan

| #   | Milestone                                               | Duration  |
| --- | ------------------------------------------------------- | --------- |
| 1   | Generator harness (Stage 9 Phase C — shared groundwork) | 2–3 weeks |
| 2   | Bake-off of the base models via the eval harness        | 3–5 days  |
| 3   | SFT iterations to the eval plateau                      | 1–2 weeks |
| 4   | RLVR (if the eval gate isn't reached after SFT)         | 1 week    |
| 5   | Quantization, shadow operation, cutover                 | 1 week    |

## Risks

- **Weight provenance**: the base's training data is not auditable
  (memorization of others' code is possible). Mitigation: Apache-2.0
  base, output used only internally as a retrieval selector — the model
  produces no shipped code artifacts.
- **Upstream dependency**: the base model can be deprecated → archive the
  weights and exact revision locally; the training recipe is
  reproducible, switching the base only costs phases 2–4.
- **Template drift**: the base model's chat template must match exactly
  between training and serving (the most common failure mode for
  tool-calling fine-tunes) — a template round-trip check in CI.
- License review per candidate before the bake-off (in particular, no
  research-only variants like Qwen2.5-Coder-3B).

## Dependencies

Stage 8 (the pipeline as data generator and eval harness); Stage 9 Phase
C/D as shared building blocks — Stage 9 itself is **not** a prerequisite.

## Acceptance criteria

Identical to Stage 9: ≥ 95% of the cloud reference on verify accuracy and
end-to-end exact match on held-out repos, 100% tool-call validity with
grammar-constrained decoding, proven zero outbound network traffic —
additionally: the eval gate applies to the quantized serving artifact.
