# Laya judge training runbook (Stage 10a)

This runbook turns the pinned public-repo corpus into labelled judge data and a
fine-tuned, calibrated Laya checkpoint. GPU training is a manual maintainer gate;
the tooling is smoke-testable on CPU.

## Prerequisites

- **Close Claude Code first.** `repo-explorer-datagen generate` shares the
  per-user `codebase-memory-mcp` daemon; a running editor session competes for
  it. Pass the managed binary path explicitly:
  `--memory-command ~/.local/bin/codebase-memory-mcp`.
- Rust toolchain (edition 2024). Python ≥ 3.10 with `uv`. GPU steps need
  `laya==0.3.7` (pulls torch/transformers).

## 1. Fetch the corpus

```bash
cargo run --release -p repo-explorer-datagen -- fetch \
  --corpus crates/repo-explorer-datagen/corpus.toml --checkouts ~/corpus
```

## 2. Generate data

```bash
cargo run --release -p repo-explorer-datagen -- generate \
  --corpus crates/repo-explorer-datagen/corpus.toml \
  --checkouts ~/corpus --out ~/judge-data/v1 \
  --memory-command ~/.local/bin/codebase-memory-mcp
```

Outputs `train.jsonl`, `val.jsonl`, `test.jsonl` and `manifest.json` under
`~/judge-data/v1`. Identical inputs (corpus, checkouts, CBM version, seed)
produce byte-identical files.

## 3. Inspect

```bash
cargo run --release -p repo-explorer-datagen -- stats --data ~/judge-data/v1
```

## 4. Bake-off (recorded, no gate)

```bash
cd train/laya
uv run python bakeoff.py --data ~/judge-data/v1 --out ~/judge-data/v1/bakeoff.json --device cuda
```

## 5. Prepare, train, calibrate

```bash
cd train/laya
uv run python prepare.py --data ~/judge-data/v1 --split train --base convaiinnovations/laya --out ~/judge-data/v1/train.pt
uv run torchrun --standalone --nproc_per_node=1 train_ddp.py convaiinnovations/laya ~/judge-data/v1/train.pt ~/laya-ckpt/v1 --epochs 4
uv run python calibrate.py --checkpoint ~/laya-ckpt/v1 --data ~/judge-data/v1 --device cuda
```

## 6. Offline gate

```bash
cd train/laya
uv run python evaluate.py --checkpoint ~/laya-ckpt/v1 --data ~/judge-data/v1 \
  --out ~/laya-ckpt/v1/eval.json --device cuda --baseline ~/judge-data/v1/bakeoff.json
```

Exits 0 only when `query_top1 ≥ 0.90`, `auroc ≥ 0.90`, `ece ≤ 0.10`, and
candidate-level precision at `select_threshold ≥ 0.85`. On failure, iterate on
the data or training knobs (`--epochs`, `--max-negatives-per-query`, template
weights). 10b's `laya` mode stays off by default until this gate passes.

## CPU smoke (before a PR)

```bash
cargo run -p repo-explorer-datagen -- fetch --corpus crates/repo-explorer-datagen/corpus.toml --checkouts ~/corpus
cargo run -p repo-explorer-datagen -- generate --corpus crates/repo-explorer-datagen/corpus.toml \
  --checkouts ~/corpus --out /tmp/judge-data-smoke --memory-command ~/.local/bin/codebase-memory-mcp \
  --only chi --max-queries-per-repo 10
cd train/laya && uv run python prepare.py --data /tmp/judge-data-smoke --split test \
  --base convaiinnovations/laya --out /tmp/items.pt \
  && uv run python train_ddp.py convaiinnovations/laya /tmp/items.pt /tmp/ckpt-smoke --device cpu --max-steps 2 \
  && uv run python -c "import laya; laya.Agent('/tmp/ckpt-smoke', device='cpu')"
```

## Kaggle 2×T4 variant

Upload the `datagen` output directory as a Kaggle dataset, then run `prepare.py`
and `train_ddp.py` in cells on `GPU T4 x2`:

```python
!torchrun --standalone --nproc_per_node=2 train_ddp.py convaiinnovations/laya /kaggle/working/train.pt /kaggle/working/ckpt --epochs 4
```

Lower `--micro-batch` if training OOMs at 1024 tokens.
