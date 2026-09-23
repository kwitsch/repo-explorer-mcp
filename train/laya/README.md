# repo-explorer-laya-train

Python tooling to fine-tune and calibrate a local Laya judge checkpoint from
`repo-explorer-datagen` output (Stage 10a). Pinned to `laya==0.3.7`; CI runs
only the `numpy`-only unit tests under `tests/`.

Full runbook — fetch → generate → stats → bakeoff → prepare → train → calibrate
→ evaluate, plus the Kaggle 2×T4 variant — is in
[`docs/laya-judge-training.md`](../../docs/laya-judge-training.md).
