#!/bin/bash
uv run torchrun --standalone --nproc_per_node=1 train_ddp.py convaiinnovations/laya ~/judge-data/v1/train.pt ~/laya-ckpt/v1 --micro-batch 1 2>&1 | tee ~/laya-ckpt/train.log
