# Derived from NandhaKishorM/laya (Apache-2.0),
# notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb @ v0.3.7
# (the `%%writefile /kaggle/working/train_ddp.py` cell).
#
# Changes vs. the notebook (and nothing else):
#   * argparse replaces the positional sys.argv;
#   * BASE_MODEL goes through resolve_checkpoint (Hub id or local dir);
#   * items are read from ITEMS_PT (key "seq" from older prepare.py runs is accepted as "ids");
#   * cfg["max_len"]=MAX_LEN, cfg["head_max_len"]=HEAD_MAX_LEN;
#   * --epochs / --micro-batch replace the notebook constants;
#   * --device cpu smoke path (gloo, single rank without torchrun, no cuda/autocast/GradScaler, fp32);
#   * --max-steps stops early after N optimizer steps;
#   * temperature fitting and its calibration hold-out slice are removed
#     (calibrate.py owns calibration); saved temperature = [1.0, 1.0, 1.0],
#     no temperature_by_options.
"""Fine-tune Laya from a local JSONL-derived item set (RLCD DDP)."""

import argparse
import json
import os
import random
import socket
import time


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_model")
    ap.add_argument("items_pt")
    ap.add_argument("output_dir")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return ap.parse_args()


def collate_train_batch(items, pad_id):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def main():
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from laya.common import build_model, proper_reward  # laya 0.3.7

    from judge_train.constants import HEAD_MAX_LEN, MAX_LEN
    from judge_train.logits import resolve_checkpoint

    args = parse_args()
    cpu = args.device == "cpu"

    # Single-process launch support (smoke runs without torchrun).
    for key, value in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"),
                       ("MASTER_ADDR", "127.0.0.1")):
        os.environ.setdefault(key, value)
    os.environ.setdefault("MASTER_PORT", _free_port())

    dist.init_process_group("gloo" if cpu else "nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    model_dir = resolve_checkpoint(args.base_model)
    output_dir = args.output_dir

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = MAX_LEN
    cfg["head_max_len"] = HEAD_MAX_LEN

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))

    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    if cpu:
        ddp_model = DDP(model, find_unused_parameters=True)
    else:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_items = torch.load(args.items_pt, weights_only=False)
    for it in all_items:
        if "ids" not in it and "seq" in it:
            it["ids"] = it.pop("seq")
    my_items = all_items[rank::world_size]

    EPOCHS = args.epochs
    MICRO_BATCH = args.micro_batch  # sequences per forward pass per GPU
    GRAD_ACCUM = 4       # Effective batch = MICRO_BATCH * world_size * 4
    GROUP_SIZE = 4       # GRPO baseline samples
    LR_ENCODER = 2.5e-5  # Encoder adaptation rate
    LR_HEAD = 1.0e-4     # Head adaptation rate
    SIGMA_START = 0.4    # Exploration noise
    SIGMA_END = 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]

    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=0.01)

    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=not cpu)

    if rank == 0:
        print(f"Starting DDP training: {len(all_items)} train items | {len(my_items)} per rank | "
              f"{EPOCHS} epochs | world {world_size} | device {device}", flush=True)
    t0 = time.time()
    updates = 0
    stop = False

    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            with torch.autocast("cuda", dtype=torch.float16, enabled=not cpu):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device)
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Evaluate proper scoring reward (w_sph=0.75 for soft target matching)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient loss + full 1.0 soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler skips optimizer.step() on inf/NaN grads and shrinks the
                # scale; only advance the LR schedule when a real step happened.
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if args.max_steps is not None and updates >= args.max_steps:
                    stop = True

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | Loss: {loss.item()*GRAD_ACCUM:.4f} | "
                      f"Reward: {r.mean().item():.3f} | LR: {cur_lr:.2e}", flush=True)
            if stop:
                break

        if rank == 0:
            print(f"=== Epoch {epoch+1}/{EPOCHS} Completed in {time.time()-t0:.1f}s | "
                  f"Avg Loss: {epoch_loss/max(1, n_batches):.4f} ===", flush=True)

        dist.barrier()

        # Overwrite a single rolling checkpoint after each epoch so a crash,
        # OOM, or session timeout doesn't lose all prior training.
        if rank == 0 and not stop:
            ckpt_dir = os.path.join(output_dir, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(ckpt_sd, os.path.join(ckpt_dir, "model.safetensors"))
            model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "total_epochs": EPOCHS,
                    "avg_loss": epoch_loss / max(1, n_batches)
                }, f, indent=2)
            print(f"  Saved rolling checkpoint (epoch {epoch+1}/{EPOCHS}) to {ckpt_dir}", flush=True)
        if stop:
            break

    dist.barrier()

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
        tok.save_pretrained(os.path.join(output_dir, "tokenizer"))

        cfg["fine_tuned"] = True
        cfg["model_name"] = "repo-explorer-judge"
        cfg["temperature"] = [1.0, 1.0, 1.0]
        cfg.pop("temperature_by_options", None)
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Model saved to {output_dir} ({updates} optimizer updates)", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
