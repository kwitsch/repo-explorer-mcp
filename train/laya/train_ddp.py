# Derived from NandhaKishorM/laya (Apache-2.0),
# notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb @ v0.3.7
#
# Changes vs. the notebook (and nothing else):
#   * argparse replaces the positional sys.argv;
#   * BASE_MODEL goes through resolve_checkpoint (Hub id or local dir);
#   * items are read from ITEMS_PT;
#   * cfg["max_len"]=MAX_LEN, cfg["head_max_len"]=HEAD_MAX_LEN;
#   * --epochs / --micro-batch replace the notebook constants;
#   * --device cpu smoke path (gloo, single rank, no cuda/autocast/GradScaler, fp32);
#   * --max-steps stops early after N optimizer steps;
#   * temperature fitting and its calibration hold-out slice are removed
#     (calibrate.py owns calibration); saved temperature = [1.0, 1.0, 1.0],
#     no temperature_by_options.
"""Fine-tune Laya from a local JSONL-derived item set (RLCD DDP)."""

import argparse
import os
import socket


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


def main():
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    import laya
    from judge_train.constants import HEAD_MAX_LEN, MAX_LEN
    from judge_train.logits import resolve_checkpoint

    args = parse_args()
    cpu = args.device == "cpu"

    # Single-process launch support (smoke runs without torchrun).
    for key, value in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"),
                       ("MASTER_ADDR", "127.0.0.1")):
        os.environ.setdefault(key, value)
    os.environ.setdefault("MASTER_PORT", _free_port())

    backend = "gloo" if cpu else "nccl"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cpu") if cpu else torch.device(f"cuda:{local_rank}")
    if not cpu:
        torch.cuda.set_device(local_rank)

    directory = resolve_checkpoint(args.base_model)
    agent = laya.Agent(directory, device=str(device))  # laya 0.3.7
    agent.cfg["max_len"] = MAX_LEN
    agent.cfg["head_max_len"] = HEAD_MAX_LEN
    model = agent.model
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()  # kept from the notebook

    ddp_model = DDP(model) if cpu else DDP(model, device_ids=[local_rank])
    items = torch.load(args.items_pt)

    from laya.common import collate_items  # laya 0.3.7
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=2e-5)
    scaler = None if cpu else torch.cuda.amp.GradScaler()

    step = 0
    ddp_model.train()
    for _epoch in range(args.epochs):
        for start in range(0, len(items), args.micro_batch):
            micro = items[start:start + args.micro_batch]
            batch = collate_items(
                [(it["seq"], it["markers"]) for it in micro], agent.tok.pad_token_id
            )
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            targets = torch.tensor([it["target"] for it in micro], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            if cpu:
                loss = agent.rlcd_loss(ddp_model, batch, targets)  # laya 0.3.7 (fp32, no autocast)
                loss.backward()
                optimizer.step()
            else:
                with torch.autocast("cuda", dtype=agent.dtype):
                    loss = agent.rlcd_loss(ddp_model, batch, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        if args.max_steps is not None and step >= args.max_steps:
            break

    if int(os.environ["RANK"]) == 0:
        agent.cfg["temperature"] = [1.0, 1.0, 1.0]
        agent.cfg.pop("temperature_by_options", None)
        agent.save(args.output_dir)  # laya 0.3.7: rl_agent_config.json, model.safetensors, tokenizer/, encoder/

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
