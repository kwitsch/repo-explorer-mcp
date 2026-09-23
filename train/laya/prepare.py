"""Build a `torch.save` item list for training from a data split."""

import argparse
import json
import os


def main():
    import torch
    from transformers import AutoTokenizer
    from laya.common import QTYPES, build_sequence  # laya 0.3.7

    from judge_train import data
    from judge_train.constants import HEAD_MAX_LEN, MAX_LEN
    from judge_train.logits import resolve_checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    directory = resolve_checkpoint(args.base)
    tok = AutoTokenizer.from_pretrained(os.path.join(directory, "tokenizer"))

    rows = data.load_rows(os.path.join(args.data, f"{args.split}.jsonl"))
    items, dropped = [], 0
    for row in rows:
        state = json.loads(row["state"])
        question = json.loads(row["questions"])["relevant"]
        internal = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
        # MUST use MAX_LEN/HEAD_MAX_LEN (1024/256), not the base config's 512/192.
        seq, markers = build_sequence(tok, state, internal, MAX_LEN, HEAD_MAX_LEN)
        if len(markers) != 2:
            dropped += 1
            continue
        gold = json.loads(row["gold"])["relevant"]["probabilities"]
        target = [gold["A"], gold["B"]]
        label = 0 if gold["A"] >= gold["B"] else 1
        items.append({"seq": seq, "markers": markers, "target": target, "label": label, "qtype": QTYPES["choice"]})

    torch.save(items, args.out)
    print(json.dumps({"items": len(items), "dropped": dropped}))


if __name__ == "__main__":
    main()
