"""Export golden parity fixtures next to a Laya checkpoint (dev-only).

Usage:
  export_golden.py --checkpoint DIR --data DATA_DIR [--n 50] [--out DIR/golden.jsonl]

Writes golden.jsonl: a header line then one line per state with the exact
token ids, markers, pre-temperature logits and p_relevant computed from the raw
logits (never Agent.predict, which rounds to 4 decimals).
"""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def judge_question():
    # Mirror repo_explorer_core::judge constants.
    return {
        "type": "choice",
        "instructions": "Does this code location answer the repository search query?",
        "criteria": {
            "A": "yes, this location answers the query",
            "B": "no, this location does not answer the query",
        },
    }


def pick_temperature(agent, question_key="choice:2"):
    by = getattr(agent, "temperature_by_options", {}) or {}
    if question_key in by:
        t = by[question_key]
    else:
        t = agent.temperature[0]
    try:
        t = float(t)
    except (TypeError, ValueError):
        t = 1.0
    if not (t == t) or t in (float("inf"), float("-inf")):
        t = 1.0
    return min(5.0, max(0.5, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import laya
    from laya.common import QTYPES, build_sequence, collate_items  # laya 0.3.7

    out_path = args.out or os.path.join(args.checkpoint, "golden.jsonl")
    agent = laya.Agent(args.checkpoint, device="cpu")

    states = []
    test_path = os.path.join(args.data, "test.jsonl")
    with open(test_path) as f:
        for line in f:
            if len(states) >= args.n:
                break
            row = json.loads(line)
            states.append(json.loads(row["state"]))
    long_state = "query: x\ncandidate: a.rs:1-2 (text match)\ncode:\n" + "  let v = 1;\n" * 400
    states.append(long_state)
    states.append("query: x\n[MASK] and again [MASK]")
    states.append("query: x")

    question = agent._to_internal(judge_question())
    t = pick_temperature(agent)
    header = {
        "header": {
            "checkpoint_sha256": sha256_file(os.path.join(args.checkpoint, "model.safetensors")),
            "laya_version": laya.__version__,
            "judge_state_version": 1,
            "n": len(states),
        }
    }
    with open(out_path, "w") as out:
        out.write(json.dumps(header) + "\n")
        for state in states:
            seq, markers = build_sequence(
                agent.tok, state, question, agent.cfg["max_len"], agent.cfg["head_max_len"]
            )
            item = {"ids": seq, "markers": markers, "qtype": QTYPES["choice"]}
            b = collate_items([[item]], agent.tok.pad_token_id)
            with torch.no_grad():
                raw, _act = agent.model(
                    b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
                )
            logits = raw[0, :2].to(torch.float32).tolist()
            import math
            a, b = logits[0] / t, logits[1] / t
            m = max(a, b)
            ea, eb = math.exp(a - m), math.exp(b - m)
            p_relevant = ea / (ea + eb)
            out.write(
                json.dumps(
                    {
                        "state": state,
                        "input_ids": list(seq),
                        "markers": list(markers),
                        "logits": [logits[0], logits[1]],
                        "p_relevant": p_relevant,
                    }
                )
                + "\n"
            )
    print(f"[export_golden] wrote {len(states)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
