"""Offline acceptance gate (D5.3) on the test split."""

import argparse
import json
import math
import os
import sys


def main():
    import numpy as np
    import laya

    from judge_train import data, metrics
    from judge_train.logits import raw_logits

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()

    judge_path = os.path.join(args.checkpoint, "repo_explorer_judge.json")
    if not os.path.isfile(judge_path):
        raise SystemExit(f"missing {judge_path}; run calibrate.py first")
    with open(judge_path, encoding="utf-8") as fh:
        judge = json.load(fh)
    if judge.get("judge_state_version") != 1:
        raise SystemExit(f"judge_state_version {judge.get('judge_state_version')} != 1")
    tau = judge["select_threshold"]

    agent = laya.Agent(args.checkpoint, device=args.device)
    t = _clamped_temperature(args.checkpoint)
    rows = data.load_rows(os.path.join(args.data, "test.jsonl"))
    states = [json.loads(r["state"]) for r in rows]
    y = np.array([1.0 if r["label"] == "A" else 0.0 for r in rows])
    logits = raw_logits(agent, states)
    z = logits[:, 0] - logits[:, 1]
    p = 1.0 / (1.0 + np.exp(-z / t))

    overall = _metrics(np, rows, p, y, tau, metrics)
    report = {
        "select_threshold": tau,
        "temperature_choice": t,
        "overall": overall,
        "by_template": _breakdown(np, rows, p, y, tau, "template", metrics),
        "by_lang": _breakdown(np, rows, p, y, tau, "lang", metrics),
        "by_query_lang": _breakdown(np, rows, p, y, tau, "query_lang", metrics),
    }
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as fh:
            report["baseline"] = json.load(fh)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2)

    gate = _gate(overall)
    _print_gate(gate)
    sys.exit(0 if all(g["pass"] for g in gate) else 1)


def _metrics(np, rows, p, y, tau, metrics):
    sel = metrics.select_metrics(p, y, tau)
    return {
        "rows": len(rows),
        "queries": len({r["query_id"] for r in rows}),
        "auroc": metrics.auroc(p, y),
        "ece": metrics.ece(p, y),
        "query_top1": metrics.query_top1(rows, p),
        "none_rate": metrics.none_rate(rows, p, tau),
        "precision": sel["precision"],
        "recall": sel["recall"],
        "selected": sel["selected"],
    }


def _breakdown(np, rows, p, y, tau, key, metrics):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row[key], []).append(i)
    out = {}
    for value, idxs in sorted(groups.items()):
        sub_rows = [rows[i] for i in idxs]
        sub_p = np.asarray([p[i] for i in idxs], dtype=float)
        sub_y = np.asarray([y[i] for i in idxs], dtype=float)
        out[value] = _metrics(np, sub_rows, sub_p, sub_y, tau, metrics)
    return out


def _clamped_temperature(checkpoint):
    with open(os.path.join(checkpoint, "rl_agent_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    temps = cfg.get("temperature") or [1.0]
    return min(5.0, max(0.50, float(temps[0])))


def _ok(value, cmp, threshold):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return cmp(value, threshold)


def _gate(overall):
    return [
        {"metric": "query_top1", "value": overall["query_top1"], "threshold": ">= 0.90",
         "pass": _ok(overall["query_top1"], lambda a, b: a >= b, 0.90)},
        {"metric": "auroc", "value": overall["auroc"], "threshold": ">= 0.90",
         "pass": _ok(overall["auroc"], lambda a, b: a >= b, 0.90)},
        {"metric": "ece", "value": overall["ece"], "threshold": "<= 0.10",
         "pass": _ok(overall["ece"], lambda a, b: a <= b, 0.10)},
        {"metric": "precision@select", "value": overall["precision"], "threshold": ">= 0.85",
         "pass": _ok(overall["precision"], lambda a, b: a >= b, 0.85)},
    ]


def _print_gate(gate):
    print(f"{'metric':<17} {'value':<10} {'threshold':<10} pass")
    for g in gate:
        v = g["value"]
        vs = "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
        print(f"{g['metric']:<17} {vs:<10} {g['threshold']:<10} {'PASS' if g['pass'] else 'FAIL'}")


def _json_safe(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
