"""Fit a single `choice` temperature on val and pick a select threshold."""

import argparse
import hashlib
import json
import math
import os


def main():
    import numpy as np
    import laya

    from judge_train import data, metrics
    from judge_train.constants import BASE_MODEL
    from judge_train.logits import raw_logits

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    manifest_path = os.path.join(args.data, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("judge_state_version") != 1:
        raise SystemExit(f"data judge_state_version {manifest.get('judge_state_version')} != tool version 1")

    agent = laya.Agent(args.checkpoint, device=args.device)
    rows = data.load_rows(os.path.join(args.data, "val.jsonl"))
    states = [json.loads(r["state"]) for r in rows]
    y = np.array([1.0 if r["label"] == "A" else 0.0 for r in rows])
    logits = raw_logits(agent, states)
    z = logits[:, 0] - logits[:, 1]

    best_t = _fit_temperature(np, z, y)
    p = _sigmoid(np, z / best_t)
    tau = metrics.pick_threshold(p, y)
    _write_temperature(args.checkpoint, best_t)

    sel = metrics.select_metrics(p, y, tau)
    val_metrics = {
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
    judge = {
        "judge_state_version": 1,
        "select_threshold": tau,
        "temperature_choice": best_t,
        "base_model": BASE_MODEL,
        "laya_version": laya.__version__,
        "data_manifest_sha256": _sha256(manifest_path),
        "val_metrics": val_metrics,
    }
    with open(os.path.join(args.checkpoint, "repo_explorer_judge.json"), "w", encoding="utf-8") as fh:
        json.dump(_json_safe(judge), fh, indent=2)
    print(json.dumps({"temperature_choice": best_t, "select_threshold": tau}))


def _sigmoid(np, z):
    return 1.0 / (1.0 + np.exp(-z))


def _fit_temperature(np, z, y):
    best_t, best_nll = 0.50, float("inf")
    t = 0.50
    while t <= 5.0 + 1e-9:
        p = np.clip(_sigmoid(np, z / t), 1e-7, 1 - 1e-7)
        nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if nll < best_nll - 1e-12:  # ties -> smaller T (loop ascends)
            best_nll, best_t = nll, t
        t = round(t + 0.01, 2)
    return best_t


def _write_temperature(checkpoint, t):
    path = os.path.join(checkpoint, "rl_agent_config.json")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    temps = cfg.get("temperature")
    if not isinstance(temps, list) or not temps:
        temps = [1.0, 1.0, 1.0]
    temps[0] = t  # index 0 is `choice`
    cfg["temperature"] = temps
    cfg.pop("temperature_by_options", None)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


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
