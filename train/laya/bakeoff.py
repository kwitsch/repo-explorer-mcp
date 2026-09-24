"""G2 bake-off: score each upstream Laya checkpoint on the test split (no gate)."""

import argparse
import json
import math
import os


def main():
    import laya
    from judge_train import data, metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = data.load_rows(os.path.join(args.data, "test.jsonl"))
    if args.limit is not None:
        rows = rows[: args.limit]

    result = {}
    for name in ("english", "multilingual", "typed-decisions"):
        router = laya.Router(max_loaded=1, device=args.device)
        probs, labels = [], []
        for row in rows:
            res = router.predict(json.loads(row["state"]), json.loads(row["questions"]), model=name)
            probs.append(res["answers"]["relevant"]["probabilities"]["A"])
            labels.append(1 if row["label"] == "A" else 0)
        result[name] = _row_metrics(rows, probs, labels, metrics)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, indent=2)
    print(json.dumps({name: _json_safe(m["auroc"]) for name, m in result.items()}))


def _row_metrics(rows, probs, labels, metrics):
    import numpy as np

    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    tau = metrics.pick_threshold(p, y)
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
        "select_threshold": tau,
    }


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
