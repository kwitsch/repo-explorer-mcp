"""Judge metrics, numpy only (no torch/laya)."""

import numpy as np


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auroc(p, y):
    """Rank-based Mann-Whitney U AUROC with average ranks for ties.

    Returns nan if only one class is present.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(p)
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(p, y, bins=15):
    """Expected calibration error. conf=max(p,1-p); equal-width bins (lo,hi];
    the first bin includes 0."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan")
    conf = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(float) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / n) * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def _group_by_query(rows):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["query_id"], []).append(i)
    return groups


def query_top1(rows, p):
    """Fraction of queries-with-a-positive whose highest-p row is positive.
    Ties resolve to the lower candidate_rank. nan if there are none."""
    p = np.asarray(p, dtype=float)
    total = 0
    hits = 0
    for _, idxs in _group_by_query(rows).items():
        if not any(rows[i]["label"] == "A" for i in idxs):
            continue
        total += 1
        best = min(idxs, key=lambda i: (-p[i], int(rows[i]["candidate_rank"])))
        if rows[best]["label"] == "A":
            hits += 1
    return hits / total if total else float("nan")


def none_rate(rows, p, tau):
    """Over queries WITHOUT a positive: the fraction whose every row has p < tau/100."""
    p = np.asarray(p, dtype=float)
    thr = tau / 100.0
    total = 0
    hits = 0
    for _, idxs in _group_by_query(rows).items():
        if any(rows[i]["label"] == "A" for i in idxs):
            continue
        total += 1
        if all(p[i] < thr for i in idxs):
            hits += 1
    return hits / total if total else float("nan")


def select_metrics(p, y, tau):
    """{precision, recall, selected} for p >= tau/100. Precision is nan when
    nothing is selected."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    thr = tau / 100.0
    selected = p >= thr
    n_sel = int(selected.sum())
    tp = float((selected & (y == 1)).sum())
    n_pos = float((y == 1).sum())
    return {
        "precision": tp / n_sel if n_sel > 0 else float("nan"),
        "recall": tp / n_pos if n_pos > 0 else float("nan"),
        "selected": n_sel,
    }


def pick_threshold(p, y):
    """Smallest tau in {30,35,...,90} whose candidate-level precision >= 0.90;
    90 if none qualifies."""
    for tau in range(30, 91, 5):
        precision = select_metrics(p, y, tau)["precision"]
        if not np.isnan(precision) and precision >= 0.90:
            return tau
    return 90
