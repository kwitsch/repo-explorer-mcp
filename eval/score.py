#!/usr/bin/env python3
"""Score a Layer A run (docs/eval/real-world-test-plan.md §7.1).

Run with: uv run --with pyyaml eval/score.py results/<run-id>

Computes the metrics that are derivable from the response + filesystem + Mode-B stderr parsing
today. Metrics that need the observability patch's structured lines (`cand_recall@top_k`,
`llm_calls`, `provider_events`, `model_served`, `forced_finish`) are left as `None` and flagged
in the summary rather than silently omitted — see the "not yet available (needs patch)" note in
the printed report.

Wilson score interval is used for all proportions (n is small in the pilot; a naive normal
approximation is misleading at n<30).
"""

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent

TRUNCATION_MARKER_RE = re.compile(r"\s*\.\.\.\s*\(truncated\)\s*$", re.I)
WS_RE = re.compile(r"\s+")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (p, max(0.0, lo), min(1.0, hi))


def load_queries(repo_id: str) -> dict[str, dict]:
    path = EVAL_DIR / "queries" / f"{repo_id}.yaml"
    with open(path) as f:
        items = yaml.safe_load(f) or []
    return {item["id"]: item for item in items}


def load_repos() -> dict[str, dict]:
    import tomllib

    with open(EVAL_DIR / "repos.toml", "rb") as f:
        data = tomllib.load(f)
    return {r["id"]: r for r in data["repo"]}


def normalize_snippet(s: str | None) -> str:
    if not s:
        return ""
    s = TRUNCATION_MARKER_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip()
    return s[:100]


def path_exists(repo_path: Path, rel: str) -> bool:
    return (repo_path / rel).exists()


def file_len(repo_path: Path, rel: str) -> int | None:
    p = repo_path / rel
    if not p.exists() or not p.is_file():
        return None
    try:
        return sum(1 for _ in open(p, "rb"))
    except OSError:
        return None


def snippet_found_at(repo_path: Path, rel: str, line_start: int | None, snippet: str) -> str:
    """Returns 'ok' (matches at/near the claimed range), 'misaligned' (found elsewhere in the
    file), or 'not_found' (fabricated) — the §7.1 hallucinated classification, minus the
    'fabricated path'/'range outside file' cases which are checked separately."""
    p = repo_path / rel
    if not p.exists() or not p.is_file():
        return "not_found"
    norm_target = normalize_snippet(snippet)
    if not norm_target:
        return "ok"  # nothing to check
    try:
        lines = p.read_text(errors="replace").splitlines()
    except OSError:
        return "not_found"
    near = set()
    if line_start:
        near = set(range(max(0, line_start - 4), min(len(lines), line_start + 3)))
    found_near = False
    found_anywhere = False
    for i, line in enumerate(lines):
        nline = normalize_snippet(line)
        if norm_target and norm_target in nline:
            found_anywhere = True
            if i in near:
                found_near = True
    if found_near:
        return "ok"
    if found_anywhere:
        return "misaligned"
    return "not_found"


def normalize_finding_path(raw: str) -> str:
    return raw[2:] if raw.startswith("./") else raw


def score_finding(finding: dict, repo_path: Path) -> dict:
    loc = finding.get("location", {})
    rel = normalize_finding_path(loc.get("path", ""))
    exists = path_exists(repo_path, rel)
    flen = file_len(repo_path, rel) if exists else None
    line_start = loc.get("line_start")
    line_end = loc.get("line_end")
    line_valid = None
    if exists and line_start is not None and line_end is not None and flen is not None:
        line_valid = 1 <= line_start <= line_end <= flen
    hallucination = None
    if not exists:
        hallucination = "fabricated_path"
    elif line_valid is False:
        hallucination = "range_outside_file"
    elif finding.get("snippet"):
        cls = snippet_found_at(repo_path, rel, line_start, finding["snippet"])
        if cls == "not_found":
            hallucination = "fabricated_snippet"
        elif cls == "misaligned":
            hallucination = "misaligned_snippet"
    return {
        "path": rel,
        "path_valid": exists,
        "line_valid": line_valid,
        "symbol_only": line_start is None,
        "hallucination": hallucination,
    }


def ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and b_start <= a_end


def range_len_ok(a_start, a_end, span_start, span_end) -> bool:
    span_len = max(1, span_end - span_start + 1)
    return (a_end - a_start + 1) <= max(3 * span_len, 40)


def match_expected(response_findings: list[dict], expect: dict) -> dict:
    """§7.1 file_hit@k / range_hit@k / recall against one query's `expect` block. Files are
    ranked by first-finding position; equivalent-group membership and primary_mode (all/any)
    are honored."""
    primary = expect.get("primary", []) or []
    equivalent_groups = expect.get("equivalent", []) or []
    primary_mode = expect.get("primary_mode", "all")
    is_negative = not primary and not equivalent_groups

    target_files = {p["path"] for p in primary}
    for group in equivalent_groups:
        for member in group:
            target_files.add(member["path"])

    ranked_files = []
    seen_files = set()
    for f in response_findings:
        rel = normalize_finding_path(f.get("location", {}).get("path", ""))
        if rel not in seen_files:
            seen_files.add(rel)
            ranked_files.append(rel)

    hit_ranks = [i for i, f in enumerate(ranked_files, start=1) if f in target_files]
    file_hit_1 = bool(hit_ranks) and 1 in hit_ranks
    file_hit_3 = any(r <= 3 for r in hit_ranks)
    file_hit_any = bool(hit_ranks)

    if primary_mode == "all" and primary:
        recovered = {p["path"] for p in primary if p["path"] in seen_files}
        recall = len(recovered) / len(primary) if primary else None
    else:
        recall = 1.0 if (file_hit_any or is_negative and not response_findings) else 0.0

    # range_hit: any finding whose range overlaps a target span, within the 3x/40-line cap.
    range_hit = False
    all_spans = [(p["path"], p.get("span")) for p in primary if p.get("span")]
    for group in equivalent_groups:
        for m in group:
            if m.get("span"):
                all_spans.append((m["path"], m["span"]))
    for f in response_findings:
        rel = normalize_finding_path(f.get("location", {}).get("path", ""))
        ls, le = f.get("location", {}).get("line_start"), f.get("location", {}).get("line_end")
        if ls is None or le is None:
            continue
        for path, span in all_spans:
            if path != rel or not span:
                continue
            if ranges_overlap(ls, le, span[0], span[1]) and range_len_ok(ls, le, span[0], span[1]):
                range_hit = True

    # twin_confusion / dedupe_defect
    distractor_files = {d["path"] for d in (expect.get("distractor") or [])}
    twin_confusion = bool(distractor_files & seen_files)
    dedupe_defect = False
    for group in equivalent_groups:
        members_seen = sum(1 for m in group if m["path"] in seen_files)
        if members_seen >= 2:
            dedupe_defect = True

    negative_ok = None
    if is_negative:
        if not response_findings:
            negative_ok = 1.0
        else:
            all_valid_but_offtopic = True  # refined manually in the blind-grading pass
            negative_ok = 0.5 if all_valid_but_offtopic else 0.0

    return {
        "file_hit_1": file_hit_1,
        "file_hit_3": file_hit_3,
        "file_hit_any": file_hit_any,
        "range_hit": range_hit,
        "recall": recall,
        "twin_confusion": twin_confusion,
        "dedupe_defect": dedupe_defect,
        "negative_ok": negative_ok,
        "is_negative": is_negative,
    }


def score_run(out_dir: Path) -> dict:
    repos = load_repos()
    per_query_rows = defaultdict(list)  # (repo, query_id) -> [scored row per pass]
    all_scored = []

    for repo_dir in sorted(out_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo_id = repo_dir.name
        if repo_id not in repos:
            continue
        repo_path = Path(repos[repo_id]["path"]).expanduser()
        queries = load_queries(repo_id)

        for jsonl_path in sorted(repo_dir.glob("pass*.jsonl")):
            with open(jsonl_path) as f:
                for line in f:
                    row = json.loads(line)
                    qid = row["query_id"]
                    if qid.endswith("-warmup"):
                        continue
                    qspec = queries.get(qid)
                    if qspec is None:
                        print(f"WARNING: no query spec for {qid} in {repo_id}", file=sys.stderr)
                        continue
                    findings = []
                    hallucinations = []
                    path_valid_all = True
                    if row.get("response") and not row.get("is_error"):
                        try:
                            parsed = json.loads(row["response"])
                            findings = parsed.get("findings", [])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    for fnd in findings:
                        fscore = score_finding(fnd, repo_path)
                        if fscore["hallucination"]:
                            hallucinations.append(fscore)
                        if not fscore["path_valid"]:
                            path_valid_all = False

                    match = match_expected(findings, qspec.get("expect", {}))
                    expected_stage = qspec.get("expect", {}).get("stage")
                    stage_match = expected_stage is None or expected_stage == row.get("stage")

                    scored = {
                        **row,
                        "cat": qspec.get("cat"),
                        "sub": qspec.get("sub"),
                        "path_valid_all": path_valid_all,
                        "hallucinated": len(hallucinations) > 0,
                        "hallucination_detail": hallucinations,
                        "stage_expected": expected_stage,
                        "stage_match": stage_match,
                        "confident_wrong": (row.get("stage") == "early-exit" and not match["file_hit_any"]),
                        **match,
                    }
                    all_scored.append(scored)
                    per_query_rows[(repo_id, qid)].append(scored)

    # pass@1 per query: success across ALL passes (deterministic) vs some (flaky) vs none.
    query_summary = []
    for (repo_id, qid), rows in sorted(per_query_rows.items()):
        successes = [r["file_hit_any"] or (r["is_negative"] and r["negative_ok"] == 1.0) for r in rows]
        n_pass = len(rows)
        n_ok = sum(successes)
        classification = "pass" if n_ok == n_pass else ("deterministic_failure" if n_ok == 0 else "flaky")
        query_summary.append(
            {
                "repo": repo_id,
                "query_id": qid,
                "cat": rows[0]["cat"],
                "n_passes": n_pass,
                "n_ok": n_ok,
                "classification": classification,
                "any_hallucination": any(r["hallucinated"] for r in rows),
                "any_confident_wrong": any(r["confident_wrong"] for r in rows),
                "stages_seen": sorted({r.get("stage") for r in rows}),
            }
        )

    return {"scored_rows": all_scored, "query_summary": query_summary}


def print_report(result: dict) -> None:
    rows = result["scored_rows"]
    qs = result["query_summary"]

    print(f"\n=== Query-level pass@1 summary ({len(qs)} queries) ===")
    for q in qs:
        flag = ""
        if q["any_hallucination"]:
            flag += " HALLUCINATION"
        if q["any_confident_wrong"]:
            flag += " CONFIDENT_WRONG"
        print(
            f"  {q['repo']:12s} {q['query_id']:20s} {q['cat']:6s} "
            f"{q['n_ok']}/{q['n_passes']} {q['classification']:20s} stages={q['stages_seen']}{flag}"
        )

    print("\n=== Per-category file_hit@3 (pooled across passes, not pass@1) ===")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r["file_hit_3"])
    for cat, vals in sorted(by_cat.items()):
        p, lo, hi = wilson_ci(sum(vals), len(vals))
        print(f"  {cat:8s} n={len(vals):3d}  file_hit@3={p:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    print("\n=== Hallucinations (P0/P1 — must be zero) ===")
    any_halluc = False
    for r in rows:
        for h in r["hallucination_detail"]:
            any_halluc = True
            print(f"  {r['repo']}/{r['query_id']} pass{r['pass']}: {h['hallucination']} path={h['path']!r}")
    if not any_halluc:
        print("  none")

    print("\n=== Confident-wrong (early-exit stage that missed) ===")
    any_cw = False
    for r in rows:
        if r["confident_wrong"]:
            any_cw = True
            print(f"  {r['repo']}/{r['query_id']} pass{r['pass']}: confidence={r['confidence']}")
    if not any_cw:
        print("  none")

    print("\n=== Stage mismatches vs. §6.3 expectation ===")
    any_mismatch = False
    for r in rows:
        if not r["stage_match"]:
            any_mismatch = True
            print(
                f"  {r['repo']}/{r['query_id']} pass{r['pass']}: "
                f"expected={r['stage_expected']} observed={r['stage']}"
            )
    if not any_mismatch:
        print("  none (or no stage expectations set)")

    print("\n=== NOT YET AVAILABLE (needs the §3.1 observability patch) ===")
    print("  cand_recall@top_k, cand_rank, llm_calls, model_served, forced_finish, provider_events,")
    print("  per-leg duration_ms, index_status.commit, git_probe_ms — all None in every row above.")

    latencies = [r["latency_ms"] for r in rows]
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        print(f"\n=== Latency ===\n  n={len(latencies)}  p50={p50:.0f}ms  max={max(latencies):.0f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="results/<run-id> directory")
    ap.add_argument("--json-out", default=None, help="write the full scored rows as JSON here")
    args = ap.parse_args()

    out_dir = Path(args.run_dir)
    if not out_dir.exists():
        sys.exit(f"no such directory: {out_dir}")

    result = score_run(out_dir)
    print_report(result)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nfull scored data written to {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
