#!/usr/bin/env python3
"""Score a Layer A run (docs/eval/real-world-test-plan.md §7.1).

Run with: uv run --with pyyaml eval/score.py results/<run-id>

The installed binary is 0.5.3 (the §3.1 observability patch), so `run.py`'s rows carry the full
field set: `candidates_ranked` (the pre-stage's own ranked list, independent of what
verify/fallback ultimately returned), `provider_calls`, `llm_calls`, `forced_finish`,
`index_status`, per-leg `leg_timings`, `git_probe_ms`. This script uses all of them —
`cand_recall@top_k`/`cand_rank` (§7.4 failure attribution: was the primary ever retrieved, at
what rank, independent of what the LLM stages did with it), provider outcomes and model-drift
detection against the run's pinned model, per-leg latency, and the `exploration_failed` line for
hard failures that never reach `is_error` reporting inconsistently. A row from an older
(pre-0.5.3) binary would simply have these fields as `None`/`[]`, so a Mode-B run scores fine —
those sections just report zero rows.

Wilson score interval is used for all proportions (n is small in the pilot; a naive normal
approximation is misleading at n<30).
"""

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent

WS_RE = re.compile(r"\s+")
# A snippet line that is only an omission marker ("...", the unicode ellipsis, optionally
# annotated "(truncated)"/"[truncated]") separates two chunks of a multi-line snippet rather
# than being real quoted content — the LLM (or the tool's own compressed rendering, which the
# LLM sometimes echoes verbatim) uses this to elide the middle of a long span.
ELLIPSIS_LINE_RE = re.compile(r"^(\.\.\.|…)\s*(\(truncated\)|\[truncated\])?$", re.I)
# The tool's own rendering (render.rs's `TRUNCATION_MARKER`) appends "…[truncated]" — or, for a
# capped `read_file`, "…[truncated after N lines; request a narrower line range]" — to the *end*
# of an otherwise-real line rather than putting it on its own line (F-19: this used to make a
# legitimately truncated line fail find_chunk's exact substring match and get flagged
# fabricated_snippet/misaligned_snippet even though the untruncated portion is real, in-range
# content). Stripped off before matching so only the real prefix is compared.
TRUNCATION_SUFFIX_RE = re.compile(r"…\[truncated(?:[^\]]*)?\]\s*$")


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
    import yaml

    path = EVAL_DIR / "queries" / f"{repo_id}.yaml"
    with open(path) as f:
        items = yaml.safe_load(f) or []
    return {item["id"]: item for item in items}


def load_repos() -> dict[str, dict]:
    import tomllib

    with open(EVAL_DIR / "repos.toml", "rb") as f:
        data = tomllib.load(f)
    return {r["id"]: r for r in data["repo"]}


def load_pinned_model(out_dir: Path) -> str | None:
    """The model this run was pinned to, read from the eval config the manifest recorded — used
    to flag any `provider_calls` row whose `model_served` drifted (§3.2/§8.1: a floating alias or
    mid-run failover means the row's model differs from what the run intended)."""
    import tomllib

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    config_path = manifest.get("config")
    if not config_path or not Path(config_path).exists():
        return None
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    providers = config.get("llm", {}).get("providers", [])
    if providers and providers[0].get("models"):
        return providers[0]["models"][0]
    return None


def snippet_chunks(snippet: str) -> list[list[str]]:
    """Split a (possibly multi-line) snippet into contiguous chunks of real content, breaking
    at any line that is only an omission marker. Each chunk is a list of whitespace-normalized
    lines, in order — a chunk must match a contiguous run of file lines, but chunks themselves
    need not be adjacent (the LLM may elide the middle of a long span). A line ending in the
    tool's own truncation marker (F-19) has that suffix stripped first: a non-empty remainder is
    real content to match (just shorter than the untruncated file line), while an empty remainder
    (the marker occupied the whole line) is treated as an elision like an ELLIPSIS_LINE_RE line.
    A genuinely blank line (F-20) is real content too — e.g. inside a docstring, or between two
    top-level items quoted together — so it's kept as a "" wildcard placeholder within the
    current chunk (matching find_chunk's substring check unconditionally) rather than dropped,
    which would otherwise misalign the chunk's contiguous-run length against the file."""
    chunks: list[list[str]] = []
    current: list[str] = []
    for raw_line in snippet.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if current:
                current.append("")
            continue
        if ELLIPSIS_LINE_RE.match(stripped):
            if current:
                chunks.append(current)
                current = []
            continue
        stripped = TRUNCATION_SUFFIX_RE.sub("", stripped).strip()
        if not stripped:
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(WS_RE.sub(" ", stripped))
    if current:
        chunks.append(current)
    return chunks


def find_chunk(
    file_lines: list[str], chunk: list[str], near_range: tuple[int, int] | None = None
) -> int | None:
    """Index in file_lines where `chunk` matches a contiguous run (each chunk line must be a
    substring of the corresponding file line) — preferring a match inside `near_range` (F-17
    follow-up) when the chunk's content is duplicated elsewhere in the file (e.g. two functions
    with the same signature, or a repeated test assertion): without this, the first (possibly
    out-of-range) occurrence would be reported even when a second occurrence sits exactly at the
    claimed line range, falsely classifying a correct finding as misaligned. Falls back to the
    first match anywhere if none falls inside `near_range`. None if the chunk isn't found at all."""
    fallback: int | None = None
    for i in range(len(file_lines) - len(chunk) + 1):
        if all(chunk[j] in file_lines[i + j] for j in range(len(chunk))):
            if near_range is not None and near_range[0] <= i < near_range[1]:
                return i
            if fallback is None:
                fallback = i
    return fallback


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


def snippet_found_at(
    repo_path: Path, rel: str, line_start: int | None, line_end: int | None, snippet: str
) -> str:
    """Returns 'ok' (matches at/near the claimed range, or the range is a whole-file span, or the
    location itself is unknown, so "near" is moot), 'misaligned' (found elsewhere in the file), or
    'not_found' (fabricated) — the §7.1 hallucinated classification, minus the 'fabricated
    path'/'range outside file' cases checked separately. Matches per-chunk (see snippet_chunks) so
    a genuine multi-line quote isn't flagged fabricated just because no single physical line equals
    the whole snippet. When both bounds are known, the "near" window spans the full claimed
    [line_start, line_end] range with a ±4-line pad (not just a band around line_start), so a
    representative line cited from anywhere inside a large non-whole-file span is accepted as
    in-range. When line_end is unknown, the window stays the narrower legacy line_start-4..
    line_start+3 band (no forward pad) rather than silently widening by one line. When line_start
    itself is unknown (location genuinely unknown), no misalignment check applies at all — only
    fabrication (not_found) is possible."""
    p = repo_path / rel
    if not p.exists() or not p.is_file():
        return "not_found"
    chunks = snippet_chunks(snippet)
    if not chunks:
        return "ok"  # nothing to check
    try:
        lines = p.read_text(errors="replace").splitlines()
    except OSError:
        return "not_found"
    file_lines = [WS_RE.sub(" ", line.strip()) for line in lines]
    # A whole-file (or near-whole-file) span has no meaningful "near line_start" — a module-level
    # semantic match legitimately cites a representative line from anywhere in the file.
    whole_file = line_start is not None and line_end is not None and (line_end - line_start) >= len(lines) - 2
    near_range = None
    if line_start and not whole_file:
        upper = line_end + 4 if line_end is not None else line_start + 3
        near_range = (max(0, line_start - 4), min(len(lines), upper))
    # Every chunk must exist somewhere in the file (a chunk that's missing is fabricated
    # content, even if the rest of the snippet is real); "misaligned" only if all chunks exist
    # but at least one falls outside the claimed range. A finding with no known line_start has no
    # range to fall outside of, so it can only ever be "ok" or "not_found" here.
    any_far = False
    for chunk in chunks:
        idx = find_chunk(file_lines, chunk, near_range)
        if idx is None:
            return "not_found"
        in_near = near_range is not None and near_range[0] <= idx < near_range[1]
        if not (whole_file or line_start is None or in_near):
            any_far = True
    return "misaligned" if any_far else "ok"


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
        cls = snippet_found_at(repo_path, rel, line_start, line_end, finding["snippet"])
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
    equivalent = expect.get("equivalent", []) or []
    primary_mode = expect.get("primary_mode", "all")
    is_negative = not primary and not equivalent

    target_files = {p["path"] for p in primary}
    for member in equivalent:
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
    for m in equivalent:
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
    members_seen = sum(1 for m in equivalent if m["path"] in seen_files)
    dedupe_defect = members_seen >= 2

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


def compute_cand_recall(candidates_ranked: list[dict] | None, expect: dict) -> dict:
    """§7.4 failure attribution: was a primary/equivalent target ever in the pre-stage's own
    ranked candidate list (`candidates_ranked`, independent of what verify/fallback did with
    it), and at what rank? Distinguishes `retrieval-miss` (never a candidate) from `rank>k`
    (a candidate, just not surfaced by the LLM stage) from everything downstream of retrieval."""
    if candidates_ranked is None:
        return {"cand_recall_at_topk": None, "cand_rank": None}
    primary = expect.get("primary", []) or []
    target_files = {p["path"] for p in primary}
    for member in expect.get("equivalent", []) or []:
        target_files.add(member["path"])
    if not target_files:
        return {"cand_recall_at_topk": None, "cand_rank": None}  # negative query: nothing to find
    best_rank = None
    for c in candidates_ranked:
        rel = normalize_finding_path(c.get("path", ""))
        if rel in target_files:
            rank = c.get("rank")
            if best_rank is None or (rank is not None and rank < best_rank):
                best_rank = rank
    return {"cand_recall_at_topk": best_rank is not None, "cand_rank": best_rank}


def attribution_class(row: dict, match: dict, cand: dict) -> str | None:
    """§7.4: one failure class per failed query, ordered so the first applicable one wins."""
    if match["file_hit_any"] or (match["is_negative"] and match["negative_ok"] == 1.0):
        return None
    if row.get("timeout"):
        return "timeout"
    if row.get("is_error"):
        return "error"
    if any(h["hallucination"] in ("fabricated_path", "fabricated_snippet") for h in row.get("hallucination_detail", [])):
        return "fabricated-path"
    if cand["cand_recall_at_topk"] is False:
        return "retrieval-miss"
    if cand["cand_recall_at_topk"] is True:
        stage = row.get("stage")
        if stage == "verify":
            return "verify-rejected-correct"
        if stage == "fallback":
            return "fallback-budget" if row.get("forced_finish") else "fallback-miss"
        return "rank>k"
    return "fallback-miss" if row.get("stage") == "fallback" else None


def score_run(out_dir: Path) -> dict:
    repos = load_repos()
    pinned_model = load_pinned_model(out_dir)
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
                    cand = compute_cand_recall(row.get("candidates_ranked"), qspec.get("expect", {}))
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
                        **cand,
                    }
                    scored["attribution"] = attribution_class(scored, match, cand)
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
                "stages_seen": sorted({r.get("stage") for r in rows if r.get("stage") is not None}),
            }
        )

    return {
        "run_id": out_dir.name,
        "scored_rows": all_scored,
        "query_summary": query_summary,
        "pinned_model": pinned_model,
    }


# ---------------------------------------------------------------------------
# QW-0 efficiency metrics
# ---------------------------------------------------------------------------

# Price table for cost_per_query (plan §7.1: cost = Σ prompt × p_in + (completion + reasoning)
# × p_out). USD per 1M tokens.
#
#   Prices as of 2026-09-15. Source: https://ai.google.dev/pricing
#
# CAVEAT, read before quoting a dollar figure anywhere outside this report: these are the
# published *tier* rates (Flash / Flash-Lite) applied to every model in that tier — the
# individual 3.x per-model rates were not re-verified on the capture date. cost_per_query is a
# comparable relative signal between runs of this harness, not an invoice. A model that is not
# in this table is reported as n/a and never priced at 0 — see qw0_cost().
#
# Covers the six models in eval/config/default.toml's failover chain. When that chain changes,
# this table changes with it, or the run reports unpriced calls.
PRICES_CAPTURED = "2026-09-15"
PRICES_SOURCE = "https://ai.google.dev/pricing"
PRICES = {
    "gemini-3.8-flash": {"usd_per_1M_input": 0.30, "usd_per_1M_output": 2.50, "tier": "flash"},
    "gemini-3.7-flash": {"usd_per_1M_input": 0.30, "usd_per_1M_output": 2.50, "tier": "flash"},
    "gemini-3.6-flash": {"usd_per_1M_input": 0.30, "usd_per_1M_output": 2.50, "tier": "flash"},
    "gemini-3.5-flash": {"usd_per_1M_input": 0.30, "usd_per_1M_output": 2.50, "tier": "flash"},
    "gemini-3.5-flash-lite": {"usd_per_1M_input": 0.10, "usd_per_1M_output": 0.40, "tier": "flash-lite"},
    "gemini-3.1-flash-lite": {"usd_per_1M_input": 0.10, "usd_per_1M_output": 0.40, "tier": "flash-lite"},
}

# Every QW-0 number this block reports, in the order print_qw0_report prints them. A metric whose
# source field is absent from every row degrades to "n/a (field absent in this run)" — an older
# results/ dir predates the field, and a silent 0 would read as a real measurement.
NA_ABSENT = "n/a (field absent in this run)"


def qw0_field(row: dict, name: str):
    """One QW-0 field off a scored row. Every field reported here is emitted as its own flat
    tracing field on the `exploration complete` / cache-hit line and copied into the row by
    run.py; None means the run's binary never emitted it."""
    return row.get(name)


def llm_calls_of(row: dict) -> int | None:
    """LLM turns for one query. A cache hit made zero by construction (the memoized result is
    returned before any provider is touched), so it counts as 0 even in a results/ dir whose
    cache line predates the llm_calls field. Every other absence stays absent (None)."""
    v = qw0_field(row, "llm_calls")
    if v is None and row.get("stage") == "cache":
        return 0
    return v


def numeric_field(rows: list[dict], name: str) -> list[float]:
    """Every non-None numeric value of `name` across rows. Empty list == absent everywhere."""
    vals = []
    for r in rows:
        v = qw0_field(r, name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(v)
    return vals


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (no interpolation): the smallest value at or above the q-th rank.
    n=1 -> that value for any q; n=2, q=0.95 -> the max. Deliberate for small n — interpolating
    a p95 out of two samples invents precision the run doesn't have."""
    if not values:
        return None
    vals = sorted(values)
    k = max(1, math.ceil(q * len(vals)))
    return vals[min(k, len(vals)) - 1]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _dist(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def call_cost(pc: dict) -> float | None:
    """USD for one provider call, or None if its model_served has no price entry.

    A call that reports no token usage at all costs 0.0 and needs no price: that is a failed
    attempt (rate_limited / quota — llm/src/lib.rs's error branch logs neither model_served nor
    tokens), or, in a results/ dir written before the provider_call regex was tightened, a ghost
    row from the router's "provider call succeeded" commentary. Billing it as "unpriced" would
    turn cost_per_query into n/a for the whole run over calls that spent nothing."""
    tokens = [pc.get(k) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")]
    if all(t is None for t in tokens):
        return 0.0
    price = PRICES.get(pc.get("model_served"))
    if price is None:
        return None
    prompt = pc.get("prompt_tokens") or 0
    out = (pc.get("completion_tokens") or 0) + (pc.get("reasoning_tokens") or 0)
    return prompt / 1e6 * price["usd_per_1M_input"] + out / 1e6 * price["usd_per_1M_output"]


def row_cost(row: dict) -> float | None:
    """USD for one query, or None if any of its provider calls is unpriced. A 0-LLM row (cache /
    early-exit) has no provider calls and costs exactly 0.0 — that is a measurement, not an
    absence."""
    total = 0.0
    for pc in row.get("provider_calls") or []:
        c = call_cost(pc)
        if c is None:
            return None
        total += c
    return total


def qw0_cost(rows: list[dict]) -> dict:
    """Run-level cost. `unpriced` counts provider calls whose model has no price entry; when it
    is non-empty the run total is incomplete and must be reported n/a, never as the partial sum
    passed off as the whole."""
    total = 0.0
    priced_calls = 0
    unpriced: Counter = Counter()
    rows_priced = 0
    for r in rows:
        row_had_unpriced = False
        for pc in r.get("provider_calls") or []:
            c = call_cost(pc)
            if c is None:
                unpriced[pc.get("model_served") or "<unknown>"] += 1
                row_had_unpriced = True
            elif c > 0 or pc.get("prompt_tokens") is not None:
                # Only attempts that actually reported usage count as billed calls; a
                # zero-usage attempt (see call_cost) costs nothing and would otherwise pad
                # the denominator of "$X over N provider calls".
                total += c
                priced_calls += 1
        if not row_had_unpriced:
            rows_priced += 1
    return {
        "usd_total_priced": total,
        "priced_calls": priced_calls,
        "unpriced": dict(unpriced),
        "rows_fully_priced": rows_priced,
        "n_rows": len(rows),
        "usd_per_query": (total / len(rows)) if rows and not unpriced else None,
    }


def qw0_cache(rows: list[dict]) -> dict:
    """cache_read_tokens / prompt tokens across the run. Numerator: the per-query
    cache_read_tokens field, falling back to summing the provider calls' own cached_tokens.
    Denominator: prompt_tokens over every provider call. Either side absent -> ratio None, so
    the report says "no cache activity reported" instead of a silent 0%."""
    read = [v for v in (qw0_field(r, "cache_read_tokens") for r in rows) if v is not None]
    write = [v for v in (qw0_field(r, "cache_write_tokens") for r in rows) if v is not None]
    per_call = [pc.get("cached_tokens") for r in rows for pc in (r.get("provider_calls") or [])]
    per_call = [v for v in per_call if v is not None]
    if not read and per_call:
        read = per_call
    prompt = [pc.get("prompt_tokens") for r in rows for pc in (r.get("provider_calls") or [])]
    prompt = [v for v in prompt if v is not None]
    numerator = sum(read) if read else None
    denominator = sum(prompt) if prompt else None
    ratio = None
    if numerator is not None and denominator:
        ratio = numerator / denominator
    return {
        "cache_read_tokens": numerator,
        "cache_write_tokens": sum(write) if write else None,
        "prompt_tokens": denominator,
        "ratio": ratio,
        "reported": bool(read),
    }


def qw0_metrics(rows: list[dict]) -> dict:
    """Every QW-0 aggregate over the already-warm-up-filtered scored rows. Pure — print_qw0_report
    and the --csv-out writer both consume this and neither recomputes anything."""
    llm_calls = [v for v in (llm_calls_of(r) for r in rows) if v is not None]
    tokens_all = numeric_field(rows, "tokens")
    tokens_llm = [
        t
        for r, t in ((r, qw0_field(r, "tokens")) for r in rows)
        if isinstance(t, (int, float)) and (llm_calls_of(r) or 0) > 0
    ]
    stage_exit = Counter(r.get("stage") or "<none>" for r in rows)
    routes = [qw0_field(r, "early_exit_route") for r in rows if r.get("stage") == "early-exit"]
    routes = [x for x in routes if x is not None]
    # M-2: both fields are emitted only on the Stage-5 path, so they are denominated over
    # Stage-5 rows only — the same restriction early_exit_route gets above. Averaging
    # orientation_calls_in_loop over all rows would divide a fallback-only numerator by the
    # whole corpus and report ~0.00 for a loop that never stopped calling get_architecture.
    # Stage 5 also exits as stage=="error" (provider exhaustion) carrying truthful counts, so
    # the denominator is "the row has a count", not the clean-exit stage name.
    fallback_rows = [r for r in rows if r.get("orientation_calls_in_loop") is not None]
    # M-1: the hit rate's denominator is every scored row, because every scored row is a row
    # that could have hit — the corpus repeats each query once per pass and each pass runs a
    # fresh server process, which is exactly the cross-session repetition M-1 exists for.
    # The layer split gets its own presence test rather than riding that denominator: a
    # pre-M-1 binary emits stage=="cache" with no cache_layer, so the total rate stays a real
    # measurement while l1/l2 report absent instead of a fabricated 0%.
    cache_rows = [r for r in rows if r.get("stage") == "cache"]
    layers = [x for x in (qw0_field(r, "cache_layer") for r in cache_rows) if x is not None]
    return {
        "n_rows": len(rows),
        "llm_calls": _dist(llm_calls),
        "stage_exit": dict(stage_exit),
        "early_exit_route": dict(Counter(routes)) if routes else None,
        "tokens_all": _dist(tokens_all),
        "tokens_llm_rows": _dist(tokens_llm),
        "cache": qw0_cache(rows),
        "cost": qw0_cost(rows),
        "latency_ms": _dist(numeric_field(rows, "latency_ms")),
        "total_ms": _dist(numeric_field(rows, "total_ms")),
        "brief_tokens": _dist(numeric_field(fallback_rows, "brief_tokens")),
        "orientation_calls_in_loop": _dist(numeric_field(fallback_rows, "orientation_calls_in_loop")),
        "cache_hits": {
            "n_rows": len(rows),
            "hit_rows": len(cache_rows),
            "hit_rate": len(cache_rows) / len(rows) if rows else None,
            "cache_hit_l1": layers.count("l1") / len(rows) if layers and rows else None,
            "cache_hit_l2": layers.count("l2") / len(rows) if layers and rows else None,
            "layer_reported": bool(layers),
        },
        # Denominated over the cache rows alone (numeric_field drops the None every non-cache
        # row carries): these measure what a hit avoided, so a non-hit has nothing to add and
        # must not pad the denominator toward zero.
        "turns_saved_by_cache": _dist(numeric_field(rows, "turns_saved_by_cache")),
        "tokens_saved_by_cache": _dist(numeric_field(rows, "tokens_saved_by_cache")),
    }


def _fmt_dist(d: dict | None, unit: str = "", keys=("mean", "p95")) -> str:
    if d is None:
        return NA_ABSENT
    parts = [f"{k}={d[k]:.1f}{unit}" for k in keys if d.get(k) is not None]
    return f"n={d['n']} " + " ".join(parts)


def print_qw0_report(agg: dict) -> None:
    n = agg["n_rows"]
    print(f"\n=== QW-0 efficiency (over {n} scored rows; warm-up excluded) ===")

    print(f"  llm_turns_per_query: {_fmt_dist(agg['llm_calls'])}")

    print("  stage_exit:")
    for stage, count in sorted(agg["stage_exit"].items(), key=lambda kv: -kv[1]):
        share = count / n if n else 0.0
        print(f"    {stage:12s} {count:4d}  {share:5.1%}")
        if stage == "early-exit":
            routes = agg["early_exit_route"]
            if routes is None:
                print(f"      early_exit_route: {NA_ABSENT}")
            else:
                for route, rn in sorted(routes.items(), key=lambda kv: -kv[1]):
                    print(f"      route={route:16s} {rn:4d}  {rn / count:5.1%} of early-exit")

    # tokens=0 by construction on cache and early-exit rows, so the all-rows mean is a
    # per-query cost figure and the LLM-rows-only mean is a per-LLM-query cost figure. Both,
    # always — either one alone reads as the other.
    print(f"  tokens_per_query (all rows):         {_fmt_dist(agg['tokens_all'], keys=('mean', 'p50', 'p95'))}")
    print(f"  tokens_per_query (LLM-touching only):{_fmt_dist(agg['tokens_llm_rows'], keys=('mean', 'p50', 'p95'))}")

    cache = agg["cache"]
    if cache["ratio"] is None:
        why = "provider reports no cache activity" if not cache["reported"] else "no prompt tokens recorded"
        print(f"  cached_ratio: n/a - {why}")
    else:
        note = " (provider reported zero cached tokens on every call)" if cache["ratio"] == 0 else ""
        print(
            f"  cached_ratio: {cache['ratio']:.1%}  "
            f"({cache['cache_read_tokens']} cache-read / {cache['prompt_tokens']} prompt tokens){note}"
        )

    cost = agg["cost"]
    if cost["usd_per_query"] is not None:
        print(f"  cost_per_query: ${cost['usd_per_query']:.5f}  (${cost['usd_total_priced']:.4f} over {n} rows, {cost['priced_calls']} provider calls)")
    elif cost["unpriced"]:
        # Before "no priced calls": when EVERY call ran on an unpriced model, priced_calls is 0
        # too, and naming the models is the useful report — not "no priced provider calls".
        unp = ", ".join(f"{m} x{c}" for m, c in sorted(cost["unpriced"].items()))
        print(f"  cost_per_query: n/a - unpriced models served: {unp}")
        if cost["priced_calls"]:
            partial = cost["usd_total_priced"] / cost["rows_fully_priced"] if cost["rows_fully_priced"] else 0.0
            print(f"    (priced subset only: ${partial:.5f}/query over {cost['priced_calls']} of "
                  f"{cost['priced_calls'] + sum(cost['unpriced'].values())} provider calls)")
    else:
        print("  cost_per_query: n/a - no priced provider calls in this run")
    print(f"    price table: {PRICES_SOURCE}, captured {PRICES_CAPTURED} (tier rates - see PRICES in score.py)")

    print(f"  latency_per_query:   {_fmt_dist(agg['latency_ms'], unit='ms')}")
    print(f"  total_ms (server-side): {_fmt_dist(agg['total_ms'], unit='ms')}")

    # M-2 repo brief. Both are Stage-5-only measurements: `n` is the number of fallback rows
    # that reported the field, NOT the run's row count. A run whose binary predates M-2, or one
    # with no fallback row at all, prints n/a here — never 0.
    print(f"  brief_tokens (fallback rows):        {_fmt_dist(agg['brief_tokens'], keys=('mean', 'p50', 'p95'))}")
    print(f"  orientation_calls_in_loop (fallback rows): {_fmt_dist(agg['orientation_calls_in_loop'])}")

    # M-1 persistent result cache. All three rates share one denominator — every scored row —
    # and the line says so, because a hit rate is only readable against the set of rows that
    # could have hit. The two saved distributions are denominated over the cache rows instead:
    # they answer "what did a hit avoid", which a non-hit row cannot contribute to.
    ch = agg["cache_hits"]
    if ch["hit_rate"] is None:
        print(f"  cache_hit_rate: {NA_ABSENT}")
    else:
        print(
            f"  cache_hit_rate: {ch['hit_rate']:.1%}  "
            f"({ch['hit_rows']} of {ch['n_rows']} scored rows served from cache)"
        )
        if ch["layer_reported"]:
            print(
                f"    cache_hit_l1: {ch['cache_hit_l1']:.1%}   cache_hit_l2: {ch['cache_hit_l2']:.1%}"
                f"   (share of the same {ch['n_rows']} rows)"
            )
        else:
            print(f"    cache_hit_l1 / cache_hit_l2: {NA_ABSENT}")
    print(f"  turns_saved_by_cache (cache rows):   {_fmt_dist(agg['turns_saved_by_cache'])}")
    print(f"  tokens_saved_by_cache (cache rows):  {_fmt_dist(agg['tokens_saved_by_cache'], keys=('mean', 'p50', 'p95'))}")


CSV_COLUMNS = [
    "run_id",
    "repo",
    "pass",
    "query_id",
    "cat",
    "stage",
    "early_exit_route",
    "llm_calls",
    "tokens",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_usd",
    "latency_ms",
    "total_ms",
    "confidence",
    "candidate_count",
    "brief_tokens",
    "orientation_calls_in_loop",
    "cache_layer",
    "turns_saved_by_cache",
    "tokens_saved_by_cache",
]


def qw0_csv_rows(result: dict) -> list[dict]:
    run_id = result.get("run_id")
    out = []
    for r in result["scored_rows"]:
        calls = r.get("provider_calls") or []
        cost = row_cost(r)
        out.append(
            {
                "run_id": run_id,
                "repo": r.get("repo"),
                "pass": r.get("pass"),
                "query_id": r.get("query_id"),
                "cat": r.get("cat"),
                "stage": r.get("stage"),
                "early_exit_route": qw0_field(r, "early_exit_route"),
                "llm_calls": llm_calls_of(r),
                "tokens": qw0_field(r, "tokens"),
                "prompt_tokens": sum(pc.get("prompt_tokens") or 0 for pc in calls),
                "completion_tokens": sum(pc.get("completion_tokens") or 0 for pc in calls),
                "reasoning_tokens": sum(pc.get("reasoning_tokens") or 0 for pc in calls),
                "cache_read_tokens": qw0_field(r, "cache_read_tokens"),
                "cache_write_tokens": qw0_field(r, "cache_write_tokens"),
                "cost_usd": f"{cost:.6f}" if cost is not None else "",
                "latency_ms": r.get("latency_ms"),
                "total_ms": qw0_field(r, "total_ms"),
                "confidence": r.get("confidence"),
                "candidate_count": qw0_field(r, "candidate_count"),
                "brief_tokens": qw0_field(r, "brief_tokens"),
                "orientation_calls_in_loop": qw0_field(r, "orientation_calls_in_loop"),
                "cache_layer": qw0_field(r, "cache_layer"),
                "turns_saved_by_cache": qw0_field(r, "turns_saved_by_cache"),
                "tokens_saved_by_cache": qw0_field(r, "tokens_saved_by_cache"),
            }
        )
    return out


def flatten_agg(agg: dict, prefix: str = "") -> list[tuple[str, str]]:
    """The aggregate dict as flat metric,value pairs for the sibling aggregate CSV."""
    pairs = []
    for key, val in agg.items():
        name = f"{prefix}{key}"
        if isinstance(val, dict):
            pairs.extend(flatten_agg(val, f"{name}."))
        else:
            pairs.append((name, "" if val is None else str(val)))
    return pairs


def write_qw0_csv(result: dict, path: Path, qw0_agg: dict) -> Path:
    """One row per scored query at `path`, plus a sibling <stem>.aggregate.csv of metric,value
    pairs. stdlib csv, no pandas — this is 32 rows, not a dataframe. `qw0_agg` is
    `qw0_metrics(result["scored_rows"])`, already computed once by `print_report`."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(qw0_csv_rows(result))
    agg_path = path.with_name(path.stem + ".aggregate" + path.suffix)
    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(flatten_agg(qw0_agg))
    return agg_path


def print_report(result: dict) -> dict:
    """Prints the full report and returns the QW-0 aggregate dict, so a caller that also
    wants the CSV (`write_qw0_csv`) doesn't have to redo the whole aggregation pass."""
    rows = result["scored_rows"]
    qs = result["query_summary"]
    pinned_model = result.get("pinned_model")

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

    print("\n=== Per-category file_hit@3 (pooled across passes, not pass@1; excludes negative queries) ===")
    by_cat = defaultdict(list)
    for r in rows:
        if r["is_negative"]:
            continue  # scored via negative_ok below — file_hit@3=0 is the correct outcome here
        by_cat[r["cat"]].append(r["file_hit_3"])
    for cat, vals in sorted(by_cat.items()):
        p, lo, hi = wilson_ci(sum(vals), len(vals))
        print(f"  {cat:8s} n={len(vals):3d}  file_hit@3={p:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    print("\n=== Negative queries (negative_ok: 1.0=correctly empty, 0.5=off-topic hedge, 0.0=confident wrong) ===")
    neg_rows = [r for r in rows if r["is_negative"]]
    if neg_rows:
        avg = sum(r["negative_ok"] for r in neg_rows) / len(neg_rows)
        print(f"  n={len(neg_rows):3d}  mean negative_ok={avg:.2f}")
        for r in neg_rows:
            print(f"  {r['repo']}/{r['query_id']} pass{r['pass']}: negative_ok={r['negative_ok']}")
    else:
        print("  none")

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
            # PR #47: an expected=early-exit mismatch may be explained by disk-verification
            # rejecting every early-exit candidate (agent.rs `result_from_candidates`) rather
            # than F-03 pre-stage nondeterminism — surface that cause when present.
            cause = ""
            if r["stage_expected"] == "early-exit" and r.get("early_exit_fallthrough"):
                reasons = r.get("early_exit_dropped_candidates") or []
                cause = f" (early-exit disk-verification rejected all candidates: {reasons[0]!r})" if reasons else " (early-exit disk-verification rejected all candidates)"
            print(
                f"  {r['repo']}/{r['query_id']} pass{r['pass']}: "
                f"expected={r['stage_expected']} observed={r['stage']}{cause}"
            )
    if not any_mismatch:
        print("  none (or no stage expectations set)")

    print("\n=== Failure attribution (§7.4) ===")
    by_attr = defaultdict(int)
    for r in rows:
        if r.get("attribution"):
            by_attr[r["attribution"]] += 1
    if by_attr:
        for attr, n in sorted(by_attr.items(), key=lambda kv: -kv[1]):
            print(f"  {attr:26s} {n}")
    else:
        print("  none (no failed queries, or candidates_ranked unavailable — Mode B run)")

    cand_rows = [r for r in rows if r.get("cand_recall_at_topk") is not None]
    if cand_rows:
        hits = sum(1 for r in cand_rows if r["cand_recall_at_topk"])
        p, lo, hi = wilson_ci(hits, len(cand_rows))
        print(f"\n=== Candidate recall@top_k (pre-stage, before any LLM stage) ===")
        print(f"  n={len(cand_rows)}  cand_recall@top_k={p:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
        ranked = [r["cand_rank"] for r in cand_rows if r["cand_rank"] is not None]
        if ranked:
            ranked.sort()
            print(f"  cand_rank (of hits): median={ranked[len(ranked)//2]}  max={max(ranked)}")

    provider_rows = [pc for r in rows for pc in r.get("provider_calls", [])]
    if provider_rows:
        print(f"\n=== Provider calls ({len(provider_rows)} total) ===")
        by_outcome = defaultdict(int)
        for pc in provider_rows:
            by_outcome[pc.get("outcome")] += 1
        for outcome, n in sorted(by_outcome.items(), key=lambda kv: -kv[1]):
            print(f"  outcome={outcome or '<none>':14s} {n}")
        models_served = sorted({pc["model_served"] for pc in provider_rows if pc.get("model_served")})
        print(f"  models served this run: {models_served}")
        if pinned_model:
            drift = sorted({m for m in models_served if m != pinned_model})
            print(f"  pinned model: {pinned_model!r}  drift: {drift or 'none'}")
        else:
            print("  pinned model: unknown (manifest.json/config not found — can't check drift)")
        provider_events = sum(1 for pc in provider_rows if pc.get("outcome") != "ok")
        print(f"  provider_events (non-ok outcomes): {provider_events}")
    else:
        print("\n=== Provider calls === none recorded (every query early-exited, or Mode B)")

    forced = [r for r in rows if r.get("forced_finish")]
    print(f"\n=== Forced finish ({len(forced)} rows) ===")
    for r in forced:
        print(f"  {r['repo']}/{r['query_id']} pass{r['pass']}: stage={r['stage']} tokens={r['tokens']}")

    leg_rows = [lt for r in rows for lt in r.get("leg_timings", [])]
    if leg_rows:
        print(f"\n=== Per-leg latency ({len(leg_rows)} legs across {len(rows)} calls) ===")
        by_leg = defaultdict(list)
        for lt in leg_rows:
            by_leg[lt.get("leg")].append(lt.get("duration_ms") or 0)
        for leg, durs in sorted(by_leg.items()):
            durs.sort()
            print(f"  {leg:10s} n={len(durs):4d}  median={durs[len(durs)//2]:.0f}ms  max={max(durs):.0f}ms")

    index_statuses = [r["index_status"] for r in rows if r.get("index_status")]
    if index_statuses:
        by_status = defaultdict(int)
        for s in index_statuses:
            by_status[s] += 1
        print(f"\n=== Index status ({len(index_statuses)} calls with a status) ===")
        for s, n in sorted(by_status.items(), key=lambda kv: -kv[1]):
            print(f"  {s:16s} {n}")

    latencies = [r["latency_ms"] for r in rows]
    if latencies:
        # Same definition as the aggregate CSV's latency_ms.p50 — two spellings of "p50"
        # over one list put two different numbers in one run's artifacts.
        p50 = percentile(latencies, 0.50)
        print(f"\n=== Latency ===\n  n={len(latencies)}  p50={p50:.0f}ms  max={max(latencies):.0f}ms")

    git_probe = [r["git_probe_ms"] for r in rows if r.get("git_probe_ms") is not None]
    if git_probe:
        git_probe.sort()
        print(f"  git_probe_ms: median={git_probe[len(git_probe)//2]}  max={max(git_probe)}")

    agg = qw0_metrics(rows)
    print_qw0_report(agg)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="results/<run-id> directory")
    ap.add_argument("--json-out", default=None, help="write the full scored rows as JSON here")
    ap.add_argument(
        "--csv-out",
        default=None,
        help="write the QW-0 per-query metrics CSV here (plus a sibling <stem>.aggregate.csv)",
    )
    args = ap.parse_args()

    out_dir = Path(args.run_dir)
    if not out_dir.exists():
        sys.exit(f"no such directory: {out_dir}")

    result = score_run(out_dir)
    qw0_agg = print_report(result)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nfull scored data written to {args.json_out}", file=sys.stderr)

    if args.csv_out:
        csv_path = Path(args.csv_out)
        agg_path = write_qw0_csv(result, csv_path, qw0_agg)
        print(f"QW-0 per-query CSV written to {csv_path} (aggregates: {agg_path})", file=sys.stderr)


if __name__ == "__main__":
    main()
