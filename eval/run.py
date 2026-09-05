#!/usr/bin/env python3
"""Layer A direct MCP driver (docs/eval/real-world-test-plan.md §8.1).

Run with: uv run --with mcp --with pyyaml eval/run.py [options]

Spawns the INSTALLED repo-explorer-mcp binary (~/.local/bin/repo-explorer-mcp), pointed at a
pinned checkout via cwd, with REPO_EXPLORER_CONFIG set to an eval config variant. Speaks MCP
over stdio (Python `mcp` SDK). Each repo x pass gets a fresh server process — an identical query
repeated within one process is a cache hit (§2 Stage 0), so independent attempts require
independent processes, never a repeated call on the same session.

Mode A vs Mode B (§3.1): this script works against the unpatched 0.5.2 binary today (Mode B).
It parses the two existing stage-indicating INFO lines and the `retrieval pre-stage complete`
line by SEQUENTIAL SLICING of the stderr log between a call's start and end timestamps, which is
fragile for concurrent calls (fine here — this script runs sequentially by design, §8.1 step 3).
Once the observability patch lands, the `req_id`-tagged lines make correlation exact instead of
timestamp-based, and the additional fields (candidates dump, llm_calls, provider calls) become
parseable — those extraction points are marked TODO(patch) below rather than guessed at.

Pilot scope: only the repos actually present in eval/repos.toml are run (today: self, requests).
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
BINARY = Path.home() / ".local" / "bin" / "repo-explorer-mcp"

# INFO/DEBUG lines this script can already parse from an unpatched 0.5.2 binary (§2). ANSI escape
# codes are present in the raw stderr (the fmt layer colours output even to a pipe) — stripped
# before any regex runs.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CACHE_LINE_RE = re.compile(r'exploration served from query cache')
COMPLETE_LINE_RE = re.compile(r'exploration complete\b.*?path="?(?P<path>[a-z-]+)"?.*?tokens=(?P<tokens>\d+)')
RETRIEVAL_LINE_RE = re.compile(
    r"retrieval pre-stage complete\b.*?candidates=(?P<candidates>\d+).*?confidence=(?P<confidence>\d+)"
)
ESCALATE_RE = re.compile(r"verification escalated to the fallback loop")
LEG_FAILED_RE = re.compile(r'retrieval leg failed\b.*?leg="?(?P<leg>\w+)"?.*?error=(?P<error>.*)')
RATE_LIMIT_ISERROR_RE = re.compile(r"rate limit|exhausted or cooling down|429|RESOURCE_EXHAUSTED", re.I)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


@dataclass
class QuerySpec:
    id: str
    cat: str
    query: str
    scope_hint: str | None = None
    max_results: int | None = None
    lang: str = "en"
    sub: str | None = None
    expect: dict = field(default_factory=dict)
    negative: dict | None = None
    notes: str | None = None


def load_queries(repo_id: str) -> list[QuerySpec]:
    path = EVAL_DIR / "queries" / f"{repo_id}.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    return [QuerySpec(**{k: v for k, v in item.items() if k in QuerySpec.__dataclass_fields__}) for item in raw]


def load_repos() -> list[dict]:
    with open(EVAL_DIR / "repos.toml", "rb") as f:
        import tomllib

        data = tomllib.load(f)
    return data["repo"]


def clone_path(repo: dict) -> Path:
    return Path(repo["path"]).expanduser()


def checkout_pin(repo: dict) -> None:
    path = clone_path(repo)
    pin = repo["pin"]
    subprocess.run(["git", "-C", str(path), "checkout", "--quiet", "--detach", pin], check=True)
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        print(f"WARNING: {repo['id']} is not clean at pin {pin}:\n{status}", file=sys.stderr)


class StderrTap:
    """The server's stderr, redirected straight to a real file (stdio_client's `errlog` wants a
    real fd via `.fileno()` for the subprocess spawn — a pure-Python write() sink doesn't work).
    Because this script issues calls strictly sequentially on one server process, the file's
    LINE COUNT before and after a call exactly delimits that call's own log lines — no fuzzy
    wall-clock slicing needed. (Once the observability patch lands, its `req_id` span makes this
    exact even for concurrent calls, e.g. R-17; this line-count approach is a Mode-B-appropriate
    stand-in that happens to be exact for the sequential case this script always runs.)"""

    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "w+")

    def fileno(self) -> int:
        return self._fh.fileno()

    def line_count(self) -> int:
        self._fh.flush()
        with open(self.path) as f:
            return sum(1 for _ in f)

    def lines_between(self, start_line: int, end_line: int) -> list[str]:
        self._fh.flush()
        with open(self.path) as f:
            lines = f.readlines()
        return [strip_ansi(l.rstrip("\n")) for l in lines[start_line:end_line]]

    def close(self) -> None:
        self._fh.close()


def parse_stage(lines: list[str]) -> dict:
    """Mode-B stage/candidate/confidence extraction from a slice of stderr lines belonging to
    one call. Returns a dict with keys: stage, tokens, candidates, confidence, escalated,
    leg_failures (list of {leg, error})."""
    out = {
        "stage": None,
        "tokens": None,
        "candidates": None,
        "confidence": None,
        "escalated": False,
        "leg_failures": [],
    }
    for line in lines:
        if m := CACHE_LINE_RE.search(line):
            out["stage"] = "cache"
            out["tokens"] = 0
        elif m := COMPLETE_LINE_RE.search(line):
            out["stage"] = m.group("path")
            out["tokens"] = int(m.group("tokens"))
        if m := RETRIEVAL_LINE_RE.search(line):
            out["candidates"] = int(m.group("candidates"))
            out["confidence"] = int(m.group("confidence"))
        if ESCALATE_RE.search(line):
            out["escalated"] = True
        if m := LEG_FAILED_RE.search(line):
            out["leg_failures"].append({"leg": m.group("leg"), "error": m.group("error")})
    return out


def query_cache_key(q: QuerySpec) -> str:
    # Mirrors crates/repo-explorer-agent/src/cache.rs query_key(): trim+lowercase text, then
    # scope_hint, then max_results, joined. Used only to label rows here, not to predict caching.
    return f"{q.query.strip().lower()}|{q.scope_hint or ''}|{q.max_results if q.max_results is not None else ''}"


async def run_one_repo_pass(repo: dict, queries: list[QuerySpec], config: Path, pass_n: int, out_dir: Path) -> list[dict]:
    repo_id = repo["id"]
    pass_dir = out_dir / repo_id
    pass_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = pass_dir / f"pass{pass_n}.stderr.log"
    jsonl_path = pass_dir / f"pass{pass_n}.jsonl"

    checkout_pin(repo)

    tap = StderrTap(stderr_path)
    # `StdioServerParameters.env` REPLACES (merged only with the SDK's own minimal default env,
    # not the current process's), so the current environment must be passed through explicitly —
    # §8.1 step 2 says "env = current env + REPO_EXPLORER_CONFIG + NO_COLOR", not "env = only
    # these two". This is how GOOGLE_API_KEY (or whichever provider key the config names) reaches
    # the child.
    import os

    server_params = StdioServerParameters(
        command=str(BINARY),
        args=[],
        env={**os.environ, "REPO_EXPLORER_CONFIG": str(config.resolve()), "NO_COLOR": "1"},
        cwd=str(clone_path(repo)),
    )

    rows = []
    seq = 0
    try:
        async with stdio_client(server_params, errlog=tap) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert "explore_repository" in names, f"explore_repository missing from tools/list: {names}"

                async def call(q: QuerySpec, warm_up: bool, timeout: float) -> dict:
                    nonlocal seq
                    seq += 1
                    args = {"query": q.query}
                    if q.scope_hint:
                        args["scope_hint"] = q.scope_hint
                    if q.max_results is not None:
                        args["max_results"] = q.max_results
                    t0 = time.time()
                    line_before = tap.line_count()
                    is_error = False
                    timed_out = False
                    response_text = None
                    try:
                        result = await asyncio.wait_for(session.call_tool("explore_repository", args), timeout=timeout)
                        # mcp SDK's CallToolResult exposes this as `is_error` (snake_case) in
                        # Python even though the wire field is `isError` — verified against
                        # types.CallToolResult.model_fields during Phase-0 harness validation.
                        is_error = bool(getattr(result, "is_error", False))
                        response_text = "".join(
                            c.text for c in result.content if getattr(c, "type", None) == "text"
                        )
                    except TimeoutError:
                        timed_out = True
                        response_text = None
                    t1 = time.time()
                    line_after = tap.line_count()
                    stage_info = parse_stage(tap.lines_between(line_before, line_after))
                    row = {
                        "row_id": hashlib.sha256(f"{repo_id}-{pass_n}-{q.id}-{seq}".encode()).hexdigest()[:16],
                        "repo": repo_id,
                        "pin": repo["pin"],
                        "pass": pass_n,
                        "call_seq": seq,
                        "query_id": q.id if not warm_up else f"{q.id}-warmup",
                        "cat": q.cat,
                        "query": q.query,
                        "scope_hint": q.scope_hint,
                        "max_results": q.max_results,
                        "query_cache_key": query_cache_key(q),
                        "ts_start": t0,
                        "ts_end": t1,
                        "latency_ms": round((t1 - t0) * 1000, 1),
                        "is_error": is_error,
                        "timeout": timed_out,
                        "response": response_text,
                        **stage_info,
                    }
                    rows.append(row)
                    return row

                # Warm-up: first query of the corpus, generous timeout (§8.1 step 2: 900s; a
                # pilot smoke run may reasonably use a shorter value via --warmup-timeout).
                warm_up_timeout = 900
                first = queries[0]
                print(f"[{repo_id} pass{pass_n}] warm-up: {first.id}", file=sys.stderr)
                warm_row = await call(first, warm_up=True, timeout=warm_up_timeout)
                print(f"  -> stage={warm_row['stage']} latency={warm_row['latency_ms']}ms", file=sys.stderr)

                # Shuffled corpus, seed = pass number (§8.1 step 3).
                order = list(queries)
                random.Random(pass_n).shuffle(order)
                for q in order:
                    print(f"[{repo_id} pass{pass_n}] {q.id}: {q.query[:60]!r}", file=sys.stderr)
                    row = await call(q, warm_up=False, timeout=300)
                    print(
                        f"  -> stage={row['stage']} candidates={row['candidates']} "
                        f"confidence={row['confidence']} tokens={row['tokens']} "
                        f"latency={row['latency_ms']}ms error={row['is_error']} timeout={row['timeout']}",
                        file=sys.stderr,
                    )
                    if row["is_error"] and row["response"] and RATE_LIMIT_ISERROR_RE.search(row["response"]):
                        print(f"  rate-limited; pausing 60s before continuing", file=sys.stderr)
                        await asyncio.sleep(60)
    finally:
        tap.close()

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[{repo_id} pass{pass_n}] wrote {len(rows)} rows -> {jsonl_path}", file=sys.stderr)
    return rows


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", nargs="*", default=None, help="repo ids to run (default: all in repos.toml)")
    ap.add_argument("--passes", type=int, default=2, help="number of passes (Phase 1 default: 2)")
    ap.add_argument("--config", default=str(EVAL_DIR / "config" / "default.toml"))
    ap.add_argument("--out", default=None, help="results directory (default: results/<run-id>)")
    args = ap.parse_args()

    if not BINARY.exists():
        sys.exit(f"installed binary not found: {BINARY}")

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    repos = load_repos()
    if args.repos:
        repos = [r for r in repos if r["id"] in args.repos]
    if not repos:
        sys.exit("no matching repos in eval/repos.toml")

    manifest = {
        "run_id": run_id,
        "binary": str(BINARY),
        "binary_version": subprocess.run([str(BINARY), "--version"], capture_output=True, text=True).stdout.strip(),
        "config": args.config,
        "passes": args.passes,
        "repos": [r["id"] for r in repos],
        "rtk_version": subprocess.run(["rtk", "--version"], capture_output=True, text=True).stdout.strip(),
        "git_version": subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip(),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"run_id={run_id} out_dir={out_dir}", file=sys.stderr)
    print(json.dumps(manifest, indent=2), file=sys.stderr)

    for repo in repos:
        queries = load_queries(repo["id"])
        for pass_n in range(1, args.passes + 1):
            await run_one_repo_pass(repo, queries, Path(args.config), pass_n, out_dir)

    print(f"done: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
