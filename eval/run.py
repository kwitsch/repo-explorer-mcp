#!/usr/bin/env python3
"""Layer A direct MCP driver (docs/eval/real-world-test-plan.md §8.1).

Run with: uv run --with mcp --with pyyaml eval/run.py [options]

Spawns the INSTALLED repo-explorer-mcp binary (~/.local/bin/repo-explorer-mcp), pointed at a
pinned checkout via cwd, with REPO_EXPLORER_CONFIG set to an eval config variant. Speaks MCP
over stdio (Python `mcp` SDK). Each repo x pass gets a fresh server process — an identical query
repeated within one process is a cache hit (§2 Stage 0), so independent attempts require
independent processes, never a repeated call on the same session.

Mode A (§3.1): the installed binary is 0.5.3, which carries the observability patch, so this
script runs in full Mode A. Every call is wrapped server-side in a `tracing::info_span!("explore",
req_id = ...)` (`crates/repo-explorer-mcp/src/server.rs` `build_req_id`) — `req_id` = the first 8
hex chars of sha256(the same normalized query-cache-key `AgentLoop::query_cache_key` uses) plus a
per-process 0-based counter incremented once per `explore_repository` call (`fetch_add` returns
the PRE-increment value). Since this script issues calls strictly sequentially on one server
process (§8.1 step 3), that counter is entirely predictable from this side too — `_ReqCounter`
below mirrors it — so every stderr line belonging to one call is matched by its req_id counter
suffix rather than guessed from a line-count or wall-clock window — verified against real 0.5.3
stderr output captured live on 2026-09-05, not derived from the source diff alone.
Field quoting follows tracing_subscriber's own rule, confirmed against live output: a bare field
name (`leg`, `path`, `outcome`, `model_served`) is quoted in the log line, `%field` (Display,
e.g. `%ctx`, `%tool_names_joined(...)`) is not, `?field`/serde_json-built fields render as
Debug/JSON (`["a","b"]`, `[{...}]`) — `extract_fields()` handles both without needing to know
which convention a given field used.

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

# ANSI escape codes are present in raw stderr whenever NO_COLOR isn't honoured by the reader
# (the fmt layer colours output even to a pipe) — stripped defensively before any regex runs,
# even though this script also sets NO_COLOR=1 in the child's environment.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Every line inside the `explore{req_id=<hash>-<n>}:` span carries this prefix (server.rs
# build_req_id/info_span). <n> is this process's 0-based per-call counter — see _ReqCounter.
REQ_ID_RE = re.compile(r"explore\{req_id=[0-9a-f]+-(?P<n>\d+)\}:")

# The message (last positional string arg to the tracing macro) that names each line, and the
# free-text-field lines that can't go through the generic key=value extractor below.
MESSAGE_RES = {
    "cache": re.compile(r"exploration served from query cache"),
    "complete": re.compile(r"exploration complete\b"),
    "retrieval_complete": re.compile(r"retrieval pre-stage complete\b"),
    "retrieval_patterns": re.compile(r"retrieval patterns\b"),
    "leg_done": re.compile(r"retrieval leg done\b"),
    "leg_failed": re.compile(r"retrieval leg failed\b"),
    "candidates": re.compile(r"retrieval candidates\b"),
    "index_status": re.compile(r"index status\b"),
    "provider_call": re.compile(r"provider call\b"),
    "verify_action": re.compile(r"verify action\b"),
    "fallback_turn": re.compile(r"fallback turn\b"),
    "escalated": re.compile(r"verification escalated to the fallback loop"),
    "exploration_failed": re.compile(r"exploration failed\b"),
}
RATE_LIMIT_ISERROR_RE = re.compile(r"rate limit|exhausted or cooling down|429|RESOURCE_EXHAUSTED", re.I)

# Generic `key=value` tokenizer for the field portion of a tracing line, handling every quoting
# convention seen live: a quoted string (`path="early-exit"`), a JSON-ish bracketed value — a
# Debug-formatted Vec<String> (`identifiers=["a", "b"]`) or a real serde_json array/object
# (`candidates=[{...}]`) — or a bare token (number, `true`/`false`, or an unquoted %Display string
# like `ctx=defined` or a comma-joined `tool_names=grep,find`). Non-greedy on brackets so several
# bracketed fields on one line (retrieval patterns has four) don't collapse into one match; this
# only breaks if a value itself contains a literal `]`, e.g. inside a path/literal/symbol token —
# an accepted, rare limitation for a stats harness, not a general-purpose log parser.
FIELD_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\[.*?\]|\{.*?\}|\S+)')


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def extract_fields(line: str) -> dict:
    """Parse every `key=value` pair after the tracing message on one log line. Values are
    coerced: quoted -> str (quotes stripped), bracketed -> json.loads (falls back to the raw
    bracket text on decode failure), `true`/`false` -> bool, else int -> float -> str."""
    fields: dict = {}
    for key, raw in FIELD_RE.findall(line):
        if raw.startswith('"') and raw.endswith('"'):
            fields[key] = raw[1:-1]
        elif raw and raw[0] in "[{":
            try:
                fields[key] = json.loads(raw)
            except json.JSONDecodeError:
                fields[key] = raw
        elif raw == "true":
            fields[key] = True
        elif raw == "false":
            fields[key] = False
        else:
            try:
                fields[key] = int(raw)
            except ValueError:
                try:
                    fields[key] = float(raw)
                except ValueError:
                    fields[key] = raw
    return fields


def line_kind(line: str) -> str | None:
    for kind, pattern in MESSAGE_RES.items():
        if pattern.search(line):
            return kind
    return None


def req_id_counter(line: str) -> int | None:
    m = REQ_ID_RE.search(line)
    return int(m.group("n")) if m else None


class _ReqCounter:
    """Mirrors the server's per-process `AtomicU64` request counter (server.rs `build_req_id`):
    starts at 0, one value consumed per `explore_repository` call in call order. Since this
    script issues calls strictly sequentially on one server process (never concurrently, except
    the dedicated R-17 robustness case which uses its own script path), the value this side
    expects for the Nth call always matches what the server actually assigned — no guessing."""

    def __init__(self):
        self._next = 0

    def next(self) -> int:
        n = self._next
        self._next += 1
        return n


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

    `lines_for(n, approx_before, approx_after)` is the primary accessor: it takes a rough
    line-count window (cheap to compute, bounds how much of the file to re-read) and filters it
    to exactly the lines whose `explore{req_id=...-n}:` counter matches `n` — the req_id is the
    ground truth; the window is just there so a multi-thousand-line file isn't rescanned in full
    per call. Falls back to the raw window (unfiltered) if no line in it carries a req_id at all,
    which degrades gracefully against a pre-0.5.3 binary."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "w+")

    def fileno(self) -> int:
        return self._fh.fileno()

    def line_count(self) -> int:
        self._fh.flush()
        with open(self.path) as f:
            return sum(1 for _ in f)

    def lines_for(self, n: int, approx_before: int, approx_after: int) -> list[str]:
        self._fh.flush()
        with open(self.path) as f:
            raw = f.readlines()
        window = [strip_ansi(l.rstrip("\n")) for l in raw[approx_before:approx_after]]
        matched = [l for l in window if req_id_counter(l) == n]
        if matched:
            return matched
        if any(req_id_counter(l) is not None for l in window):
            # req_id lines exist in this window but none match `n` — a real association miss,
            # not just an unpatched binary. Surface the window as-is rather than silently
            # returning nothing, so parse_call_lines still gets a chance at the old plain lines.
            return window
        return window  # pre-0.5.3 binary: no req_id anywhere, line-count window is all there is

    def close(self) -> None:
        self._fh.close()


def parse_call_lines(lines: list[str]) -> dict:
    """Full §7.1/§3.1 field extraction from one call's stderr lines (see `StderrTap.lines_for`).
    Keeps the original key names (stage, tokens, candidates, confidence, escalated,
    leg_failures) score.py already consumes, and adds every field the observability patch
    introduced: llm_calls, forced_finish, index_status, git_probe_ms, retrieval_patterns,
    leg_timings, candidates_ranked, provider_calls, verify_actions, fallback_turns,
    exploration_failed."""
    out = {
        "stage": None,
        "tokens": None,
        "candidates": None,
        "confidence": None,
        "escalated": False,
        "leg_failures": [],
        "llm_calls": None,
        "forced_finish": None,
        "index_status": None,
        "index_duration_ms": None,
        "git_probe_ms": None,
        "retrieval_patterns": None,
        "leg_timings": [],
        "candidates_ranked": None,
        "provider_calls": [],
        "verify_actions": [],
        "fallback_turns": [],
        "exploration_failed": None,
    }
    for line in lines:
        kind = line_kind(line)
        if kind is None:
            continue
        f = extract_fields(line)
        if kind == "cache":
            out["stage"] = "cache"
            out["tokens"] = f.get("tokens", 0)
            out["git_probe_ms"] = f.get("git_probe_ms")
        elif kind == "complete":
            out["stage"] = f.get("path")
            out["tokens"] = f.get("tokens")
            out["llm_calls"] = f.get("llm_calls")
            out["forced_finish"] = f.get("forced_finish")
            out["index_status"] = f.get("index_status")
            out["git_probe_ms"] = f.get("git_probe_ms")
        elif kind == "retrieval_complete":
            out["candidates"] = f.get("candidates")
            out["confidence"] = f.get("confidence")
        elif kind == "retrieval_patterns":
            out["retrieval_patterns"] = {
                k: f.get(k) for k in ("literals", "identifiers", "path_tokens", "grep_patterns")
            }
        elif kind == "leg_done":
            out["leg_timings"].append(
                {"leg": f.get("leg"), "ctx": f.get("ctx"), "hits": f.get("hits"), "duration_ms": f.get("duration_ms")}
            )
        elif kind == "leg_failed":
            # `error=` has no bracket/quote wrapper and can contain '=' itself (JSON error bodies
            # from codebase-memory-mcp do), so the generic tokenizer only gets a truncated prefix
            # — take everything from the first `error=` to end of line instead.
            m = re.search(r'leg="?(?P<leg>\w+)"?.*?\berror=(?P<error>.*)$', line)
            if m:
                out["leg_failures"].append({"leg": m.group("leg"), "error": m.group("error")})
        elif kind == "candidates":
            m = re.search(r"candidates=(\[.*\])\s*$", line)
            if m:
                try:
                    out["candidates_ranked"] = json.loads(m.group(1))
                except json.JSONDecodeError:
                    out["candidates_ranked"] = None
        elif kind == "index_status":
            out["index_status"] = f.get("outcome")
            out["index_duration_ms"] = f.get("duration_ms")
        elif kind == "provider_call":
            out["provider_calls"].append(
                {
                    "provider": f.get("provider"),
                    "model_requested": f.get("model_requested"),
                    "model_served": f.get("model_served"),
                    "attempt": f.get("attempt"),
                    "outcome": f.get("outcome"),
                    "latency_ms": f.get("latency_ms"),
                    "prompt_tokens": f.get("prompt_tokens"),
                    "completion_tokens": f.get("completion_tokens"),
                    "reasoning_tokens": f.get("reasoning_tokens"),
                }
            )
        elif kind == "verify_action":
            out["verify_actions"].append({"turn": f.get("turn"), "action": f.get("action")})
        elif kind == "fallback_turn":
            out["fallback_turns"].append(
                {
                    "turn": f.get("turn"),
                    "tool_names": f.get("tool_names"),
                    "rejected_single_call": f.get("rejected_single_call"),
                    "strikes": f.get("strikes"),
                    "budget_spent": f.get("budget_spent"),
                }
            )
        elif kind == "escalated":
            out["escalated"] = True
        elif kind == "exploration_failed":
            m = re.search(r'error_class="?(?P<cls>\w+)"?\s+message=(?P<msg>.*)$', line)
            if m:
                out["exploration_failed"] = {"error_class": m.group("cls"), "message": m.group("msg")}
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
    req_counter = _ReqCounter()
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
                    req_id_n = req_counter.next()
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
                    stage_info = parse_call_lines(tap.lines_for(req_id_n, line_before, line_after))
                    row = {
                        "row_id": hashlib.sha256(f"{repo_id}-{pass_n}-{q.id}-{seq}".encode()).hexdigest()[:16],
                        "repo": repo_id,
                        "pin": repo["pin"],
                        "pass": pass_n,
                        "call_seq": seq,
                        "req_id_n": req_id_n,
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
                print(
                    f"  -> stage={warm_row['stage']} index_status={warm_row['index_status']} "
                    f"latency={warm_row['latency_ms']}ms",
                    file=sys.stderr,
                )

                # Shuffled corpus, seed = pass number (§8.1 step 3).
                order = list(queries)
                random.Random(pass_n).shuffle(order)
                for q in order:
                    print(f"[{repo_id} pass{pass_n}] {q.id}: {q.query[:60]!r}", file=sys.stderr)
                    row = await call(q, warm_up=False, timeout=300)
                    models_served = sorted({pc["model_served"] for pc in row["provider_calls"] if pc.get("model_served")})
                    print(
                        f"  -> stage={row['stage']} candidates={row['candidates']} "
                        f"confidence={row['confidence']} tokens={row['tokens']} llm_calls={row['llm_calls']} "
                        f"forced_finish={row['forced_finish']} model_served={models_served or None} "
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
