#!/usr/bin/env python3
"""Assert-based self-check for eval/score.py's snippet_found_at window (F-17),
snippet_chunks' blank-line handling (F-20), and the QW-0 efficiency aggregates
(percentile edge cases, graceful degradation on rows predating the QW-0 fields,
the unknown-model price fallback, and run.py's exploration-complete line parse).

No test framework, no fixtures, no results/ directory, no external corpus:
writes a synthetic file to a tempdir and asserts the classifications directly.
Run with:  uv run --with pyyaml eval/test_score.py   (prints OK; non-zero exit on failure)
"""
import contextlib
import io
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import (
    NA_ABSENT,
    PRICES,
    call_cost,
    llm_calls_of,
    load_queries,
    percentile,
    qw0_cache,
    qw0_cost,
    qw0_csv_rows,
    qw0_metrics,
    print_qw0_report,
    snippet_found_at,
)

# run.py imports the MCP client SDK at module scope purely to drive the server; the log-line
# parsers under test need none of it. Stubbing the import keeps this file runnable with just
# pyyaml, the way its docstring promises, instead of pulling `mcp` into the test path.
for _name in ("mcp", "mcp.client", "mcp.client.stdio"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["mcp"].ClientSession = object
sys.modules["mcp"].StdioServerParameters = object
sys.modules["mcp.client.stdio"].stdio_client = None

from run import parse_call_lines  # noqa: E402


# One `exploration complete` line exactly as the QW-0 telemetry contract specifies it: the new
# fields appended to the existing ones, each as its own flat tracing field.
COMPLETE_LINE = (
    "2026-09-15T07:19:41.767542Z  INFO explore{req_id=03bccc96-7}: repo_explorer_agent::agent: "
    'exploration complete path="verify" tokens=8653 llm_calls=2 forced_finish=false '
    'index_status="UpToDate" git_probe_ms=3 confidence=72 candidate_count=4 '
    'early_exit_route="none" cache_read_tokens=1024 cache_write_tokens=256 total_ms=8121'
)
PROVIDER_LINE = (
    "2026-09-15T07:19:44.421971Z  INFO explore{req_id=03bccc96-7}: repo_explorer_llm: "
    "provider call provider=gemini model_requested=gemini-3.7-flash "
    'model_served="gemini-3.7-flash" attempt=1 outcome="ok" latency_ms=2512 '
    "prompt_tokens=2388 completion_tokens=56 reasoning_tokens=36 "
    "cached_tokens=2048 cache_creation_tokens=0"
)


def _row(**kw) -> dict:
    base = {"repo": "self", "pass": 1, "query_id": "P1", "cat": "P1", "stage": "verify", "latency_ms": 100.0}
    base.update(kw)
    return base


def check_metrics_parse() -> None:
    """The contract's exploration-complete line, parsed end to end by run.py."""
    out = parse_call_lines([COMPLETE_LINE])
    assert out["stage"] == "verify", out["stage"]
    assert out["tokens"] == 8653
    assert out["llm_calls"] == 2
    assert out["candidate_count"] == 4
    assert out["early_exit_route"] == "none"
    assert out["cache_read_tokens"] == 1024
    assert out["cache_write_tokens"] == 256
    assert out["total_ms"] == 8121

    # A pre-QW-0 binary emits none of the new fields: they stay None, never 0.
    old = parse_call_lines(['INFO: exploration complete path="verify" tokens=1'])
    assert old["candidate_count"] is None and old["early_exit_route"] is None, old

    # The cache-hit line exits before the retrieval pre-stage, so `confidence` is never
    # measured on it. It must stay absent in the row — including for an older binary that
    # still logged the constructor's placeholder 0, which would read as a real score.
    cache = parse_call_lines([
        "INFO: exploration served from query cache "
        'path="cache" tokens=0 llm_calls=0 git_probe_ms=4 confidence=0 '
        'early_exit_route="none" cache_read_tokens=0 cache_write_tokens=0 total_ms=4'
    ])
    assert cache["stage"] == "cache", cache["stage"]
    assert cache["confidence"] is None, cache["confidence"]

    # Provider-call cache fields.
    pc = parse_call_lines([PROVIDER_LINE])["provider_calls"][0]
    assert pc["cached_tokens"] == 2048 and pc["cache_creation_tokens"] == 0, pc
    assert pc["prompt_tokens"] == 2388


def check_provider_call_ghosts() -> None:
    """core/src/llm.rs's router commentary shares the first two words of the real per-attempt
    line. Counting it doubled provider calls and made every cost aggregate unusable."""
    ghosts = [
        "DEBUG explore{req_id=a-0}: repo_explorer_core::llm: provider call succeeded "
        "provider=gemini model=gemini-3.7-flash",
        "WARN explore{req_id=a-0}: repo_explorer_core::llm: provider call failed, no failover "
        "provider=gemini",
        "WARN explore{req_id=a-0}: repo_explorer_agent::verify: verification stage provider "
        "call failed; escalating error=boom",
    ]
    assert parse_call_lines(ghosts)["provider_calls"] == []
    assert len(parse_call_lines([PROVIDER_LINE] + ghosts)["provider_calls"]) == 1


def check_percentile_edges() -> None:
    assert percentile([], 0.95) is None
    # n=1: the only sample is every percentile.
    assert percentile([7], 0.95) == 7
    assert percentile([7], 0.50) == 7
    # n=2, nearest-rank: p95 -> ceil(1.9)=2 -> the max; p50 -> ceil(1.0)=1 -> the lower.
    assert percentile([3, 9], 0.95) == 9
    assert percentile([3, 9], 0.50) == 3
    assert percentile([9, 3], 0.95) == 9  # input order must not matter
    # n=20: p95 -> ceil(19)=19 -> the 19th smallest (0-based index 18).
    vals = list(range(20))
    assert percentile(vals, 0.95) == 18
    assert percentile(vals, 0.50) == 9


def check_qw0_aggregates() -> None:
    rows = [
        _row(query_id="a", stage="verify", llm_calls=2, tokens=1000, latency_ms=100.0),
        _row(query_id="b", stage="fallback", llm_calls=6, tokens=30000, latency_ms=900.0),
        _row(query_id="c", stage="early-exit", llm_calls=0, tokens=0, latency_ms=10.0, early_exit_route="confidence"),
        _row(query_id="d", stage="early-exit", llm_calls=0, tokens=0, latency_ms=12.0, early_exit_route="unique-symbol"),
        _row(query_id="e", stage="cache", tokens=0, latency_ms=5.0),  # no llm_calls field at all
    ]
    agg = qw0_metrics(rows)
    assert agg["n_rows"] == 5
    # The cache row counts as 0 turns (definitional), so n=5 not 4.
    assert agg["llm_calls"]["n"] == 5
    assert abs(agg["llm_calls"]["mean"] - 8 / 5) < 1e-9
    assert agg["llm_calls"]["p95"] == 6
    assert agg["stage_exit"] == {"verify": 1, "fallback": 1, "early-exit": 2, "cache": 1}
    assert agg["early_exit_route"] == {"confidence": 1, "unique-symbol": 1}
    # Both token views: all rows (zeros included) vs LLM-touching rows only.
    assert agg["tokens_all"]["n"] == 5 and abs(agg["tokens_all"]["mean"] - 31000 / 5) < 1e-9
    assert agg["tokens_llm_rows"]["n"] == 2 and agg["tokens_llm_rows"]["mean"] == 15500
    assert agg["latency_ms"]["p95"] == 900.0
    assert llm_calls_of(rows[4]) == 0
    assert llm_calls_of(_row(stage="verify")) is None


def check_graceful_degradation() -> None:
    """A row set from a pre-QW-0 binary: none of the new fields exist. Every affected metric
    must come back None ("n/a (field absent in this run)" in the report), never 0."""
    rows = [
        _row(query_id="a", stage="verify", llm_calls=2, tokens=1000),
        _row(query_id="b", stage="early-exit", llm_calls=0, tokens=0),
    ]
    agg = qw0_metrics(rows)
    assert agg["early_exit_route"] is None, agg["early_exit_route"]
    assert agg["total_ms"] is None
    assert agg["cache"]["ratio"] is None and agg["cache"]["reported"] is False
    assert agg["cost"]["usd_per_query"] == 0.0  # no provider calls at all == nothing spent
    # The existing fields still aggregate.
    assert agg["llm_calls"]["mean"] == 1.0
    assert agg["tokens_all"]["mean"] == 500.0

    # Completely empty row set must not divide by zero anywhere.
    empty = qw0_metrics([])
    assert empty["llm_calls"] is None and empty["tokens_all"] is None
    assert empty["cost"]["usd_per_query"] is None

    flat_rows = [_row(query_id="a", stage="early-exit", llm_calls=0, tokens=0,
                      early_exit_route="unique-symbol", total_ms=42)]
    flat_agg = qw0_metrics(flat_rows)
    assert flat_agg["early_exit_route"] == {"unique-symbol": 1}, flat_agg["early_exit_route"]
    assert flat_agg["total_ms"]["mean"] == 42


def check_all_unpriced_report() -> None:
    """A run where EVERY provider call ran on an unpriced model: priced_calls is 0 too, so the
    report must still name the models instead of claiming there were no provider calls."""
    rows = [_row(query_id="a", provider_calls=[
        {"model_served": "mystery-1", "prompt_tokens": 100, "completion_tokens": 10}
    ])]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_qw0_report(qw0_metrics(rows))
    out = buf.getvalue()
    assert "unpriced models served: mystery-1 x1" in out, out
    assert "no priced provider calls" not in out, out
    # Nothing was priced, so the partial-subset line (a 0-denominator ratio) must not print.
    assert "priced subset only" not in out, out


def check_priced_subset_report() -> None:
    """A run mixing priced and unpriced rows: the "priced subset only" per-query figure must
    divide by the rows that actually had priced calls (rows_fully_priced), not by every scored
    row -- else a run mostly on an unpriced model wildly understates the priced subset's cost."""
    priced_call = {"model_served": "gemini-3.5-flash-lite", "prompt_tokens": 5_000_000, "completion_tokens": 0}
    unpriced_call = {"model_served": "mystery-1", "prompt_tokens": 100, "completion_tokens": 10}
    rows = [
        _row(query_id="a", provider_calls=[priced_call]),
        _row(query_id="b", provider_calls=[priced_call]),
        _row(query_id="c", provider_calls=[unpriced_call]),
        _row(query_id="d", provider_calls=[unpriced_call]),
        _row(query_id="e", provider_calls=[unpriced_call]),
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_qw0_report(qw0_metrics(rows))
    out = buf.getvalue()
    # $0.50 per priced call (5M input tokens @ $0.10/1M) over 2 fully-priced rows -> $0.50/query.
    # Dividing by all 5 rows (the bug) would print $0.20000/query instead.
    assert "priced subset only: $0.50000/query over 2 of 5 provider calls" in out, out


def check_price_fallback() -> None:
    priced = {"model_served": "gemini-3.5-flash-lite", "prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert abs(call_cost(priced) - (0.10 + 0.40)) < 1e-9
    # Reasoning tokens bill as output (plan §7.1).
    assert abs(call_cost({**priced, "reasoning_tokens": 1_000_000}) - (0.10 + 0.80)) < 1e-9

    # Unknown model WITH usage -> None (unpriced), never 0.
    unknown = {"model_served": "gemini-9.9-ultra", "prompt_tokens": 500, "completion_tokens": 5}
    assert call_cost(unknown) is None
    agg = qw0_cost([_row(provider_calls=[priced]), _row(provider_calls=[unknown])])
    assert agg["usd_per_query"] is None, "an unpriced model must not be silently costed at 0"
    assert agg["unpriced"] == {"gemini-9.9-ultra": 1}
    assert agg["priced_calls"] == 1

    # A failed attempt reports no usage at all: free, and not "unpriced".
    failed = {"model_served": None, "outcome": "rate_limited"}
    assert call_cost(failed) == 0.0
    only_failed = qw0_cost([_row(provider_calls=[priced, failed])])
    assert only_failed["unpriced"] == {} and only_failed["usd_per_query"] is not None
    assert only_failed["priced_calls"] == 1, "a zero-usage attempt must not pad the call count"

    # Every model in eval/config/default.toml's failover chain is priced.
    for model in (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ):
        assert model in PRICES, f"unpriced model in the eval failover chain: {model}"


def check_cache_ratio() -> None:
    call = {"model_served": "gemini-3.5-flash", "prompt_tokens": 1000, "completion_tokens": 10, "cached_tokens": 400}
    # Per-query field wins.
    r = _row(cache_read_tokens=400, cache_write_tokens=100, provider_calls=[call])
    c = qw0_cache([r])
    assert c["ratio"] == 0.4 and c["cache_write_tokens"] == 100
    # No per-query field: fall back to the provider calls' own cached_tokens.
    c = qw0_cache([_row(provider_calls=[call])])
    assert c["ratio"] == 0.4 and c["reported"] is True
    # Cache reported but genuinely zero: 0.0, distinct from absent.
    c = qw0_cache([_row(cache_read_tokens=0, provider_calls=[{**call, "cached_tokens": 0}])])
    assert c["ratio"] == 0.0 and c["reported"] is True
    # Nothing reported anywhere: None, so the report says n/a rather than 0%.
    c = qw0_cache([_row(provider_calls=[{k: v for k, v in call.items() if k != "cached_tokens"}])])
    assert c["ratio"] is None and c["reported"] is False


def check_csv_rows() -> None:
    rows = [
        _row(query_id="a", stage="cache", tokens=0, latency_ms=5.0),
        _row(
            query_id="b",
            stage="verify",
            llm_calls=2,
            tokens=1000,
            provider_calls=[
                {"model_served": "gemini-3.5-flash", "prompt_tokens": 900, "completion_tokens": 100,
                 "reasoning_tokens": 10}
            ],
        ),
    ]
    out = qw0_csv_rows({"run_id": "20260915T000000", "scored_rows": rows})
    assert out[0]["run_id"] == "20260915T000000"
    assert out[0]["llm_calls"] == 0, "cache row must export 0 turns, matching the aggregate"
    assert out[1]["prompt_tokens"] == 900 and out[1]["reasoning_tokens"] == 10
    assert float(out[1]["cost_usd"]) > 0
    # An unpriced model exports an empty cost cell, not a zero.
    unpriced = qw0_csv_rows(
        {"run_id": "x", "scored_rows": [_row(provider_calls=[{"model_served": "nope", "prompt_tokens": 1}])]}
    )
    assert unpriced[0]["cost_usd"] == ""


def check_repo_brief_metrics() -> None:
    """M-2's two Stage-5 fields: parsed off the same `exploration complete` line by run.py, and
    aggregated over fallback rows ONLY. A row set from a pre-M-2 binary must report them absent,
    never as a zero that would look like "the loop stopped re-orienting"."""
    # run.py half: the names must match the server's tracing fields verbatim, or score.py sees
    # nothing and silently prints n/a forever.
    parsed = parse_call_lines([
        "INFO explore{req_id=a-0}: repo_explorer_agent::agent: exploration complete "
        'path="fallback" tokens=30000 llm_calls=4 brief_tokens=742 orientation_calls_in_loop=0'
    ])
    assert parsed["stage"] == "fallback"
    assert parsed["brief_tokens"] == 742, parsed
    assert parsed["orientation_calls_in_loop"] == 0, parsed
    assert parse_call_lines([COMPLETE_LINE])["brief_tokens"] is None  # pre-M-2 line

    # Pre-M-2 rows: absent everywhere -> None (NA_ABSENT in the report), not an n=0/mean=0 dist.
    old = qw0_metrics([_row(query_id="a", stage="fallback", llm_calls=6, tokens=30000)])
    assert old["brief_tokens"] is None, old["brief_tokens"]
    assert old["orientation_calls_in_loop"] is None, old["orientation_calls_in_loop"]

    # Denominator is "the row carries a count", not the stage name: a non-Stage-5 row never
    # carries one, and a Stage-5 run that died in the provider chain (stage=="error") does.
    agg = qw0_metrics([
        _row(query_id="a", stage="fallback", llm_calls=4, tokens=30000,
             brief_tokens=700, orientation_calls_in_loop=1),
        _row(query_id="b", stage="error", llm_calls=5, tokens=31000,
             brief_tokens=900, orientation_calls_in_loop=1),
        _row(query_id="c", stage="verify", llm_calls=2, tokens=1000),
    ])
    assert agg["brief_tokens"]["n"] == 2 and agg["brief_tokens"]["mean"] == 800
    assert agg["orientation_calls_in_loop"]["n"] == 2
    assert agg["orientation_calls_in_loop"]["mean"] == 1.0, agg["orientation_calls_in_loop"]

    # A measured zero is a measurement, not an absence — this is the acceptance gate's reading.
    zeroed = qw0_metrics([_row(query_id="a", stage="fallback", brief_tokens=640,
                               orientation_calls_in_loop=0)])
    assert zeroed["orientation_calls_in_loop"]["mean"] == 0.0

    # Report + CSV both carry them.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_qw0_report(old)
    assert f"orientation_calls_in_loop (fallback rows): {NA_ABSENT}" in buf.getvalue(), buf.getvalue()
    csv_row = qw0_csv_rows({"run_id": "x", "scored_rows": [
        _row(query_id="a", stage="fallback", brief_tokens=640, orientation_calls_in_loop=0)
    ]})[0]
    assert csv_row["brief_tokens"] == 640 and csv_row["orientation_calls_in_loop"] == 0


def _write_file(dir_path: Path, name: str, n_lines: int) -> None:
    body = "\n".join(f"SENTINEL_{i:03d}_line_content" for i in range(n_lines))
    (dir_path / name).write_text(body + "\n")


def check_queries_ascii() -> None:
    for repo_id in ("self", "requests"):
        for item_id, item in load_queries(repo_id).items():
            q = item["query"]
            assert q.isascii(), f"{repo_id}: non-ASCII query in {item_id}: {q!r}"


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _write_file(repo, "sample.py", 40)

        # 1. Widened window: in-range hit far past line_start+3 but inside
        #    line_end+4 (chunk at file index 32; claimed span 5..35).
        #    Pre-fix -> "misaligned"; post-fix must be "ok".
        assert snippet_found_at(
            repo, "sample.py", 5, 35, "SENTINEL_032_line_content"
        ) == "ok"

        # 2. No over-correction: chunk genuinely outside the widened window
        #    (chunk at index 37; claimed span 5..10, window upper bound 14).
        assert snippet_found_at(
            repo, "sample.py", 5, 10, "SENTINEL_037_line_content"
        ) == "misaligned"

        # 3. Fabricated chunk: not present anywhere in the file.
        assert snippet_found_at(
            repo, "sample.py", 5, 35, "TOTALLY_ABSENT_SENTINEL_STRING"
        ) == "not_found"

        # 4. line_end is None must not raise TypeError; chunk (index 5) sits
        #    inside the legacy line_start-4..line_start+3 fallback window.
        assert snippet_found_at(
            repo, "sample.py", 5, None, "SENTINEL_005_line_content"
        ) == "ok"

        # 5. Tightened fallback: line_end is None must NOT silently gain the
        #    forward pad. Chunk at index 8 sits just past the legacy
        #    line_start+3 upper bound (window is indices 1..7), so it must be
        #    "misaligned" — it would wrongly be "ok" if the fallback still
        #    added the full +4 pad instead of the old +3.
        assert snippet_found_at(
            repo, "sample.py", 5, None, "SENTINEL_008_line_content"
        ) == "misaligned"

        # 6. Location genuinely unknown (line_start=None, the DTO convention
        #    for a symbol-only finding): a real in-file snippet must be "ok",
        #    never "misaligned" — there is no known range for it to fall
        #    outside of.
        assert snippet_found_at(
            repo, "sample.py", None, None, "SENTINEL_020_line_content"
        ) == "ok"

        # 7. Location unknown but the snippet is fabricated: still "not_found".
        assert snippet_found_at(
            repo, "sample.py", None, None, "TOTALLY_ABSENT_SENTINEL_STRING"
        ) == "not_found"

        # 8. F-20: a real blank line inside an otherwise-real multi-line
        #    snippet (e.g. a blank docstring line) must not break contiguous
        #    chunk matching. blank.py has a genuine blank line at index 2
        #    (0-based) between SENTINEL_001 and SENTINEL_003 — the same shape
        #    a docstring with a blank line, or two module members quoted
        #    together, produces.
        (repo / "blank.py").write_text(
            "SENTINEL_000_line_content\n"
            "SENTINEL_001_line_content\n"
            "\n"
            "SENTINEL_003_line_content\n"
        )
        assert snippet_found_at(
            repo,
            "blank.py",
            2,
            4,
            "SENTINEL_001_line_content\n\nSENTINEL_003_line_content",
        ) == "ok"

        # 9. F-17 follow-up: duplicated content (e.g. two functions with the
        #    same signature) must prefer the occurrence inside the claimed
        #    range over an earlier out-of-range duplicate. dup.py repeats
        #    "DUP_line_content" at indices 2 and 20 (1-based lines 3, 21);
        #    claiming line 21 must be "ok", not "misaligned" from latching
        #    onto the line-3 occurrence.
        (repo / "dup.py").write_text(
            "\n".join(
                "DUP_line_content" if i in (2, 20) else f"filler_{i:03d}"
                for i in range(30)
            )
            + "\n"
        )
        assert snippet_found_at(repo, "dup.py", 21, 21, "DUP_line_content") == "ok"

        # QW-0 efficiency block (eval/score.py) + run.py's QW-0 log parsing.
        check_metrics_parse()
        check_provider_call_ghosts()
        check_percentile_edges()
        check_qw0_aggregates()
        check_graceful_degradation()
        check_price_fallback()
        check_all_unpriced_report()
        check_priced_subset_report()
        check_cache_ratio()
        check_csv_rows()
        check_repo_brief_metrics()

        # English-only invariant: every eval query string is pure ASCII
        # (scoped to item["query"]; notes/comments keep their non-ASCII
        # punctuation).
        check_queries_ascii()

    print("OK")


if __name__ == "__main__":
    main()
