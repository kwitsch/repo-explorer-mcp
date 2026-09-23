"""HTTP latency reference for the P2 gate (stdlib only).

Usage:
  bench_http.py --url http://127.0.0.1:8765 --golden DIR/golden.jsonl
                [--concurrency 4] [--runs 20]

Sends one POST /v1/systemone per state (the 10b request body, model
"typed-decisions"), at most --concurrency in flight, and prints p50/p95 ms.
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def judge_body(state):
    return {
        "model": "typed-decisions",
        "state": state,
        "questions": {
            "relevant": {
                "type": "choice",
                "instructions": "Does this code location answer the repository search query?",
                "criteria": {
                    "A": "yes, this location answers the query",
                    "B": "no, this location does not answer the query",
                },
            }
        },
    }


def post_one(url, state):
    data = json.dumps(judge_body(state)).encode()
    req = urllib.request.Request(
        url + "/v1/systemone", data=data, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()


def one_call(url, states, pool):
    list(pool.map(lambda s: post_one(url, s), states))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    states = []
    with open(args.golden) as f:
        next(f)  # header
        for line in f:
            if not line.strip():
                continue
            states.append(json.loads(line)["state"])
            if len(states) >= 12:
                break

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for _ in range(3):
            one_call(args.url, states, pool)
        ms = []
        for _ in range(args.runs):
            t = time.perf_counter()
            one_call(args.url, states, pool)
            ms.append((time.perf_counter() - t) * 1000.0)

    ms.sort()
    def q(p):
        return ms[round((len(ms) - 1) * p)]
    print(json.dumps({"p50_ms": round(q(0.5), 2), "p95_ms": round(q(0.95), 2)}))


if __name__ == "__main__":
    main()
