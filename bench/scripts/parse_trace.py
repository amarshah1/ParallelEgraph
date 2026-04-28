"""Parse PE_TRACE stderr (`[pe] round=...`) into CSV.

Input lines look like:
  [pe] round=  0 work=    1234 frontier=    5678 next=     900 \
  consolidate= 1.234ms frontier= 0.567ms semisort= 2.345ms

Emits CSV columns:
  workload,parlay_threads,round,work,frontier,next,
  consolidate_ms,frontier_ms,semisort_ms

Run as:
  python3 bench/scripts/parse_trace.py <workload> <threads> < trace.log > trace.csv
"""
from __future__ import annotations

import argparse
import re
import sys


LINE_RE = re.compile(
    r"^\[pe\] round=\s*(?P<round>\d+)\s+"
    r"work=\s*(?P<work>\d+)\s+"
    r"frontier=\s*(?P<frontier>\d+)\s+"
    r"next=\s*(?P<next>\d+)\s+"
    r"consolidate=\s*(?P<cons>[0-9.]+)ms\s+"
    r"frontier=\s*(?P<front>[0-9.]+)ms\s+"
    r"semisort=\s*(?P<semi>[0-9.]+)ms"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("threads", type=int)
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()

    if not args.no_header:
        print(
            "workload,parlay_threads,round,work,frontier,next,"
            "consolidate_ms,frontier_ms,semisort_ms"
        )

    for line in sys.stdin:
        m = LINE_RE.search(line)
        if not m:
            continue
        print(
            f"{args.workload},{args.threads},"
            f"{m['round']},{m['work']},{m['frontier']},{m['next']},"
            f"{m['cons']},{m['front']},{m['semi']}"
        )


if __name__ == "__main__":
    main()
