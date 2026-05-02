#!/usr/bin/env python3
"""Recover the partially-completed XL random run in this folder.

Two things to fix:
  1. `random.csv` has 310 rows with workload="custom" because of a bug in
     _retag_random_csv_rows that's since been fixed. The `merges` column
     uniquely identifies each ladder rung, so we map merges → label.
  2. `random_trace.csv` was never written (run_random_xl never finished
     normally). Reconstruct it by parsing every trace log under
     random_traces/ — filenames already encode the right (label, threads).
"""

from __future__ import annotations
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# n_merges is the unambiguous fingerprint of each ladder rung.
MERGES_TO_LABEL = {
    "200000":  "XL",
    "400000":  "2XL",
    "800000":  "4XL",
    "1600000": "8XL",
    "3200000": "16XL",
    "6400000": "32XL",
}

# ---------------------------------------------------------------------------
# (1) random.csv: rewrite workload column where it equals "custom".
# ---------------------------------------------------------------------------

src = HERE / "random.csv"
backup = HERE / "random.csv.bak"
if not backup.exists():
    backup.write_bytes(src.read_bytes())
    print(f"backup: {backup}")

with open(src, newline="") as f:
    rdr = csv.reader(f)
    header = next(rdr)
    rows = list(rdr)

merges_idx = header.index("merges")
workload_idx = header.index("workload")
fixed = 0
for row in rows:
    if row[workload_idx] != "custom":
        continue
    label = MERGES_TO_LABEL.get(row[merges_idx])
    if label is None:
        raise SystemExit(f"unrecognized merges={row[merges_idx]!r}; "
                         f"expand MERGES_TO_LABEL")
    row[workload_idx] = label
    fixed += 1

with open(src, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(f"random.csv: retagged {fixed} rows")

# ---------------------------------------------------------------------------
# (2) random_trace.csv: rebuild from the per-file trace logs.
# ---------------------------------------------------------------------------

# Mirrors run_all_benchmarks.parse_trace_log.
TRACE_RE = re.compile(
    r"^\[pe\] round=\s*(?P<round>\d+)\s+"
    r"work=\s*(?P<work>\d+)\s+"
    r"frontier=\s*(?P<frontier>\d+)\s+"
    r"next=\s*(?P<next>\d+)\s+"
    r"consolidate=\s*(?P<cons>[0-9.]+)ms\s+"
    r"frontier=\s*(?P<front>[0-9.]+)ms\s+"
    r"semisort=\s*(?P<semi>[0-9.]+)ms"
    r"(?:\s+\(keyed=\s*(?P<keyed>[0-9.]+)ms"
    r"\s+group_by=\s*(?P<group_by>[0-9.]+)ms"
    r"\s+per_group=\s*(?P<per_group>[0-9.]+)ms\))?"
)
RANDOM_BANNER_RE = re.compile(r"^\[bench\] running (\S+) \.\.\.")
NAME_RE = re.compile(r"^(?P<label>.+?)_T(?P<t>\d+)\.log$")
TRACE_COLUMNS = [
    "workload", "parlay_threads", "round", "work", "frontier", "next",
    "consolidate_ms", "frontier_ms", "semisort_ms",
    "keyed_ms", "group_by_ms", "per_group_ms",
]

trace_dir = HERE / "random_traces"
out_rows: list[list] = []
for log in sorted(trace_dir.iterdir()):
    m = NAME_RE.match(log.name)
    if not m:
        continue
    file_label = m.group("label")
    threads = int(m.group("t"))
    text = log.read_text(errors="replace")
    # The banner in older trace logs may still say "custom" (the retag
    # function only ran on stderr that was passed through it; some logs
    # were written verbatim). Anchor on the filename's label and use the
    # banner only as a divider between workloads in case multiple appear
    # (they shouldn't, since each invocation runs one PE_BENCH_CUSTOM).
    current = file_label
    for line in text.splitlines():
        mb = RANDOM_BANNER_RE.match(line)
        if mb is not None:
            tok = mb.group(1)
            current = file_label if tok == "custom" else tok
            continue
        mm = TRACE_RE.match(line)
        if mm is None:
            continue
        out_rows.append([
            current, threads,
            mm["round"], mm["work"], mm["frontier"], mm["next"],
            mm["cons"], mm["front"], mm["semi"],
            mm["keyed"] or "", mm["group_by"] or "", mm["per_group"] or "",
        ])

with open(HERE / "random_trace.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(TRACE_COLUMNS)
    w.writerows(out_rows)
print(f"random_trace.csv: wrote {len(out_rows)} rows")
