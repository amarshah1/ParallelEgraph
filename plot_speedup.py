#!/usr/bin/env python3
"""Make per-family log-log speedup plots from a sweep CSV.

Usage:
    python plot_speedup.py [csv_path] [out_dir]

Defaults: csv_path=synthetic_sweep.csv, out_dir=plots
"""

import os
import re
import sys
from collections import defaultdict
import csv

import matplotlib.pyplot as plt
import matplotlib as mpl


FAMILIES = ["chain", "grid", "cube", "quartic", "quintic", "exp"]
FAMILY_RE = re.compile(r"^(chain|grid|cube|quartic|quintic|exp)_n(\d+)_")


def family_and_n(filename: str):
    m = FAMILY_RE.match(filename)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def load_runs(csv_path: str):
    """Returns {family: {filename: {threads: close_s}}} skipping non-OK rows."""
    runs = {f: defaultdict(dict) for f in FAMILIES}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fam, _ = family_and_n(row["file"])
            if fam is None:
                continue
            try:
                threads = int(row["threads"])
                close = float(row["close_s"])
            except (ValueError, KeyError):
                continue
            # Skip timeouts / errors so they don't pollute speedup
            if row.get("result", "").lower() not in ("sat", "unsat"):
                continue
            # Skip rows with no closure measurement (close_s missing or 0).
            if close <= 0.0:
                continue
            runs[fam][row["file"]][threads] = close
    return runs


def plot_family(fam: str, files: dict, out_dir: str):
    """One plot: x=threads, y=speedup, one line per file."""
    if not files:
        print(f"  {fam}: no data, skipping")
        return

    # Sort files by `n` so the legend reads in order.
    by_n = sorted(files.items(), key=lambda kv: family_and_n(kv[0])[1] or 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = mpl.colormaps.get_cmap("viridis").resampled(max(1, len(by_n)))

    all_thread_counts = set()
    for filename, by_threads in by_n:
        if 1 not in by_threads:
            print(f"  skipping {filename}: no T=1 baseline")
            continue
        baseline = by_threads[1]
        thread_list = sorted(by_threads.keys())
        all_thread_counts.update(thread_list)
        speedups = [baseline / by_threads[t] for t in thread_list]
        n = family_and_n(filename)[1]
        ax.plot(thread_list, speedups,
                marker="o", linewidth=1.2, markersize=4,
                color=cmap(by_n.index((filename, by_threads))),
                label=f"n={n}")

    if all_thread_counts:
        # Ideal-scaling reference (y = x).
        tmin, tmax = min(all_thread_counts), max(all_thread_counts)
        ax.plot([tmin, tmax], [tmin, tmax],
                "--", color="gray", linewidth=1.0, label="ideal (y=x)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Threads (PARLAY_NUM_THREADS)")
    ax.set_ylabel("Closure speedup vs. T=1 (close_s_1 / close_s_T)")
    ax.set_title(f"{fam}: closure speedup vs cores (per benchmark)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8, ncol=2 if len(by_n) > 8 else 1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"speedup_{fam}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}  ({len(by_n)} benchmarks)")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_sweep.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"

    print(f"Reading {csv_path}")
    runs = load_runs(csv_path)
    for fam in FAMILIES:
        plot_family(fam, runs[fam], out_dir)


if __name__ == "__main__":
    main()
