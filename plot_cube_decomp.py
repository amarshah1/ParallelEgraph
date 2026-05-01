#!/usr/bin/env python3
"""Log-log speedup plot for cube_decomp.csv.

X axis: number of cores (parlay_threads)
Y axis: speedup over T=1 (wallclock_T1 / wallclock_T)

Encoding:
  * color  -> initial cube size k (one color per k)
  * line / marker style -> decomposition rate d (one style per d)

Dropped: configurations whose T=1 time is < 1ms (k=5) since speedups
there are dominated by scheduler/startup noise.

Usage:
    python3 plot_cube_decomp.py runs/<ts>/cube_decomp.csv
    python3 plot_cube_decomp.py --out fig.png runs/<ts>/cube_decomp.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: str):
    times: dict[tuple[int, int, int], float] = {}
    for r in csv.DictReader(open(path)):
        d = int(r["d"])
        k = int(r["n"])
        t = int(r["parlay_threads"])
        ms = float(r["wallclock_ms"])
        times[(d, k, t)] = ms
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="cube_decomp.csv path")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <csv-dir>/cube_decomp_speedup.png)")
    ap.add_argument("--min-t1-ms", type=float, default=1.0,
                    help="drop configs whose T=1 baseline is below this (default 1.0ms)")
    args = ap.parse_args()

    times = load(args.csv)
    ds = sorted({d for (d, _, _) in times})
    ks = sorted({k for (_, k, _) in times})
    threads = sorted({t for (_, _, t) in times})
    if not threads:
        sys.exit("no data")

    # Color per k, linestyle/marker per d.
    cmap = plt.get_cmap("viridis", max(len(ks), 2))
    color_for_k = {k: cmap(i / max(len(ks) - 1, 1)) for i, k in enumerate(ks)}
    linestyles = ["-", "--", "-.", ":", (0, (1, 1))]
    markers    = ["o", "s", "^", "D", "v"]
    style_for_d = {d: (linestyles[i % len(linestyles)],
                       markers[i % len(markers)])
                   for i, d in enumerate(ds)}

    fig, ax = plt.subplots(figsize=(8, 6))
    ideal = [t for t in threads]
    ax.plot(threads, ideal, color="grey", linestyle=":", linewidth=1,
            label="linear (ideal)", zorder=0)

    plotted_any = False
    for d in ds:
        ls, mk = style_for_d[d]
        for k in ks:
            t1 = times.get((d, k, threads[0]))
            if t1 is None or t1 < args.min_t1_ms:
                continue
            ys = []
            xs = []
            for t in threads:
                ms = times.get((d, k, t))
                if ms is None or ms <= 0:
                    continue
                xs.append(t)
                ys.append(t1 / ms)
            if not xs:
                continue
            ax.plot(xs, ys, color=color_for_k[k], linestyle=ls, marker=mk,
                    markersize=6, linewidth=1.5,
                    label=f"d={d}, k={k}")
            plotted_any = True

    if not plotted_any:
        sys.exit(f"no configs above --min-t1-ms={args.min_t1_ms}")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup (T=1 / T)")
    ax.set_title("cube workload — strong-scaling speedup\n"
                 "(color = k, linestyle/marker = d)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()

    out = args.out or str(Path(args.csv).with_name("cube_decomp_speedup.png"))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
