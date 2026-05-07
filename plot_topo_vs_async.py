#!/usr/bin/env python3
"""Plot par_async speedup vs the sequential nelson_topo_iter baseline.

For each present input CSV under `<run_dir>/`:

  synthetic.csv    → synthetic_async_speedup_vs_nelson.png    (color=(fam,n), style=d)
  random.csv       → random_async_speedup_vs_nelson.png       (color=workload)
  cube_decomp.csv  → cube_decomp_async_speedup_vs_nelson.png  (color=(fam,n), style=d)
  egg.csv          → egg_async_speedup_vs_nelson.png          (color=top-N files by close_s)

  speedup(T) = nelson_topo_iter / par_async(T)

Sequential measurements at T>1 are inflated by parlay-worker contention
when the C++ binary runs every algo in one process at PARLAY_NUM_THREADS=T.
The plotter takes the T=1 baseline only; if T=1 is missing it falls back
to the cross-T median.

random/synthetic/cube_decomp use the C++ binaries' shared schema and
report wallclock_ms. egg uses egraph-cc's per-invocation schema and
reports close_s (closure phase only).

All values are medians across trials. Axes are log-2.

Usage:
    python3 plot_topo_vs_async.py runs/<ts>
    python3 plot_topo_vs_async.py --top-n 12 runs/<ts>
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


PAR_ALGO = "par_async"
BASELINE_ALGO = "nelson_topo_iter"


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(open(path)))


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _collect_synth(rows: list[dict[str, str]]):
    """{(fam, n, d, t, algo): median_ms}."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        try:
            fam = r["family"]; n = int(r["n"]); d = int(r["d"])
            t   = int(r["parlay_threads"])
            ms  = float(r["wallclock_ms"])
            algo = r["algorithm"]
        except (KeyError, ValueError):
            continue
        buckets[(fam, n, d, t, algo)].append(ms)
    return {k: median(v) for k, v in buckets.items() if v}


def _collect_random(rows: list[dict[str, str]]):
    """{(workload, t, algo): median_ms}."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        try:
            wl = r["workload"]
            t  = int(r["parlay_threads"])
            ms = float(r["wallclock_ms"])
            algo = r["algorithm"]
        except (KeyError, ValueError):
            continue
        buckets[(wl, t, algo)].append(ms)
    return {k: median(v) for k, v in buckets.items() if v}


def _collect_egg(rows: list[dict[str, str]]):
    """{(file, t, algo): median_close_seconds}.

    Egg's schema is per-invocation: file, expected, algorithm, threads,
    trial, result, wall_s, ..., close_s. We use close_s because that's
    the closure-phase time which is what the parallel algorithms target;
    wall_s additionally includes parse/build/check/dtor and would dilute
    the comparison.

    Skips rows whose `result` is timeout/error/empty.
    """
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        try:
            f = r["file"]
            t = int(r["threads"])
            close = float(r["close_s"])
            algo = r["algorithm"]
        except (KeyError, ValueError):
            continue
        if r.get("result", "").lower() in ("timeout", "error", ""):
            continue
        buckets[(f, t, algo)].append(close)
    return {k: median(v) for k, v in buckets.items() if v}


def plot_synth(med, out_path: Path):
    threads = sorted({t for (_, _, _, t, _) in med})
    triples = sorted({(fam, n, d) for (fam, n, d, _, a) in med
                      if a in (PAR_ALGO, BASELINE_ALGO)})
    if not threads or not triples:
        print(f"  skip {out_path.name}: no rows")
        return

    # nelson_topo_iter is sequential, but if it was run inside the same
    # process as the parallel algos, parlay's worker pool stays warm and
    # interferes with the "sequential" measurement at high T (cache
    # contention, NUMA placement). Empirically the T=128 baseline can be
    # ~2x the T=1 baseline. Use T=1 only; fall back to the cross-T median
    # when T=1 is missing so older CSVs still produce a plot.
    base_by_triple: dict[tuple, float] = {}
    base_t = 1 if 1 in threads else None
    for (fam, n, d) in triples:
        if base_t is not None and (fam, n, d, base_t, BASELINE_ALGO) in med:
            base_by_triple[(fam, n, d)] = med[(fam, n, d, base_t, BASELINE_ALGO)]
            continue
        vals = [med[(fam, n, d, t, BASELINE_ALGO)]
                for t in threads
                if (fam, n, d, t, BASELINE_ALGO) in med]
        if vals:
            base_by_triple[(fam, n, d)] = median(vals)

    if not base_by_triple:
        print(f"  skip {out_path.name}: no {BASELINE_ALGO} rows")
        return

    color_keys = sorted({(fam, n) for (fam, n, _) in base_by_triple})
    d_values   = sorted({d for (_, _, d) in base_by_triple})
    cmap = plt.get_cmap("viridis", max(len(color_keys), 2))
    color_for_ck = {ck: cmap(i / max(len(color_keys) - 1, 1))
                    for i, ck in enumerate(color_keys)}
    linestyles = ["-", "--", "-.", ":", (0, (1, 1))]
    markers    = ["o", "s", "^", "D", "v"]
    style_for_d = {d: (linestyles[i % len(linestyles)],
                       markers[i % len(markers)])
                   for i, d in enumerate(d_values)}

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(threads, threads, color="grey", linestyle=":", linewidth=1,
            label="linear (ideal)", zorder=0)

    plotted = 0
    for d in d_values:
        ls, mk = style_for_d[d]
        for ck in color_keys:
            fam, n = ck
            base = base_by_triple.get((fam, n, d))
            if base is None:
                continue
            xs, ys = [], []
            for t in threads:
                par = med.get((fam, n, d, t, PAR_ALGO))
                if par and par > 0:
                    xs.append(t); ys.append(base / par)
            if xs:
                ax.plot(xs, ys, color=color_for_ck[ck], linestyle=ls,
                        marker=mk, markersize=5, linewidth=1.4,
                        label=f"d={d} {fam} n={n}")
                plotted += 1

    if plotted == 0:
        print(f"  skip {out_path.name}: no overlapping cells")
        plt.close(fig)
        return

    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(threads); ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel(f"speedup ({BASELINE_ALGO} / {PAR_ALGO})")
    ax.set_title(f"synthetic — {PAR_ALGO} vs sequential {BASELINE_ALGO}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, out_path)


def plot_egg(med, out_path: Path, top_n: int):
    """Egg uses close_s as the time metric. Pick the top-N files by
    par_async closure time at the largest T (those are the ones with
    enough work to scale; cheap files just measure overhead)."""
    threads = sorted({t for (_, t, _) in med})
    if not threads:
        print(f"  skip {out_path.name}: no rows")
        return
    t_max = threads[-1]

    # Files with both a baseline (T=1) and a par_async measurement
    # at every-T. Rank by par_async at T_max.
    par_at_max: dict[str, float] = {}
    for (f, t, algo), v in med.items():
        if algo == PAR_ALGO and t == t_max:
            par_at_max[f] = v
    if not par_at_max:
        print(f"  skip {out_path.name}: no {PAR_ALGO} rows at T={t_max}")
        return
    top_files = [f for f, _ in
                 sorted(par_at_max.items(), key=lambda x: -x[1])[:top_n]]

    # Sequential baseline: T=1 only (avoid the parlay-worker contention
    # inflation on egraph-cc invocations at higher T).
    base_t = 1 if 1 in threads else None
    base_by_file: dict[str, float] = {}
    for f in top_files:
        if base_t is not None and (f, base_t, BASELINE_ALGO) in med:
            base_by_file[f] = med[(f, base_t, BASELINE_ALGO)]
            continue
        vals = [med[(f, t, BASELINE_ALGO)]
                for t in threads
                if (f, t, BASELINE_ALGO) in med]
        if vals:
            base_by_file[f] = median(vals)

    if not base_by_file:
        print(f"  skip {out_path.name}: no {BASELINE_ALGO} rows for top files")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(threads, threads, color="grey", linestyle=":", linewidth=1,
            label="linear (ideal)", zorder=0)
    cmap = plt.get_cmap("tab10")

    plotted = 0
    for i, f in enumerate(top_files):
        base = base_by_file.get(f)
        if base is None:
            continue
        xs, ys = [], []
        for t in threads:
            par = med.get((f, t, PAR_ALGO))
            if par and par > 0:
                xs.append(t); ys.append(base / par)
        if xs:
            short = f.split(".")[0]
            ax.plot(xs, ys, marker="o", color=cmap(i % 10),
                    linewidth=1.5, label=short)
            plotted += 1

    if plotted == 0:
        print(f"  skip {out_path.name}: no overlapping cells")
        plt.close(fig)
        return

    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(threads); ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel(f"speedup ({BASELINE_ALGO} / {PAR_ALGO})  [close_s]")
    ax.set_title(f"egg — top-{top_n} longest-close files, "
                 f"{PAR_ALGO} vs sequential {BASELINE_ALGO}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, out_path)


def plot_random(med, out_path: Path):
    threads = sorted({t for (_, t, _) in med})
    workloads = sorted({wl for (wl, _, a) in med
                        if a in (PAR_ALGO, BASELINE_ALGO)})
    if not threads or not workloads:
        print(f"  skip {out_path.name}: no rows")
        return

    # Sequential baseline: prefer the T=1 measurement. Higher-T samples
    # of nelson_topo_iter are inflated by parlay-worker contention when
    # the C++ binary runs every algo in one process at PARLAY_NUM_THREADS=T,
    # so taking a cross-T median would over-estimate the baseline by ~2x
    # at the high end and make par_async look better than it is.
    base_by_wl: dict[str, float] = {}
    base_t = 1 if 1 in threads else None
    for wl in workloads:
        if base_t is not None and (wl, base_t, BASELINE_ALGO) in med:
            base_by_wl[wl] = med[(wl, base_t, BASELINE_ALGO)]
            continue
        vals = [med[(wl, t, BASELINE_ALGO)]
                for t in threads
                if (wl, t, BASELINE_ALGO) in med]
        if vals:
            base_by_wl[wl] = median(vals)

    if not base_by_wl:
        print(f"  skip {out_path.name}: no {BASELINE_ALGO} rows")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(threads, threads, color="grey", linestyle=":", linewidth=1,
            label="linear (ideal)", zorder=0)
    cmap = plt.get_cmap("tab10")

    plotted = 0
    for i, wl in enumerate(sorted(base_by_wl)):
        base = base_by_wl[wl]
        xs, ys = [], []
        for t in threads:
            par = med.get((wl, t, PAR_ALGO))
            if par and par > 0:
                xs.append(t); ys.append(base / par)
        if xs:
            ax.plot(xs, ys, marker="o", color=cmap(i % 10),
                    linewidth=1.5, label=wl)
            plotted += 1

    if plotted == 0:
        print(f"  skip {out_path.name}: no overlapping cells")
        plt.close(fig)
        return

    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(threads); ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel(f"speedup ({BASELINE_ALGO} / {PAR_ALGO})")
    ax.set_title(f"random — {PAR_ALGO} vs sequential {BASELINE_ALGO}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    _save(fig, out_path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir",
                    help="runs/<ts>/ folder; processes synthetic.csv, "
                         "random.csv, cube_decomp.csv, and egg.csv if "
                         "present")
    ap.add_argument("--top-n", type=int, default=10,
                    help="N for top-N longest-close egg files (default 10)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")

    figs = run_dir / "figs"
    figs.mkdir(exist_ok=True)
    print(f"writing figures under {figs}")

    any_csv = False

    synth_csv = run_dir / "synthetic.csv"
    if synth_csv.exists():
        any_csv = True
        print(f"reading {synth_csv}")
        med = _collect_synth(_read_csv(synth_csv))
        if med:
            plot_synth(med, figs / "synthetic_async_speedup_vs_nelson.png")
        else:
            print("  no usable rows in synthetic.csv")

    rand_csv = run_dir / "random.csv"
    if rand_csv.exists():
        any_csv = True
        print(f"reading {rand_csv}")
        med = _collect_random(_read_csv(rand_csv))
        if med:
            plot_random(med, figs / "random_async_speedup_vs_nelson.png")
        else:
            print("  no usable rows in random.csv")

    cube_csv = run_dir / "cube_decomp.csv"
    if cube_csv.exists():
        any_csv = True
        print(f"reading {cube_csv}")
        # cube_decomp.csv shares synthetic_bench's schema, so plot_synth works.
        med = _collect_synth(_read_csv(cube_csv))
        if med:
            plot_synth(med, figs / "cube_decomp_async_speedup_vs_nelson.png")
        else:
            print("  no usable rows in cube_decomp.csv")

    egg_csv = run_dir / "egg.csv"
    if egg_csv.exists():
        any_csv = True
        print(f"reading {egg_csv}")
        med = _collect_egg(_read_csv(egg_csv))
        if med:
            plot_egg(med, figs / "egg_async_speedup_vs_nelson.png",
                     args.top_n)
        else:
            print("  no usable rows in egg.csv")

    if not any_csv:
        sys.exit(f"no synthetic.csv, random.csv, cube_decomp.csv, or "
                 f"egg.csv under {run_dir}")
    print("done.")


if __name__ == "__main__":
    main()
