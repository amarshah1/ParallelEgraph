#!/usr/bin/env python3
"""Plot par_async speedup vs the sequential nelson_topo_iter baseline.

For each present input CSV under `<run_dir>/`:

  synthetic.csv    → synthetic_async_speedup_vs_nelson.png    (color=(fam,n), style=d)
  random.csv       → random_async_speedup_vs_nelson.png       (color=workload)
  cube_decomp.csv  → cube_decomp_async_speedup_vs_nelson.png  (color=(fam,n), style=d)
  egg.csv          → egg_async_speedup_vs_nelson.png          (color=top-N files by close_s)
  gates.csv        → gates_async_speedup.png                  (color=top-N files by close_ms)

  speedup(T) = nelson_topo_iter / par_async(T)

Sequential measurements at T>1 are inflated by parlay-worker contention
when the C++ binary runs every algo in one process at PARLAY_NUM_THREADS=T.
The plotter takes the T=1 baseline only; if T=1 is missing it falls back
to the cross-T median.

random/synthetic/cube_decomp use the C++ binaries' shared schema and
report wallclock_ms. egg uses egraph-cc's per-invocation schema and
reports close_s (closure phase only). gates uses gates_bench's schema
which reports close_ms; gates_bench has NO sequential baseline (it
only runs the two par algorithms), so the gates plot uses
par_topo_iter(T=1) as the surrogate baseline — the closest available
"best-rounds-based-CC at one core" measurement.

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
# Use the minimum across these per-workload as the baseline. Different
# sequential algos win on different shapes — nelson_simple is fastest
# on dense / shallow inputs, nelson_topo_iter on deeper DAGs — so the
# fairest "speedup over best sequential" picks the per-workload min.
BASELINE_ALGOS = ("nelson_topo_iter", "nelson_simple")
BASELINE_TAG = " / ".join(BASELINE_ALGOS) + " (min)"


def _best_baseline(med: dict, key_prefix: tuple, threads: list[int]):
    """Return the per-workload baseline = min over BASELINE_ALGOS of the
    T=1 measurement (or cross-T median if T=1 is missing). Returns None
    if neither baseline algo has any rows for this workload.

    `key_prefix` is the leading tuple of the med-dict key, e.g.
    (fam, n, d) for synth, (file,) for egg/gates, (workload,) for random.
    The full key is `key_prefix + (t, algo)`.
    """
    base_t = 1 if 1 in threads else None
    candidates = []
    for algo in BASELINE_ALGOS:
        if base_t is not None and (key_prefix + (base_t, algo)) in med:
            candidates.append(med[key_prefix + (base_t, algo)])
            continue
        vals = [med[key_prefix + (t, algo)] for t in threads
                if (key_prefix + (t, algo)) in med]
        if vals:
            candidates.append(median(vals))
    return min(candidates) if candidates else None


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


def _collect_gates(rows: list[dict[str, str]]):
    """{(file, t, algo): median_close_ms}.

    gates_bench schema: file, suite, n_gates, n_literals, n_not_terms,
    total_classes, algorithm, trial, parlay_threads, read_s, parse_s,
    build_s, close_ms. Closure-only ms is what we want — read/parse/
    build are file-loading overhead, separately reported.
    """
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        try:
            f = r["file"]
            t = int(r["parlay_threads"])
            ms = float(r["close_ms"])
            algo = r["algorithm"]
        except (KeyError, ValueError):
            continue
        buckets[(f, t, algo)].append(ms)
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
                      if a == PAR_ALGO or a in BASELINE_ALGOS})
    if not threads or not triples:
        print(f"  skip {out_path.name}: no rows")
        return

    # Per-workload baseline = min(nelson_topo_iter, nelson_simple) at T=1
    # (or cross-T median if T=1 is missing). Sequential rows at T>1 are
    # inflated by parlay-worker contention when the C++ binary runs every
    # algo in one process at PARLAY_NUM_THREADS=T — empirically the
    # T=128 baseline can be ~2x the T=1 baseline.
    base_by_triple: dict[tuple, float] = {}
    for triple in triples:
        b = _best_baseline(med, triple, threads)
        if b is not None:
            base_by_triple[triple] = b

    if not base_by_triple:
        print(f"  skip {out_path.name}: no {BASELINE_TAG} rows")
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
    ax.set_ylabel(f"speedup ({BASELINE_TAG} / {PAR_ALGO})")
    ax.set_title(f"synthetic — {PAR_ALGO} vs sequential {BASELINE_TAG}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, out_path)


def plot_egg(med, out_path: Path, top_n: int, phase_label: str = "egg"):
    """Egg uses close_s as the time metric. Pick the top-N files by
    par_async closure time at the largest T (those are the ones with
    enough work to scale; cheap files just measure overhead).

    `phase_label` shows up in the title; pass the CSV stem so
    custom_smt phases (e.g. smt_benchmarks.csv) get a self-identifying
    plot.
    """
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

    # Per-file baseline = min(BASELINE_ALGOS) at T=1.
    base_by_file: dict[str, float] = {}
    for f in top_files:
        b = _best_baseline(med, (f,), threads)
        if b is not None:
            base_by_file[f] = b

    if not base_by_file:
        print(f"  skip {out_path.name}: no {BASELINE_TAG} rows for top files")
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
    ax.set_ylabel(f"speedup ({BASELINE_TAG} / {PAR_ALGO})  [close_s]")
    ax.set_title(f"{phase_label} — top-{top_n} longest-close files, "
                 f"{PAR_ALGO} vs sequential {BASELINE_TAG}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, out_path)


def plot_random(med, out_path: Path):
    threads = sorted({t for (_, t, _) in med})
    workloads = sorted({wl for (wl, _, a) in med
                        if a == PAR_ALGO or a in BASELINE_ALGOS})
    if not threads or not workloads:
        print(f"  skip {out_path.name}: no rows")
        return

    # Per-workload baseline = min(BASELINE_ALGOS) at T=1.
    base_by_wl: dict[str, float] = {}
    for wl in workloads:
        b = _best_baseline(med, (wl,), threads)
        if b is not None:
            base_by_wl[wl] = b

    if not base_by_wl:
        print(f"  skip {out_path.name}: no {BASELINE_TAG} rows")
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
    ax.set_ylabel(f"speedup ({BASELINE_TAG} / {PAR_ALGO})")
    ax.set_title(f"random — {PAR_ALGO} vs sequential {BASELINE_TAG}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    _save(fig, out_path)


def plot_gates(med, out_path: Path, top_n: int):
    """gates_bench can be run with any subset of {nelson_simple,
    nelson_topo_iter, par_close, par_topo_iter, par_async,
    par_async_cont}, so the plotter picks a baseline based on what's
    actually in the CSV. Preference order:

      1. min(nelson_simple, nelson_topo_iter) at T=1 — best
         sequential baseline. Picked per-file so we always compare
         par_async against the seq algo that actually won on that
         workload.
      2. par_topo_iter(T=1) — surrogate; closest "best-rounds-based-
         CC at one core" measurement available.
      3. par_async(T=1) — last-resort self-relative baseline. The
         resulting speedup is just par_async(T=1)/par_async(T) and
         degenerates to 1.0 at T=1, but it still surfaces async's
         strong-scaling curve when no other algorithm was run.

    Plot lines:
      * par_async(T) — solid, the algorithm we're evaluating.
      * par_topo_iter(T) — dashed, plotted only when par_topo_iter
        rows are present; serves as a parallel-scaling reference.

    Pick top-N files by par_async closure cost at T_max so we focus
    on workloads with enough work for parallelism to matter.
    """
    threads = sorted({t for (_, t, _) in med})
    if not threads:
        print(f"  skip {out_path.name}: no rows")
        return
    if 1 not in threads:
        print(f"  skip {out_path.name}: no T=1 measurement to use as baseline")
        return
    t_max = threads[-1]

    # Discover which algorithms are present at all.
    algos_present = {a for (_, _, a) in med}

    # Rank files by par_async closure cost at T_max — the more work
    # par_async did, the more interesting the scaling.
    cost_at_max: dict[str, float] = {}
    for (f, t, algo), v in med.items():
        if algo == PAR_ALGO and t == t_max:
            cost_at_max[f] = v
    if not cost_at_max:
        print(f"  skip {out_path.name}: no {PAR_ALGO} rows at T={t_max}")
        return
    top_files = [f for f, _ in
                 sorted(cost_at_max.items(), key=lambda x: -x[1])[:top_n]]

    # Pick baseline. First preference: per-file min over BASELINE_ALGOS
    # (the sequential algos) at T=1. Fall back through par_topo_iter and
    # par_async if no sequential rows are present.
    chosen_baseline: str | None = None
    base_by_file: dict[str, float] = {}
    seq_present = [a for a in BASELINE_ALGOS if a in algos_present]
    if seq_present:
        for f in top_files:
            vals = [med[(f, 1, a)] for a in seq_present
                    if (f, 1, a) in med and med[(f, 1, a)] > 0]
            if vals:
                base_by_file[f] = min(vals)
        if base_by_file:
            chosen_baseline = BASELINE_TAG
    # Fallbacks: par_topo_iter@T=1 then par_async@T=1.
    if chosen_baseline is None:
        for cand in ("par_topo_iter", PAR_ALGO):
            if cand not in algos_present:
                continue
            candidate_base: dict[str, float] = {}
            for f in top_files:
                v = med.get((f, 1, cand))
                if v is not None and v > 0:
                    candidate_base[f] = v
            if candidate_base:
                chosen_baseline = cand
                base_by_file = candidate_base
                break

    if chosen_baseline is None:
        print(f"  skip {out_path.name}: no usable baseline at T=1 "
              f"(tried {seq_present} + par_topo_iter + {PAR_ALGO})")
        return
    print(f"  gates baseline: {chosen_baseline}@T=1")

    # Did par_topo_iter rows show up for any top file? If yes, we'll
    # plot par_topo_iter as a dashed reference line alongside async.
    have_topo_curve = "par_topo_iter" in algos_present and any(
        med.get((f, t, "par_topo_iter")) for f in top_files for t in threads)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(threads, threads, color="grey", linestyle=":", linewidth=1,
            label="linear (ideal)", zorder=0)
    cmap = plt.get_cmap("tab10")

    plotted = 0
    for i, f in enumerate(top_files):
        base = base_by_file.get(f)
        if base is None:
            continue
        short = f.replace(".gates", "")
        # par_async (solid) — what we're evaluating.
        xs, ys = [], []
        for t in threads:
            par = med.get((f, t, PAR_ALGO))
            if par and par > 0:
                xs.append(t); ys.append(base / par)
        if xs:
            ax.plot(xs, ys, marker="o", color=cmap(i % 10),
                    linewidth=1.5, label=f"{short}  ({PAR_ALGO})")
            plotted += 1
        # par_topo_iter (dashed) — only if rows are available.
        if have_topo_curve:
            xs, ys = [], []
            for t in threads:
                tv = med.get((f, t, "par_topo_iter"))
                if tv and tv > 0:
                    xs.append(t); ys.append(base / tv)
            if xs:
                ax.plot(xs, ys, marker="s", linestyle="--",
                        color=cmap(i % 10), linewidth=1.0, alpha=0.6,
                        label=f"{short}  (par_topo_iter)")

    if plotted == 0:
        print(f"  skip {out_path.name}: no overlapping cells")
        plt.close(fig)
        return

    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(threads); ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("threads")
    # Title and y-axis annotate the baseline used so the reader can
    # interpret the y-values correctly. self-relative (par_async@T=1)
    # is annotated specially since it's degenerate at T=1.
    if chosen_baseline == PAR_ALGO:
        baseline_tag = f"{PAR_ALGO}@T=1 (self-relative)"
    elif chosen_baseline == BASELINE_TAG:
        baseline_tag = f"{BASELINE_TAG}@T=1"
    else:
        baseline_tag = f"{chosen_baseline}@T=1"
    ax.set_ylabel(f"speedup ({baseline_tag} / algo)  [close_ms]")
    title_extra = "" if have_topo_curve else "  (no par_topo_iter rows)"
    ax.set_title(f"gates — top-{top_n} files, {PAR_ALGO} vs "
                 f"{baseline_tag}{title_extra}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, out_path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir",
                    help="runs/<ts>/ folder; processes synthetic.csv, "
                         "random.csv, cube_decomp.csv, egg.csv, and "
                         "gates.csv if present")
    ap.add_argument("--top-n", type=int, default=10,
                    help="N for top-N longest-close egg/gates files "
                         "(default 10)")
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

    # custom_smt phases: any other CSV in the run dir produced by
    # run_egg (same schema as egg.csv). Phase name = CSV stem.
    handled = {"synthetic.csv", "random.csv", "cube_decomp.csv",
               "egg.csv", "gates.csv"}
    for csv_path in sorted(run_dir.glob("*.csv")):
        if csv_path.name in handled:
            continue
        any_csv = True
        phase = csv_path.stem
        print(f"reading {csv_path}")
        med = _collect_egg(_read_csv(csv_path))
        if med:
            plot_egg(med, figs / f"{phase}_async_speedup_vs_nelson.png",
                     args.top_n, phase_label=phase)
        else:
            print(f"  no usable rows in {csv_path.name}")

    gates_csv = run_dir / "gates.csv"
    if gates_csv.exists():
        any_csv = True
        print(f"reading {gates_csv}")
        med = _collect_gates(_read_csv(gates_csv))
        if med:
            plot_gates(med, figs / "gates_async_speedup.png", args.top_n)
        else:
            print("  no usable rows in gates.csv")

    if not any_csv:
        sys.exit(f"no synthetic.csv, random.csv, cube_decomp.csv, "
                 f"egg.csv, or gates.csv under {run_dir}")
    print("done.")


if __name__ == "__main__":
    main()
