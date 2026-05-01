#!/usr/bin/env python3
"""Generate (A)/(B)/(C) figures for a runs/<ts>/ output folder.

For each category present (random, synthetic, cube_decomp, egg):
  (A) per-round stacked-bar charts (consolidate / frontier_build /
      semi:keyed / semi:group_by_key / semi:per_group / semi:canon+dedup)
      — one figure per (workload, threads). All values are MEDIAN across
      trials (per-trial rounds with the same index are aggregated).

  (B) log-log speedup-vs-cores plot.
        random / synthetic / cube_decomp: speedup = nelson_seq_T1 /
          par_close_T (so both per_close_T1 and per_close_T_max get
          plotted on the same scale, both relative to the sequential
          baseline). Median across trials.
        egg: no nelson baseline available — falls back to self-relative
          (par_T1 / par_T) on the top-N longest-close files.

  (C) per-round time line plots (x = round, y = ms).
        random / synthetic / cube_decomp: one figure per representative
          configuration. We pick the largest k per family (synthetic
          / cube_decomp) or all 6 random workloads, plotted at T=1 and
          T=max.
        egg: top-N longest-close files at T=1 and T=max.

Figures land in runs/<ts>/figs/.

Usage:
    python3 plot_results.py runs/<ts>
    python3 plot_results.py --top-n 10 runs/<ts>
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import pandas as pd


# ---- shared helpers --------------------------------------------------------

PHASE_STACKS = [
    ("consolidate_ms",     "consolidate"),
    ("frontier_ms",        "frontier_build"),
    ("keyed_ms",           "semi: keyed"),
    ("group_by_ms",        "semi: group_by_key"),
    ("per_group_ms",       "semi: per_group"),
    ("semi_other_ms",      "semi: canon+dedup"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(open(path)))


def _f(s: str) -> float | None:
    """Float-or-None parse; trace CSVs leave keyed/group_by/per_group blank
    for older traces."""
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _should_skip(path: Path) -> bool:
    """Return True iff `path` already exists. Set PE_PLOT_FORCE=1 to
    overwrite. Closes any open figure to avoid leaks when callers bail
    early."""
    if os.environ.get("PE_PLOT_FORCE"):
        return False
    if path.exists():
        print(f"  skip (exists) {path}")
        return True
    return False


# ---- (A) per-round stacked bars -------------------------------------------

def plot_trace_bars(trace_csv: Path, out_dir: Path, label: str):
    """One stacked-bar fig per (workload, threads). MEDIAN across trials.

    Trace CSV rows are per (workload, threads, round, trial-instance);
    closure_compare repeats each workload's parallel-trial loop, so the
    same `round` index appears multiple times per (workload, threads).
    We take the median over those repetitions for each phase column.

    pandas read+groupby: ~10× faster than per-row Python aggregation on
    the 1.4M-row synthetic_trace.csv.
    """
    if not trace_csv.exists():
        return

    phase_cols = ["consolidate_ms", "frontier_ms",
                  "keyed_ms", "group_by_ms", "per_group_ms", "semisort_ms"]
    usecols = ["workload", "parlay_threads", "round"] + phase_cols
    df = pd.read_csv(trace_csv, usecols=lambda c: c in usecols)
    if df.empty:
        return
    # Coerce to numeric; missing trace columns get filled with NaN→0.
    for c in phase_cols:
        if c not in df.columns:
            df[c] = float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["parlay_threads"] = pd.to_numeric(df["parlay_threads"],
                                          errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["workload", "parlay_threads", "round"])

    med = (df.groupby(["workload", "parlay_threads", "round"], sort=True)
             [phase_cols].median()
             .reset_index())
    med = med.fillna(0.0)
    med["semi_other_ms"] = (med["semisort_ms"]
                             - med["keyed_ms"]
                             - med["group_by_ms"]
                             - med["per_group_ms"]).clip(lower=0.0)

    bar_dir = out_dir / "trace_bars"
    for (wl, t), grp in med.groupby(["workload", "parlay_threads"], sort=True):
        if grp.empty:
            continue
        # Sanitize workload for filename.
        safe = str(wl).replace("/", "_").replace(" ", "_")
        out_path = bar_dir / f"{label}_{safe}_T{int(t)}.png"
        if _should_skip(out_path):
            continue
        grp = grp.sort_values("round")
        xs = grp["round"].to_numpy()
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        bottom = [0.0] * len(grp)
        for col, lab in PHASE_STACKS:
            ys = grp[col].to_numpy()
            ax.bar(xs, ys, bottom=bottom, label=lab)
            bottom = [b + y for b, y in zip(bottom, ys)]
        ax.set_xlabel("round")
        ax.set_ylabel("ms (median across trials)")
        ax.set_title(f"{label}: per-round phase breakdown — {wl}, T={int(t)}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, out_path)


# ---- (B) log-log speedup ---------------------------------------------------

def _median_ms(rows: list[dict[str, str]],
               key_fn,
               ms_field: str) -> dict[tuple, float]:
    """Group rows by key_fn(row) → median(float(row[ms_field]))."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        try:
            v = float(r[ms_field])
        except (KeyError, ValueError, TypeError):
            continue
        k = key_fn(r)
        if k is None:
            continue
        buckets[k].append(v)
    return {k: median(v) for k, v in buckets.items() if v}


def plot_speedup_random(csv_path: Path, out_dir: Path):
    """X=threads, Y=nelson_T1/par_T, color=workload name."""
    if not csv_path.exists():
        return
    if _should_skip(out_dir / "random_speedup.png"):
        return
    rows = _read_csv(csv_path)
    # closure_compare_bench's CSV uses `workload` as the per-row label.
    name_col = "workload" if rows and "workload" in rows[0] else "name"
    # Per (name, algorithm, threads) -> median ms
    par = _median_ms(rows,
        key_fn=lambda r: (r[name_col], int(r["parlay_threads"]))
            if r["algorithm"] == "par_close" else None,
        ms_field="wallclock_ms")
    nel = _median_ms(rows,
        key_fn=lambda r: (r[name_col], int(r["parlay_threads"]))
            if r["algorithm"] == "nelson_seq" else None,
        ms_field="wallclock_ms")
    # Nelson is sequential; closure_compare runs it at every thread
    # count anyway. Take the min thread's median as the baseline (or
    # any — they should match within noise).
    nelson_baseline: dict[str, float] = {}
    for (name, _t), v in nel.items():
        nelson_baseline.setdefault(name, v)
        nelson_baseline[name] = min(nelson_baseline[name], v)
    # alternative: keep all-threads median
    nel_per_name: dict[str, list[float]] = defaultdict(list)
    for (name, _t), v in nel.items():
        nel_per_name[name].append(v)
    nelson_med: dict[str, float] = {n: median(vs) for n, vs in nel_per_name.items()}

    threads = sorted({t for (_, t) in par})
    names   = sorted({n for (n, _) in par})
    if not threads or not names:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    ax.plot(threads, threads, color="grey", linestyle=":", linewidth=1,
            label="linear (ideal)", zorder=0)
    for i, n in enumerate(names):
        baseline = nelson_med.get(n)
        if baseline is None:
            continue
        ys, xs = [], []
        for t in threads:
            ms = par.get((n, t))
            if ms and ms > 0:
                xs.append(t); ys.append(baseline / ms)
        if xs:
            ax.plot(xs, ys, marker="o", color=cmap(i % 10), label=n,
                    linewidth=1.5)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(threads); ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup (nelson_seq / par_close)")
    ax.set_title("random — strong-scaling speedup vs sequential")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    _save(fig, out_dir / "random_speedup.png")


def _plot_synth_like_speedup(csv_path: Path, out_dir: Path,
                              fig_name: str, title: str):
    """Common (synthetic / cube_decomp) speedup plot.

    color = (family, n) and linestyle/marker = d. nelson_seq is run at
    one thread count per (family, n, d) — we use that as the baseline.
    """
    if not csv_path.exists():
        return
    if _should_skip(out_dir / fig_name):
        return
    rows = _read_csv(csv_path)

    par_med: dict[tuple[str, int, int, int], float] = {}  # (fam, n, d, t)
    nel_med: dict[tuple[str, int, int], float] = {}        # (fam, n, d)

    par_buckets: dict[tuple, list[float]] = defaultdict(list)
    nel_buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        try:
            fam = r["family"]; n = int(r["n"]); d = int(r["d"])
            t   = int(r["parlay_threads"])
            ms  = float(r["wallclock_ms"])
        except (KeyError, ValueError):
            continue
        if r["algorithm"] == "par_close":
            par_buckets[(fam, n, d, t)].append(ms)
        elif r["algorithm"] == "nelson_seq":
            nel_buckets[(fam, n, d)].append(ms)
    par_med = {k: median(v) for k, v in par_buckets.items()}
    nel_med = {k: median(v) for k, v in nel_buckets.items()}

    threads = sorted({t for (_, _, _, t) in par_med})
    if not threads:
        return

    # Color = (family, n); linestyle/marker = d.
    color_keys = sorted({(fam, n) for (fam, n, _, _) in par_med})
    d_values = sorted({d for (_, _, d, _) in par_med})
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
    for d in d_values:
        ls, mk = style_for_d[d]
        for ck in color_keys:
            fam, n = ck
            base = nel_med.get((fam, n, d))
            if base is None:
                continue
            xs, ys = [], []
            for t in threads:
                ms = par_med.get((fam, n, d, t))
                if ms and ms > 0:
                    xs.append(t); ys.append(base / ms)
            if xs:
                ax.plot(xs, ys, color=color_for_ck[ck], linestyle=ls,
                        marker=mk, markersize=5, linewidth=1.4,
                        label=f"d={d} {fam} n={n}")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(threads); ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup (nelson_seq / par_close)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, out_dir / fig_name)


def plot_speedup_synthetic(csv_path: Path, out_dir: Path):
    _plot_synth_like_speedup(csv_path, out_dir,
        fig_name="synthetic_speedup.png",
        title="synthetic — speedup vs nelson_seq (color=(family,n), style=d)")


def plot_speedup_cube_decomp(csv_path: Path, out_dir: Path):
    _plot_synth_like_speedup(csv_path, out_dir,
        fig_name="cube_decomp_speedup.png",
        title="cube_decomp — speedup vs nelson_seq (color=(cube,k), style=d)")


def plot_speedup_egg(csv_path: Path, out_dir: Path, top_n: int):
    """Egg has no nelson baseline → use self-relative (T1/T) on top-N
    longest-close files."""
    if not csv_path.exists():
        return
    if _should_skip(out_dir / "egg_speedup.png"):
        return
    rows = _read_csv(csv_path)

    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        try:
            f = r["file"]; t = int(r["threads"])
            close = float(r["close_s"])
        except (KeyError, ValueError):
            continue
        if r.get("result", "").lower() in ("timeout", "error", ""):
            continue
        buckets[(f, t)].append(close)
    close_med = {k: median(v) for k, v in buckets.items()}
    threads = sorted({t for (_, t) in close_med})
    if not threads:
        return
    t_max = threads[-1]

    # Rank files by close_s at T=t_max (parallel-best); pick top-N.
    by_file_max: dict[str, float] = {}
    for (f, t), v in close_med.items():
        if t == t_max:
            by_file_max[f] = v
    top_files = sorted(by_file_max.items(), key=lambda x: -x[1])[:top_n]
    top_names = [f for f, _ in top_files]
    if not top_names:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(threads, threads, color="grey", linestyle=":", linewidth=1,
            label="linear (ideal)", zorder=0)
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(top_names):
        base = close_med.get((f, threads[0]))
        if base is None or base <= 0:
            continue
        xs, ys = [], []
        for t in threads:
            ms = close_med.get((f, t))
            if ms and ms > 0:
                xs.append(t); ys.append(base / ms)
        if xs:
            short = f.split(".")[0]
            ax.plot(xs, ys, marker="o", color=cmap(i % 10),
                    linewidth=1.5, label=short)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(threads); ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel("self-relative speedup (T1 / T)")
    ax.set_title(f"egg — top-{top_n} longest-close files (self-relative)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, out_dir / "egg_speedup.png")


# ---- (C) per-round time lines ---------------------------------------------

def _per_round_medians(trace_csv: Path) -> dict[tuple[str, int], list[tuple[int, float]]]:
    """(workload, threads) → sorted list of (round, median_total_ms)."""
    if not trace_csv.exists():
        return {}
    sum_cols = ["consolidate_ms", "frontier_ms", "semisort_ms"]
    usecols = ["workload", "parlay_threads", "round"] + sum_cols
    df = pd.read_csv(trace_csv, usecols=lambda c: c in usecols)
    if df.empty:
        return {}
    for c in sum_cols:
        if c not in df.columns:
            df[c] = float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["parlay_threads"] = pd.to_numeric(df["parlay_threads"],
                                          errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["workload", "parlay_threads", "round"])
    df["__total"] = df[sum_cols].sum(axis=1)
    med = (df.groupby(["workload", "parlay_threads", "round"], sort=True)
             ["__total"].median()
             .reset_index())
    out: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for wl, t, rd, tot in med.itertuples(index=False, name=None):
        out[(str(wl), int(t))].append((int(rd), float(tot)))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def plot_rounds_random(trace_csv: Path, out_dir: Path):
    by_wlt = _per_round_medians(trace_csv)
    if not by_wlt:
        return
    threads = sorted({t for (_, t) in by_wlt})
    if not threads:
        return
    t1, t_max = threads[0], threads[-1]
    workloads = sorted({w for (w, _) in by_wlt})
    for t_pick in [t1, t_max]:
        out_path = out_dir / f"random_rounds_T{t_pick}.png"
        if _should_skip(out_path):
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("tab10")
        for i, w in enumerate(workloads):
            rs = by_wlt.get((w, t_pick))
            if not rs:
                continue
            xs = [r for r, _ in rs]; ys = [v for _, v in rs]
            ax.plot(xs, ys, marker="o", color=cmap(i % 10), label=w,
                    linewidth=1.5)
        ax.set_xlabel("round")
        ax.set_ylabel("ms per round (median)")
        ax.set_title(f"random — per-round time, T={t_pick}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        _save(fig, out_path)


def plot_rounds_synthlike(trace_csv: Path, out_dir: Path,
                          fig_prefix: str, title_prefix: str,
                          family_filter: list[str] | None = None):
    """For synthetic/cube_decomp: pick the largest n per family. Plot at
    T=1 and T=max."""
    by_wlt = _per_round_medians(trace_csv)
    if not by_wlt:
        return
    threads = sorted({t for (_, t) in by_wlt})
    if not threads:
        return
    t1, t_max = threads[0], threads[-1]

    # Workloads here look like "cube_n10" or "cube_d2_k55" — group by
    # the prefix before the last "_<axis>=<n>" element. We just take the
    # workload with the largest numeric tail per non-tail prefix.
    workloads = sorted({w for (w, _) in by_wlt})
    # Pick top per "family" — heuristic: split on "_", reduce the trailing
    # group while keeping a representative sample. We just pick all and
    # plot at most 8 to keep figures readable.
    if family_filter:
        workloads = [w for w in workloads
                     if any(w.startswith(p) for p in family_filter)]
    workloads = workloads[:12]

    for t_pick in [t1, t_max]:
        out_path = out_dir / f"{fig_prefix}_rounds_T{t_pick}.png"
        if _should_skip(out_path):
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        cmap = plt.get_cmap("tab10")
        for i, w in enumerate(workloads):
            rs = by_wlt.get((w, t_pick))
            if not rs:
                continue
            xs = [r for r, _ in rs]; ys = [v for _, v in rs]
            ax.plot(xs, ys, marker="o", color=cmap(i % 10), label=w,
                    linewidth=1.3, markersize=4)
        ax.set_xlabel("round")
        ax.set_ylabel("ms per round (median)")
        ax.set_title(f"{title_prefix} — per-round time, T={t_pick}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)
        _save(fig, out_path)


def plot_rounds_egg(trace_csv: Path, egg_csv: Path, out_dir: Path,
                     top_n: int):
    by_wlt = _per_round_medians(trace_csv)
    if not by_wlt or not egg_csv.exists():
        return
    threads = sorted({t for (_, t) in by_wlt})
    if not threads:
        return
    t1, t_max = threads[0], threads[-1]

    # Pick top-N longest-close files at T=t_max.
    rows = _read_csv(egg_csv)
    close_buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        try:
            f = r["file"]; t = int(r["threads"])
            close = float(r["close_s"])
        except (KeyError, ValueError):
            continue
        if r.get("result", "").lower() in ("timeout", "error", ""):
            continue
        close_buckets[(f, t)].append(close)
    by_file_max: dict[str, float] = {}
    for (f, t), v in close_buckets.items():
        if t == t_max:
            by_file_max[f] = median(v)
    top_files = [f for f, _ in
                 sorted(by_file_max.items(), key=lambda x: -x[1])[:top_n]]
    # Trace's `workload` field is the file *stem* (no .smt2). Strip suffix.
    stems = [f.removesuffix(".smt2") for f in top_files]

    for t_pick in [t1, t_max]:
        out_path = out_dir / f"egg_rounds_T{t_pick}.png"
        if _should_skip(out_path):
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        cmap = plt.get_cmap("tab10")
        for i, stem in enumerate(stems):
            rs = by_wlt.get((stem, t_pick))
            if not rs:
                continue
            xs = [r for r, _ in rs]; ys = [v for _, v in rs]
            short = stem.split(".")[0]
            ax.plot(xs, ys, marker="o", color=cmap(i % 10), label=short,
                    linewidth=1.3, markersize=4)
        ax.set_xlabel("round")
        ax.set_ylabel("ms per round (median)")
        ax.set_title(f"egg — top-{top_n} longest-close, per-round, T={t_pick}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)
        _save(fig, out_path)


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="runs/<ts>/ folder")
    ap.add_argument("--top-n", type=int, default=10,
                    help="N for top-N longest-close egg files (default 10)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")
    figs = run_dir / "figs"
    figs.mkdir(exist_ok=True)
    print(f"writing figures under {figs}")

    # (A) per-round bar charts
    print("[A] per-round stacked bars")
    plot_trace_bars(run_dir / "random_trace.csv", figs, "random")
    plot_trace_bars(run_dir / "synthetic_trace.csv", figs, "synthetic")
    plot_trace_bars(run_dir / "cube_decomp_trace.csv", figs, "cube_decomp")
    plot_trace_bars(run_dir / "egg_trace.csv", figs, "egg")

    # (B) log-log speedup
    print("[B] speedup vs threads")
    plot_speedup_random(run_dir / "random.csv", figs)
    plot_speedup_synthetic(run_dir / "synthetic.csv", figs)
    plot_speedup_cube_decomp(run_dir / "cube_decomp.csv", figs)
    plot_speedup_egg(run_dir / "egg.csv", figs, args.top_n)

    # (C) per-round time
    print("[C] per-round time lines")
    plot_rounds_random(run_dir / "random_trace.csv", figs)
    plot_rounds_synthlike(run_dir / "synthetic_trace.csv", figs,
                          fig_prefix="synthetic",
                          title_prefix="synthetic")
    plot_rounds_synthlike(run_dir / "cube_decomp_trace.csv", figs,
                          fig_prefix="cube_decomp",
                          title_prefix="cube_decomp")
    plot_rounds_egg(run_dir / "egg_trace.csv", run_dir / "egg.csv",
                    figs, args.top_n)
    print("done.")


if __name__ == "__main__":
    main()
