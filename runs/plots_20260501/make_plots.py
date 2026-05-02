#!/usr/bin/env python3
"""Generate plots from the 20260501_015054_psc + 20260501_101233_psc_egg runs.

Charts:
  (1) Log-log speedup-vs-threads (y = nelson_seq / par_close, except egg
      which uses close_s @ T=1 / close_s @ Tn since there is no seq row):
        random_loglog.png
        synthetic_loglog.png        (top 15 by T=1 par_close time)
        cube_decomp_loglog.png      (color=d, linestyle=k)
        egg_loglog.png              (top 10 by T=1 close_s)
  (2) Per-round phase-time stacked bars, one chart per workload, for all
      four suites:
        random_phasebars/<workload>.png
        synthetic_phasebars/<family_n>.png
        cube_decomp_phasebars/<d_k>.png
        egg_phasebars/<file>.png
  (3) Frontier-work-per-round line plot:
        cube_decomp_round_sizes.png   (all (d,k) combos that exist)
        egg_round_sizes.png           (all egg files that have trace data)
"""

from __future__ import annotations
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


def _integer_x_ticks(ax):
    """Force the x-axis to integer-only major ticks."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Global styling — applies to every figure in this script.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.45,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

ROOT = Path("/ocean/projects/cis230065p/ashah12/ParallelEgraph/runs")
RUN_MAIN = ROOT / "20260501_015054_psc"
RUN_EGG = ROOT / "20260501_101233_psc_egg"
RUN_RANDOM_XL = ROOT / "20260501_130934"        # XL→8XL ladder run
RUN_RANDOM_XL_HEAVY = ROOT / "20260501_165123"   # 16XL + 32XL run
OUT = ROOT / "plots_20260501"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_csv_concat(*paths):
    """Read multiple CSVs that share a schema and concatenate the rows."""
    rows: list[dict] = []
    for p in paths:
        if Path(p).exists():
            rows.extend(read_csv(p))
    return rows


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ---------------------------------------------------------------------------
# (1) log-log speedup-vs-threads
# ---------------------------------------------------------------------------

def _group_par_close_times(rows, key_fn):
    """rows of par_close → {label: {threads: [ms,...]}}"""
    s = defaultdict(lambda: defaultdict(list))
    for r in rows:
        label = key_fn(r)
        if label is None:
            continue
        s[label][int(r["parlay_threads"])].append(float(r["wallclock_ms"]))
    return s


def _seq_baseline(rows, key_fn):
    """rows of nelson_seq → {label: median ms}.

    nelson_seq is sequential and the C++ binaries record one row per
    (workload, threads, trial) but the time itself is thread-invariant —
    we collapse across threads and take the median of all trials.
    """
    by_label = defaultdict(list)
    for r in rows:
        label = key_fn(r)
        if label is None:
            continue
        by_label[label].append(float(r["wallclock_ms"]))
    return {lbl: median(vs) for lbl, vs in by_label.items()}


def loglog_speedup(par_rows, seq_rows, key_fn, out_path,
                   max_lines=None,
                   line_style_fn=None,    # (label) -> dict of plot kwargs
                   sort_for_top=None,     # (label, par_series) -> sort key
                   custom_legend=None,    # callable(ax) to draw a custom legend
                   show_data_legend=True):
    par_series = _group_par_close_times(par_rows, key_fn)
    seq_med = _seq_baseline(seq_rows, key_fn)

    labels = list(par_series.keys())
    if max_lines is not None and len(labels) > max_lines:
        if sort_for_top is None:
            sort_for_top = lambda lbl, s: \
                -median(s[min(s)]) if s else 0
        labels.sort(key=lambda lbl: sort_for_top(lbl, par_series[lbl]))
        labels = labels[:max_lines]

    fig, ax = plt.subplots(figsize=(8, 6))
    all_threads: set[int] = set()
    for label in labels:
        s = par_series[label]
        if label not in seq_med:
            continue
        baseline = seq_med[label]
        thr = sorted(s.keys())
        all_threads.update(thr)
        speeds = [baseline / median(s[t]) for t in thr]
        kwargs = dict(marker="o", markersize=4, alpha=0.85, label=str(label))
        if line_style_fn is not None:
            kwargs.update(line_style_fn(label))
        ax.plot(thr, speeds, **kwargs)
    if all_threads:
        ref = sorted(all_threads)
        ax.plot(ref, ref, color="black", linestyle=":", linewidth=1.2,
                label="linear (y = x)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup over sequential")

    if custom_legend is not None:
        custom_legend(ax)
    elif show_data_legend:
        if len(labels) <= 12:
            ax.legend(loc="best")
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
                      fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}  ({len(labels)} lines)")


def _egg_stem(name: str) -> str:
    """Strip a trailing .smt2 so egg.csv (with extension) and the
    egg_traces/<stem>_T<thr>.log basenames refer to the same workload key."""
    return name[:-len(".smt2")] if name.endswith(".smt2") else name


def _egg_color_map():
    """Shared {stem: rgba} for the egg log-log and egg round-size charts.

    Ordering: median close_s @ T=1 desc (slowest workload first), so the
    viridis ramp encodes "expensive vs cheap" the same way in both plots
    and a given color always means the same benchmark.

    Files that didn't run at T=1 (no baseline available in egg.csv) get
    a fallback gray; they will still be plotted, just without a stable
    cross-chart color since there is no baseline to rank them by.
    """
    egg_rows = [r for r in read_csv(RUN_EGG / "egg.csv")
                if r["result"] in ("sat", "unsat") and r["close_s"]]
    t1_ms = defaultdict(list)
    for r in egg_rows:
        if int(r["threads"]) != 1:
            continue
        try:
            t1_ms[r["file"]].append(float(r["close_s"]) * 1000.0)
        except ValueError:
            pass
    ranked_files = sorted(t1_ms, key=lambda f: -median(t1_ms[f]))
    cmap = plt.get_cmap("viridis")
    n = max(len(ranked_files) - 1, 1)
    color_by_stem = {_egg_stem(f): cmap(i / n)
                     for i, f in enumerate(ranked_files)}
    return color_by_stem, ranked_files  # ranked_files keeps the .smt2 suffix


RANDOM_WORKLOAD_ORDER = ["small", "deep-s", "medium", "deep-m",
                         "large", "deep-l",
                         "XL", "2XL", "4XL", "8XL", "16XL", "32XL"]


def _random_color_map():
    """Shared {workload: rgba} for the random log-log and round-size charts.

    Ordered small→deep-l→XL→32XL so each depth-1/depth-3 size pair sits
    adjacent on the viridis ramp and the XL ladder continues past `deep-l`;
    same color = same workload across plots.
    """
    cmap = plt.get_cmap("viridis")
    n = max(len(RANDOM_WORKLOAD_ORDER) - 1, 1)
    return {w: cmap(i / n) for i, w in enumerate(RANDOM_WORKLOAD_ORDER)}


def _cube_styling(crows):
    """Shared color/linestyle/marker scheme for cube_decomp charts.

    color = k (≈ cube width), linestyle+marker = d (≈ tree depth / fan-in).
    Returns helpers usable from both loglog_speedup and the round-size
    chart so the two share an identical legend.
    """
    d_values = sorted({int(r["d"]) for r in crows})
    k_values = sorted({int(r["n"]) for r in crows})

    k_cmap = plt.get_cmap("viridis")
    if len(k_values) == 1:
        k_color = {k_values[0]: k_cmap(0.5)}
    else:
        k_color = {k: k_cmap(i / (len(k_values) - 1))
                   for i, k in enumerate(k_values)}

    style_cycle = [
        ("-",  "o"),
        ("--", "s"),
        ("-.", "^"),
        (":",  "D"),
        ((0, (3, 1, 1, 1)), "v"),
    ]
    d_style = {d: style_cycle[i % len(style_cycle)]
               for i, d in enumerate(d_values)}

    def label_from_row(r):
        return f"d{r['d']}_k{r['n']}"

    def parse(label):
        # Accepts both "d{d}_k{k}" (loglog) and "cube_d{d}_k{k}" (trace).
        m = re.search(r"d(\d+)_k(\d+)", label)
        return int(m.group(1)), int(m.group(2))

    def style_from_label(label):
        d, k = parse(label)
        ls, mk = d_style[d]
        return dict(color=k_color[k], linestyle=ls, marker=mk,
                    markersize=5, label="_nolegend_")

    def build_legend(ax, *, include_ref=True):
        k_handles = [
            Line2D([], [], color=k_color[k], linewidth=2.4, label=f"k = {k}")
            for k in k_values
        ]
        d_handles = [
            Line2D([], [], color="0.25", linestyle=d_style[d][0],
                   marker=d_style[d][1], markersize=6, linewidth=1.6,
                   label=f"d = {d}")
            for d in d_values
        ]
        leg1 = ax.legend(
            handles=k_handles,
            title="color: k  (≈ cube width)",
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            title_fontsize=9, alignment="left",
        )
        ax.add_artist(leg1)
        bottom_handles = list(d_handles)
        if include_ref:
            bottom_handles.append(
                Line2D([], [], color="black", linestyle=":", linewidth=1.2,
                       label="linear (y = x)"))
        ax.legend(
            handles=bottom_handles,
            title="line / marker: d  (≈ tree depth / fan-in)",
            loc="upper left", bbox_to_anchor=(1.02, 0.55),
            title_fontsize=9, alignment="left",
        )

    return {
        "label_from_row": label_from_row,
        "parse_label": parse,
        "style_from_label": style_from_label,
        "build_legend": build_legend,
    }


def make_loglog_plots():
    # random — color matches random_round_sizes.png so a given color always
    # means the same workload across the two charts. No legend (the colors
    # aren't self-describing without one, but that's intentional: the
    # round-size chart carries the legend for the whole pair).
    # Merges the original 6-workload run with the XL ladder run (XL→8XL,
    # plus 16/32XL when those land in the same folder).
    rrows = read_csv_concat(RUN_MAIN / "random.csv",
                            RUN_RANDOM_XL / "random.csv",
                            RUN_RANDOM_XL_HEAVY / "random.csv")
    random_colors = _random_color_map()

    def random_style(label):
        c = random_colors.get(label)
        return dict(color=c) if c is not None else {}

    loglog_speedup(
        [r for r in rrows if r["algorithm"] == "par_close"],
        [r for r in rrows if r["algorithm"] == "nelson_seq"],
        key_fn=lambda r: r["workload"],
        out_path=OUT / "random_loglog.png",
        line_style_fn=random_style,
    )

    # synthetic — top 15 by T=1 par_close time
    srows = read_csv(RUN_MAIN / "synthetic.csv")
    sort_top = lambda lbl, s: -median(s.get(1, [0.0])) if s else 0
    loglog_speedup(
        [r for r in srows if r["algorithm"] == "par_close"],
        [r for r in srows if r["algorithm"] == "nelson_seq"],
        key_fn=lambda r: f"{r['family']}_n{r['n']}",
        out_path=OUT / "synthetic_loglog.png",
        max_lines=15,
        sort_for_top=sort_top,
    )

    crows = read_csv(RUN_MAIN / "cube_decomp.csv")
    cube = _cube_styling(crows)
    loglog_speedup(
        [r for r in crows if r["algorithm"] == "par_close"],
        [r for r in crows if r["algorithm"] == "nelson_seq"],
        key_fn=cube["label_from_row"],
        out_path=OUT / "cube_decomp_loglog.png",
        line_style_fn=cube["style_from_label"],
        show_data_legend=False,
    )

    # egg: speedup uses close_s@T=1 as per-file baseline
    egg_rows = [r for r in read_csv(RUN_EGG / "egg.csv")
                if r["result"] in ("sat", "unsat") and r["close_s"]]

    # close-time per (file, threads) in ms
    file_t_close_ms = defaultdict(lambda: defaultdict(list))
    for r in egg_rows:
        try:
            ms = float(r["close_s"]) * 1000.0
        except ValueError:
            continue
        file_t_close_ms[r["file"]][int(r["threads"])].append(ms)

    # All files that have a T=1 baseline (no top-N truncation).
    files_with_t1 = {f: median(t1) for f, ts in file_t_close_ms.items()
                     if (t1 := ts.get(1))}
    all_files = sorted(files_with_t1, key=lambda f: -files_with_t1[f])

    # Shared color map (also used by the egg round-size chart) so a given
    # color always corresponds to the same benchmark across the two plots.
    color_by_stem, _ = _egg_color_map()

    fig, ax = plt.subplots(figsize=(8, 6))
    all_threads: set[int] = set()
    for f in all_files:
        ts = file_t_close_ms[f]
        if 1 not in ts:
            continue
        t1 = median(ts[1])
        thr = sorted(ts.keys())
        all_threads.update(thr)
        speeds = [t1 / median(ts[t]) for t in thr]
        ax.plot(thr, speeds, marker="o", markersize=2.5, alpha=0.55,
                linewidth=0.9, color=color_by_stem.get(_egg_stem(f), "0.6"))
    if all_threads:
        ref = sorted(all_threads)
        ax.plot(ref, ref, color="black", linestyle=":", linewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup over sequential")
    fig.tight_layout()
    fig.savefig(OUT / "egg_loglog.png")
    plt.close(fig)
    print(f"  → {OUT / 'egg_loglog.png'}  ({len(all_files)} lines)")


# ---------------------------------------------------------------------------
# (2) per-round phase bar charts for every workload in every suite
# ---------------------------------------------------------------------------

def _phase_bars_from_trace(trace_rows, out_dir: Path, suite: str):
    """trace_rows: dicts with workload, parlay_threads, round, *_ms.

    One bar chart per workload, stacked phases, at the highest thread count
    that appears for that workload.
    """
    out_dir.mkdir(exist_ok=True)
    # Group: workload -> threads available
    by_w_t = defaultdict(set)
    for r in trace_rows:
        by_w_t[r["workload"]].add(int(r["parlay_threads"]))

    # workload -> round -> phase -> [ms across trials]
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    target_t = {w: max(ts) for w, ts in by_w_t.items()}
    for r in trace_rows:
        w = r["workload"]
        if int(r["parlay_threads"]) != target_t[w]:
            continue
        rd = int(r["round"])
        for phase, key in [("consolidate", "consolidate_ms"),
                           ("frontier",    "frontier_ms"),
                           ("semisort",    "semisort_ms")]:
            v = r.get(key, "")
            if v == "" or v is None:
                continue
            try:
                bucket[w][rd][phase].append(float(v))
            except ValueError:
                pass

    n_made = 0
    for w in sorted(bucket):
        rounds = sorted(bucket[w].keys())
        if not rounds:
            continue
        cons = [median(bucket[w][rd]["consolidate"]) for rd in rounds]
        fron = [median(bucket[w][rd]["frontier"])    for rd in rounds]
        semi = [median(bucket[w][rd]["semisort"])    for rd in rounds]

        fig, ax = plt.subplots(figsize=(max(6, len(rounds) * 0.4), 5))
        x = np.arange(len(rounds))
        ax.bar(x, cons, label="consolidate", color="#4C72B0")
        ax.bar(x, fron, bottom=cons, label="frontier", color="#55A868")
        bottoms2 = [a + b for a, b in zip(cons, fron)]
        ax.bar(x, semi, bottom=bottoms2, label="semisort", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in rounds],
                           rotation=0 if len(rounds) <= 30 else 90,
                           fontsize=7)
        ax.set_xlabel("round")
        ax.set_ylabel("median time per phase (ms)")
        ax.set_title(f"{suite} / {w}  (T={target_t[w]})")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        # Sanitize filename (egg files have dots / slashes are fine but stay safe)
        fname = re.sub(r"[^A-Za-z0-9._-]+", "_", w) + ".png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        n_made += 1
    print(f"  → {out_dir}/ ({n_made} charts)")


def make_random_combined_phasebar(out_path: Path):
    """Single stacked bar chart aggregating all 6 random benchmarks.

    For each (workload, round), compute that round's share of the workload's
    total measured time, split by phase. For each round, take the median of
    each phase-share *only across workloads that reached that round*. Stack
    consolidate/frontier/semisort and emit one bar per round index.
    """
    rows = read_csv_concat(RUN_MAIN / "random_trace.csv",
                           RUN_RANDOM_XL / "random_trace.csv",
                           RUN_RANDOM_XL_HEAVY / "random_trace.csv")
    by_w_t = defaultdict(set)
    for r in rows:
        by_w_t[r["workload"]].add(int(r["parlay_threads"]))
    target_t = {w: max(ts) for w, ts in by_w_t.items()}

    # workload -> round -> phase -> [ms across trials]
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        w = r["workload"]
        if int(r["parlay_threads"]) != target_t[w]:
            continue
        rd = int(r["round"])
        for phase, key in [("consolidate", "consolidate_ms"),
                           ("frontier",    "frontier_ms"),
                           ("semisort",    "semisort_ms")]:
            v = r.get(key, "")
            if not v:
                continue
            try:
                bucket[w][rd][phase].append(float(v))
            except ValueError:
                pass

    # Collapse trials → one (workload, round, phase) → median ms.
    per_w_r = {
        w: {rd: {p: median(vs) for p, vs in phases.items()}
            for rd, phases in rounds.items()}
        for w, rounds in bucket.items()
    }
    # Workload total = sum across all its rounds and phases.
    w_total = {w: sum(sum(p.values()) for p in rounds.values())
               for w, rounds in per_w_r.items() if rounds}

    # Round -> phase -> [fraction-of-workload-total over workloads that
    # reached this round].
    round_phase_fracs = defaultdict(lambda: defaultdict(list))
    for w, rounds in per_w_r.items():
        total = w_total.get(w, 0.0)
        if total <= 0:
            continue
        for rd, phases in rounds.items():
            for phase, ms in phases.items():
                round_phase_fracs[rd][phase].append(ms / total)

    rounds_sorted = sorted(round_phase_fracs.keys())
    cons = [median(round_phase_fracs[rd]["consolidate"])
            for rd in rounds_sorted]
    fron = [median(round_phase_fracs[rd]["frontier"])
            for rd in rounds_sorted]
    semi = [median(round_phase_fracs[rd]["semisort"])
            for rd in rounds_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(rounds_sorted) * 0.35), 5.5))
    x = np.arange(len(rounds_sorted))
    ax.bar(x, cons, label="consolidate", color="#4C72B0")
    ax.bar(x, fron, bottom=cons, label="frontier", color="#55A868")
    bottoms2 = [a + b for a, b in zip(cons, fron)]
    ax.bar(x, semi, bottom=bottoms2, label="semisort", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(r)) for r in rounds_sorted],
                       rotation=0 if len(rounds_sorted) <= 30 else 90,
                       fontsize=8)
    ax.set_xlabel("round")
    ax.set_ylabel("median fraction of total time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}  ({len(rounds_sorted)} rounds aggregated "
          f"across {len(per_w_r)} workloads)")


def make_phase_bars():
    _phase_bars_from_trace(
        read_csv(RUN_MAIN / "random_trace.csv"),
        OUT / "random_phasebars", "random")
    _phase_bars_from_trace(
        read_csv(RUN_MAIN / "synthetic_trace.csv"),
        OUT / "synthetic_phasebars", "synthetic")
    _phase_bars_from_trace(
        read_csv(RUN_MAIN / "cube_decomp_trace.csv"),
        OUT / "cube_decomp_phasebars", "cube_decomp")
    _phase_bars_from_trace(
        build_egg_trace_rows(RUN_EGG / "egg_traces"),
        OUT / "egg_phasebars", "egg")


# ---------------------------------------------------------------------------
# (3) frontier-size vs round
# ---------------------------------------------------------------------------

def _round_size_series(trace_rows):
    """workload → ordered list of (round, median work) at that workload's max T."""
    by_w_t = defaultdict(set)
    for r in trace_rows:
        by_w_t[r["workload"]].add(int(r["parlay_threads"]))
    target_t = {w: max(ts) for w, ts in by_w_t.items()}

    flat = defaultdict(lambda: defaultdict(list))
    for r in trace_rows:
        if int(r["parlay_threads"]) != target_t[r["workload"]]:
            continue
        try:
            flat[r["workload"]][int(r["round"])].append(int(r["work"]))
        except ValueError:
            pass

    return {w: {rd: median(vs) for rd, vs in rds.items()}
            for w, rds in flat.items()}


def _round_size_chart_plain(trace_rows, out_path, color_by_workload=None,
                            workload_order=None, show_legend=False,
                            line_alpha=0.55, line_width=0.9,
                            marker_size=2.5):
    """Generic round-size chart, no legend by default.

    color_by_workload, if provided, is a {workload: rgba} map (any missing
    workloads fall back to gray). Used by the egg chart to share its color
    map with the egg log-log so the same color means the same benchmark.
    Default: viridis ramp by alphabetical order of the workload key.

    workload_order, if provided, fixes the line drawing + legend order
    (any workloads it doesn't mention get appended alphabetically).
    """
    series = _round_size_series(trace_rows)
    if workload_order is not None:
        present = set(series.keys())
        ordered = [w for w in workload_order if w in present]
        ordered += sorted(present - set(ordered))
        workloads = ordered
    else:
        workloads = sorted(series.keys())

    if color_by_workload is None:
        cmap = plt.get_cmap("viridis")
        n = max(len(workloads) - 1, 1)
        color_for = {w: cmap(i / n) for i, w in enumerate(workloads)}
    else:
        color_for = {w: color_by_workload.get(w, "0.6") for w in workloads}

    fig, ax = plt.subplots(figsize=(9, 6))
    for w in workloads:
        rounds = sorted(series[w].keys())
        ys = [series[w][rd] for rd in rounds]
        ax.plot(rounds, ys, marker="o", markersize=marker_size,
                alpha=line_alpha, linewidth=line_width,
                color=color_for[w], label=str(w))
    _integer_x_ticks(ax)
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("frontier work")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}  ({len(workloads)} lines)")


def _cube_round_size_chart(out_path):
    """Cube round-size chart, sharing color/linestyle/marker + legend with
    the cube log-log so the two read together."""
    crows = read_csv(RUN_MAIN / "cube_decomp.csv")
    cube = _cube_styling(crows)
    series = _round_size_series(read_csv(RUN_MAIN / "cube_decomp_trace.csv"))
    workloads = sorted(series.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    for w in workloads:
        rounds = sorted(series[w].keys())
        ys = [series[w][rd] for rd in rounds]
        kwargs = cube["style_from_label"](w)
        kwargs["alpha"] = 0.85
        ax.plot(rounds, ys, **kwargs)
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("frontier work")
    _integer_x_ticks(ax)
    cube["build_legend"](ax, include_ref=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}  ({len(workloads)} lines)")


def make_round_size_charts():
    _cube_round_size_chart(OUT / "cube_decomp_round_sizes.png")

    # random: 6 baked-in workloads + the XL ladder, paired ordering
    # small→deep-l→XL→32XL. Color map shared with random_loglog.png.
    _round_size_chart_plain(
        read_csv_concat(RUN_MAIN / "random_trace.csv",
                        RUN_RANDOM_XL / "random_trace.csv",
                        RUN_RANDOM_XL_HEAVY / "random_trace.csv"),
        OUT / "random_round_sizes.png",
        color_by_workload=_random_color_map(),
        workload_order=RANDOM_WORKLOAD_ORDER,
        show_legend=True,
        line_alpha=0.9, line_width=1.4, marker_size=4,
    )

    egg_color_by_stem, _ = _egg_color_map()
    _round_size_chart_plain(
        build_egg_trace_rows(RUN_EGG / "egg_traces"),
        OUT / "egg_round_sizes.png",
        color_by_workload=egg_color_by_stem,
    )


def build_egg_trace_rows(trace_dir: Path):
    """egg_traces/<file>_T<thr>.log → trace_csv-shaped rows."""
    TRACE_RE = re.compile(
        r"^\[pe\] round=\s*(?P<round>\d+)\s+"
        r"work=\s*(?P<work>\d+)\s+"
        r"frontier=\s*(?P<frontier>\d+)\s+"
        r"next=\s*(?P<next>\d+)\s+"
        r"consolidate=\s*(?P<cons>[0-9.]+)ms\s+"
        r"frontier=\s*(?P<front>[0-9.]+)ms\s+"
        r"semisort=\s*(?P<semi>[0-9.]+)ms"
    )
    name_re = re.compile(r"^(?P<stem>.+?)_T(?P<t>\d+)\.log$")
    out = []
    for p in sorted(trace_dir.iterdir()):
        m = name_re.match(p.name)
        if not m:
            continue
        stem = m.group("stem")
        t = int(m.group("t"))
        text = p.read_text(errors="replace")
        for line in text.splitlines():
            mm = TRACE_RE.match(line)
            if not mm:
                continue
            out.append({
                "workload": stem,
                "parlay_threads": str(t),
                "round": mm.group("round"),
                "work": mm.group("work"),
                "frontier": mm.group("frontier"),
                "next": mm.group("next"),
                "consolidate_ms": mm.group("cons"),
                "frontier_ms": mm.group("front"),
                "semisort_ms": mm.group("semi"),
            })
    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== (1) log-log speedup vs threads ===")
    make_loglog_plots()
    # Phase bars deliberately not regenerated — keep existing ones.
    # print("=== (2) per-round phase bars (all suites) ===")
    # make_phase_bars()
    print("=== (2b) random combined phase bar ===")
    make_random_combined_phasebar(OUT / "random_phasebar_combined.png")
    print("=== (3) round sizes ===")
    make_round_size_charts()
    print(f"\nAll plots in: {OUT}")
