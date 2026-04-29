"""Plot the CSVs produced by run.py into PNGs under bench/results/.

Subcommands:
  strong-scaling  — speedup vs threads, one line per workload
  wallclock       — par_close ms vs threads, log-log
  trace-rounds    — per-round time breakdown (consolidate/frontier/semisort)
  components      — phase wallclock vs threads
  workload-sweep  — par_close ms vs depth and vs merge_frac
  smt             — par_close ms vs n, one line per family
  all             — every plot in turn

Each subcommand reads its CSV from bench/results/ and writes a PNG there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _save(fig, name: str) -> Path:
    out = RESULTS / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.relative_to(RESULTS.parents[1])}")
    return out


def _read(csv: Path) -> pd.DataFrame:
    if not csv.exists():
        sys.exit(f"missing {csv} — run `python3 bench/scripts/run.py ...` first")
    return pd.read_csv(csv)


# ------------------------- strong scaling -------------------------


def plot_strong_scaling(_args):
    df = _read(RESULTS / "strong_scaling.csv")
    par = df[df["algorithm"] == "par_close"].copy()
    # Take median across trials per (workload, parlay_threads).
    med = (
        par.groupby(["workload", "parlay_threads"])["wallclock_ms"]
        .median()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    workloads = sorted(med["workload"].unique())
    for w in workloads:
        sub = med[med["workload"] == w].sort_values("parlay_threads")
        # Speedup = T1 / TN where T1 is wallclock at smallest thread count.
        baseline = sub.iloc[0]["wallclock_ms"]
        speedup = baseline / sub["wallclock_ms"]
        ax.plot(sub["parlay_threads"], speedup, marker="o", label=w)

    # Ideal y=x line.
    threads = sorted(med["parlay_threads"].unique())
    ax.plot(threads, threads, "k--", alpha=0.4, label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("parlay threads")
    ax.set_ylabel("speedup vs T=min")
    ax.set_title("parallel_close strong scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "fig_strong_scaling.png")


def plot_wallclock(_args):
    df = _read(RESULTS / "strong_scaling.csv")
    par = df[df["algorithm"] == "par_close"].copy()
    med = (
        par.groupby(["workload", "parlay_threads"])["wallclock_ms"]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in sorted(med["workload"].unique()):
        sub = med[med["workload"] == w].sort_values("parlay_threads")
        ax.plot(sub["parlay_threads"], sub["wallclock_ms"], marker="o", label=w)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("parlay threads")
    ax.set_ylabel("par_close wallclock (ms)")
    ax.set_title("parallel_close wallclock vs threads")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_wallclock.png")


# ------------------------- per-round trace -------------------------


def plot_trace_rounds(_args):
    df = _read(RESULTS / "trace.csv")
    # If sub-phase columns are present, decompose semisort into them plus a
    # "canon+other" residual = semisort_ms - (keyed + group_by + per_group).
    has_sub = {"keyed_ms", "group_by_ms", "per_group_ms"}.issubset(df.columns)
    cols = ["consolidate_ms", "frontier_ms", "semisort_ms"]
    if has_sub:
        cols += ["keyed_ms", "group_by_ms", "per_group_ms"]

    grp = (
        df.groupby(["workload", "parlay_threads", "round"])[cols]
        .mean()
        .reset_index()
    )

    for (w, t), sub in grp.groupby(["workload", "parlay_threads"]):
        sub = sub.sort_values("round").copy()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        bottom = [0.0] * len(sub)
        if has_sub:
            sub["semisort_other_ms"] = (
                sub["semisort_ms"]
                - sub["keyed_ms"].fillna(0)
                - sub["group_by_ms"].fillna(0)
                - sub["per_group_ms"].fillna(0)
            ).clip(lower=0)
            stacks = [
                ("consolidate_ms", "consolidate"),
                ("frontier_ms", "frontier_build"),
                ("keyed_ms", "semi: keyed"),
                ("group_by_ms", "semi: group_by_key"),
                ("per_group_ms", "semi: per_group"),
                ("semisort_other_ms", "semi: canon+dedup"),
            ]
        else:
            stacks = [
                ("consolidate_ms", "consolidate"),
                ("frontier_ms", "frontier_build"),
                ("semisort_ms", "semisort"),
            ]
        for col, label in stacks:
            vals = sub[col].fillna(0)
            ax.bar(sub["round"], vals, bottom=bottom, label=label)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xlabel("round")
        ax.set_ylabel("ms (mean across trials)")
        ax.set_title(f"per-round phase breakdown — {w}, T={t}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, f"fig_trace_rounds_{w}_T{t}.png")


# ------------------------- components -------------------------


def plot_components(_args):
    df = _read(RESULTS / "components.csv")
    med = (
        df.groupby(["phase", "parlay_threads", "workload"])["wallclock_ms"]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for phase in ("consolidate", "semisort"):
        sub = med[med["phase"] == phase].sort_values("parlay_threads")
        if sub.empty:
            continue
        ax.plot(sub["parlay_threads"], sub["wallclock_ms"], marker="o", label=phase)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("parlay threads")
    ax.set_ylabel("ms (median, round-0 in isolation)")
    workloads = sorted(med["workload"].unique())
    suffix = workloads[0] if len(workloads) == 1 else ",".join(workloads)
    ax.set_title(f"phase wallclock vs threads — {suffix}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_components.png")


# ------------------------- workload sweep -------------------------


def plot_workload_sweep(_args):
    df = _read(RESULTS / "workload_sweep.csv")
    par = df[df["algorithm"] == "par_close"].copy()
    par["merge_frac"] = par["merges"] / par["leaves"]
    med = (
        par.groupby(["depth", "merge_frac"])["wallclock_ms"]
        .median()
        .reset_index()
    )

    # vs depth (one line per merge_frac)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mf in sorted(med["merge_frac"].unique()):
        sub = med[med["merge_frac"] == mf].sort_values("depth")
        ax.plot(sub["depth"], sub["wallclock_ms"], marker="o",
                label=f"merge_frac={mf:.2f}")
    ax.set_xlabel("depth (function-node levels)")
    ax.set_ylabel("par_close wallclock (ms)")
    ax.set_title("workload sweep: par_close vs depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "fig_workload_depth.png")

    # vs merge_frac (one line per depth)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in sorted(med["depth"].unique()):
        sub = med[med["depth"] == d].sort_values("merge_frac")
        ax.plot(sub["merge_frac"], sub["wallclock_ms"], marker="o",
                label=f"depth={d}")
    ax.set_xlabel("merge_frac (merges / leaves)")
    ax.set_ylabel("par_close wallclock (ms)")
    ax.set_title("workload sweep: par_close vs merge_frac")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "fig_workload_merge_frac.png")


# ------------------------- smt -------------------------


def plot_width_grid(_args):
    df = _read(RESULTS / "width_grid.csv")
    par = df[df.algorithm == "par_close"].copy()
    nel = df[df.algorithm == "nelson_seq"].copy()
    if par.empty:
        return

    leaves_per_workload = par.groupby("workload")["leaves"].first()
    workload_order = leaves_per_workload.sort_values().index.tolist()

    # Use highest measured T per workload
    par_at_max_t = par.groupby("workload").apply(
        lambda g: g[g.parlay_threads == g.parlay_threads.max()][
            "wallclock_ms"
        ].median(),
        include_groups=False,
    )
    par_at_max_t = par_at_max_t.reindex(workload_order)
    leaves = leaves_per_workload.reindex(workload_order)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(leaves.values, par_at_max_t.values, marker="o", label="par_close")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_leaves")
    ax.set_ylabel("par_close wallclock (ms)")
    ax.set_title("Width-axis scaling (T_max)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, "fig_width_grid_wallclock.png")

    if not nel.empty:
        nel_med = nel.groupby("workload")["wallclock_ms"].median()
        common = par_at_max_t.dropna().index.intersection(nel_med.index)
        if len(common) >= 2:
            speedup = (nel_med.loc[common] / par_at_max_t.loc[common]).reindex(
                workload_order
            ).dropna()
            leaves_s = leaves.reindex(speedup.index)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(leaves_s.values, speedup.values, marker="o")
            ax.set_xscale("log")
            ax.set_xlabel("n_leaves")
            ax.set_ylabel("speedup over Nelson")
            ax.set_title("Width-axis: speedup over Nelson at T_max")
            ax.grid(True, which="both", alpha=0.3)
            _save(fig, "fig_width_grid_speedup.png")


def plot_depth_sweep(_args):
    df = _read(RESULTS / "depth_sweep.csv")
    med = (
        df.groupby(["depth", "algorithm"])["wallclock_ms"].median().reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in sorted(med["algorithm"].unique()):
        sub = med[med["algorithm"] == algo].sort_values("depth")
        ax.plot(sub["depth"], sub["wallclock_ms"], marker="o", label=algo)
    ax.set_xlabel("depth (function-node levels)")
    ax.set_ylabel("wallclock (ms)")
    ax.set_yscale("log")
    ax.set_title("Depth sweep (xl, T_max)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_depth_sweep.png")


def plot_merge_density(_args):
    df = _read(RESULTS / "merge_density.csv")
    par = df[df.algorithm == "par_close"].copy()
    if par.empty:
        return
    par["merge_frac"] = par["merges"] / par["leaves"]
    med = par.groupby("merge_frac")["wallclock_ms"].median().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(med["merge_frac"], med["wallclock_ms"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("merge_frac (n_merges / n_leaves)")
    ax.set_ylabel("par_close wallclock (ms)")
    ax.set_title("Merge-density sweep (xl-d3, T_max)")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_merge_density.png")


def plot_nfns_sweep(_args):
    df = _read(RESULTS / "nfns_sweep.csv")
    par = df[df.algorithm == "par_close"].copy()
    if par.empty:
        return
    med = par.groupby("fns")["wallclock_ms"].median().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(med["fns"], med["wallclock_ms"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_fns (operators per level)")
    ax.set_ylabel("par_close wallclock (ms)")
    ax.set_title("n_fns sweep (xl-d3, T_max)")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_nfns_sweep.png")


def plot_round_by_size(_args):
    df = _read(RESULTS / "trace.csv")
    has_sub = {"keyed_ms", "group_by_ms", "per_group_ms"}.issubset(df.columns)
    cols = ["consolidate_ms", "frontier_ms", "semisort_ms"]
    if has_sub:
        cols += ["keyed_ms", "group_by_ms", "per_group_ms"]

    # Mean per (workload, round), then sum across rounds → total per workload
    grp = df.groupby(["workload", "round"])[cols].mean().reset_index()
    agg = grp.groupby("workload")[cols].sum()

    order = ["large", "xl-d3", "2xl-d3", "4xl-d3", "8xl-d3", "16xl-d3"]
    agg = agg.reindex([w for w in order if w in agg.index])
    if agg.empty:
        return

    if has_sub:
        agg["semi_other_ms"] = (
            agg["semisort_ms"]
            - agg[["keyed_ms", "group_by_ms", "per_group_ms"]].sum(axis=1)
        ).clip(lower=0)
        stacks = [
            ("consolidate_ms", "consolidate"),
            ("frontier_ms", "frontier_build"),
            ("keyed_ms", "semi: keyed"),
            ("group_by_ms", "semi: group_by_key"),
            ("per_group_ms", "semi: per_group"),
            ("semi_other_ms", "semi: canon+dedup"),
        ]
    else:
        stacks = [
            ("consolidate_ms", "consolidate"),
            ("frontier_ms", "frontier"),
            ("semisort_ms", "semisort"),
        ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = [0.0] * len(agg)
    labels = agg.index.tolist()
    for col, label in stacks:
        vals = agg[col].fillna(0).values
        ax.bar(labels, vals, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel("workload")
    ax.set_ylabel("total ms across all rounds (mean over trials)")
    ax.set_title("Phase decomposition by workload size")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "fig_round_by_size.png")


def plot_smt(_args):
    df = _read(RESULTS / "smt.csv")
    df = df[df["family"] != ""]  # skip non-matching filenames (regression cases)
    par = df[df["algorithm"] == "par_close"].copy()
    med = (
        par.groupby(["family", "n"])["wallclock_ms"]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam in sorted(med["family"].unique()):
        sub = med[med["family"] == fam].sort_values("n")
        ax.plot(sub["n"], sub["wallclock_ms"], marker="o", label=fam)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (family parameter)")
    ax.set_ylabel("par_close wallclock (ms)")
    ax.set_title("synthetic SMT-LIB families")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_smt_scaling.png")


# ------------------------- all -------------------------


def plot_all(args):
    for fn in (
        plot_strong_scaling,
        plot_wallclock,
        plot_components,
        plot_trace_rounds,
        plot_workload_sweep,
        plot_smt,
        plot_width_grid,
        plot_depth_sweep,
        plot_merge_density,
        plot_nfns_sweep,
        plot_round_by_size,
    ):
        try:
            fn(args)
        except SystemExit as e:
            # Skip a missing CSV but keep going for the rest.
            print(f"  skip: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [
        ("strong-scaling", plot_strong_scaling),
        ("wallclock", plot_wallclock),
        ("trace-rounds", plot_trace_rounds),
        ("components", plot_components),
        ("workload-sweep", plot_workload_sweep),
        ("smt", plot_smt),
        ("width-grid", plot_width_grid),
        ("depth-sweep", plot_depth_sweep),
        ("merge-density", plot_merge_density),
        ("nfns-sweep", plot_nfns_sweep),
        ("round-by-size", plot_round_by_size),
        ("all", plot_all),
    ]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
