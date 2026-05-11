#!/usr/bin/env python3
"""Print summary tables for whatever CSVs exist under bench/results/.

Run as:
    python3 bench/scripts/summarize.py                   # to stdout
    python3 bench/scripts/summarize.py > summary.txt     # capture

Each section quietly skips when its CSV is missing, so it works for
partial sweeps as well as full runs of bench/scripts/run.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results"


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def strong_scaling(results: Path) -> None:
    csv = results / "strong_scaling.csv"
    if not csv.exists():
        return
    ss = pd.read_csv(csv)
    par = ss[ss.algorithm == "par_parents"]
    med = (
        par.groupby(["workload", "parlay_threads"])["wallclock_ms"]
        .median()
        .reset_index()
    )
    wide = (
        med.pivot(index="workload", columns="parlay_threads", values="wallclock_ms")
        .sort_index(axis=1)
    )
    _section("par_parents median ms (workload × T)")
    print(wide.to_string())
    _section("speedup vs T=min")
    print(wide.rdiv(wide.iloc[:, 0], axis=0).to_string())
    if wide.shape[1] >= 2:
        _section("T=min / T=max speedup")
        print((wide.iloc[:, 0] / wide.iloc[:, -1]).to_string())

    nel = ss[ss.algorithm == "nelson_seq"]
    if not nel.empty:
        nm = nel.groupby("workload")["wallclock_ms"].median()
        nm = nm.reindex(wide.index)
        _section("par@T=max vs nelson_seq speedup")
        print((nm / wide.iloc[:, -1]).to_string())


def components(results: Path) -> None:
    csv = results / "components.csv"
    if not csv.exists():
        return
    cp = pd.read_csv(csv)
    cp_w = (
        cp.groupby(["phase", "parlay_threads"])["wallclock_ms"]
        .median()
        .reset_index()
        .pivot(index="phase", columns="parlay_threads", values="wallclock_ms")
        .sort_index(axis=1)
    )
    _section("component_bench median ms (phase × T)")
    print(cp_w.to_string())
    _section("component speedup vs T=min")
    print(cp_w.rdiv(cp_w.iloc[:, 0], axis=0).to_string())


def trace_rounds(results: Path) -> None:
    csv = results / "trace.csv"
    if not csv.exists():
        return
    tr = pd.read_csv(csv)
    has_sub = {"keyed_ms", "group_by_ms", "per_group_ms"}.issubset(tr.columns)
    cols = ["consolidate_ms", "frontier_ms", "semisort_ms"]
    if has_sub:
        cols += ["keyed_ms", "group_by_ms", "per_group_ms"]

    for (w, t), sub in tr.groupby(["workload", "parlay_threads"]):
        g = sub.groupby("round")[cols].mean()
        if has_sub:
            g["semi_other_ms"] = (
                g["semisort_ms"]
                - g[["keyed_ms", "group_by_ms", "per_group_ms"]].sum(axis=1)
            ).clip(lower=0)
        g["round_total_ms"] = g[
            ["consolidate_ms", "frontier_ms", "semisort_ms"]
        ].sum(axis=1)
        _section(f"trace mean per-round ms — {w}, T={t}")
        print(g.round(3).to_string())


def union_style(results: Path) -> None:
    csv = results / "union_style.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    _section("union_style median ms")
    print(df.groupby("union_style")["wallclock_ms"].median().to_string())


def dnc_cutoff(results: Path) -> None:
    csv = results / "dnc_cutoff.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    _section("dnc_cutoff median ms")
    print(df.groupby("dnc_cutoff")["wallclock_ms"].median().to_string())


def smt(results: Path) -> None:
    csv = results / "smt.csv"
    if not csv.exists():
        return
    sm = pd.read_csv(csv)
    sm = sm[sm["family"].notna() & (sm["family"] != "")]
    par = sm[sm.algorithm == "par_parents"]
    if par.empty:
        return
    sm_w = (
        par.groupby(["family", "n"])["wallclock_ms"]
        .median()
        .reset_index()
        .pivot(index="family", columns="n", values="wallclock_ms")
        .sort_index(axis=1)
    )
    _section("smt par_parents median ms (family × n)")
    print(sm_w.to_string())


def workload_sweep(results: Path) -> None:
    csv = results / "workload_sweep.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    par = df[df.algorithm == "par_parents"].copy()
    if par.empty:
        return
    par["merge_frac"] = par["merges"] / par["leaves"]
    med = (
        par.groupby(["depth", "merge_frac"])["wallclock_ms"]
        .median()
        .reset_index()
        .pivot(index="depth", columns="merge_frac", values="wallclock_ms")
        .sort_index(axis=1)
    )
    _section("workload_sweep par_parents median ms (depth × merge_frac)")
    print(med.to_string())


def width_grid(results: Path) -> None:
    csv = results / "width_grid.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    par = df[df.algorithm == "par_parents"].copy()
    if par.empty:
        return

    leaves_per_workload = par.groupby("workload")["leaves"].first()
    workload_order = leaves_per_workload.sort_values().index.tolist()

    med = (
        par.groupby(["workload", "parlay_threads"])["wallclock_ms"]
        .median()
        .reset_index()
    )
    pivot = (
        med.pivot(index="workload", columns="parlay_threads", values="wallclock_ms")
        .reindex(workload_order)
        .sort_index(axis=1)
    )
    _section("width grid: par_parents median ms (workload × T)")
    print(pivot.to_string())

    nel = df[df.algorithm == "nelson_seq"]
    if not nel.empty:
        nel_med = nel.groupby("workload")["wallclock_ms"].median().reindex(
            workload_order
        )
        _section("width grid: nelson_seq median ms")
        print(nel_med.dropna().to_string())
        if pivot.shape[1] >= 1:
            par_at_max = pivot.iloc[:, -1]
            speedup = nel_med / par_at_max
            _section("width grid: speedup over nelson at T_max")
            print(speedup.dropna().to_string())


def depth_sweep(results: Path) -> None:
    csv = results / "depth_sweep.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    pivot = (
        df.groupby(["depth", "algorithm"])["wallclock_ms"]
        .median()
        .reset_index()
        .pivot(index="algorithm", columns="depth", values="wallclock_ms")
        .sort_index(axis=1)
    )
    _section("depth sweep: median wallclock ms (algorithm × depth)")
    print(pivot.to_string())


def merge_density(results: Path) -> None:
    csv = results / "merge_density.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    par = df[df.algorithm == "par_parents"].copy()
    if par.empty:
        return
    par["merge_frac"] = par["merges"] / par["leaves"]
    med = par.groupby("merge_frac")["wallclock_ms"].median()
    _section("merge_density: par_parents median ms by merge_frac")
    print(med.to_string())


def nfns_sweep(results: Path) -> None:
    csv = results / "nfns_sweep.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    par = df[df.algorithm == "par_parents"].copy()
    if par.empty:
        return
    med = par.groupby("fns")["wallclock_ms"].median()
    _section("nfns_sweep: par_parents median ms by n_fns")
    print(med.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"missing results dir: {args.results_dir}")

    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    strong_scaling(args.results_dir)
    width_grid(args.results_dir)
    depth_sweep(args.results_dir)
    merge_density(args.results_dir)
    nfns_sweep(args.results_dir)
    components(args.results_dir)
    trace_rounds(args.results_dir)
    workload_sweep(args.results_dir)
    union_style(args.results_dir)
    dnc_cutoff(args.results_dir)
    smt(args.results_dir)


if __name__ == "__main__":
    main()
