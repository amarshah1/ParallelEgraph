"""Bench driver: sweep a parameter, collect CSV results.

Subcommands:
  strong-scaling   — closure_compare across PARLAY_NUM_THREADS
  workload-sweep   — closure_compare across (depth, merge-fraction) at fixed T
  trace            — one PE_TRACE run, parsed into CSV
  components       — component_bench across PARLAY_NUM_THREADS
  smt              — smt_bench over the gen_bench.py synthetic families
  all              — strong-scaling, components, trace, workload-sweep, smt

All outputs land in bench/results/. Re-running overwrites by default;
pass --append to add rows to an existing CSV.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import (
    BUILD_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    SYNTHETIC_DIR,
    append_csv,
    ensure_built,
    ensure_results_dir,
    run_capture,
    threads_default,
)

CC_BENCH = "closure_compare_bench"
COMPONENT_BENCH = "component_bench"
SMT_BENCH = "smt_bench"
PARSE_TRACE = REPO_ROOT / "bench" / "scripts" / "parse_trace.py"

CC_HEADER = (
    "workload,leaves,fns,nodes,merges,depth,algorithm,trial,"
    "parlay_threads,union_style,dnc_cutoff,wallclock_ms"
)
COMPONENT_HEADER = "phase,trial,parlay_threads,workload,wallclock_ms"
SMT_HEADER = (
    "file,family,n,classes,equalities,algorithm,trial,parlay_threads,"
    "wallclock_ms"
)
TRACE_HEADER = (
    "workload,parlay_threads,round,work,frontier,next,"
    "consolidate_ms,frontier_ms,semisort_ms"
)


def reset_csv(path: Path, header: str, append: bool) -> None:
    if append and path.exists():
        return
    path.write_text(header + "\n")


# ------------------------- subcommands -------------------------


def cmd_strong_scaling(args):
    bin_path = ensure_built(CC_BENCH)
    out = ensure_results_dir() / "strong_scaling.csv"
    reset_csv(out, CC_HEADER, args.append)
    threads = args.threads or threads_default()
    workloads = args.workloads or [None]  # None = all 6
    for t in threads:
        for w in workloads:
            env = {
                "PARLAY_NUM_THREADS": str(t),
                "PE_BENCH_FORMAT": "csv",
            }
            if w:
                env["PE_BENCH_ONLY"] = w
            print(f"[strong-scaling] T={t} workload={w or 'all'}")
            tmp = Path(tempfile.mktemp(suffix=".csv"))
            run_capture([str(bin_path)], env_overrides=env, stdout_path=tmp)
            append_csv(out, tmp)
            tmp.unlink(missing_ok=True)
    print(f"  → {out.relative_to(REPO_ROOT)}")


def cmd_workload_sweep(args):
    bin_path = ensure_built(CC_BENCH)
    out = ensure_results_dir() / "workload_sweep.csv"
    reset_csv(out, CC_HEADER, args.append)
    # Hold (leaves, fns, nodes) ≈ medium-scale; sweep depth and merge_frac.
    leaves = 10_000
    fns = 8
    nodes = 200_000
    threads = args.threads[0] if args.threads else (os.cpu_count() or 8)
    for depth in args.depths:
        for merge_frac in args.merge_fracs:
            merges = max(1, int(leaves * merge_frac))
            spec = f"{leaves},{fns},{nodes},{merges},{depth}"
            env = {
                "PARLAY_NUM_THREADS": str(threads),
                "PE_BENCH_FORMAT": "csv",
                "PE_BENCH_CUSTOM": spec,
            }
            print(f"[workload-sweep] depth={depth} merge_frac={merge_frac}")
            tmp = Path(tempfile.mktemp(suffix=".csv"))
            run_capture([str(bin_path)], env_overrides=env, stdout_path=tmp)
            append_csv(out, tmp)
            tmp.unlink(missing_ok=True)
    print(f"  → {out.relative_to(REPO_ROOT)}")


def cmd_trace(args):
    bin_path = ensure_built(CC_BENCH)
    out = ensure_results_dir() / "trace.csv"
    reset_csv(out, TRACE_HEADER, args.append)
    threads = args.threads[0] if args.threads else (os.cpu_count() or 8)
    workload = args.trace_workload
    env = {
        "PARLAY_NUM_THREADS": str(threads),
        "PE_TRACE": "1",
        "PE_BENCH_ONLY": workload,
        "PE_BENCH_SKIP_NELSON": "1",
    }
    print(f"[trace] workload={workload} T={threads}")
    log = Path(tempfile.mktemp(suffix=".log"))
    devnull = Path(os.devnull)
    run_capture(
        [str(bin_path)],
        env_overrides=env,
        stdout_path=devnull,
        stderr_path=log,
    )
    # Pipe the trace through parse_trace.py and append.
    parse_cmd = [
        sys.executable,
        str(PARSE_TRACE),
        workload,
        str(threads),
        "--no-header",
    ]
    with open(log, "rb") as f, open(out, "ab") as o:
        subprocess.run(parse_cmd, stdin=f, stdout=o, check=True)
    log.unlink(missing_ok=True)
    print(f"  → {out.relative_to(REPO_ROOT)}")


def cmd_components(args):
    bin_path = ensure_built(COMPONENT_BENCH)
    out = ensure_results_dir() / "components.csv"
    reset_csv(out, COMPONENT_HEADER, args.append)
    threads = args.threads or threads_default()
    workload = args.component_workload
    for t in threads:
        env = {
            "PARLAY_NUM_THREADS": str(t),
            "PE_BENCH_FORMAT": "csv",
            "PE_COMPONENT_WORKLOAD": workload,
        }
        print(f"[components] T={t} workload={workload}")
        tmp = Path(tempfile.mktemp(suffix=".csv"))
        run_capture([str(bin_path)], env_overrides=env, stdout_path=tmp)
        append_csv(out, tmp)
        tmp.unlink(missing_ok=True)
    print(f"  → {out.relative_to(REPO_ROOT)}")


def cmd_smt(args):
    bin_path = ensure_built(SMT_BENCH)
    out = ensure_results_dir() / "smt.csv"
    reset_csv(out, SMT_HEADER, args.append)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    # gen_bench.py sweep <n1> <n2> <step> generates all 4 families per n.
    n1, n2, step = args.smt_range
    print(f"[smt] generating sweep n={n1}..{n2} step {step}")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "gen_bench.py"),
            "sweep",
            str(n1),
            str(n2),
            str(step),
            str(SYNTHETIC_DIR),
        ],
        check=True,
    )
    threads = args.threads[0] if args.threads else (os.cpu_count() or 8)
    env = {"PARLAY_NUM_THREADS": str(threads)}
    if args.smt_skip_nelson:
        env["PE_BENCH_SKIP_NELSON"] = "1"
    print(f"[smt] running smt_bench T={threads}")
    tmp = Path(tempfile.mktemp(suffix=".csv"))
    run_capture(
        [str(bin_path), str(SYNTHETIC_DIR)],
        env_overrides=env,
        stdout_path=tmp,
    )
    append_csv(out, tmp)
    tmp.unlink(missing_ok=True)
    print(f"  → {out.relative_to(REPO_ROOT)}")


def cmd_all(args):
    cmd_strong_scaling(args)
    cmd_components(args)
    cmd_trace(args)
    cmd_workload_sweep(args)
    cmd_smt(args)


# ------------------------- argparse -------------------------


def main():
    # Shared flags live on a parent so each subcommand inherits them and
    # argparse doesn't trip over greedy nargs='+' eating the subcommand name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--threads",
        type=int,
        nargs="+",
        help="thread counts to sweep (default: powers of two up to nproc)",
    )
    common.add_argument(
        "--workloads",
        nargs="+",
        help="restrict closure_compare to these workload names",
    )
    common.add_argument(
        "--append",
        action="store_true",
        help="append to existing CSVs instead of overwriting",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ss = sub.add_parser(
        "strong-scaling", parents=[common],
        help="closure_compare across thread counts",
    )
    p_ss.set_defaults(func=cmd_strong_scaling)

    p_ws = sub.add_parser(
        "workload-sweep", parents=[common],
        help="depth × merge-fraction sweep",
    )
    p_ws.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p_ws.add_argument(
        "--merge-fracs",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.4],
        dest="merge_fracs",
    )
    p_ws.set_defaults(func=cmd_workload_sweep)

    p_tr = sub.add_parser("trace", parents=[common], help="single PE_TRACE run")
    p_tr.add_argument("--trace-workload", default="large")
    p_tr.set_defaults(func=cmd_trace)

    p_cp = sub.add_parser(
        "components", parents=[common],
        help="component_bench across thread counts",
    )
    p_cp.add_argument("--component-workload", default="large")
    p_cp.set_defaults(func=cmd_components)

    p_sm = sub.add_parser(
        "smt", parents=[common],
        help="smt_bench over generated synthetic families",
    )
    p_sm.add_argument(
        "--smt-range",
        type=int,
        nargs=3,
        default=[3, 9, 1],
        metavar=("N1", "N2", "STEP"),
    )
    p_sm.add_argument("--smt-skip-nelson", action="store_true")
    p_sm.set_defaults(func=cmd_smt)

    p_al = sub.add_parser(
        "all", parents=[common],
        help="run every subcommand sequentially",
    )
    p_al.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p_al.add_argument(
        "--merge-fracs",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.4],
        dest="merge_fracs",
    )
    p_al.add_argument("--trace-workload", default="large")
    p_al.add_argument("--component-workload", default="large")
    p_al.add_argument(
        "--smt-range",
        type=int,
        nargs=3,
        default=[3, 9, 1],
        metavar=("N1", "N2", "STEP"),
    )
    p_al.add_argument("--smt-skip-nelson", action="store_true")
    p_al.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
