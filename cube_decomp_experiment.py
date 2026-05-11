#!/usr/bin/env python3
"""Cube workload sweep across two axes:

  * d (decomposition rate / g-tree fan-in) ∈ {2, 3, 4, 5}
    Larger d → shallower g-tree → frontier shrinks by a factor of d per round.
  * k (cube parameter) ∈ {5, 55, 105, 155} (default; tweakable via --ks)

For each (d, k) we time both nelson_seq and par_parents at every thread count
(default 1, 2, 4, 8). Output goes to a timestamped folder under runs/:

  runs/<ts>/
    cube_decomp.csv         — one row per (d, k, threads, algorithm, trial)
    cube_decomp_traces/
      d<d>_k<k>_T<threads>.log   — PE_TRACE=1 stderr per invocation

Usage:
    python3 cube_decomp_experiment.py
    python3 cube_decomp_experiment.py --threads-sweep 1,2,4,8
    python3 cube_decomp_experiment.py --ks 5,55,105,155 --ds 2,3,4,5
    python3 cube_decomp_experiment.py --skip-nelson  # parallel-only
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_THREADS = "1,2,4,8"
DEFAULT_KS      = "5,55,105,155"
DEFAULT_DS      = "2,3,4,5"


def cmake_build():
    print("Configuring CMake (Release)...", flush=True)
    subprocess.run(
        ["cmake", "-B", "build", "-S", ".", "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    print("Building synthetic_bench...", flush=True)
    subprocess.run(
        ["cmake", "--build", "build", "-j", "--target", "synthetic_bench"],
        check=True,
    )


def run_one(binary: str, k: int, d: int, threads: int,
            skip_nelson: bool, emit_header: bool,
            warmup: int, trials: int,
            numactl_prefix: list[str]) -> tuple[str, str]:
    """One synthetic_bench invocation. Returns (stdout_csv, stderr_trace)."""
    cmd = list(numactl_prefix) + [binary]
    env = os.environ.copy()
    env["PE_SYNTH_FAMILIES"] = "cube"
    env["PE_SYNTH_NS"] = str(k)
    env["PE_SYNTH_D"]  = str(d)
    env["PE_BENCH_FORMAT"] = "csv"
    env["PE_BENCH_TRIALS"] = str(trials)
    env["PE_BENCH_WARMUP"] = str(warmup)
    if emit_header:
        env["PE_BENCH_HEADER"] = "1"
    if skip_nelson:
        env["PE_BENCH_SKIP_NELSON"] = "1"
    env["PE_TRACE"] = "1"
    env["PARLAY_NUM_THREADS"] = str(threads)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"synthetic_bench failed at d={d} k={k} T={threads}")
    return proc.stdout, proc.stderr


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads-sweep", default=DEFAULT_THREADS,
                    help=f"comma-separated thread counts (default {DEFAULT_THREADS})")
    ap.add_argument("--ks", default=DEFAULT_KS,
                    help=f"comma-separated cube k values (default {DEFAULT_KS})")
    ap.add_argument("--ds", default=DEFAULT_DS,
                    help=f"comma-separated d values (default {DEFAULT_DS})")
    ap.add_argument("--skip-nelson", action="store_true",
                    help="skip the sequential baseline; parallel only")
    ap.add_argument("--warmup", type=int, default=0,
                    help="warmup runs per (d,k,threads) (default: 0 — fast)")
    ap.add_argument("--trials", type=int, default=1,
                    help="measured trials per (d,k,threads) (default: 1 — fast)")
    ap.add_argument("--out-root", default="runs",
                    help="parent dir for the timestamped folder")
    ap.add_argument("--no-numactl", action="store_true",
                    help="don't prepend `numactl -i all` even if available")
    args = ap.parse_args()

    try:
        thread_counts = sorted(int(t) for t in args.threads_sweep.split(",") if t)
        ks            = [int(t) for t in args.ks.split(",") if t]
        ds            = [int(t) for t in args.ds.split(",") if t]
    except ValueError:
        sys.exit("--threads-sweep/--ks/--ds expect comma-separated integers")
    if any(d < 2 for d in ds):
        sys.exit("d must be >= 2 (synthetic_bench enforces this too)")

    cmake_build()
    binary = "./build/synthetic_bench"

    numactl_prefix: list[str] = []
    if not args.no_numactl:
        which = shutil.which("numactl")
        if which:
            numactl_prefix = [which, "-i", "all"]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "cube_decomp_traces"
    trace_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "cube_decomp.csv"

    print(f"Output folder: {out_dir}")
    print(f"  d sweep:       {ds}")
    print(f"  k sweep:       {ks}")
    print(f"  thread sweep:  {thread_counts}")
    print(f"  numactl:       {' '.join(numactl_prefix) if numactl_prefix else 'none'}")
    print()

    # Loop order: (d, k) outermost, threads innermost. For each (d, k) we
    # run nelson once on the first thread count, then par_parents at every
    # thread count. Lets us track "this workload's full thread curve" and
    # avoids re-running the thread-independent sequential baseline.
    first = True
    nelson_done: set[tuple[int, int]] = set()
    with open(csv_path, "w") as csv_out:
        for d in ds:
            for k in ks:
                for t in thread_counts:
                    skip_nelson = args.skip_nelson or (d, k) in nelson_done
                    nel_tag = "" if not skip_nelson else " (par-only)"
                    print(f"  [cube d={d} k={k}] threads={t}{nel_tag}",
                          flush=True)
                    csv_text, trace_text = run_one(
                        binary, k=k, d=d, threads=t,
                        skip_nelson=skip_nelson,
                        emit_header=first,
                        warmup=args.warmup, trials=args.trials,
                        numactl_prefix=numactl_prefix)
                    csv_out.write(csv_text)
                    csv_out.flush()
                    (trace_dir / f"d{d}_k{k}_T{t}.log").write_text(trace_text)
                    first = False
                    nelson_done.add((d, k))

    print()
    print(f"Done. {csv_path}")
    print(f"      traces: {trace_dir}")


if __name__ == "__main__":
    main()
