#!/usr/bin/env python3
"""Run a solver on all .smt2 files in a folder and record per-file timing.

Usage:
    python bench.py <folder>                           # use our solver (debug)
    python bench.py <folder> --release                 # use our solver (release)
    python bench.py <folder> --solver /usr/bin/z3      # use z3
    python bench.py <folder> --solver cvc5             # use cvc5
    python bench.py <folder> --csv results.csv         # write CSV output
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time


def build_our_solver(release: bool) -> str:
    """Build our solver and return the binary path."""
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    print("Building solver...", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    profile = "release" if release else "debug"
    return os.path.join("target", profile, "parallel-egraph")


def run_one(solver: str, path: str) -> tuple[str, float, str]:
    """Run the solver on a single file. Returns (result, elapsed_secs, error)."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [solver, path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            return ("ERROR", elapsed, proc.stderr.strip())
        return (proc.stdout.strip(), elapsed, "")
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return ("TIMEOUT", elapsed, "exceeded 300s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Directory containing .smt2 files")
    parser.add_argument("--solver", help="Path to external solver binary (e.g. z3, cvc5)")
    parser.add_argument("--debug", action="store_true", help="Use debug build (only for our solver, default is release)")
    parser.add_argument("--csv", dest="csv_file", help="Write results to CSV")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.smt2")))
    if not files:
        print(f"No .smt2 files found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    if args.solver:
        solver = args.solver
        solver_name = os.path.basename(solver)
    else:
        solver = build_our_solver(not args.debug)
        solver_name = "parallel-egraph"

    # Warmup: run the first benchmark once without recording
    print(f"\nWarmup: {os.path.basename(files[0])}", flush=True)
    run_one(solver, files[0])

    results = []
    total_time = 0.0

    print(f"\nSolver: {solver_name}")
    print(f"{'File':<50} {'Result':<10} {'Time (s)':>10}")
    print("-" * 72)

    for path in files:
        name = os.path.basename(path)
        result, elapsed, error = run_one(solver, path)
        total_time += elapsed
        results.append((name, result, elapsed, error))

        status = result if not error else f"{result}: {error}"
        print(f"{name:<50} {status:<10} {elapsed:>10.4f}")

    print("-" * 72)
    print(f"{'TOTAL':<50} {'':<10} {total_time:>10.4f}")
    print(f"\n{len(files)} files, {total_time:.4f}s total")

    if args.csv_file:
        with open(args.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "result", "time_secs", "error"])
            for name, result, elapsed, error in results:
                writer.writerow([name, result, f"{elapsed:.6f}", error])
        print(f"Results written to {args.csv_file}")


if __name__ == "__main__":
    main()
