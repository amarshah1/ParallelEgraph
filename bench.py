#!/usr/bin/env python3
"""Run a solver on all .smt2 files in a folder and record per-file timing.

When using our solver, pass --timing to get a detailed phase breakdown
(parse, build, merge, rebuild, check, solve, total) reported by the solver
itself, independent of process-launch overhead.

Usage:
    python bench.py <folder>                           # use our solver (release)
    python bench.py <folder> --debug                   # use our solver (debug)
    python bench.py <folder> --solver /usr/bin/z3      # use z3
    python bench.py <folder> --csv results.csv         # write CSV output
    python bench.py <folder> --parallel                # parallel mode
    python bench.py <folder> --timing                  # show phase breakdown
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time


TIMING_RE = re.compile(
    r"timing:\s+"
    r"parse=(\S+)\s+build=(\S+)\s+merge=(\S+)\s+"
    r"rebuild=(\S+)\s+check=(\S+)\s+solve=(\S+)\s+total=(\S+)"
)


def build_our_solver(release: bool) -> str:
    """Build our solver and return the binary path."""
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    print("Building solver...", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    profile = "release" if release else "debug"
    return os.path.join("target", profile, "parallel-egraph")


def parse_timing_line(stderr: str) -> dict | None:
    """Parse a timing: line from solver stderr into a dict of floats."""
    m = TIMING_RE.search(stderr)
    if not m:
        return None
    keys = ["parse", "build", "merge", "rebuild", "check", "solve", "total"]
    return {k: float(m.group(i + 1)) for i, k in enumerate(keys)}


def run_one(solver: str, path: str, parallel: bool, timing: bool) -> dict:
    """Run the solver on a single file. Returns a result dict."""
    cmd = [solver, path]
    if parallel:
        cmd.append("-p")
    if timing:
        cmd.append("--timing")

    wall_start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        wall_s = time.perf_counter() - wall_start
        if proc.returncode != 0:
            return {"result": "ERROR", "wall_s": wall_s, "error": proc.stderr.strip()}

        info = {"result": proc.stdout.strip(), "wall_s": wall_s, "error": ""}
        if timing:
            t = parse_timing_line(proc.stderr)
            if t:
                info.update(t)
        return info

    except subprocess.TimeoutExpired:
        wall_s = time.perf_counter() - wall_start
        return {"result": "TIMEOUT", "wall_s": wall_s, "error": "exceeded 300s"}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", help="Directory containing .smt2 files")
    parser.add_argument("--solver", help="Path to external solver binary (e.g. z3, cvc5)")
    parser.add_argument("--debug", action="store_true", help="Use debug build")
    parser.add_argument("--parallel", action="store_true", help="Parallel mode")
    parser.add_argument("--timing", action="store_true", help="Report per-phase timing from solver")
    parser.add_argument("--csv", dest="csv_file", help="Write results to CSV")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.smt2")))
    if not files:
        print(f"No .smt2 files found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    if args.solver:
        solver = args.solver
        solver_name = os.path.basename(solver)
        if args.timing:
            print("Warning: --timing only works with our solver, ignored for external solvers",
                  file=sys.stderr)
            args.timing = False
    else:
        solver = build_our_solver(not args.debug)
        solver_name = "parallel-egraph"
        # Always use detailed timing for our solver
        args.timing = True

    # Warmup
    print(f"\nWarmup: {os.path.basename(files[0])}", flush=True)
    run_one(solver, files[0], args.parallel, args.timing)

    results = []

    print(f"\nSolver: {solver_name}{'  [parallel]' if args.parallel else ''}")

    if args.timing:
        hdr = f"{'File':<40} {'Result':<7} {'Wall':>7} {'Parse':>7} {'Build':>7} {'Merge':>7} {'Rebuild':>8} {'Check':>7} {'Solve':>7}"
        print(hdr)
        print("-" * len(hdr))
    else:
        print(f"{'File':<50} {'Result':<10} {'Time (s)':>10}")
        print("-" * 72)

    total_wall = 0.0
    total_solve = 0.0

    for path in files:
        name = os.path.basename(path)
        info = run_one(solver, path, args.parallel, args.timing)
        total_wall += info["wall_s"]
        results.append((name, info))

        if args.timing and "solve" in info:
            total_solve += info["solve"]
            print(
                f"{name:<40} {info['result']:<7} "
                f"{info['wall_s']:>7.4f} {info['parse']:>7.4f} {info['build']:>7.4f} "
                f"{info['merge']:>7.4f} {info['rebuild']:>8.4f} {info['check']:>7.4f} "
                f"{info['solve']:>7.4f}"
            )
        else:
            status = info["result"] if not info.get("error") else f"{info['result']}: {info['error']}"
            print(f"{name:<50} {status:<10} {info['wall_s']:>10.4f}")

    if args.timing:
        print("-" * len(hdr))
        print(f"{'TOTAL':<40} {'':<7} {total_wall:>7.4f} {'':<7} {'':<7} {'':<7} {'':<8} {'':<7} {total_solve:>7.4f}")
    else:
        print("-" * 72)
        print(f"{'TOTAL':<50} {'':<10} {total_wall:>10.4f}")

    print(f"\n{len(files)} files, wall={total_wall:.4f}s", end="")
    if args.timing:
        print(f", solve={total_solve:.4f}s", end="")
    print()

    if args.csv_file:
        with open(args.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            if args.timing:
                writer.writerow(["file", "result", "wall_s", "parse_s", "build_s",
                                 "merge_s", "rebuild_s", "check_s", "solve_s", "error"])
                for name, info in results:
                    writer.writerow([
                        name, info["result"],
                        f"{info['wall_s']:.6f}",
                        f"{info.get('parse', ''):.6f}" if "parse" in info else "",
                        f"{info.get('build', ''):.6f}" if "build" in info else "",
                        f"{info.get('merge', ''):.6f}" if "merge" in info else "",
                        f"{info.get('rebuild', ''):.6f}" if "rebuild" in info else "",
                        f"{info.get('check', ''):.6f}" if "check" in info else "",
                        f"{info.get('solve', ''):.6f}" if "solve" in info else "",
                        info.get("error", ""),
                    ])
            else:
                writer.writerow(["file", "result", "time_secs", "error"])
                for name, info in results:
                    writer.writerow([name, info["result"], f"{info['wall_s']:.6f}", info.get("error", "")])
        print(f"Results written to {args.csv_file}")


if __name__ == "__main__":
    main()
