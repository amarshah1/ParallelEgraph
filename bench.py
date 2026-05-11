#!/usr/bin/env python3
"""Run egraph-cc (or another QF_UF solver) on every .smt2 in a folder.

Reports per-file wall-clock time and sat/unsat. If filenames follow the
`*_sat.smt2` / `*_unsat.smt2` convention (as the synthetic benchmarks
do), --check verifies the answers.

Usage:
    python bench.py synthetic_benchmarks
    python bench.py synthetic_benchmarks --threads 8
    python bench.py synthetic_benchmarks --threads-sweep 1,2,4,8 --csv out.csv
    python bench.py synthetic_benchmarks --solver /usr/local/bin/z3
    python bench.py synthetic_benchmarks --check          # verify sat/unsat
    python bench.py synthetic_benchmarks --debug          # debug build
    python bench.py synthetic_benchmarks --pattern 'cube*' # filter by basename
    python bench.py synthetic_benchmarks --timeout 60      # per-file wall budget
"""

import argparse
import csv
import datetime
import fnmatch
import glob
import os
import re
import shutil
import subprocess
import sys
import time


# Per-phase timing emitted by egraph-cc --timing on stderr:
#   timing: read=0.082 parse=0.281 build=1.193 close=1.922 check=0.000 dtor=0.748
TIMING_RE = re.compile(
    r"timing:\s+"
    r"read=(\S+)\s+parse=(\S+)\s+build=(\S+)\s+"
    r"close=(\S+)\s+check=(\S+)\s+dtor=(\S+)"
)
TIMING_KEYS = ["read", "parse", "build", "close", "check", "dtor"]


def parse_timing_line(stderr: str) -> dict | None:
    m = TIMING_RE.search(stderr)
    if not m:
        return None
    return {k: float(m.group(i + 1)) for i, k in enumerate(TIMING_KEYS)}


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_our_solver(release: bool) -> str:
    """Configure and build egraph-cc; return its path."""
    build_dir = "build"
    if not os.path.isdir(build_dir):
        build_type = "Release" if release else "Debug"
        print(f"Configuring CMake ({build_type})...", flush=True)
        subprocess.run(
            ["cmake", "-B", build_dir, "-S", ".",
             f"-DCMAKE_BUILD_TYPE={build_type}"],
            check=True, capture_output=True,
        )
    print("Building egraph-cc...", flush=True)
    subprocess.run(
        ["cmake", "--build", build_dir, "-j", "--target", "egraph-cc"],
        check=True, capture_output=True,
    )
    return os.path.join(build_dir, "egraph-cc")


# ---------------------------------------------------------------------------
# Filename → expected answer (sat/unsat)
# ---------------------------------------------------------------------------

_EXPECTED_RE = re.compile(r"_(sat|unsat)\.smt2$")


def expected_from_name(path: str):
    """Return 'sat' / 'unsat' inferred from filename, or None if unknown."""
    m = _EXPECTED_RE.search(os.path.basename(path))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(solver: str, path: str, threads: int | None,
            timeout: float | None, extra_args: list[str],
            phase_timing: bool = False,
            sequential: bool = False,
            numactl_prefix: list[str] | None = None) -> dict:
    """Run the solver on a single file. Returns a result dict.

    `threads`, when set, is passed to the child via PARLAY_NUM_THREADS so
    parlay's scheduler picks it up. (Our solver has no --parallel flag —
    parallelism is always on; thread count is the dial.)

    `phase_timing` requests our solver's --timing output; the parsed
    phase fields are merged into the result dict.

    `sequential` adds --sequential to the solver invocation (only
    meaningful for our own egraph-cc; ignored args are caller's problem).

    `numactl_prefix`, if set, is prepended to the command line. Typical
    use: ["numactl", "-i", "all"] to interleave allocations across NUMA
    nodes and avoid contention on a single node's memory.
    """
    cmd = []
    if numactl_prefix:
        cmd.extend(numactl_prefix)
    cmd.append(solver)
    cmd.extend(extra_args)
    if phase_timing:
        cmd.append("--timing")
    if sequential:
        cmd.append("--sequential")
    cmd.append(path)
    env = os.environ.copy()
    if threads is not None:
        env["PARLAY_NUM_THREADS"] = str(threads)

    wall_start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        wall_s = time.perf_counter() - wall_start
        stdout = proc.stdout.strip()
        if proc.returncode != 0:
            return {"result": "ERROR", "wall_s": wall_s,
                    "error": proc.stderr.strip()[:200]}
        # z3 / cvc5 sometimes preface "sat"/"unsat" with junk or
        # "(error ...)" lines. Take the first line that's exactly one of
        # the two answers.
        result = "UNKNOWN"
        for line in stdout.splitlines():
            line = line.strip()
            if line in ("sat", "unsat", "unknown"):
                result = line
                break
        info = {"result": result, "wall_s": wall_s, "error": ""}
        if phase_timing:
            t = parse_timing_line(proc.stderr)
            if t is not None:
                info.update(t)
        return info
    except subprocess.TimeoutExpired:
        wall_s = time.perf_counter() - wall_start
        return {"result": "TIMEOUT", "wall_s": wall_s,
                "error": f"exceeded {timeout}s"}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def fmt_check(result: str, expected: str | None) -> str:
    """Return 'OK', 'FAIL', or '?'."""
    if expected is None:
        return "?"
    if result.lower() == expected.lower():
        return "OK"
    return "FAIL"


def print_table_header(check: bool, threads_col: bool, phase: bool,
                       algo_col: bool):
    parts = [f"{'File':<40}"]
    if algo_col:
        parts.append(f"{'Algo':<10}")
    parts.extend([f"{'Result':<8}", f"{'Wall(s)':>10}"])
    if phase:
        parts.extend([f"{'Read':>7}", f"{'Parse':>7}", f"{'Build':>7}",
                      f"{'Close':>7}", f"{'Check':>7}", f"{'Dtor':>7}"])
    if check:
        parts.append(f"{'OK?':<4}")
    if threads_col:
        parts.append(f"{'Thr':>4}")
    line = " ".join(parts)
    print(line)
    print("-" * len(line))
    return line


def print_table_row(name: str, info: dict, expected: str | None,
                    check: bool, threads: int | None, phase: bool,
                    algorithm: str | None):
    parts = [f"{name:<40}"]
    if algorithm is not None:
        parts.append(f"{algorithm:<10}")
    parts.extend([
        f"{info['result']:<8}",
        f"{info['wall_s']:>10.4f}",
    ])
    if phase:
        for k in TIMING_KEYS:
            v = info.get(k)
            parts.append(f"{v:>7.4f}" if v is not None else f"{'-':>7}")
    if check:
        parts.append(f"{fmt_check(info['result'], expected):<4}")
    if threads is not None:
        parts.append(f"{threads:>4}")
    print(" ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="directory containing .smt2 files")
    parser.add_argument("--solver",
                        help="external solver binary (e.g. z3, cvc5). "
                             "Default: build and use ./build/egraph-cc")
    parser.add_argument("--debug", action="store_true",
                        help="use debug build (only for our solver)")
    parser.add_argument("--threads", type=int, default=None,
                        help="PARLAY_NUM_THREADS to pass to the child")
    parser.add_argument("--threads-sweep", default=None,
                        help="comma-separated thread counts; runs every file at "
                             "every thread count (e.g. '1,2,4,8')")
    parser.add_argument("--check", action="store_true",
                        help="verify sat/unsat against filename suffix")
    parser.add_argument("--pattern", default="*",
                        help="glob filter on basename (default: '*')")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-file wall budget in seconds")
    parser.add_argument("--csv", dest="csv_file",
                        help="append per-run rows to this CSV path")
    parser.add_argument("--solver-arg", action="append", default=[],
                        help="repeatable: extra arg passed to the solver "
                             "verbatim (e.g. --solver-arg=--smt2)")
    parser.add_argument("--no-numactl", action="store_true",
                        help="don't prepend `numactl -i all` to runs even "
                             "if numactl is available. Default: use it "
                             "when present (helps on multi-NUMA machines; "
                             "no-op on macOS / single-socket boxes).")
    parser.add_argument("--sequential", action="store_true",
                        help="run sequential_close_nelson alongside "
                             "parallel_parents. Adds one row per file with "
                             "algorithm=sequential.")
    args = parser.parse_args()
    extra = list(args.solver_arg)

    # Decide whether to prepend numactl. Default is "yes if available".
    numactl_prefix: list[str] | None = None
    if not args.no_numactl and not args.solver:
        which = shutil.which("numactl")
        if which:
            numactl_prefix = [which, "-i", "all"]

    files = sorted(glob.glob(os.path.join(args.folder, "*.smt2")))
    files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), args.pattern)]
    if not files:
        print(f"No .smt2 files matched in {args.folder}", file=sys.stderr)
        sys.exit(1)

    if args.solver:
        solver = args.solver
        solver_name = os.path.basename(solver)
        # External solvers don't speak our --timing dialect.
        phase_timing = False
    else:
        solver = build_our_solver(release=not args.debug)
        solver_name = "egraph-cc"
        # Always request phase timing from our own solver. It's nearly
        # free to print and lets us separate read/parse/build/close/dtor.
        phase_timing = True

    # Determine the thread-count axis.
    if args.threads_sweep:
        try:
            thread_counts = [int(t) for t in args.threads_sweep.split(",") if t]
        except ValueError:
            print("--threads-sweep expects comma-separated integers",
                  file=sys.stderr)
            sys.exit(2)
    else:
        thread_counts = [args.threads]  # may contain a single None

    # Default CSV path includes timestamp when not explicitly given.
    if args.csv_file is None and args.solver is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.csv_file = f"synthetic_sweep_{ts}.csv"

    # Warmup: run the first file once at the smallest non-None thread count
    # so the solver binary is page-cached and parlay's scheduler is warm.
    warmup_threads = next((t for t in thread_counts if t is not None), None)
    print(f"Warmup: {os.path.basename(files[0])}", flush=True)
    run_one(solver, files[0], warmup_threads, args.timeout, extra,
            phase_timing=phase_timing, numactl_prefix=numactl_prefix)

    print()
    suffix = []
    if args.threads is not None and not args.threads_sweep:
        suffix.append(f"threads={args.threads}")
    if args.threads_sweep:
        suffix.append(f"threads-sweep={args.threads_sweep}")
    if args.sequential:
        suffix.append("with-sequential")
    if numactl_prefix:
        suffix.append(f"numactl={' '.join(numactl_prefix[1:])}")
    if extra:
        suffix.append(f"extra={extra}")
    print(f"Solver: {solver_name}" + (f"  ({', '.join(suffix)})" if suffix else ""))

    # An "algorithm" column shows up if we ran the sequential variant or
    # if the user asked for it via --sequential. Only meaningful with
    # our own egraph-cc (external solvers don't have --sequential).
    show_algo_col = args.sequential and not args.solver

    print_table_header(check=args.check,
                       threads_col=(args.threads_sweep is not None),
                       phase=phase_timing,
                       algo_col=show_algo_col)

    # rows entries: (name, expected, algorithm, threads, info)
    rows = []
    n_runs = 0
    n_failures = 0
    n_timeouts = 0
    total_wall = 0.0

    def record(name, expected, algorithm, t, info):
        nonlocal n_runs, n_failures, n_timeouts, total_wall
        n_runs += 1
        total_wall += info["wall_s"]
        if info["result"] == "TIMEOUT":
            n_timeouts += 1
        if args.check and expected is not None and \
           info["result"].lower() not in (expected.lower(), "timeout", "error"):
            n_failures += 1
        print_table_row(name, info, expected, args.check, t,
                        phase=phase_timing,
                        algorithm=algorithm if show_algo_col else None)
        rows.append((name, expected, algorithm, t, info))

    for path in files:
        name = os.path.basename(path)
        expected = expected_from_name(path)

        # Sequential first: a single run, no thread sweep, no
        # PARLAY_NUM_THREADS dependency. The "threads" column is left
        # blank in the CSV for sequential rows.
        if args.sequential and not args.solver:
            info = run_one(solver, path, threads=None, timeout=args.timeout,
                           extra_args=extra, phase_timing=phase_timing,
                           sequential=True, numactl_prefix=numactl_prefix)
            record(name, expected, "sequential", None, info)

        # Parallel sweep across thread counts.
        for t in thread_counts:
            info = run_one(solver, path, t, args.timeout, extra,
                           phase_timing=phase_timing,
                           numactl_prefix=numactl_prefix)
            record(name, expected, "parallel", t, info)
        sys.stdout.flush()

    print()
    summary = [f"{n_runs} runs", f"wall={total_wall:.4f}s"]
    if args.check:
        summary.append(f"failures={n_failures}")
    if n_timeouts:
        summary.append(f"timeouts={n_timeouts}")
    print(", ".join(summary))

    if args.csv_file:
        # Append-only: caller can run sweeps in multiple invocations and
        # merge them into one CSV. Only writes the header if file is new.
        # Phase columns (read_s/parse_s/build_s/close_s/check_s/dtor_s)
        # come from egraph-cc's --timing line; they're empty when running
        # an external solver. The `algorithm` column is "parallel" or
        # "sequential" (or external solver name when not ours).
        write_header = not os.path.exists(args.csv_file)
        with open(args.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["solver", "algorithm", "file", "expected",
                                 "threads", "result", "wall_s",
                                 "read_s", "parse_s", "build_s",
                                 "close_s", "check_s", "dtor_s",
                                 "check", "error"])
            for name, expected, algorithm, t, info in rows:
                phase_cells = [
                    f"{info[k]:.6f}" if k in info else ""
                    for k in TIMING_KEYS
                ]
                writer.writerow([
                    solver_name, algorithm, name,
                    expected if expected else "",
                    t if t is not None else "",
                    info["result"],
                    f"{info['wall_s']:.6f}",
                    *phase_cells,
                    fmt_check(info["result"], expected) if args.check else "",
                    info.get("error", ""),
                ])
        print(f"CSV {'created' if write_header else 'appended'}: {args.csv_file}")


if __name__ == "__main__":
    main()
