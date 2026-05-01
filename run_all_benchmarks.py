#!/usr/bin/env python3
"""Drive all four benchmark suites end-to-end into one timestamped folder.

What it runs:
  (a) random       — bench/closure_compare.cpp's six baked-in random DAGs
                     at each thread count, in CSV mode, one trace file per
                     thread count.
  (b) synthetic    — bench/synthetic_bench.cpp via ./build/synthetic_bench
                     across all (family, n, threads) triples. One trace
                     file per (family, n, threads). d=2 by default — this
                     phase is for the polynomial-arity sweep.
  (c) cube_decomp  — also bench/synthetic_bench, but cube only, swept over
                     two axes: the decomposition rate d ∈ {2,3,4,5} (g-tree
                     fan-in) and the cube parameter k. One trace file per
                     (d, k, threads). Designed to surface frontier/round
                     scaling differences across decomposition rates.
  (d) egg          — egraph-cc on every .smt2 in cc-benchmarks/smt-grounded
                     (a git submodule of this repo) across all thread
                     counts. Master CSV mirrors bench.py's schema
                     (file, threads, trial, result, wall_s, phase fields).

All four use a 1 warmup + 5 measured trials policy. The synthetic /
cube_decomp binaries bake that into PE_BENCH_TRIALS / PE_BENCH_WARMUP
env (overridable); egg loops here in the driver.

Layout of the output folder (auto-created):
  runs/<YYYYMMDD_HHMMSS>/
    random.csv         random_trace.csv
    synthetic.csv      synthetic_trace.csv
    cube_decomp.csv    cube_decomp_trace.csv
    egg.csv            egg_trace.csv
    random_traces/T<n>.log
    synthetic_traces/<family>/<family>_n<N>_T<thr>.log
    cube_decomp_traces/d<d>_k<k>_T<thr>.log
    egg_traces/<basename>_T<thr>.log

PE_TRACE=1 is set on every invocation so the trace files contain the
per-round semisort/consolidate breakdowns.

Usage:
    python3 run_all_benchmarks.py
    python3 run_all_benchmarks.py --threads-sweep 1,2,4,8
    python3 run_all_benchmarks.py --skip random
    python3 run_all_benchmarks.py --egg-pattern 'demo.*'   # filter cc-benchmarks
    python3 run_all_benchmarks.py --egg-timeout 30          # tighter than 120s
"""

import argparse
import csv as csvmod
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_THREADS = "1,2,4,8,16,32,64,128,144"
DEFAULT_EGG_DIR = "cc-benchmarks/smt-grounded"
CC_BENCHMARKS_LOCAL = "cc-benchmarks"

# All three benchmark suites use a 1 warmup + 5 measured trials policy.
# closure_compare and synthetic_bench bake this in at compile time
# (constexpr WARMUP/TRIALS); egg/egraph-cc loops in this driver.
EGG_WARMUP = 1
EGG_TRIALS = 5

# egraph-cc emits this on stderr when --timing is passed. PE_TRACE=1 mixes
# round= lines into the same stream, so we anchor on the final "timing:" line.
TIMING_RE = re.compile(
    r"timing:\s+read=(\S+)\s+parse=(\S+)\s+build=(\S+)\s+"
    r"close=(\S+)\s+check=(\S+)\s+dtor=(\S+)"
)
TIMING_KEYS = ["read", "parse", "build", "close", "check", "dtor"]

# PE_TRACE=1 line, optionally with the (keyed=... group_by=... per_group=...)
# semisort sub-phase tail. Mirrors bench/scripts/parse_trace.py.
TRACE_RE = re.compile(
    r"^\[pe\] round=\s*(?P<round>\d+)\s+"
    r"work=\s*(?P<work>\d+)\s+"
    r"frontier=\s*(?P<frontier>\d+)\s+"
    r"next=\s*(?P<next>\d+)\s+"
    r"consolidate=\s*(?P<cons>[0-9.]+)ms\s+"
    r"frontier=\s*(?P<front>[0-9.]+)ms\s+"
    r"semisort=\s*(?P<semi>[0-9.]+)ms"
    r"(?:\s+\(keyed=\s*(?P<keyed>[0-9.]+)ms"
    r"\s+group_by=\s*(?P<group_by>[0-9.]+)ms"
    r"\s+per_group=\s*(?P<per_group>[0-9.]+)ms\))?"
)
# closure_compare prints "[bench] running <name> ..." before each workload's
# warmup+trial loop — that's the boundary marker we use to attribute round
# lines to the right workload in a multi-workload trace stream.
RANDOM_BANNER_RE = re.compile(r"^\[bench\] running (\S+) \.\.\.")
TRACE_COLUMNS = [
    "workload", "parlay_threads", "round", "work", "frontier", "next",
    "consolidate_ms", "frontier_ms", "semisort_ms",
    "keyed_ms", "group_by_ms", "per_group_ms",
]


def _trace_row_from_match(m: re.Match, workload: str, threads: int) -> list:
    return [
        workload, threads,
        m["round"], m["work"], m["frontier"], m["next"],
        m["cons"], m["front"], m["semi"],
        m["keyed"] or "", m["group_by"] or "", m["per_group"] or "",
    ]


def parse_trace_log(log_text: str, threads: int,
                    default_workload: str = "",
                    workload_re: re.Pattern | None = None) -> list[list]:
    """Parse a PE_TRACE stderr log into trace.csv rows.

    `workload_re` is an optional banner regex with one capture group naming
    the workload that follows. When provided, round lines after a banner
    match are attributed to that workload; before the first banner they get
    `default_workload`. When None, every round line uses `default_workload`
    (for single-workload logs from synthetic_bench / egraph-cc).
    """
    rows: list[list] = []
    current = default_workload
    for line in log_text.splitlines():
        if workload_re is not None:
            mb = workload_re.match(line)
            if mb is not None:
                current = mb.group(1)
                continue
        m = TRACE_RE.match(line)
        if m is None:
            continue
        rows.append(_trace_row_from_match(m, current, threads))
    return rows


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def cmake_configure_build(targets: list[str]):
    """Run cmake configure (idempotent) then build the listed targets."""
    build_dir = "build"
    print(f"Configuring CMake (Release)...", flush=True)
    subprocess.run(
        ["cmake", "-B", build_dir, "-S", ".", "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    print(f"Building {', '.join(targets)}...", flush=True)
    subprocess.run(
        ["cmake", "--build", build_dir, "-j"] +
        sum([["--target", t] for t in targets], []),
        check=True,
    )


def ensure_cc_benchmarks() -> str:
    """Populate the cc-benchmarks submodule if needed; return smt-grounded path.

    cc-benchmarks/ is a git submodule of this repo. A fresh `git clone`
    leaves the submodule directory empty until `git submodule update
    --init` runs — we do that here so the egg phase always finds its
    .smt2 files without the user having to remember the extra step.
    """
    if os.path.isdir(DEFAULT_EGG_DIR):
        return DEFAULT_EGG_DIR
    print(f"Initializing submodule: {CC_BENCHMARKS_LOCAL}", flush=True)
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive",
         CC_BENCHMARKS_LOCAL],
        check=True,
    )
    if not os.path.isdir(DEFAULT_EGG_DIR):
        sys.exit(f"submodule init succeeded but {DEFAULT_EGG_DIR} still missing")
    return DEFAULT_EGG_DIR


# ---------------------------------------------------------------------------
# (a) random / closure_compare
# ---------------------------------------------------------------------------

def run_random(out_dir: Path, thread_counts: list[int]):
    """closure_compare_bench with PE_BENCH_FORMAT=csv at each thread count.

    The binary runs all 6 baked-in workloads in one invocation. Header is
    emitted on the first invocation only; subsequent invocations append.
    Per-round PE_TRACE output is captured to random_traces/T<n>.log AND
    parsed into random_trace.csv (one row per workload×thread×round trial)
    so the existing per-round bar-chart plotter can consume it.
    """
    csv_path = out_dir / "random.csv"
    trace_csv_path = out_dir / "random_trace.csv"
    trace_dir = out_dir / "random_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    binary = "./build/closure_compare_bench"
    first = True
    all_trace_rows: list[list] = []
    with open(csv_path, "w") as csv_out:
        for t in thread_counts:
            print(f"  [random] threads={t}", flush=True)
            env = os.environ.copy()
            env["PE_BENCH_FORMAT"] = "csv"
            if first:
                env["PE_BENCH_HEADER"] = "1"
            env["PE_TRACE"] = "1"
            env["PARLAY_NUM_THREADS"] = str(t)
            t0 = time.perf_counter()
            proc = subprocess.run([binary], capture_output=True, text=True,
                                  env=env)
            wall = time.perf_counter() - t0
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
                raise RuntimeError(f"closure_compare_bench failed at T={t}")
            csv_out.write(proc.stdout)
            csv_out.flush()
            (trace_dir / f"T{t}.log").write_text(proc.stderr)
            all_trace_rows.extend(parse_trace_log(
                proc.stderr, threads=t, workload_re=RANDOM_BANNER_RE))
            first = False
            print(f"    wrote {trace_dir / f'T{t}.log'}  ({wall:.1f}s)",
                  flush=True)

    with open(trace_csv_path, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(TRACE_COLUMNS)
        w.writerows(all_trace_rows)
    print(f"  → {csv_path}")
    print(f"  → {trace_csv_path}  ({len(all_trace_rows)} rows)")


# ---------------------------------------------------------------------------
# (b) synthetic / synthetic_bench
# ---------------------------------------------------------------------------

# These mirror synthetic_bench.py's defaults — kept in sync manually.
SYNTH_FAMILY_NS = {
    "chain":   list(range(5600, 7101, 500)),
    "grid":    list(range(160, 201, 5)),
    "cube":    list(range(110, 146, 5)),
    "quartic": list(range(38, 54, 3)),
    "quintic": list(range(18, 24)),
}


def run_synthetic(out_dir: Path, thread_counts: list[int]):
    """synthetic_bench binary, one invocation per (family, n, threads).

    Per-round PE_TRACE output is captured per-invocation to
    synthetic_traces/<family>/<family>_n<N>_T<thr>.log AND merged into
    synthetic_trace.csv with workload="<family>_n<N>" so the bar-chart
    plotter can consume it.
    """
    csv_path = out_dir / "synthetic.csv"
    trace_csv_path = out_dir / "synthetic_trace.csv"
    trace_root = out_dir / "synthetic_traces"
    trace_root.mkdir(parents=True, exist_ok=True)

    binary = "./build/synthetic_bench"
    first = True
    nelson_done: set[tuple[str, int]] = set()
    all_trace_rows: list[list] = []
    with open(csv_path, "w") as csv_out:
        for fam, ns in SYNTH_FAMILY_NS.items():
            (trace_root / fam).mkdir(parents=True, exist_ok=True)
            for n in ns:
                for t in thread_counts:
                    skip_nelson = (fam, n) in nelson_done
                    label = f"+nelson" if not skip_nelson else "par_only"
                    print(f"  [synthetic] {fam} n={n} threads={t} ({label})",
                          flush=True)
                    env = os.environ.copy()
                    env["PE_SYNTH_FAMILIES"] = fam
                    env["PE_SYNTH_NS"] = str(n)
                    env["PE_BENCH_FORMAT"] = "csv"
                    if first:
                        env["PE_BENCH_HEADER"] = "1"
                    if skip_nelson:
                        env["PE_BENCH_SKIP_NELSON"] = "1"
                    env["PE_TRACE"] = "1"
                    env["PARLAY_NUM_THREADS"] = str(t)
                    proc = subprocess.run([binary], capture_output=True,
                                          text=True, env=env)
                    if proc.returncode != 0:
                        sys.stderr.write(proc.stderr)
                        raise RuntimeError(
                            f"synthetic_bench failed at {fam} n={n} T={t}")
                    csv_out.write(proc.stdout)
                    csv_out.flush()
                    trace_path = trace_root / fam / f"{fam}_n{n}_T{t}.log"
                    trace_path.write_text(proc.stderr)
                    all_trace_rows.extend(parse_trace_log(
                        proc.stderr, threads=t,
                        default_workload=f"{fam}_n{n}"))
                    first = False
                    nelson_done.add((fam, n))

    with open(trace_csv_path, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(TRACE_COLUMNS)
        w.writerows(all_trace_rows)
    print(f"  → {csv_path}")
    print(f"  → {trace_csv_path}  ({len(all_trace_rows)} rows)")


# ---------------------------------------------------------------------------
# (c) cube_decomp / synthetic_bench cube only, swept over (d, k)
# ---------------------------------------------------------------------------

# d ∈ {2,3,4,5}: g-tree fan-in (decomposition rate). d=1 is excluded
# because a unary chain over k³ leaves blows up to k³ rounds — infeasible
# at k=155.
CUBE_DECOMP_DS = [2, 3, 4, 5]
CUBE_DECOMP_KS = [5, 55, 105, 155]


def run_cube_decomp(out_dir: Path, thread_counts: list[int]):
    """synthetic_bench (cube only) swept over d ∈ CUBE_DECOMP_DS and
    k ∈ CUBE_DECOMP_KS at every thread count. CSV row schema matches
    synthetic.csv (same binary), with the `d` column carrying the axis.
    Trace logs land in cube_decomp_traces/d<d>_k<k>_T<thr>.log.

    nelson_seq runs once per (d, k) on the first thread count; subsequent
    thread counts skip nelson via PE_BENCH_SKIP_NELSON=1.
    """
    csv_path = out_dir / "cube_decomp.csv"
    trace_csv_path = out_dir / "cube_decomp_trace.csv"
    trace_dir = out_dir / "cube_decomp_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    binary = "./build/synthetic_bench"
    first = True
    nelson_done: set[tuple[int, int]] = set()
    all_trace_rows: list[list] = []
    with open(csv_path, "w") as csv_out:
        for d in CUBE_DECOMP_DS:
            for k in CUBE_DECOMP_KS:
                for t in thread_counts:
                    skip_nelson = (d, k) in nelson_done
                    label = "+nelson" if not skip_nelson else "par_only"
                    print(f"  [cube_decomp] d={d} k={k} threads={t} ({label})",
                          flush=True)
                    env = os.environ.copy()
                    env["PE_SYNTH_FAMILIES"] = "cube"
                    env["PE_SYNTH_NS"] = str(k)
                    env["PE_SYNTH_D"]  = str(d)
                    env["PE_BENCH_FORMAT"] = "csv"
                    if first:
                        env["PE_BENCH_HEADER"] = "1"
                    if skip_nelson:
                        env["PE_BENCH_SKIP_NELSON"] = "1"
                    env["PE_TRACE"] = "1"
                    env["PARLAY_NUM_THREADS"] = str(t)
                    proc = subprocess.run([binary], capture_output=True,
                                          text=True, env=env)
                    if proc.returncode != 0:
                        sys.stderr.write(proc.stderr)
                        raise RuntimeError(
                            f"synthetic_bench failed at d={d} k={k} T={t}")
                    csv_out.write(proc.stdout)
                    csv_out.flush()
                    trace_path = trace_dir / f"d{d}_k{k}_T{t}.log"
                    trace_path.write_text(proc.stderr)
                    # Workload tag carries d so plotters can group across d.
                    all_trace_rows.extend(parse_trace_log(
                        proc.stderr, threads=t,
                        default_workload=f"cube_d{d}_k{k}"))
                    first = False
                    nelson_done.add((d, k))

    with open(trace_csv_path, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(TRACE_COLUMNS)
        w.writerows(all_trace_rows)
    print(f"  → {csv_path}")
    print(f"  → {trace_csv_path}  ({len(all_trace_rows)} rows)")


# ---------------------------------------------------------------------------
# (d) egg / egraph-cc on cc-benchmarks
# ---------------------------------------------------------------------------

EXPECTED_RE = re.compile(r"_(sat|unsat)\.smt2$|\.(?:sat|unsat)\.smt2$")


def expected_from_name(path: str):
    """Mirror bench.py: try to parse sat/unsat hint from filename."""
    m = re.search(r"(?:^|[._])(sat|unsat)\.smt2$", os.path.basename(path))
    return m.group(1) if m else None


def parse_timing(stderr: str) -> dict | None:
    m = TIMING_RE.search(stderr)
    if not m:
        return None
    return {k: float(m.group(i + 1)) for i, k in enumerate(TIMING_KEYS)}


def parse_result(stdout: str) -> str:
    for line in stdout.splitlines():
        line = line.strip()
        if line in ("sat", "unsat", "unknown"):
            return line
    return "UNKNOWN"


def run_egg(out_dir: Path, thread_counts: list[int],
            egg_dir: str, pattern: str, timeout: float):
    csv_path = out_dir / "egg.csv"
    trace_csv_path = out_dir / "egg_trace.csv"
    trace_dir = out_dir / "egg_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(egg_dir, "*.smt2")))
    if pattern != "*":
        import fnmatch
        files = [f for f in files
                 if fnmatch.fnmatch(os.path.basename(f), pattern)]
    if not files:
        print(f"  [egg] no .smt2 files matched in {egg_dir}", flush=True)
        return
    print(f"  [egg] {len(files)} files × {len(thread_counts)} threads",
          flush=True)

    numactl = shutil.which("numactl")
    numactl_prefix = [numactl, "-i", "all"] if numactl else []

    def run_invocation(path: str, t: int):
        """One invocation. Returns (wall_s, result, timing, error, trace)."""
        cmd = list(numactl_prefix) + [binary, "--timing", path]
        env = os.environ.copy()
        env["PE_TRACE"] = "1"
        env["PARLAY_NUM_THREADS"] = str(t)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  env=env, timeout=timeout)
            wall = time.perf_counter() - t0
            if proc.returncode != 0:
                return (wall, "ERROR", None,
                        proc.stderr.strip()[:200], proc.stderr)
            return (wall, parse_result(proc.stdout),
                    parse_timing(proc.stderr), "", proc.stderr)
        except subprocess.TimeoutExpired as e:
            wall = time.perf_counter() - t0
            partial = ""
            if e.stderr:
                partial = (e.stderr.decode("utf-8", "replace")
                           if isinstance(e.stderr, bytes) else e.stderr)
            return (wall, "TIMEOUT", None, f"exceeded {timeout}s", partial)

    binary = "./build/egraph-cc"
    all_trace_rows: list[list] = []
    with open(csv_path, "w", newline="") as f:
        writer = csvmod.writer(f)
        writer.writerow(["file", "expected", "threads", "trial", "result",
                         "wall_s", *[f"{k}_s" for k in TIMING_KEYS], "error"])
        for path in files:
            name = os.path.basename(path)
            stem = name[:-len(".smt2")] if name.endswith(".smt2") else name
            expected = expected_from_name(path) or ""
            for t in thread_counts:
                # Warmup invocations: page-cache solver, prime the parlay
                # scheduler, but don't record. If a warmup hits TIMEOUT or
                # ERROR, skip this (file, t) cell — trials would just spin
                # for the same reason.
                bail = False
                for w in range(EGG_WARMUP):
                    _, w_result, _, _, _ = run_invocation(path, t)
                    if w_result in ("TIMEOUT", "ERROR"):
                        # Record one row so the CSV still reflects the
                        # cell. Trials are skipped.
                        writer.writerow([name, expected, t, -1, w_result,
                                         "", "", "", "", "", "", "",
                                         "warmup failed"])
                        f.flush()
                        print(f"  [egg] {name} T={t:<3} warmup={w_result} "
                              f"(skipping trials)", flush=True)
                        bail = True
                        break
                if bail:
                    continue
                # Measured trials. One log file per (file, t) covering all
                # trials, with banner lines so each trial's rounds can be
                # disambiguated downstream if needed.
                trace_chunks: list[str] = []
                for trial in range(EGG_TRIALS):
                    wall, result, timing, error, trace_text = \
                        run_invocation(path, t)
                    trace_chunks.append(
                        f"=== trial {trial} (T={t}) ===\n{trace_text}")
                    all_trace_rows.extend(parse_trace_log(
                        trace_text, threads=t, default_workload=stem))
                    row = [name, expected, t, trial, result, f"{wall:.6f}"]
                    row += [f"{timing[k]:.6f}" if timing else ""
                            for k in TIMING_KEYS]
                    row.append(error)
                    writer.writerow(row)
                    f.flush()
                    close_s = f"{timing['close']:.4f}s" if timing else "-"
                    print(f"  [egg] {name} T={t:<3} trial={trial} "
                          f"{result:<8} close={close_s} wall={wall:.2f}s",
                          flush=True)
                (trace_dir / f"{stem}_T{t}.log").write_text(
                    "".join(trace_chunks))

    with open(trace_csv_path, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(TRACE_COLUMNS)
        w.writerows(all_trace_rows)
    print(f"  → {csv_path}")
    print(f"  → {trace_csv_path}  ({len(all_trace_rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads-sweep", default=DEFAULT_THREADS,
                    help=f"comma-separated thread counts (default: {DEFAULT_THREADS})")
    ap.add_argument("--skip", action="append", default=[],
                    choices=["random", "synthetic", "cube_decomp", "egg"],
                    help="skip a phase; repeatable")
    ap.add_argument("--out-root", default="runs",
                    help="parent dir for the timestamped run folder "
                         "(default: runs/)")
    ap.add_argument("--egg-dir", default=DEFAULT_EGG_DIR,
                    help=f"directory of .smt2 files (default: {DEFAULT_EGG_DIR})")
    ap.add_argument("--egg-pattern", default="*",
                    help="basename glob filter for egg files (default: '*')")
    ap.add_argument("--egg-timeout", type=float, default=120.0,
                    help="per-file wall budget for egg phase (default: 120s)")
    args = ap.parse_args()

    try:
        thread_counts = [int(t) for t in args.threads_sweep.split(",") if t]
        thread_counts = sorted(set(thread_counts))
    except ValueError:
        sys.exit("--threads-sweep expects comma-separated integers")

    skip = set(args.skip)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")
    print(f"Thread sweep: {thread_counts}")
    print(f"Skipping: {sorted(skip) if skip else 'nothing'}")
    print()

    # Build all needed targets up front. egraph-cc is needed for egg;
    # closure_compare_bench for random; synthetic_bench for both
    # synthetic and cube_decomp.
    targets = []
    if "random" not in skip:        targets.append("closure_compare_bench")
    if "synthetic" not in skip or \
       "cube_decomp" not in skip:   targets.append("synthetic_bench")
    if "egg" not in skip:           targets.append("egraph-cc")
    # De-dup while preserving order.
    seen: set[str] = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]
    if targets:
        cmake_configure_build(targets)
    print()

    if "egg" not in skip:
        ensure_cc_benchmarks()

    if "random" not in skip:
        print("=== (a) random ===")
        run_random(out_dir, thread_counts)
        print()

    if "synthetic" not in skip:
        print("=== (b) synthetic ===")
        run_synthetic(out_dir, thread_counts)
        print()

    if "cube_decomp" not in skip:
        print("=== (c) cube_decomp ===")
        run_cube_decomp(out_dir, thread_counts)
        print()

    if "egg" not in skip:
        print("=== (d) egg / cc-benchmarks ===")
        run_egg(out_dir, thread_counts, args.egg_dir, args.egg_pattern,
                args.egg_timeout)
        print()

    print(f"Done. All artifacts under: {out_dir}")


if __name__ == "__main__":
    main()
