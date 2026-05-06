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


def _numactl_prefix() -> list[str]:
    numactl = shutil.which("numactl")
    return [numactl, "-i", "all"] if numactl else []


NUMACTL_PREFIX = _numactl_prefix()


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

# XL ladder: doubles n_nodes / n_merges each step, starting from a 2× scale
# of the `large` workload's nodes/merges. depth=3 across the board (no
# separate deep-XL family). Driven via PE_BENCH_CUSTOM so no
# closure_compare.cpp recompile is needed.
# Tuple layout: (n_leaves, n_fns, n_nodes, n_merges, depth).
RANDOM_XL_LADDER: dict[str, tuple[int, int, int, int, int]] = {
    "XL":   (50_000, 16,  2_000_000,   200_000, 3),
    "2XL":  (50_000, 16,  4_000_000,   400_000, 3),
    "4XL":  (50_000, 16,  8_000_000,   800_000, 3),
    "8XL":  (50_000, 16, 16_000_000, 1_600_000, 3),
    "16XL": (50_000, 16, 32_000_000, 3_200_000, 3),
    "32XL": (50_000, 16, 64_000_000, 6_400_000, 3),
}


def _retag_random_csv_rows(stdout: str, label: str, emit_header: bool) -> str:
    """Rewrite the `workload` column in closure_compare CSV stdout from
    'custom' to `label`. Drops the header line unless emit_header is True
    (so we only print one header in the merged CSV).

    Note: only the very first invocation has PE_BENCH_HEADER=1, so every
    subsequent invocation's stdout is *headerless* — we therefore retag
    by recognizing rows that start with 'custom,' rather than gating on
    a header sighting.
    """
    out: list[str] = []
    for line in stdout.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("workload,"):
            if emit_header:
                out.append(line)
            continue
        # Data row from PE_BENCH_CUSTOM: closure_compare prints
        # "custom,leaves,fns,..." — swap the leading token for our label.
        if stripped.startswith("custom,"):
            comma = line.find(",")
            out.append(f"{label}{line[comma:]}")
            continue
        out.append(line)
    return "".join(out)


def _retag_trace_banner(stderr: str, label: str) -> str:
    """closure_compare emits "[bench] running custom ..." per workload. The
    XL ladder runs one invocation per label, so there's a single banner per
    invocation — rewrite it so parse_trace_log attributes round lines to
    the correct ladder rung instead of the constant string "custom".
    """
    return re.sub(r"^\[bench\] running custom \.\.\.",
                  f"[bench] running {label} ...",
                  stderr, flags=re.MULTILINE)


def run_random_xl(out_dir: Path, thread_counts: list[int],
                  labels: list[str] | None = None):
    """Same as run_random, but iterates the XL ladder via PE_BENCH_CUSTOM.

    One closure_compare_bench invocation per (label, threads) cell. The
    stdout `workload` column gets rewritten from 'custom' to the ladder
    label so the merged CSV is self-describing; the stderr trace banner
    gets the same treatment so per-round attribution survives parsing.

    If `labels` is given, only those rungs of RANDOM_XL_LADDER run (in
    the requested order). Useful for resuming a partial sweep.
    """
    csv_path = out_dir / "random.csv"
    trace_csv_path = out_dir / "random_trace.csv"
    trace_dir = out_dir / "random_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    if labels is None:
        ladder = list(RANDOM_XL_LADDER.items())
    else:
        unknown = [l for l in labels if l not in RANDOM_XL_LADDER]
        if unknown:
            sys.exit(f"unknown XL ladder labels: {unknown}. "
                     f"valid: {list(RANDOM_XL_LADDER)}")
        ladder = [(l, RANDOM_XL_LADDER[l]) for l in labels]

    binary = "./build/closure_compare_bench"
    first = True
    all_trace_rows: list[list] = []
    with open(csv_path, "w") as csv_out:
        for label, params in ladder:
            spec = ",".join(str(x) for x in params)
            # nelson_seq is the sequential baseline — its time doesn't
            # depend on PARLAY_NUM_THREADS, so we only run it on the first
            # thread count for each ladder rung and skip it on the rest.
            for i, t in enumerate(thread_counts):
                skip_nelson = i > 0
                tag = "par_only" if skip_nelson else "+nelson"
                print(f"  [random-xl] {label} threads={t} ({tag})",
                      flush=True)
                env = os.environ.copy()
                env["PE_BENCH_FORMAT"] = "csv"
                if first:
                    env["PE_BENCH_HEADER"] = "1"
                if skip_nelson:
                    env["PE_BENCH_SKIP_NELSON"] = "1"
                env["PE_TRACE"] = "1"
                env["PARLAY_NUM_THREADS"] = str(t)
                env["PE_BENCH_CUSTOM"] = spec
                t0 = time.perf_counter()
                proc = subprocess.run(NUMACTL_PREFIX + [binary],
                                      capture_output=True, text=True, env=env)
                wall = time.perf_counter() - t0
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr)
                    raise RuntimeError(
                        f"closure_compare_bench failed at {label} T={t}")
                csv_out.write(_retag_random_csv_rows(
                    proc.stdout, label, emit_header=first))
                csv_out.flush()
                retagged_stderr = _retag_trace_banner(proc.stderr, label)
                (trace_dir / f"{label}_T{t}.log").write_text(retagged_stderr)
                all_trace_rows.extend(parse_trace_log(
                    retagged_stderr, threads=t,
                    default_workload=label,
                    workload_re=RANDOM_BANNER_RE))
                first = False
                print(f"    wrote {trace_dir / f'{label}_T{t}.log'} "
                      f"({wall:.1f}s)", flush=True)

    with open(trace_csv_path, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(TRACE_COLUMNS)
        w.writerows(all_trace_rows)
    print(f"  → {csv_path}")
    print(f"  → {trace_csv_path}  ({len(all_trace_rows)} rows)")


def run_random(out_dir: Path, thread_counts: list[int],
               also_async: bool = False):
    """closure_compare_bench with PE_BENCH_FORMAT=csv at each thread count.

    The binary runs all 6 baked-in workloads in one invocation. Header is
    emitted on the first invocation only; subsequent invocations append.
    Per-round PE_TRACE output is captured to random_traces/T<n>.log AND
    parsed into random_trace.csv (one row per workload×thread×round trial)
    so the existing per-round bar-chart plotter can consume it.

    When `also_async`, runs a second pass per thread count with
    PE_USE_ASYNC=1 and PE_BENCH_SKIP_NELSON=1 so the async closure is
    timed alongside BSP. The CSV's `algorithm` column distinguishes them
    (nelson_seq / par_close / par_close_async).
    """
    csv_path = out_dir / "random.csv"
    trace_csv_path = out_dir / "random_trace.csv"
    trace_dir = out_dir / "random_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    binary = "./build/closure_compare_bench"
    first = True
    all_trace_rows: list[list] = []
    with open(csv_path, "w") as csv_out:
        for i, t in enumerate(thread_counts):
            # nelson_seq is sequential — its time doesn't depend on
            # PARLAY_NUM_THREADS. Run it only on the first thread count
            # for each (workload) and skip on subsequent thread counts
            # via PE_BENCH_SKIP_NELSON=1.
            bsp_skip_nelson = i > 0
            passes: list[tuple[str, dict[str, str]]] = [
                ("bsp", {"PE_BENCH_SKIP_NELSON": "1"} if bsp_skip_nelson
                        else {}),
            ]
            if also_async:
                passes.append(("async", {
                    "PE_USE_ASYNC": "1",
                    "PE_BENCH_SKIP_NELSON": "1",
                }))
            for pass_label, pass_env in passes:
                tag = ""
                if pass_label == "bsp":
                    tag = " (par_only)" if bsp_skip_nelson else " (+nelson)"
                print(f"  [random] threads={t} ({pass_label}{tag})",
                      flush=True)
                env = os.environ.copy()
                env["PE_BENCH_FORMAT"] = "csv"
                if first:
                    env["PE_BENCH_HEADER"] = "1"
                env["PE_TRACE"] = "1"
                env["PARLAY_NUM_THREADS"] = str(t)
                env.update(pass_env)
                t0 = time.perf_counter()
                proc = subprocess.run(NUMACTL_PREFIX + [binary],
                                      capture_output=True, text=True, env=env)
                wall = time.perf_counter() - t0
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr)
                    raise RuntimeError(
                        f"closure_compare_bench failed at T={t} {pass_label}")
                csv_out.write(proc.stdout)
                csv_out.flush()
                (trace_dir / f"T{t}_{pass_label}.log").write_text(proc.stderr)
                all_trace_rows.extend(parse_trace_log(
                    proc.stderr, threads=t, workload_re=RANDOM_BANNER_RE))
                first = False
                print(f"    wrote {trace_dir / f'T{t}_{pass_label}.log'} "
                      f"({wall:.1f}s)", flush=True)

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


def run_synthetic(out_dir: Path, thread_counts: list[int],
                  also_async: bool = False):
    """synthetic_bench binary, one invocation per (family, n, threads).

    Per-round PE_TRACE output is captured per-invocation to
    synthetic_traces/<family>/<family>_n<N>_T<thr>.log AND merged into
    synthetic_trace.csv with workload="<family>_n<N>" so the bar-chart
    plotter can consume it.

    When `also_async`, runs a second invocation per (family, n, threads)
    with PE_USE_ASYNC=1 and PE_BENCH_SKIP_NELSON=1 — nelson is captured
    once in the BSP pass per (family, n) so the async pass always skips
    it. CSV's `algorithm` column distinguishes par_close vs
    par_close_async.
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
                    # Pass spec: (label, extra_env). BSP first (may carry
                    # nelson on the first thread count for this (fam,n));
                    # async second, always parallel-only.
                    skip_nelson = (fam, n) in nelson_done
                    bsp_label = "+nelson" if not skip_nelson else "par_only"
                    passes: list[tuple[str, dict[str, str], str]] = [
                        ("bsp", {} if not skip_nelson else
                                {"PE_BENCH_SKIP_NELSON": "1"},
                         bsp_label),
                    ]
                    if also_async:
                        passes.append(("async", {
                            "PE_USE_ASYNC": "1",
                            "PE_BENCH_SKIP_NELSON": "1",
                        }, "async"))
                    for pass_label, pass_env, status in passes:
                        print(f"  [synthetic] {fam} n={n} threads={t} "
                              f"({pass_label} {status})", flush=True)
                        env = os.environ.copy()
                        env["PE_SYNTH_FAMILIES"] = fam
                        env["PE_SYNTH_NS"] = str(n)
                        env["PE_BENCH_FORMAT"] = "csv"
                        if first:
                            env["PE_BENCH_HEADER"] = "1"
                        env["PE_TRACE"] = "1"
                        env["PARLAY_NUM_THREADS"] = str(t)
                        env.update(pass_env)
                        proc = subprocess.run(NUMACTL_PREFIX + [binary],
                                              capture_output=True,
                                              text=True, env=env)
                        if proc.returncode != 0:
                            sys.stderr.write(proc.stderr)
                            raise RuntimeError(
                                f"synthetic_bench failed at {fam} n={n} "
                                f"T={t} {pass_label}")
                        csv_out.write(proc.stdout)
                        csv_out.flush()
                        trace_name = (f"{fam}_n{n}_T{t}_{pass_label}.log"
                                      if also_async
                                      else f"{fam}_n{n}_T{t}.log")
                        trace_path = trace_root / fam / trace_name
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


def run_cube_decomp(out_dir: Path, thread_counts: list[int],
                    also_async: bool = False):
    """synthetic_bench (cube only) swept over d ∈ CUBE_DECOMP_DS and
    k ∈ CUBE_DECOMP_KS at every thread count. CSV row schema matches
    synthetic.csv (same binary), with the `d` column carrying the axis.
    Trace logs land in cube_decomp_traces/d<d>_k<k>_T<thr>.log.

    nelson_seq runs once per (d, k) on the first thread count; subsequent
    thread counts skip nelson via PE_BENCH_SKIP_NELSON=1.

    When `also_async`, each (d, k, threads) cell does a second invocation
    with PE_USE_ASYNC=1, always skipping nelson (already captured by
    BSP pass).
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
                    bsp_label = "+nelson" if not skip_nelson else "par_only"
                    passes: list[tuple[str, dict[str, str], str]] = [
                        ("bsp", {} if not skip_nelson else
                                {"PE_BENCH_SKIP_NELSON": "1"},
                         bsp_label),
                    ]
                    if also_async:
                        passes.append(("async", {
                            "PE_USE_ASYNC": "1",
                            "PE_BENCH_SKIP_NELSON": "1",
                        }, "async"))
                    for pass_label, pass_env, status in passes:
                        print(f"  [cube_decomp] d={d} k={k} threads={t} "
                              f"({pass_label} {status})", flush=True)
                        env = os.environ.copy()
                        env["PE_SYNTH_FAMILIES"] = "cube"
                        env["PE_SYNTH_NS"] = str(k)
                        env["PE_SYNTH_D"]  = str(d)
                        env["PE_BENCH_FORMAT"] = "csv"
                        if first:
                            env["PE_BENCH_HEADER"] = "1"
                        env["PE_TRACE"] = "1"
                        env["PARLAY_NUM_THREADS"] = str(t)
                        env.update(pass_env)
                        proc = subprocess.run(NUMACTL_PREFIX + [binary],
                                              capture_output=True,
                                              text=True, env=env)
                        if proc.returncode != 0:
                            sys.stderr.write(proc.stderr)
                            raise RuntimeError(
                                f"synthetic_bench failed at d={d} k={k} "
                                f"T={t} {pass_label}")
                        csv_out.write(proc.stdout)
                        csv_out.flush()
                        trace_name = (f"d{d}_k{k}_T{t}_{pass_label}.log"
                                      if also_async
                                      else f"d{d}_k{k}_T{t}.log")
                        (trace_dir / trace_name).write_text(proc.stderr)
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


_UNIT_TO_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}


def _parse_duration_seconds(token: str) -> float:
    """Parse '35.23ms' / '1.4s' / '0.5us' → seconds. Bare numbers are seconds."""
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([a-zµ]*)$", token)
    if not m:
        raise ValueError(f"unrecognized duration token: {token!r}")
    value, unit = float(m.group(1)), m.group(2)
    if unit == "":
        return value
    if unit not in _UNIT_TO_SECONDS:
        raise ValueError(f"unknown time unit in {token!r}")
    return value * _UNIT_TO_SECONDS[unit]


def parse_timing(stderr: str) -> dict | None:
    m = TIMING_RE.search(stderr)
    if not m:
        return None
    return {k: _parse_duration_seconds(m.group(i + 1))
            for i, k in enumerate(TIMING_KEYS)}


def parse_result(stdout: str) -> str:
    for line in stdout.splitlines():
        line = line.strip()
        if line in ("sat", "unsat", "unknown"):
            return line
    return "UNKNOWN"


def run_egg(out_dir: Path, thread_counts: list[int],
            egg_dir: str, pattern: str, timeout: float,
            also_async: bool = False):
    """egraph-cc per (file, threads, trial) under egg_dir.

    When `also_async`, each (file, threads) cell runs a second sweep of
    1 warmup + 5 trials with PE_USE_ASYNC=1. CSV gains an `algorithm`
    column = `par_close` or `par_close_async` so downstream plots can
    group correctly. egraph-cc has no sequential mode, so there is no
    nelson_seq row in egg.csv (intentionally; sequential lives in the
    synthetic / random phases).
    """
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
    print(f"  [egg] {len(files)} files × {len(thread_counts)} threads"
          + (" × {bsp,async}" if also_async else ""),
          flush=True)

    def run_invocation(path: str, t: int, use_async: bool):
        """One invocation. Returns (wall_s, result, timing, error, trace)."""
        cmd = NUMACTL_PREFIX + [binary, "--timing", path]
        env = os.environ.copy()
        env["PE_TRACE"] = "1"
        env["PARLAY_NUM_THREADS"] = str(t)
        if use_async:
            env["PE_USE_ASYNC"] = "1"
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
    algos = [("par_close", False)]
    if also_async:
        algos.append(("par_close_async", True))

    all_trace_rows: list[list] = []
    with open(csv_path, "w", newline="") as f:
        writer = csvmod.writer(f)
        writer.writerow(["file", "expected", "algorithm", "threads", "trial",
                         "result", "wall_s",
                         *[f"{k}_s" for k in TIMING_KEYS], "error"])
        for path in files:
            name = os.path.basename(path)
            stem = name[:-len(".smt2")] if name.endswith(".smt2") else name
            expected = expected_from_name(path) or ""
            for t in thread_counts:
                for algo_name, use_async in algos:
                    # Warmup invocations: page-cache solver, prime the
                    # parlay scheduler, but don't record. If a warmup hits
                    # TIMEOUT or ERROR, skip this (file, t, algo) cell.
                    bail = False
                    for w in range(EGG_WARMUP):
                        _, w_result, _, _, _ = run_invocation(
                            path, t, use_async)
                        if w_result in ("TIMEOUT", "ERROR"):
                            writer.writerow([name, expected, algo_name, t, -1,
                                             w_result, "", "", "", "", "",
                                             "", "", "warmup failed"])
                            f.flush()
                            print(f"  [egg] {name} algo={algo_name} "
                                  f"T={t:<3} warmup={w_result} "
                                  f"(skipping trials)", flush=True)
                            bail = True
                            break
                    if bail:
                        continue
                    # Measured trials. One log file per (file, t, algo).
                    trace_chunks: list[str] = []
                    for trial in range(EGG_TRIALS):
                        wall, result, timing, error, trace_text = \
                            run_invocation(path, t, use_async)
                        trace_chunks.append(
                            f"=== trial {trial} (T={t} {algo_name}) ===\n"
                            f"{trace_text}")
                        all_trace_rows.extend(parse_trace_log(
                            trace_text, threads=t, default_workload=stem))
                        row = [name, expected, algo_name, t, trial, result,
                               f"{wall:.6f}"]
                        row += [f"{timing[k]:.6f}" if timing else ""
                                for k in TIMING_KEYS]
                        row.append(error)
                        writer.writerow(row)
                        f.flush()
                        close_s = f"{timing['close']:.4f}s" if timing else "-"
                        print(f"  [egg] {name} algo={algo_name} "
                              f"T={t:<3} trial={trial} "
                              f"{result:<8} close={close_s} wall={wall:.2f}s",
                              flush=True)
                    log_name = (f"{stem}_T{t}_{algo_name}.log"
                                if also_async else f"{stem}_T{t}.log")
                    (trace_dir / log_name).write_text("".join(trace_chunks))

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
    ap.add_argument("--random-xl-only", action="store_true",
                    help="run only the random phase, and only the XL→32XL "
                         "ladder via PE_BENCH_CUSTOM. Implies --skip "
                         "synthetic --skip cube_decomp --skip egg.")
    ap.add_argument("--random-xl-labels", default=None,
                    help="comma-separated subset of XL ladder rungs to run "
                         "(e.g. '16XL,32XL'). Only meaningful with "
                         "--random-xl-only. Defaults to the full ladder.")
    ap.add_argument("--also-async", action="store_true",
                    help="run the async-rounds closure alongside BSP at "
                         "every (workload, threads) cell. Roughly doubles "
                         "the wall-clock of each phase. CSVs gain an "
                         "`algorithm` column distinguishing par_close vs "
                         "par_close_async (egg.csv) — random/synthetic/"
                         "cube_decomp already had one, populated by the "
                         "binaries themselves.")
    ap.add_argument("--warmup", type=int, default=None,
                    help="warmup runs per (workload, threads, algorithm) "
                         "cell. Default: 1 (matches existing benches). "
                         "Applied uniformly to all four phases via "
                         "PE_BENCH_WARMUP env (random/synthetic/"
                         "cube_decomp) and the egg-phase loop variable.")
    ap.add_argument("--trials", type=int, default=None,
                    help="measured trials per cell (default 5). Applied "
                         "uniformly to all four phases via "
                         "PE_BENCH_TRIALS env and the egg-phase loop "
                         "variable.")
    args = ap.parse_args()

    # Apply warmup/trials overrides:
    #   - C++ binaries (closure_compare_bench, synthetic_bench, smt_bench)
    #     read PE_BENCH_WARMUP / PE_BENCH_TRIALS from env. Set those once
    #     globally so every subprocess inherits them.
    #   - The egg phase's loop is in this driver — patch the module-level
    #     EGG_WARMUP / EGG_TRIALS so run_egg picks up the new values.
    global EGG_WARMUP, EGG_TRIALS
    if args.warmup is not None:
        if args.warmup < 0:
            sys.exit("--warmup must be >= 0")
        os.environ["PE_BENCH_WARMUP"] = str(args.warmup)
        EGG_WARMUP = args.warmup
    if args.trials is not None:
        if args.trials < 1:
            sys.exit("--trials must be >= 1")
        os.environ["PE_BENCH_TRIALS"] = str(args.trials)
        EGG_TRIALS = args.trials

    try:
        thread_counts = [int(t) for t in args.threads_sweep.split(",") if t]
        thread_counts = sorted(set(thread_counts))
    except ValueError:
        sys.exit("--threads-sweep expects comma-separated integers")

    skip = set(args.skip)
    if args.random_xl_only:
        skip.update({"synthetic", "cube_decomp", "egg"})
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
        if args.random_xl_only:
            xl_labels = None
            if args.random_xl_labels:
                xl_labels = [s.strip() for s in args.random_xl_labels.split(",")
                             if s.strip()]
            tag = (f" ({','.join(xl_labels)})" if xl_labels else "")
            print(f"=== (a) random (XL ladder{tag}) ===")
            run_random_xl(out_dir, thread_counts, labels=xl_labels)
        else:
            print("=== (a) random ===")
            run_random(out_dir, thread_counts, also_async=args.also_async)
        print()

    if "synthetic" not in skip:
        print("=== (b) synthetic ===")
        run_synthetic(out_dir, thread_counts, also_async=args.also_async)
        print()

    if "cube_decomp" not in skip:
        print("=== (c) cube_decomp ===")
        run_cube_decomp(out_dir, thread_counts, also_async=args.also_async)
        print()

    if "egg" not in skip:
        print("=== (d) egg / cc-benchmarks ===")
        run_egg(out_dir, thread_counts, args.egg_dir, args.egg_pattern,
                args.egg_timeout, also_async=args.also_async)
        print()

    print(f"Done. All artifacts under: {out_dir}")


if __name__ == "__main__":
    main()
