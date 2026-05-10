#!/usr/bin/env python3
"""All-phase driver for the par_topo_iter vs par_async comparison.

Phases:
  random       — closure_compare_bench (6 baked-in workloads)
  synthetic    — synthetic_bench across (family, n, threads)
  cube_decomp  — synthetic_bench (cube only) across (d, k, threads)
  egg          — egraph-cc on each .smt2 in cc-benchmarks/smt-grounded,
                 dispatched per-algorithm via --sequential / PE_USE_*
  gates        — gates_bench on each .gates in
                 miter-cc-benchmarks/{iwls22,hwmcc12}, runs both par
                 algorithms only (no sequential); CSV schema is gates'
                 native (file, suite, n_gates, ..., algorithm, ...)

Each phase emits the same CSV schema the corresponding C++ binary already
prints, so the existing plotters (`plot_results.py`, `plot_topo_vs_async.py`)
can consume the output unchanged.

Sequential algorithms (every algo in the C++ output not starting with
`par_`) only depend on the workload, not on PARLAY_NUM_THREADS. Running
them at every T duplicates measurements and inflates wall time. Solution:
- random/synthetic/cube_decomp binaries honour PE_BENCH_PAR_ONLY=1 (set
  on every T>1 invocation; T=1 runs the full set).
- egg dispatches one egraph-cc invocation per (file, algo, T); the driver
  simply skips sequential algos when T>1.

Usage:
    python3 compare_topo_vs_async.py
    python3 compare_topo_vs_async.py --threads-sweep 1,4,8
    python3 compare_topo_vs_async.py --skip cube_decomp
    python3 compare_topo_vs_async.py --warmup 1 --trials 5
"""

import argparse
import csv as csvmod
import datetime
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


# Mini sweep: enough to surface the topo_iter vs async difference,
# without burning hours. cube is the workload where async lost to BSP
# in earlier runs, so it's the most discriminating.
DEFAULT_FAMILY_NS = {
    "chain":       [3000, 6000],
    "grid":        [120, 175],
    "cube":        [80, 120, 145],
    "quartic":     [25, 45],
    "quintic":     [12, 18],
    "mixed_depth": [30, 50, 80, 110, 145],
}

# cube_decomp axis: g-tree fan-in d × cube parameter k.
CUBE_DECOMP_DS = [2, 3, 4, 5]
CUBE_DECOMP_KS = [5, 55, 105, 155]

# gates phase: where to find .gates files and which suites to walk by
# default. *_with_not variants are skipped by default since the
# gates_bench builder synthesizes NOT terms automatically — running
# both would double-count.
GATES_DEFAULT_ROOT   = "../miter-benchmarks/miter-cc-benchmarks"
GATES_DEFAULT_SUITES = ["iwls22", "hwmcc12"]

# Algorithms surfaced in the per-phase summary table (in column order).
# Mapping to the binaries' CSV `algorithm` tags:
#   nelson_seq        — Nelson's original sequential CC
#   nelson_topo       — sequential topo (sometimes unsound on cross-depth
#                        initial unions; kept as historical baseline)
#   nelson_topo_iter  — sequential topo_iter (sound, rounds-based)
#   par_close         — BSP parallel CC (parents_-frontier)
#   par_topo_iter     — parallel topo_iter (rounds-based, sound)
#   par_async         — async-rounds CC, integer-sort + run-walk
#                        semisort (production)
#   par_async_gbk     — same as par_async but using parlay::group_by_key
#                        (hash table + per-bucket sequence allocation);
#                        kept as A/B baseline for the integer-sort swap
#   par_async_cont    — truly-async CC (semisorter + unioner via
#                        parlay::par_do, deque mailbox, drain-gated
#                        BSP-with-overlap; same dirty-filter as
#                        par_async but pipelined within each round)
#   par_naive         — naive rounds CC (semisort all non-leaves every
#                        round; no dirty filter; the "ablation" against
#                        par_async that quantifies what filtering buys)
# Default algo set: parallel best (par_async) + sequential best baseline
# (nelson_simple). Other algos are still recognized via --algos but are
# off by default. Add "par_async_gbk", "nelson_seq", etc. via --algos
# when comparing more thoroughly.
ALGOS_OF_INTEREST = [
    "nelson_simple",
    "par_async",
]
ALGO_HEADERS = {
    "nelson_simple":  "nl_simple",
    "par_async":      "par_async",
}

# Sequential = every algo not starting with "par_". Run only at T=1.
SEQ_ALGOS = [a for a in ALGOS_OF_INTEREST if not a.startswith("par_")]


def cmake_build(targets: list[str]):
    print("Configuring CMake (Release)...", flush=True)
    subprocess.run(
        ["cmake", "-B", "build", "-S", ".", "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    print(f"Building {', '.join(targets)}...", flush=True)
    subprocess.run(
        ["cmake", "--build", "build", "-j"]
        + sum([["--target", t] for t in targets], []),
        check=True,
    )


def _numactl_prefix(no_numactl: bool) -> list[str]:
    if no_numactl:
        return []
    which = shutil.which("numactl")
    return [which, "-i", "all"] if which else []


def _run_binary(binary: str, *, env: dict, numactl_prefix: list[str],
                phase_label: str) -> str:
    cmd = list(numactl_prefix) + [binary]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{binary} failed during {phase_label}")
    return proc.stdout


# ---------------------------------------------------------------------------
# Phase: random / closure_compare_bench
# ---------------------------------------------------------------------------

# XL ladder: doubles n_nodes / n_merges each step. Driven via
# PE_BENCH_CUSTOM (leaves, fns, nodes, merges, depth) so closure_compare
# runs one large workload per invocation. Mirrors run_all_benchmarks.py's
# RANDOM_XL_LADDER; kept in sync manually.
RANDOM_XL_LADDER: dict[str, tuple[int, int, int, int, int]] = {
    "XL":   (50_000, 16,  2_000_000,   200_000, 3),
    "2XL":  (50_000, 16,  4_000_000,   400_000, 3),
    "4XL":  (50_000, 16,  8_000_000,   800_000, 3),
    "8XL":  (50_000, 16, 16_000_000, 1_600_000, 3),
    "16XL": (50_000, 16, 32_000_000, 3_200_000, 3),
    "32XL": (50_000, 16, 64_000_000, 6_400_000, 3),
}


def _retag_random_xl_rows(stdout: str, label: str, emit_header: bool) -> str:
    """closure_compare prints `custom,leaves,fns,...` rows when driven
    via PE_BENCH_CUSTOM. Rewrite the leading workload column from
    `custom` to the ladder label so the merged CSV is self-describing.
    Drops the header line unless emit_header is True."""
    out: list[str] = []
    for line in stdout.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("workload,"):
            if emit_header:
                out.append(line)
            continue
        if stripped.startswith("custom,"):
            comma = line.find(",")
            out.append(f"{label}{line[comma:]}")
            continue
        out.append(line)
    return "".join(out)


def run_random_xl(out_dir: Path, thread_counts: list[int],
                  *, warmup: int, trials: int,
                  labels: list[str] | None,
                  algos: list[str] | None,
                  numactl_prefix: list[str],
                  append: bool = False):
    csv_path = out_dir / "random.csv"
    binary = "./build/closure_compare_bench"
    if labels is None:
        ladder = list(RANDOM_XL_LADDER.items())
    else:
        unknown = [l for l in labels if l not in RANDOM_XL_LADDER]
        if unknown:
            sys.exit(f"unknown XL ladder labels: {unknown}. "
                     f"valid: {list(RANDOM_XL_LADDER)}")
        ladder = [(l, RANDOM_XL_LADDER[l]) for l in labels]
    mode = "a" if append else "w"
    print(f"=== random (XL ladder, {[l for l, _ in ladder]}) "
          f"-> {csv_path}{' (append)' if append else ''}", flush=True)
    first = not append
    with open(csv_path, mode) as out:
        for label, params in ladder:
            spec = ",".join(str(x) for x in params)
            for t in thread_counts:
                par_only = t > 1
                tag = "par_only" if par_only else "+seq"
                print(f"  [random-xl] {label} T={t} ({tag})", flush=True)
                env = os.environ.copy()
                env["PE_BENCH_FORMAT"] = "csv"
                env["PE_BENCH_TRIALS"] = str(trials)
                env["PE_BENCH_WARMUP"] = str(warmup)
                env["PARLAY_NUM_THREADS"] = str(t)
                env["PE_BENCH_CUSTOM"] = spec
                if first:
                    env["PE_BENCH_HEADER"] = "1"
                if par_only:
                    env["PE_BENCH_PAR_ONLY"] = "1"
                if algos:
                    env["PE_BENCH_ALGOS"] = ",".join(algos)
                stdout = _run_binary(binary, env=env,
                                     numactl_prefix=numactl_prefix,
                                     phase_label=f"random-xl {label} T={t}")
                out.write(_retag_random_xl_rows(
                    stdout, label, emit_header=first))
                out.flush()
                first = False
    return csv_path


def run_random(out_dir: Path, thread_counts: list[int],
               *, warmup: int, trials: int,
               algos: list[str] | None,
               numactl_prefix: list[str],
               append: bool = False):
    csv_path = out_dir / "random.csv"
    binary = "./build/closure_compare_bench"
    mode = "a" if append else "w"
    print(f"=== random (closure_compare_bench) → {csv_path}"
          f"{' (append)' if append else ''}", flush=True)
    first = not append  # skip CSV header on append so we don't duplicate it
    with open(csv_path, mode) as out:
        for t in thread_counts:
            par_only = t > 1
            tag = "par_only" if par_only else "+seq"
            print(f"  [random] T={t} ({tag})", flush=True)
            env = os.environ.copy()
            env["PE_BENCH_FORMAT"] = "csv"
            env["PE_BENCH_TRIALS"] = str(trials)
            env["PE_BENCH_WARMUP"] = str(warmup)
            env["PARLAY_NUM_THREADS"] = str(t)
            if first:
                env["PE_BENCH_HEADER"] = "1"
            if par_only:
                env["PE_BENCH_PAR_ONLY"] = "1"
            if algos:
                env["PE_BENCH_ALGOS"] = ",".join(algos)
            stdout = _run_binary(binary, env=env,
                                 numactl_prefix=numactl_prefix,
                                 phase_label=f"random T={t}")
            out.write(stdout); out.flush()
            first = False
    return csv_path


# ---------------------------------------------------------------------------
# Phase: synthetic / synthetic_bench across (family, n, threads)
# ---------------------------------------------------------------------------

def run_synthetic(out_dir: Path, families: list[str],
                  family_ns: dict[str, list[int]],
                  override_ns: list[int] | None,
                  thread_counts: list[int],
                  *, warmup: int, trials: int,
                  algos: list[str] | None,
                  numactl_prefix: list[str]):
    csv_path = out_dir / "synthetic.csv"
    binary = "./build/synthetic_bench"
    print(f"=== synthetic (synthetic_bench) → {csv_path}", flush=True)
    first = True
    with open(csv_path, "w") as out:
        for fam in families:
            ns = override_ns if override_ns is not None else family_ns[fam]
            for n in ns:
                for t in thread_counts:
                    par_only = t > 1
                    tag = "par_only" if par_only else "+seq"
                    print(f"  [synthetic] {fam} n={n} T={t} ({tag})",
                          flush=True)
                    env = os.environ.copy()
                    env["PE_SYNTH_FAMILIES"] = fam
                    env["PE_SYNTH_NS"] = str(n)
                    env["PE_BENCH_FORMAT"] = "csv"
                    env["PE_BENCH_TRIALS"] = str(trials)
                    env["PE_BENCH_WARMUP"] = str(warmup)
                    env["PARLAY_NUM_THREADS"] = str(t)
                    if first:
                        env["PE_BENCH_HEADER"] = "1"
                    if par_only:
                        env["PE_BENCH_PAR_ONLY"] = "1"
                    if algos:
                        env["PE_BENCH_ALGOS"] = ",".join(algos)
                    stdout = _run_binary(
                        binary, env=env, numactl_prefix=numactl_prefix,
                        phase_label=f"synthetic {fam} n={n} T={t}")
                    out.write(stdout); out.flush()
                    first = False
    return csv_path


# ---------------------------------------------------------------------------
# Phase: cube_decomp / synthetic_bench (cube only) across (d, k, threads)
# ---------------------------------------------------------------------------

def run_cube_decomp(out_dir: Path, thread_counts: list[int],
                    *, warmup: int, trials: int,
                    algos: list[str] | None,
                    numactl_prefix: list[str]):
    csv_path = out_dir / "cube_decomp.csv"
    binary = "./build/synthetic_bench"
    print(f"=== cube_decomp (synthetic_bench cube only) → {csv_path}",
          flush=True)
    first = True
    with open(csv_path, "w") as out:
        for d in CUBE_DECOMP_DS:
            for k in CUBE_DECOMP_KS:
                for t in thread_counts:
                    par_only = t > 1
                    tag = "par_only" if par_only else "+seq"
                    print(f"  [cube_decomp] d={d} k={k} T={t} ({tag})",
                          flush=True)
                    env = os.environ.copy()
                    env["PE_SYNTH_FAMILIES"] = "cube"
                    env["PE_SYNTH_NS"] = str(k)
                    env["PE_SYNTH_D"] = str(d)
                    env["PE_BENCH_FORMAT"] = "csv"
                    env["PE_BENCH_TRIALS"] = str(trials)
                    env["PE_BENCH_WARMUP"] = str(warmup)
                    env["PARLAY_NUM_THREADS"] = str(t)
                    if first:
                        env["PE_BENCH_HEADER"] = "1"
                    if par_only:
                        env["PE_BENCH_PAR_ONLY"] = "1"
                    if algos:
                        env["PE_BENCH_ALGOS"] = ",".join(algos)
                    stdout = _run_binary(
                        binary, env=env, numactl_prefix=numactl_prefix,
                        phase_label=f"cube_decomp d={d} k={k} T={t}")
                    out.write(stdout); out.flush()
                    first = False
    return csv_path


# ---------------------------------------------------------------------------
# Phase: egg / egraph-cc on .smt2 files, dispatched per algorithm
# ---------------------------------------------------------------------------

DEFAULT_EGG_DIR = "cc-benchmarks/smt-grounded"
CC_BENCHMARKS_LOCAL = "cc-benchmarks"

# (algorithm tag, --sequential value, env-var to set, env-var value).
# Sequential entries pass `seq_arg` and leave env empty; parallel entries
# leave `seq_arg` None and set the chosen PE_USE_* var. nelson_topo_iter
# and nelson_dst depend on the egraph-cc patches landing this same
# revision (--sequential=topo_iter / =dst).
EGG_ALGOS: list[tuple[str, str | None, str | None]] = [
    ("nelson_seq",       "nelson",     None),
    ("nelson_topo",      "topo",       None),
    ("nelson_topo_iter", "topo_iter",  None),
    ("nelson_dst",       "dst",        None),
    ("nelson_simple",    "simple",     None),
    ("par_close",        None,         None),
    ("par_topo_iter",    None,         "PE_USE_TOPO"),
    ("par_async",        None,         "PE_USE_ASYNC"),
    ("par_async_gbk",    None,         "PE_USE_ASYNC_GBK"),
    ("par_async_cont",   None,         "PE_USE_ASYNC_CONT"),
    ("par_async_min_id", None,         "PE_USE_ASYNC_MIN_ID"),
    ("par_naive",        None,         "PE_USE_NAIVE"),
]

EGG_TIMING_RE = re.compile(
    r"timing:\s+read=(\S+)\s+parse=(\S+)\s+build=(\S+)\s+"
    r"close=(\S+)\s+check=(\S+)\s+dtor=(\S+)"
)
EGG_TIMING_KEYS = ["read", "parse", "build", "close", "check", "dtor"]
EGG_UNIT_TO_S = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}


def _parse_egg_duration(token: str) -> float:
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([a-zµ]*)$", token)
    if not m:
        raise ValueError(f"unrecognized duration token: {token!r}")
    value, unit = float(m.group(1)), m.group(2)
    if unit == "":
        return value
    if unit not in EGG_UNIT_TO_S:
        raise ValueError(f"unknown time unit in {token!r}")
    return value * EGG_UNIT_TO_S[unit]


def _egg_parse_timing(stderr: str) -> dict | None:
    m = EGG_TIMING_RE.search(stderr)
    if not m:
        return None
    return {k: _parse_egg_duration(m.group(i + 1))
            for i, k in enumerate(EGG_TIMING_KEYS)}


def _egg_parse_result(stdout: str) -> str:
    for line in stdout.splitlines():
        line = line.strip()
        if line in ("sat", "unsat", "unknown"):
            return line
    return "UNKNOWN"


def _egg_expected(path: str) -> str:
    m = re.search(r"(?:^|[._])(sat|unsat)\.smt2$", os.path.basename(path))
    return m.group(1) if m else ""


def _ensure_cc_benchmarks(egg_dir: str) -> str:
    """Initialize the cc-benchmarks submodule if smt-grounded is missing."""
    if os.path.isdir(egg_dir):
        return egg_dir
    print(f"Initializing submodule: {CC_BENCHMARKS_LOCAL}", flush=True)
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive",
         CC_BENCHMARKS_LOCAL],
        check=True,
    )
    if not os.path.isdir(egg_dir):
        sys.exit(f"submodule init succeeded but {egg_dir} is still missing")
    return egg_dir


def run_egg(out_dir: Path, thread_counts: list[int],
            *, egg_dir: str, pattern: str, timeout: float,
            warmup: int, trials: int,
            algos: list[str] | None,
            numactl_prefix: list[str]):
    csv_path = out_dir / "egg.csv"
    binary = "./build/egraph-cc"
    _ensure_cc_benchmarks(egg_dir)

    files = sorted(glob.glob(os.path.join(egg_dir, "*.smt2")))
    if pattern != "*":
        import fnmatch
        files = [f for f in files
                 if fnmatch.fnmatch(os.path.basename(f), pattern)]
    if not files:
        print(f"  [egg] no .smt2 files matched in {egg_dir}", flush=True)
        return None

    egg_algos = EGG_ALGOS
    if algos:
        wanted = set(algos)
        egg_algos = [row for row in EGG_ALGOS if row[0] in wanted]
        missing = wanted - {row[0] for row in EGG_ALGOS}
        if missing:
            print(f"  [egg] WARN: --algos contains {sorted(missing)} which "
                  "egraph-cc does not support; ignoring.", flush=True)

    # (algo, T) cells: skip sequential algos when T>1.
    cells: list[tuple[str, str | None, str | None, int]] = []
    for t in thread_counts:
        for algo, seq_arg, env_var in egg_algos:
            if seq_arg is not None and t != 1:
                continue
            cells.append((algo, seq_arg, env_var, t))

    print(f"=== egg ({len(files)} files × {len(cells)} cells) "
          f"→ {csv_path}", flush=True)

    def run_invocation(path: str, t: int, seq_arg: str | None,
                       env_var: str | None):
        cmd = list(numactl_prefix) + [binary, "--timing"]
        if seq_arg is not None:
            cmd.append(f"--sequential={seq_arg}")
        cmd.append(path)
        env = os.environ.copy()
        env["PARLAY_NUM_THREADS"] = str(t)
        if env_var is not None:
            env[env_var] = "1"
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  env=env, timeout=timeout)
            wall = time.perf_counter() - t0
            if proc.returncode != 0:
                return (wall, "ERROR", None,
                        proc.stderr.strip()[:200])
            return (wall, _egg_parse_result(proc.stdout),
                    _egg_parse_timing(proc.stderr), "")
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - t0
            return (wall, "TIMEOUT", None, f"exceeded {timeout}s")

    with open(csv_path, "w", newline="") as f:
        writer = csvmod.writer(f)
        writer.writerow(["file", "expected", "algorithm", "threads", "trial",
                         "result", "wall_s",
                         *[f"{k}_s" for k in EGG_TIMING_KEYS], "error"])
        for path in files:
            name = os.path.basename(path)
            expected = _egg_expected(path)
            for algo, seq_arg, env_var, t in cells:
                # Warmup; if it fails (TIMEOUT/ERROR) skip the trial loop.
                bail = False
                for _ in range(warmup):
                    _, w_result, _, _ = run_invocation(
                        path, t, seq_arg, env_var)
                    if w_result in ("TIMEOUT", "ERROR"):
                        writer.writerow([name, expected, algo, t, -1,
                                         w_result, "", *[""] * 6,
                                         "warmup failed"])
                        f.flush()
                        print(f"  [egg] {name} algo={algo} T={t} "
                              f"warmup={w_result} (skipping trials)",
                              flush=True)
                        bail = True
                        break
                if bail:
                    continue
                for trial in range(trials):
                    wall, result, timing, error = run_invocation(
                        path, t, seq_arg, env_var)
                    row = [name, expected, algo, t, trial, result,
                           f"{wall:.6f}"]
                    row += [f"{timing[k]:.6f}" if timing else ""
                            for k in EGG_TIMING_KEYS]
                    row.append(error)
                    writer.writerow(row)
                    f.flush()
                    close_s = (f"{timing['close']:.4f}s"
                               if timing else "-")
                    print(f"  [egg] {name} algo={algo} T={t} trial={trial} "
                          f"{result:<8} close={close_s} wall={wall:.2f}s",
                          flush=True)
    return csv_path


# ---------------------------------------------------------------------------
# Phase: gates / gates_bench on miter-cc-benchmarks
# ---------------------------------------------------------------------------

def run_gates(out_dir: Path, thread_counts: list[int],
              gates_root: str, suites: list[str],
              files_glob: str | None,
              *, warmup: int, trials: int,
              algos: list[str] | None,
              numactl_prefix: list[str]):
    """gates_bench across all matched .gates files at every thread
    count. By default runs all three algorithms (nelson_topo_iter,
    par_topo_iter, par_async); --algos restricts via PE_BENCH_ALGOS.
    nelson_topo_iter is sequential — its time doesn't depend on
    PARLAY_NUM_THREADS, but we re-run it at every T anyway because
    the per-(file, threads) cell is the unit of measurement here and
    skipping it would create awkward gaps in the CSV. The duplicated
    measurements should be near-identical; downstream plotters average
    across T or pick the T=1 row.

    CSV schema is gates_bench's native:
      file,suite,n_gates,n_literals,n_not_terms,total_classes,
      algorithm,trial,parlay_threads,read_s,parse_s,build_s,close_ms.

    One subprocess per (file, threads) cell, mirroring run_egg's
    per-file granularity. Two reasons we don't batch all files into
    one subprocess: (a) `subprocess.run(capture_output=True)` only
    flushes stdout to the parent on process exit, so a multi-hour
    invocation that gets killed leaves *zero bytes* on disk; (b) the
    largest file (6s125-xits-opt at 7.8M gates) takes 30s+ closure at
    T=1, so a multi-file batch can run minutes between checkpoints.
    Per-file invocations checkpoint after each file at the cost of a
    few hundred ms of fork+load overhead per file (negligible relative
    to closure time on the files we care about).
    """
    csv_path = out_dir / "gates.csv"
    binary = "./build/gates_bench"

    # Resolve input files: explicit glob wins; otherwise walk suites.
    if files_glob:
        files = sorted(glob.glob(files_glob))
    else:
        files = []
        for suite in suites:
            files += sorted(glob.glob(os.path.join(gates_root, suite,
                                                    "*.gates")))
    if not files:
        print(f"  [gates] no .gates files matched (root={gates_root}, "
              f"suites={suites}, glob={files_glob})")
        return csv_path

    print(f"=== gates (gates_bench) → {csv_path}", flush=True)
    print(f"  [gates] {len(files)} file(s) × {len(thread_counts)} thread "
          f"count(s)", flush=True)

    first = True
    with open(csv_path, "w") as out:
        for t in thread_counts:
            for fpath in files:
                fname = os.path.basename(fpath)
                print(f"  [gates] T={t} {fname}", flush=True)
                t0 = time.perf_counter()
                cmd = list(numactl_prefix) + [binary, fpath]
                env = os.environ.copy()
                env["PE_BENCH_TRIALS"] = str(trials)
                env["PE_BENCH_WARMUP"] = str(warmup)
                env["PARLAY_NUM_THREADS"] = str(t)
                if first:
                    env["PE_BENCH_HEADER"] = "1"
                # gates_bench understands a comma-separated list of EXACT
                # algo names: nelson_topo_iter, par_topo_iter, par_async.
                # Only forward the subset our --algos whitelist allows.
                if algos:
                    gates_algos = [a for a in algos
                                   if a in ("nelson_topo_iter",
                                            "par_topo_iter",
                                            "par_async")]
                    if gates_algos:
                        env["PE_BENCH_ALGOS"] = ",".join(gates_algos)
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      env=env)
                wall = time.perf_counter() - t0
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr)
                    raise RuntimeError(
                        f"gates_bench failed at T={t} on {fname}")
                # Mirror the bench's [gates_bench] progress / sizing
                # line and write the trial rows out immediately so a
                # killed run leaves a checkpointed CSV.
                if proc.stderr:
                    sys.stderr.write(proc.stderr)
                out.write(proc.stdout); out.flush()
                first = False
                print(f"    done in {wall:.1f}s", flush=True)
    return csv_path


def summarize_gates(csv_path: Path):
    """Custom summary for the gates phase: pairs par_topo_iter vs
    par_async per (file, threads), prints both medians + ratio + each
    algorithm's strong-scaling speedup against its own T=1.
    """
    rows = list(csvmod.DictReader(open(csv_path)))
    if not rows:
        print(f"  (empty: {csv_path})")
        return
    buckets: dict[tuple, list[float]] = {}
    file_meta: dict[str, tuple[int, int]] = {}
    for r in rows:
        try:
            ms = float(r["close_ms"])
        except (KeyError, ValueError):
            continue
        key = (r["file"], int(r["parlay_threads"]), r["algorithm"])
        buckets.setdefault(key, []).append(ms)
        file_meta[r["file"]] = (int(r["n_gates"]), int(r["total_classes"]))
    medians = {k: statistics.median(v) for k, v in buckets.items() if v}

    files = sorted({k[0] for k in medians})
    threads = sorted({k[1] for k in medians})

    print()
    print(f"{'file':<32} {'thr':>4} | {'topo_ms':>10} {'async_ms':>10}"
          f" | {'topo/async':>10} {'topo_T1/T':>10} {'async_T1/T':>11}"
          f" | {'gates':>10} {'classes':>10}")
    print("-" * 120)
    for f in files:
        gates, total = file_meta.get(f, (0, 0))
        topo_t1 = medians.get((f, threads[0], "par_topo_iter"))
        async_t1 = medians.get((f, threads[0], "par_async"))
        for t in threads:
            tm = medians.get((f, t, "par_topo_iter"))
            am = medians.get((f, t, "par_async"))
            tm_s = f"{tm:>10.2f}" if tm is not None else f"{'-':>10}"
            am_s = f"{am:>10.2f}" if am is not None else f"{'-':>10}"
            ratio = (f"{tm/am:>9.2f}x"
                     if (tm is not None and am and am > 0) else f"{'-':>10}")
            t_spd = (f"{topo_t1/tm:>9.2f}x"
                     if (tm and topo_t1) else f"{'-':>10}")
            a_spd = (f"{async_t1/am:>10.2f}x"
                     if (am and async_t1) else f"{'-':>11}")
            short = f.replace(".gates", "")
            print(f"{short:<32} {t:>4} | {tm_s} {am_s} | "
                  f"{ratio} {t_spd} {a_spd} | {gates:>10} {total:>10}")
        print()


# ---------------------------------------------------------------------------
# Per-phase summary
# ---------------------------------------------------------------------------

def summarize(csv_path: Path, group_keys: list[str]):
    """Print one summary block per CSV file using `group_keys` as the
    workload identity (e.g. ['family','n'] for synthetic, ['workload']
    for random, ['family','n','d'] for cube_decomp)."""
    rows = list(csvmod.DictReader(open(csv_path)))
    if not rows:
        print(f"  (empty: {csv_path})")
        return
    buckets: dict[tuple, list[float]] = {}
    for r in rows:
        algo = r["algorithm"]
        if algo not in ALGOS_OF_INTEREST:
            continue
        try:
            wl = tuple(r[k] for k in group_keys)
            t = int(r["parlay_threads"])
            ms = float(r["wallclock_ms"])
        except (KeyError, ValueError):
            continue
        buckets.setdefault((wl, t, algo), []).append(ms)
    medians = {k: statistics.median(v) for k, v in buckets.items() if v}

    workloads = sorted({wl for (wl, _, _) in medians})
    threads = sorted({t for (_, t, _) in medians})
    if not workloads:
        return

    col_w = {a: max(len(ALGO_HEADERS[a]), 10) for a in ALGOS_OF_INTEREST}
    wl_label = "/".join(group_keys)
    fixed = f"{wl_label:<20} {'thr':>4} | "
    header_cells = [f"{ALGO_HEADERS[a]:>{col_w[a]}}" for a in ALGOS_OF_INTEREST]
    print()
    print(f"--- {csv_path.name} ---")
    print(fixed + " ".join(header_cells)
          + " | "
          + f"{'simple/async':>12}")
    total_w = (len(fixed) + sum(col_w.values()) + len(ALGOS_OF_INTEREST) - 1
               + len(" | ") + 12)
    print("-" * total_w)

    # Sequential algos only run at T=1; cache that median per workload.
    seq_at_t1: dict[tuple, dict[str, float]] = {}
    for wl in workloads:
        seq_at_t1[wl] = {a: medians[(wl, 1, a)]
                         for a in SEQ_ALGOS if (wl, 1, a) in medians}

    for wl in workloads:
        for t in threads:
            cells = []
            vals: dict[str, float | None] = {}
            for a in ALGOS_OF_INTEREST:
                if a in SEQ_ALGOS:
                    # Display the T=1 value on every row for context.
                    v = seq_at_t1[wl].get(a)
                else:
                    v = medians.get((wl, t, a))
                vals[a] = v
                cells.append(f"{v:>{col_w[a]}.2f}" if v is not None
                             else f"{'-':>{col_w[a]}}")
            # Headline ratio: sequential baseline / parallel best —
            # the parallel speedup over single-threaded simple.
            pa = vals.get("par_async")
            ns = vals.get("nelson_simple")
            ratio_simple_async = (f"{ns/pa:>11.2f}x"
                                  if (ns and pa and pa > 0)
                                  else f"{'-':>12}")
            wl_str = "/".join(str(x) for x in wl)
            print(f"{wl_str:<20} {t:>4} | "
                  + " ".join(cells)
                  + f" | {ratio_simple_async}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads-sweep", default="1,4,8",
                    help="comma-separated thread counts (default: 1,4,8). "
                         "T=1 runs sequential + parallel; T>1 runs only "
                         "parallel algos via PE_BENCH_PAR_ONLY=1.")
    ap.add_argument("--families", default=None,
                    help=f"comma-separated subset of "
                         f"{','.join(DEFAULT_FAMILY_NS)} for the synthetic "
                         f"phase (default: all)")
    ap.add_argument("--ns", default=None,
                    help="comma-separated n values; applied to every "
                         "selected family (default: per-family ranges)")
    ap.add_argument("--skip", action="append", default=[],
                    choices=["random", "synthetic", "cube_decomp", "egg",
                             "gates"],
                    help="skip a phase; repeatable")
    ap.add_argument("--algos", default="par_async,nelson_simple",
                    help="comma-separated algorithm whitelist. Default: "
                         "par_async,nelson_simple (the parallel and "
                         "sequential headline baselines). Other valid "
                         "names: nelson_seq, nelson_topo, "
                         "nelson_topo_iter, nelson_dst, par_close, "
                         "par_topo_iter, par_async_gbk, par_async_cont, "
                         "par_async_min_id, par_naive. Pass 'all' to "
                         "run every recognized algo. Sets PE_BENCH_ALGOS "
                         "for the C++ binaries and filters the egg "
                         "dispatch loop, so unwanted algos are never run.")
    ap.add_argument("--random-modes", default="default",
                    help="comma-separated random-phase modes. Valid: "
                         "'default' (6 baked-in workloads) and 'xl' "
                         "(XL/2XL/.../32XL ladder via PE_BENCH_CUSTOM). "
                         "Pass both ('default,xl') to merge their rows "
                         "into one random.csv.")
    ap.add_argument("--xl-labels", default=None,
                    help="comma-separated subset of XL ladder rungs to run "
                         f"(valid: {','.join(RANDOM_XL_LADDER)}). Only used "
                         "when 'xl' is in --random-modes. Default: all rungs.")
    ap.add_argument("--egg-dir", default=DEFAULT_EGG_DIR,
                    help=f"directory of .smt2 files for the egg phase "
                         f"(default: {DEFAULT_EGG_DIR})")
    ap.add_argument("--egg-pattern", default="*",
                    help="basename glob filter for egg files (default: '*')")
    ap.add_argument("--egg-timeout", type=float, default=120.0,
                    help="per-invocation wall budget for egg (default: 120s)")
    ap.add_argument("--gates-root", default=GATES_DEFAULT_ROOT,
                    help=f"miter-cc-benchmarks root for the gates phase "
                         f"(default: {GATES_DEFAULT_ROOT})")
    ap.add_argument("--gates-suites", default=",".join(GATES_DEFAULT_SUITES),
                    help=f"comma-separated suite dirs under --gates-root "
                         f"(default: {','.join(GATES_DEFAULT_SUITES)})")
    ap.add_argument("--gates-files", default=None,
                    help="explicit glob of .gates files (overrides "
                         "--gates-root/--gates-suites)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup runs (default 1)")
    ap.add_argument("--trials", type=int, default=3,
                    help="measured trials (default 3)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: runs/topo_vs_async_<ts>/)")
    ap.add_argument("--no-numactl", action="store_true",
                    help="don't prepend `numactl -i all`")
    args = ap.parse_args()

    try:
        thread_counts = sorted(set(int(t) for t in
                                   args.threads_sweep.split(",") if t))
    except ValueError:
        sys.exit("--threads-sweep expects comma-separated integers")
    if not thread_counts:
        sys.exit("--threads-sweep empty")
    if 1 not in thread_counts:
        print("WARN: T=1 not in --threads-sweep; sequential algos will not "
              "be measured. Add 1 if you want a baseline.", file=sys.stderr)

    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        unknown = [f for f in families if f not in DEFAULT_FAMILY_NS]
        if unknown:
            sys.exit(f"unknown families: {','.join(unknown)}")
    else:
        families = list(DEFAULT_FAMILY_NS)

    override_ns = None
    if args.ns:
        try:
            override_ns = [int(x) for x in args.ns.split(",") if x.strip()]
        except ValueError:
            sys.exit("--ns expects comma-separated integers")

    valid_algos = {row[0] for row in EGG_ALGOS}
    algos: list[str] | None = None
    if args.algos:
        if args.algos.strip() == "all":
            # Sentinel: run every recognized algo (no filter passed to
            # the binaries, no filter on the egg dispatch loop).
            algos = None
        else:
            algos = [a.strip() for a in args.algos.split(",") if a.strip()]
            unknown = [a for a in algos if a not in valid_algos]
            if unknown:
                sys.exit(f"unknown algos: {','.join(unknown)}. "
                         f"valid: {sorted(valid_algos)}")

    valid_random_modes = {"default", "xl"}
    random_modes = [m.strip() for m in args.random_modes.split(",")
                    if m.strip()]
    unknown = [m for m in random_modes if m not in valid_random_modes]
    if unknown:
        sys.exit(f"unknown random modes: {','.join(unknown)}. "
                 f"valid: {sorted(valid_random_modes)}")
    if not random_modes:
        sys.exit("--random-modes is empty")
    # Preserve user-given order in case it matters (it doesn't downstream;
    # plotting groups by workload). Dedup while preserving first occurrence.
    seen: set[str] = set()
    random_modes = [m for m in random_modes
                    if not (m in seen or seen.add(m))]

    xl_labels = None
    if args.xl_labels:
        xl_labels = [s.strip() for s in args.xl_labels.split(",")
                     if s.strip()]
        if "xl" not in random_modes:
            print("WARN: --xl-labels given but 'xl' not in --random-modes; "
                  "ignoring.", file=sys.stderr)
            xl_labels = None

    skip = set(args.skip)
    targets: list[str] = []
    if "random" not in skip:
        targets.append("closure_compare_bench")
    if ("synthetic" not in skip) or ("cube_decomp" not in skip):
        targets.append("synthetic_bench")
    if "egg" not in skip:
        targets.append("egraph-cc")
    if "gates" not in skip:
        targets.append("gates_bench")
    if not targets:
        sys.exit("nothing to do (everything skipped)")
    cmake_build(targets)

    numactl_prefix = _numactl_prefix(args.no_numactl)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.out) if args.out
               else Path("runs") / f"topo_vs_async_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output:        {out_dir}")
    print(f"Phases:        {[p for p in ('random','synthetic','cube_decomp','egg') if p not in skip]}")
    if "random" not in skip:
        rm_label = ",".join(random_modes)
        if "xl" in random_modes:
            rm_label += f" (xl={xl_labels or list(RANDOM_XL_LADDER)})"
        print(f"Random modes:  {rm_label}")
    print(f"Threads:       {thread_counts}")
    print(f"Families:      {families}")
    if override_ns:
        print(f"Override ns:   {override_ns}")
    print(f"Warmup/trials: {args.warmup}/{args.trials}")
    if algos:
        print(f"Algos:         {algos}  (PE_BENCH_ALGOS filter active)")
    else:
        print(f"Algos shown:   {ALGOS_OF_INTEREST}")
        print(f"  (sequential, T=1 only): {SEQ_ALGOS}")
    print(f"numactl:       "
          f"{' '.join(numactl_prefix) if numactl_prefix else 'none'}")
    print()

    csv_paths: list[tuple[Path, list[str]]] = []
    if "random" not in skip:
        # Run each requested mode; second/third call appends so they all
        # land in one random.csv that downstream plotters can consume.
        first = True
        last_p = None
        for m in random_modes:
            if m == "default":
                last_p = run_random(out_dir, thread_counts,
                                    warmup=args.warmup, trials=args.trials,
                                    algos=algos,
                                    numactl_prefix=numactl_prefix,
                                    append=not first)
            else:  # "xl"
                last_p = run_random_xl(out_dir, thread_counts,
                                       warmup=args.warmup,
                                       trials=args.trials,
                                       labels=xl_labels, algos=algos,
                                       numactl_prefix=numactl_prefix,
                                       append=not first)
            first = False
        if last_p is not None:
            csv_paths.append((last_p, ["workload"]))
    if "synthetic" not in skip:
        p = run_synthetic(out_dir, families, DEFAULT_FAMILY_NS, override_ns,
                          thread_counts,
                          warmup=args.warmup, trials=args.trials,
                          algos=algos,
                          numactl_prefix=numactl_prefix)
        csv_paths.append((p, ["family", "n"]))
    if "cube_decomp" not in skip:
        p = run_cube_decomp(out_dir, thread_counts,
                            warmup=args.warmup, trials=args.trials,
                            algos=algos,
                            numactl_prefix=numactl_prefix)
        csv_paths.append((p, ["family", "n", "d"]))

    if "egg" not in skip:
        p = run_egg(out_dir, thread_counts,
                    egg_dir=args.egg_dir, pattern=args.egg_pattern,
                    timeout=args.egg_timeout,
                    warmup=args.warmup, trials=args.trials,
                    algos=algos,
                    numactl_prefix=numactl_prefix)
        # egg.csv has a different schema (file/algorithm/threads, no
        # family/n/wallclock_ms) — `summarize` won't grok it. Skip in the
        # per-phase summary; downstream plotters handle it directly.

    gates_csv: Path | None = None
    if "gates" not in skip:
        gates_csv = run_gates(
            out_dir, thread_counts,
            gates_root=args.gates_root,
            suites=[s.strip() for s in args.gates_suites.split(",") if s.strip()],
            files_glob=args.gates_files,
            warmup=args.warmup, trials=args.trials,
            algos=algos,
            numactl_prefix=numactl_prefix)

    for p, keys in csv_paths:
        summarize(p, keys)
    if gates_csv is not None and gates_csv.exists():
        summarize_gates(gates_csv)

    print()
    print(f"Done. CSVs under: {out_dir}")


if __name__ == "__main__":
    main()
