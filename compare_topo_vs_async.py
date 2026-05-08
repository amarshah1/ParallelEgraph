#!/usr/bin/env python3
"""All-phase driver for the par_topo_iter vs par_async comparison.

Phases:
  random       — closure_compare_bench (6 baked-in workloads)
  synthetic    — synthetic_bench across (family, n, threads)
  cube_decomp  — synthetic_bench (cube only) across (d, k, threads)
  egg          — egraph-cc on each .smt2 in cc-benchmarks/smt-grounded,
                 dispatched per-algorithm via --sequential / PE_USE_*

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

# Algorithms surfaced in the per-phase summary table (in column order).
# Mapping to the binaries' CSV `algorithm` tags:
#   nelson_seq        — Nelson's original sequential CC
#   nelson_topo       — sequential topo (sometimes unsound on cross-depth
#                        initial unions; kept as historical baseline)
#   nelson_topo_iter  — sequential topo_iter (sound, rounds-based)
#   par_close         — BSP parallel CC (parents_-frontier)
#   par_topo_iter     — parallel topo_iter (rounds-based, sound)
#   par_async         — async-rounds CC (mark-based dirty filter)
#   par_async_cont    — truly-async CC (semisorter + unioner via
#                        parlay::par_do, deque mailbox, drain-gated
#                        BSP-with-overlap; same dirty-filter as
#                        par_async but pipelined within each round)
#   par_naive         — naive rounds CC (semisort all non-leaves every
#                        round; no dirty filter; the "ablation" against
#                        par_async that quantifies what filtering buys)
ALGOS_OF_INTEREST = [
    "nelson_seq",
    "nelson_topo",
    "nelson_topo_iter",
    "par_close",
    "par_topo_iter",
    "par_async",
    "par_async_cont",
    "par_naive",
]
ALGO_HEADERS = {
    "nelson_seq":       "nelson",
    "nelson_topo":      "nl_topo",
    "nelson_topo_iter": "nl_topo_it",
    "par_close":        "par_close",
    "par_topo_iter":    "par_topo_it",
    "par_async":        "par_async",
    "par_async_cont":   "par_a_cont",
    "par_naive":        "par_naive",
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

def run_random(out_dir: Path, thread_counts: list[int],
               *, warmup: int, trials: int,
               algos: list[str] | None,
               numactl_prefix: list[str]):
    csv_path = out_dir / "random.csv"
    binary = "./build/closure_compare_bench"
    print(f"=== random (closure_compare_bench) → {csv_path}", flush=True)
    first = True
    with open(csv_path, "w") as out:
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
    ("par_close",        None,         None),
    ("par_topo_iter",    None,         "PE_USE_TOPO"),
    ("par_async",        None,         "PE_USE_ASYNC"),
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
          + f"{'topo/async':>10} {'naive/async':>12} "
          + f"{'cont/async':>11} {'topo_it_spd':>12}")
    total_w = (len(fixed) + sum(col_w.values()) + len(ALGOS_OF_INTEREST) - 1
               + len(" | ") + 10 + 1 + 12 + 1 + 11 + 1 + 12)
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
            pt = vals.get("par_topo_iter")
            pa = vals.get("par_async")
            pc = vals.get("par_async_cont")
            pn = vals.get("par_naive")
            ni = vals.get("nelson_topo_iter")
            ratio_topo_async = (f"{pt/pa:>9.2f}x"
                                if (pt and pa and pa > 0) else f"{'-':>10}")
            ratio_naive_async = (f"{pn/pa:>11.2f}x"
                                 if (pn and pa and pa > 0)
                                 else f"{'-':>12}")
            ratio_cont_async = (f"{pc/pa:>10.2f}x"
                                if (pc and pa and pa > 0)
                                else f"{'-':>11}")
            topo_it_spd = (f"{ni/pt:>11.2f}x"
                           if (ni and pt and pt > 0) else f"{'-':>12}")
            wl_str = "/".join(str(x) for x in wl)
            print(f"{wl_str:<20} {t:>4} | "
                  + " ".join(cells)
                  + f" | {ratio_topo_async} {ratio_naive_async} "
                  + f"{ratio_cont_async} {topo_it_spd}")
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
                    choices=["random", "synthetic", "cube_decomp", "egg"],
                    help="skip a phase; repeatable")
    ap.add_argument("--algos", default=None,
                    help="comma-separated algorithm whitelist. Default: "
                         "all. Names: nelson_seq, nelson_topo, "
                         "nelson_topo_iter, nelson_dst, par_close, "
                         "par_topo_iter, par_async, par_async_cont, "
                         "par_async_min_id, par_naive. Sets "
                         "PE_BENCH_ALGOS for the C++ binaries and "
                         "filters the egg dispatch loop, so unwanted "
                         "algos are never run.")
    ap.add_argument("--egg-dir", default=DEFAULT_EGG_DIR,
                    help=f"directory of .smt2 files for the egg phase "
                         f"(default: {DEFAULT_EGG_DIR})")
    ap.add_argument("--egg-pattern", default="*",
                    help="basename glob filter for egg files (default: '*')")
    ap.add_argument("--egg-timeout", type=float, default=120.0,
                    help="per-invocation wall budget for egg (default: 120s)")
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

    valid_algos = {row[0] for row in EGG_ALGOS}  # full 8 names
    algos: list[str] | None = None
    if args.algos:
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
        unknown = [a for a in algos if a not in valid_algos]
        if unknown:
            sys.exit(f"unknown algos: {','.join(unknown)}. "
                     f"valid: {sorted(valid_algos)}")

    skip = set(args.skip)
    targets: list[str] = []
    if "random" not in skip:
        targets.append("closure_compare_bench")
    if ("synthetic" not in skip) or ("cube_decomp" not in skip):
        targets.append("synthetic_bench")
    if "egg" not in skip:
        targets.append("egraph-cc")
    if not targets:
        sys.exit("nothing to do (everything skipped)")
    cmake_build(targets)

    numactl_prefix = _numactl_prefix(args.no_numactl)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (Path(args.out) if args.out
               else Path("runs") / f"topo_vs_async_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output:        {out_dir}")
    print(f"Phases:        {[p for p in ('random','synthetic','cube_decomp') if p not in skip]}")
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
        p = run_random(out_dir, thread_counts,
                       warmup=args.warmup, trials=args.trials,
                       algos=algos,
                       numactl_prefix=numactl_prefix)
        csv_paths.append((p, ["workload"]))
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

    for p, keys in csv_paths:
        summarize(p, keys)

    print()
    print(f"Done. CSVs under: {out_dir}")


if __name__ == "__main__":
    main()
