#!/usr/bin/env python3
"""FMCAD 2026 artifact driver — runs nelson_simple_inline, par_parents,
par_filter across all relevant workload phases.

Phases:
  random       — closure_compare_bench (6 baked-in workloads + XL ladder
                 via PE_BENCH_CUSTOM, opt-in via --random-modes xl)
  synthetic    — synthetic_bench across (family, n, threads)
  cube_decomp  — synthetic_bench (cube only) across (d, k, threads)
  gates        — gates_bench on each .gates in
                 miter-cc-benchmarks/{iwls22,hwmcc12}; CSV schema is gates'
                 native (file, suite, n_gates, ..., algorithm, ...)

Each phase emits the same CSV schema the corresponding C++ binary already
prints, so the existing plotter (`plot_topo_vs_async.py`) can consume the
output unchanged.

Sequential algorithms only depend on the workload, not on
PARLAY_NUM_THREADS. Running them at every T duplicates measurements and
inflates wall time. The bench binaries honour PE_BENCH_PAR_ONLY=1 (set
on every T>1 invocation; T=1 runs the full set).

Usage (canonical FMCAD invocation):
    python3 compare_topo_vs_async.py \\
      --skip egg --skip synthetic \\
      --threads-sweep 1,2,4,8,16,32,64,96,128,192 \\
      --warmup 1 --random-modes xl,default --trials 5 \\
      --algos nelson_simple_inline,par_parents,par_filter
"""

import argparse
import csv as csvmod
import datetime
import glob
import os
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
CUBE_DECOMP_KS = list(range(5, 206, 25))

# gates phase: where to find .gates files and which suites to walk by
# default. The non *_with_not variants are skipped by default since the
# gates_bench builder synthesizes NOT terms automatically — running
# both would double-count.
GATES_DEFAULT_ROOT   = "./miter-cc-benchmarks"
GATES_DEFAULT_SUITES = ["iwls22_with_not", "hwmcc12_with_not"]

# Algorithms surfaced in the per-phase summary table (in column order).
# Mapping to the binaries' CSV `algorithm` tags:
#   nelson_simple_inline — sequential CC, SigK<K> + bump-arena fallback
#   par_parents          — BSP parallel CC (parents_-frontier)
#   par_filter           — filter-mode parallel CC (last_marked_ stamps)
ALGOS_OF_INTEREST = [
    "nelson_simple_inline",
    "par_parents",
    "par_filter",
]
ALGO_HEADERS = {
    "nelson_simple_inline":  "nl_inline",
    "par_parents":           "par_parents",
    "par_filter":            "par_filter",
}

# Sequential = every algo not starting with "par_". Run only at T=1.
SEQ_ALGOS = [a for a in ALGOS_OF_INTEREST if not a.startswith("par_")]


def _has_parallel_algo(algos: "list[str] | None") -> bool:
    """`--algos` whitelist contains at least one parallel algo (or is unset,
    so every algo will run). When False, T>1 invocations are pure waste:
    PE_BENCH_PAR_ONLY=1 strips every algo from the run, but the binary still
    pays init/parse/build cost. Callers skip the T>1 cells in that case."""
    if algos is None:
        return True
    return any(a.startswith("par_") for a in algos)


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
                phase_label: str,
                trace_path: "Path | None" = None) -> str:
    cmd = list(numactl_prefix) + [binary]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{binary} failed during {phase_label}")
    # Forensic stderr capture (PE_TRACE per-round detail, build sizes,
    # etc). Each caller passes a unique path so a killed run still
    # leaves the partial traces on disk.
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as tf:
            tf.write(proc.stderr)
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
    traces_dir = out_dir / "traces" / "random_xl"
    first = not append
    with open(csv_path, mode) as out:
        any_par = _has_parallel_algo(algos)
        for label, params in ladder:
            spec = ",".join(str(x) for x in params)
            for t in thread_counts:
                par_only = t > 1
                if par_only and not any_par:
                    continue  # sequential-only: T>1 cells would be empty
                tag = "par_only" if par_only else "+seq"
                print(f"  [random-xl] {label} T={t} ({tag})", flush=True)
                env = os.environ.copy()
                env["PE_BENCH_FORMAT"] = "csv"
                env["PE_BENCH_TRIALS"] = str(trials)
                env["PE_BENCH_WARMUP"] = str(warmup)
                env["PARLAY_NUM_THREADS"] = str(t)
                env["PE_BENCH_CUSTOM"] = spec
                env["PE_TRACE"] = "1"
                if first:
                    env["PE_BENCH_HEADER"] = "1"
                if par_only:
                    env["PE_BENCH_PAR_ONLY"] = "1"
                if algos is not None:
                    env["PE_BENCH_ALGOS"] = ",".join(algos)
                stdout = _run_binary(binary, env=env,
                                     numactl_prefix=numactl_prefix,
                                     phase_label=f"random-xl {label} T={t}",
                                     trace_path=traces_dir
                                     / f"{label}__T{t:03d}.log")
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
    traces_dir = out_dir / "traces" / "random"
    first = not append  # skip CSV header on append so we don't duplicate it
    any_par = _has_parallel_algo(algos)
    with open(csv_path, mode) as out:
        for t in thread_counts:
            par_only = t > 1
            if par_only and not any_par:
                continue
            tag = "par_only" if par_only else "+seq"
            print(f"  [random] T={t} ({tag})", flush=True)
            env = os.environ.copy()
            env["PE_BENCH_FORMAT"] = "csv"
            env["PE_BENCH_TRIALS"] = str(trials)
            env["PE_BENCH_WARMUP"] = str(warmup)
            env["PARLAY_NUM_THREADS"] = str(t)
            env["PE_TRACE"] = "1"
            if first:
                env["PE_BENCH_HEADER"] = "1"
            if par_only:
                env["PE_BENCH_PAR_ONLY"] = "1"
            if algos is not None:
                env["PE_BENCH_ALGOS"] = ",".join(algos)
            stdout = _run_binary(binary, env=env,
                                 numactl_prefix=numactl_prefix,
                                 phase_label=f"random T={t}",
                                 trace_path=traces_dir / f"T{t:03d}.log")
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
    traces_dir = out_dir / "traces" / "synthetic"
    first = True
    any_par = _has_parallel_algo(algos)
    with open(csv_path, "w") as out:
        for fam in families:
            ns = override_ns if override_ns is not None else family_ns[fam]
            for n in ns:
                for t in thread_counts:
                    par_only = t > 1
                    if par_only and not any_par:
                        continue
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
                    env["PE_TRACE"] = "1"
                    if first:
                        env["PE_BENCH_HEADER"] = "1"
                    if par_only:
                        env["PE_BENCH_PAR_ONLY"] = "1"
                    if algos is not None:
                        env["PE_BENCH_ALGOS"] = ",".join(algos)
                    stdout = _run_binary(
                        binary, env=env, numactl_prefix=numactl_prefix,
                        phase_label=f"synthetic {fam} n={n} T={t}",
                        trace_path=traces_dir
                        / f"{fam}__n{n}__T{t:03d}.log")
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
    traces_dir = out_dir / "traces" / "cube_decomp"
    first = True
    any_par = _has_parallel_algo(algos)
    with open(csv_path, "w") as out:
        for d in CUBE_DECOMP_DS:
            for k in CUBE_DECOMP_KS:
                for t in thread_counts:
                    par_only = t > 1
                    if par_only and not any_par:
                        continue
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
                    env["PE_TRACE"] = "1"
                    if first:
                        env["PE_BENCH_HEADER"] = "1"
                    if par_only:
                        env["PE_BENCH_PAR_ONLY"] = "1"
                    if algos is not None:
                        env["PE_BENCH_ALGOS"] = ",".join(algos)
                    stdout = _run_binary(
                        binary, env=env, numactl_prefix=numactl_prefix,
                        phase_label=f"cube_decomp d={d} k={k} T={t}",
                        trace_path=traces_dir
                        / f"d{d}__k{k}__T{t:03d}.log")
                    out.write(stdout); out.flush()
                    first = False
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
    par_topo_iter, par_filter); --algos restricts via PE_BENCH_ALGOS.
    nelson_topo_iter is sequential — its time doesn't depend on
    PARLAY_NUM_THREADS, but we re-run it at every T anyway because
    the per-(file, threads) cell is the unit of measurement here and
    skipping it would create awkward gaps in the CSV. The duplicated
    measurements should be near-identical; downstream plotters average
    across T or pick the T=1 row.

    CSV schema is gates_bench's native:
      file,suite,n_gates,n_literals,n_not_terms,total_classes,
      algorithm,trial,parlay_threads,read_s,parse_s,build_s,close_ms.

    One subprocess per (file, threads) cell. Two reasons we don't batch
    all files into
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

    # Auto-init + auto-decompress only when pointed at the default
    # submodule checkout. Custom --gates-root values are left alone so
    # the driver never mutates a path the user supplied explicitly.
    if not files_glob and gates_root == GATES_DEFAULT_ROOT:
        # `git clone` of the parent repo creates an EMPTY placeholder
        # dir for the gitlink, so `os.path.isdir(gates_root)` would say
        # True even when the submodule was never fetched. Check that
        # the expected suite subdirs are present too — that's the real
        # "is this populated?" signal on a fresh PSC clone.
        populated = (os.path.isdir(gates_root) and
                     all(os.path.isdir(os.path.join(gates_root, s))
                         for s in suites))
        if not populated:
            print(f"Initializing submodule: {gates_root}", flush=True)
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive",
                 gates_root],
                check=True,
            )
        # The submodule ships .gates.xz; decompress per-suite (in place)
        # if any are still compressed. decompress.sh removes the .xz
        # after success, so subsequent runs are no-ops.
        decompress = os.path.join(gates_root, "decompress.sh")
        need_run = [s for s in suites
                    if os.path.isdir(os.path.join(gates_root, s))
                    and glob.glob(os.path.join(gates_root, s,
                                                "*.gates.xz"))]
        if need_run and os.path.isfile(decompress):
            print(f"Decompressing .gates.xz in: {need_run}", flush=True)
            subprocess.run(["bash", decompress, *need_run], check=True)

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

    # Per-cell PE_TRACE stderr capture, mirroring the other phases.
    traces_dir = out_dir / "traces" / "gates"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Algo names gates_bench currently understands.
    GATES_OK = ("nelson_simple", "nelson_simple_hash", "nelson_simple_arity",
                "nelson_simple_bump", "nelson_simple_arity_bump",
                "nelson_simple_inline", "nelson_topo_iter",
                "par_parents", "par_parents_gbk", "par_topo_iter",
                "par_filter", "par_filter_gbk", "par_filter_hybrid")

    first = True
    any_par = _has_parallel_algo(algos)
    with open(csv_path, "w") as out:
        for t in thread_counts:
            if t > 1 and not any_par:
                continue
            for fpath in files:
                fname = os.path.basename(fpath)
                print(f"  [gates] T={t} {fname}", flush=True)
                t0 = time.perf_counter()
                cmd = list(numactl_prefix) + [binary, fpath]
                env = os.environ.copy()
                env["PE_BENCH_TRIALS"] = str(trials)
                env["PE_BENCH_WARMUP"] = str(warmup)
                env["PARLAY_NUM_THREADS"] = str(t)
                env["PE_TRACE"] = "1"
                # Sequential algos at T>1 just duplicate the T=1 number
                # at higher noise; skip them. Mirrors run_random /
                # run_synthetic.
                if t > 1:
                    env["PE_BENCH_PAR_ONLY"] = "1"
                if first:
                    env["PE_BENCH_HEADER"] = "1"
                if algos is not None:
                    gates_algos = [a for a in algos if a in GATES_OK]
                    if not gates_algos:
                        print(f"    [gates] T={t} {fname} no algos "
                              f"recognized by gates_bench; skip",
                              flush=True)
                        continue
                    env["PE_BENCH_ALGOS"] = ",".join(gates_algos)
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      env=env)
                wall = time.perf_counter() - t0
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr)
                    raise RuntimeError(
                        f"gates_bench failed at T={t} on {fname}")
                # Forensic trace: stash stderr (carries [gates_bench]
                # sizing line + PE_TRACE per-round detail) per cell so a
                # killed driver doesn't lose it.
                stem = fname[:-len(".gates")] if fname.endswith(".gates") \
                    else fname
                trace_path = traces_dir / f"T{t:03d}__{stem}.log"
                with open(trace_path, "w") as tf:
                    tf.write(proc.stderr)
                # Also tee the bench's sizing line to the driver's stderr
                # so the operator sees progress.
                if proc.stderr:
                    sys.stderr.write(proc.stderr)
                out.write(proc.stdout); out.flush()
                first = False
                print(f"    done in {wall:.1f}s", flush=True)
    return csv_path


def _geomean(vs: list[float]) -> float | None:
    # statistics.geometric_mean raises on <=0 values; filter and return
    # None if nothing positive remains so callers can show "-".
    pos = [v for v in vs if v > 0]
    return statistics.geometric_mean(pos) if pos else None


def summarize_gates(csv_path: Path):
    """Custom summary for the gates phase: pairs par_topo_iter vs
    par_filter per (file, threads), prints both geomeans + ratio + each
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
    geomeans = {k: g for k, v in buckets.items()
                if (g := _geomean(v)) is not None}

    files = sorted({k[0] for k in geomeans})
    threads = sorted({k[1] for k in geomeans})

    print()
    print(f"{'file':<32} {'thr':>4} | {'topo_ms':>10} {'async_ms':>10}"
          f" | {'topo/async':>10} {'topo_T1/T':>10} {'async_T1/T':>11}"
          f" | {'gates':>10} {'classes':>10}")
    print("-" * 120)
    for f in files:
        gates, total = file_meta.get(f, (0, 0))
        topo_t1 = geomeans.get((f, threads[0], "par_topo_iter"))
        filter_t1 = geomeans.get((f, threads[0], "par_filter"))
        for t in threads:
            tm = geomeans.get((f, t, "par_topo_iter"))
            am = geomeans.get((f, t, "par_filter"))
            tm_s = f"{tm:>10.2f}" if tm is not None else f"{'-':>10}"
            am_s = f"{am:>10.2f}" if am is not None else f"{'-':>10}"
            ratio = (f"{tm/am:>9.2f}x"
                     if (tm is not None and am and am > 0) else f"{'-':>10}")
            t_spd = (f"{topo_t1/tm:>9.2f}x"
                     if (tm and topo_t1) else f"{'-':>10}")
            a_spd = (f"{filter_t1/am:>10.2f}x"
                     if (am and filter_t1) else f"{'-':>11}")
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
    geomeans = {k: g for k, v in buckets.items()
                if (g := _geomean(v)) is not None}

    workloads = sorted({wl for (wl, _, _) in geomeans})
    threads = sorted({t for (_, t, _) in geomeans})
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
          + f"{'simple/async':>12} {'topo_it/async':>13}")
    total_w = (len(fixed) + sum(col_w.values()) + len(ALGOS_OF_INTEREST) - 1
               + len(" | ") + 12 + 1 + 13)
    print("-" * total_w)

    # Sequential algos only run at T=1; cache that geomean per workload.
    seq_at_t1: dict[tuple, dict[str, float]] = {}
    for wl in workloads:
        seq_at_t1[wl] = {a: geomeans[(wl, 1, a)]
                         for a in SEQ_ALGOS if (wl, 1, a) in geomeans}

    for wl in workloads:
        for t in threads:
            cells = []
            vals: dict[str, float | None] = {}
            for a in ALGOS_OF_INTEREST:
                if a in SEQ_ALGOS:
                    # Display the T=1 value on every row for context.
                    v = seq_at_t1[wl].get(a)
                else:
                    v = geomeans.get((wl, t, a))
                vals[a] = v
                cells.append(f"{v:>{col_w[a]}.2f}" if v is not None
                             else f"{'-':>{col_w[a]}}")
            # Headline ratios:
            #   simple/async   = sequential baseline / par_filter (parallel
            #                    speedup over single-threaded simple). Prefer
            #                    nelson_simple_inline (the SigK+bump sound
            #                    sequential) when present; fall back to
            #                    nelson_simple (textbook) otherwise.
            #   topo_it/async  = par_topo_iter / par_filter (relative cost
            #                    of the depth-stratified parallel path).
            pa = vals.get("par_filter")
            ns = vals.get("nelson_simple_inline") or vals.get("nelson_simple")
            pt = vals.get("par_topo_iter")
            ratio_simple_async = (f"{ns/pa:>11.2f}x"
                                  if (ns and pa and pa > 0)
                                  else f"{'-':>12}")
            ratio_topo_async = (f"{pt/pa:>12.2f}x"
                                if (pt and pa and pa > 0)
                                else f"{'-':>13}")
            wl_str = "/".join(str(x) for x in wl)
            print(f"{wl_str:<20} {t:>4} | "
                  + " ".join(cells)
                  + f" | {ratio_simple_async} {ratio_topo_async}")
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
                    choices=["random", "synthetic", "cube_decomp", "gates"],
                    help="skip a phase; repeatable")
    ap.add_argument("--algos",
                    default="nelson_simple_inline,par_parents,par_filter",
                    help="comma-separated algorithm whitelist. Default: "
                         "nelson_simple_inline,par_parents,par_filter "
                         "(the three algorithms exercised by the "
                         "FMCAD artifact). Sets PE_BENCH_ALGOS for the "
                         "C++ benchmark binaries.")
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
    ap.add_argument("--shutdown-after", action="store_true",
                    help="run `shutdown -h now` after all phases complete "
                         "(for unattended overnight runs)")
    args = ap.parse_args()

    try:
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

        valid_algos = {"nelson_simple_inline", "par_parents", "par_filter"}
        algos: list[str] | None = None
        if args.algos:
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
        print(f"Phases:        {[p for p in ('random','synthetic','cube_decomp','gates') if p not in skip]}")
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
    finally:
        if args.shutdown_after:
            print("--shutdown-after set; running `shutdown -h now`", flush=True)
            rc = subprocess.run(["sudo", "shutdown", "-h", "now"]).returncode
            if rc != 0:
                sys.exit(f"shutdown failed with exit code {rc}")


if __name__ == "__main__":
    main()
