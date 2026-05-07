#!/usr/bin/env python3
"""Mini evaluation: par_topo_iter vs par_close_async on the synthetic
suite. Both algorithms are rounds-based; topo_iter resolves the
soundness issues that plagued earlier topo variants.

Pinned set of families/sizes by default; tweak via flags. Output is
synthetic_bench's native CSV plus a final summary table grouping by
(family, n, threads, algorithm).

Usage:
    python3 compare_topo_vs_async.py
    python3 compare_topo_vs_async.py --threads-sweep 1,4,8 --families cube
    python3 compare_topo_vs_async.py --warmup 1 --trials 5
"""

import argparse
import csv as csvmod
import datetime
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

# Mini sweep: enough to surface the topo_iter vs async difference,
# without burning hours. cube is the workload where async lost to BSP
# in earlier runs, so it's the most discriminating. mixed_depth is
# cube + extra cross-depth initial unions — designed to stress
# par_topo_iter (whose convergence relies on a topo order of inputs).
DEFAULT_FAMILY_NS = {
    "chain":       [3000, 6000],
    "grid":        [120, 175],
    "cube":        [80, 120, 145],
    "quartic":     [25, 45],
    "quintic":     [12, 18],
    "mixed_depth": [30, 50, 80, 110, 145],
}

# Algorithms we report in the summary, in column order. Mapping to
# synthetic_bench's CSV `algorithm` tags:
#   nelson_seq        — Nelson's original sequential CC
#   nelson_topo       — sequential topo (sometimes unsound on
#                        cross-depth initial unions; kept as historical
#                        baseline)
#   nelson_topo_iter  — sequential topo_iter (sound, rounds-based)
#   par_close         — BSP parallel CC (parents_-frontier)
#   par_topo_iter     — parallel topo_iter (rounds-based, sound)
#   par_async         — async-rounds CC (mark-based dirty filter)
ALGOS_OF_INTEREST = [
    "nelson_seq",
    "nelson_topo",
    "nelson_topo_iter",
    "par_close",
    "par_topo_iter",
    "par_async",
]
# Short header labels for the summary table.
ALGO_HEADERS = {
    "nelson_seq":       "nelson",
    "nelson_topo":      "nl_topo",
    "nelson_topo_iter": "nl_topo_it",
    "par_close":        "par_close",
    "par_topo_iter":    "par_topo_it",
    "par_async":        "par_async",
}


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


def run_one(family: str, n: int, threads: int, *,
            warmup: int, trials: int, emit_header: bool,
            numactl_prefix: list[str]) -> str:
    """One synthetic_bench invocation. Emits CSV rows for all 8
    algorithms (nelson_seq, nelson_topo, nelson_topo_iter, nelson_dst,
    par_close, par_topo_iter, par_async, par_async_min_id). Caller
    filters down to ALGOS_OF_INTEREST in summarize().
    """
    cmd = list(numactl_prefix) + ["./build/synthetic_bench"]
    env = os.environ.copy()
    env["PE_SYNTH_FAMILIES"] = family
    env["PE_SYNTH_NS"] = str(n)
    env["PE_BENCH_FORMAT"] = "csv"
    env["PE_BENCH_TRIALS"] = str(trials)
    env["PE_BENCH_WARMUP"] = str(warmup)
    env["PARLAY_NUM_THREADS"] = str(threads)
    if emit_header:
        env["PE_BENCH_HEADER"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"synthetic_bench failed: family={family} n={n} T={threads}")
    return proc.stdout


def summarize(csv_path: Path):
    rows = list(csvmod.DictReader(open(csv_path)))
    # group: (family, n, threads, algorithm) -> list[ms]
    buckets: dict[tuple, list[float]] = {}
    for r in rows:
        algo = r["algorithm"]
        if algo not in ALGOS_OF_INTEREST:
            continue
        key = (r["family"], int(r["n"]), int(r["parlay_threads"]), algo)
        buckets.setdefault(key, []).append(float(r["wallclock_ms"]))
    medians = {k: statistics.median(v) for k, v in buckets.items() if v}

    workloads = sorted({(f, n) for (f, n, _, _) in medians})
    threads = sorted({t for (_, _, t, _) in medians})

    # Column widths for each algorithm — wide enough for the header
    # plus reasonable ms values (4 digits + decimals).
    col_w = {a: max(len(ALGO_HEADERS[a]), 10) for a in ALGOS_OF_INTEREST}

    # Header
    header_cells = [f"{ALGO_HEADERS[a]:>{col_w[a]}}" for a in ALGOS_OF_INTEREST]
    fixed = f"{'family':<8} {'n':>5} {'thr':>4} | "
    print()
    print(fixed + " ".join(header_cells)
          + " | "
          + f"{'topo/async':>10} {'topo_it_spd':>12}")
    total_w = (len(fixed) + sum(col_w.values()) + len(ALGOS_OF_INTEREST) - 1
               + len(" | ") + 10 + 1 + 12)
    print("-" * total_w)

    for (fam, n) in workloads:
        for t in threads:
            cells = []
            vals: dict[str, float | None] = {}
            for a in ALGOS_OF_INTEREST:
                v = medians.get((fam, n, t, a))
                vals[a] = v
                cells.append(f"{v:>{col_w[a]}.2f}" if v is not None
                             else f"{'-':>{col_w[a]}}")
            pt = vals.get("par_topo_iter")
            pa = vals.get("par_async")
            ni = vals.get("nelson_topo_iter")
            # par_topo_iter / par_async — < 1 means topo is faster.
            ratio_topo_async = (f"{pt/pa:>9.2f}x"
                                if (pt and pa and pa > 0) else f"{'-':>10}")
            # nelson_topo_iter / par_topo_iter — speedup of par_topo_iter
            # over its sequential baseline.
            topo_it_spd = (f"{ni/pt:>11.2f}x"
                           if (ni and pt and pt > 0) else f"{'-':>12}")
            print(f"{fam:<8} {n:>5} {t:>4} | "
                  + " ".join(cells)
                  + f" | {ratio_topo_async} {topo_it_spd}")
        print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads-sweep", default="1,4,8",
                    help="comma-separated thread counts (default: 1,4,8)")
    ap.add_argument("--families", default=None,
                    help=f"comma-separated subset of "
                         f"{','.join(DEFAULT_FAMILY_NS)} (default: all)")
    ap.add_argument("--ns", default=None,
                    help="comma-separated n values; applied to every "
                         "selected family (default: per-family ranges)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup runs (default 1)")
    ap.add_argument("--trials", type=int, default=3,
                    help="measured trials (default 3)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: runs/topo_vs_async_<ts>/)")
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default: <out>/compare.csv)")
    ap.add_argument("--no-numactl", action="store_true",
                    help="don't prepend `numactl -i all`")
    args = ap.parse_args()

    try:
        thread_counts = sorted(set(int(t) for t in args.threads_sweep.split(",") if t))
    except ValueError:
        sys.exit("--threads-sweep expects comma-separated integers")

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

    cmake_build()

    numactl_prefix: list[str] = []
    if not args.no_numactl:
        which = shutil.which("numactl")
        if which:
            numactl_prefix = [which, "-i", "all"]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs") / f"topo_vs_async_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_dir / "compare.csv"

    print(f"Output: {out_dir}")
    print(f"  csv:           {csv_path}")
    print(f"  families:      {families}")
    print(f"  threads:       {thread_counts}")
    print(f"  warmup:        {args.warmup}")
    print(f"  trials:        {args.trials}")
    print(f"  algos shown:   {ALGOS_OF_INTEREST}")
    print(f"  numactl:       {' '.join(numactl_prefix) if numactl_prefix else 'none'}")
    print()

    first = True
    with open(csv_path, "w") as out:
        for fam in families:
            ns = override_ns if override_ns is not None else DEFAULT_FAMILY_NS[fam]
            for n in ns:
                for t in thread_counts:
                    print(f"  [{fam} n={n} T={t}]", flush=True)
                    text = run_one(fam, n, t,
                                   warmup=args.warmup, trials=args.trials,
                                   emit_header=first,
                                   numactl_prefix=numactl_prefix)
                    out.write(text)
                    out.flush()
                    first = False

    summarize(csv_path)
    print()
    print(f"Done. CSV: {csv_path}")


if __name__ == "__main__":
    main()
