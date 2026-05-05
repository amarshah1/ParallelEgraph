#!/usr/bin/env python3
"""Compare BSP (parallel_close) vs async-rounds (parallel_close_async_rounds)
on the synthetic benchmarks via ./build/synthetic_bench.

For each (family, n, threads) we run both algorithms — BSP first, then
async — and write per-trial CSV rows into a single file. The CSV uses
synthetic_bench's native schema, with the `algorithm` column carrying
either `par_close` (BSP) or `par_close_async` (mark-based async). At
the end we print a summary table showing the median ms per
(family, n, threads, algorithm) and the async-vs-bsp ratio.

Usage:
    python3 compare_async_vs_bsp.py
    python3 compare_async_vs_bsp.py --threads-sweep 1,2,4,8
    python3 compare_async_vs_bsp.py --families cube,quintic --ns 10,20
    python3 compare_async_vs_bsp.py --csv my_compare.csv

Output goes to runs/compare_async_<ts>/ unless --out is specified.
"""

import argparse
import csv as csvmod
import datetime
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


# Per-round trace lines look like:
#   [pe] round=  0 work=...
#   [pe-async] round=  3 R=4 dirty=...
# We just count occurrences to report round count per invocation. The
# per-round detail itself stays in stderr / trace logs.
ROUND_RE = re.compile(r"^\[pe(?:-async)?\] round=\s*(\d+)")


# Default sweep — cube/quintic at sizes that have enough work to expose
# scaling differences without taking forever. Tune via flags as needed.
DEFAULT_FAMILY_NS = {
    "chain":   [3000, 5000, 7000],
    "grid":    [120, 150, 175],
    "cube":    [80, 110, 140],
    "quartic": [25, 35, 45],
    "quintic": [12, 16, 20],
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


def run_one(family: str, n: int, threads: int, *, use_async: bool,
            warmup: int, trials: int, emit_header: bool,
            numactl_prefix: list[str],
            trace_log_dir: Path | None = None) -> tuple[str, int]:
    """Returns (stdout_csv, round_count_in_last_trial).

    PE_TRACE=1 is set so we can count rounds. The whole stderr (including
    [pe]/[pe-async] lines for every trial) is captured. The last trial's
    round count is what we report — earlier trials' counts should be
    identical for synthetic workloads (deterministic).
    """
    cmd = list(numactl_prefix) + ["./build/synthetic_bench"]
    env = os.environ.copy()
    env["PE_SYNTH_FAMILIES"] = family
    env["PE_SYNTH_NS"] = str(n)
    env["PE_BENCH_FORMAT"] = "csv"
    env["PE_BENCH_SKIP_NELSON"] = "1"
    env["PE_BENCH_TRIALS"] = str(trials)
    env["PE_BENCH_WARMUP"] = str(warmup)
    env["PARLAY_NUM_THREADS"] = str(threads)
    env["PE_TRACE"] = "1"
    if emit_header:
        env["PE_BENCH_HEADER"] = "1"
    if use_async:
        env["PE_USE_ASYNC"] = "1"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"synthetic_bench failed: family={family} n={n} T={threads} "
            f"async={use_async}")

    # Optional: dump the raw trace for debugging.
    if trace_log_dir is not None:
        algo = "async" if use_async else "bsp"
        (trace_log_dir / f"{family}_n{n}_T{threads}_{algo}.log").write_text(
            proc.stderr)

    # Count rounds in the LAST trial. Each trial's first round line is
    # round=0; we find the largest round index after the last reset.
    max_round = 0
    last_seen = -1
    for line in proc.stderr.splitlines():
        m = ROUND_RE.match(line)
        if m is None:
            continue
        r = int(m.group(1))
        if r < last_seen:
            # Reset: a new trial started. Reset our running max so we
            # report the LAST trial's round count.
            max_round = r
        else:
            max_round = r
        last_seen = r
    # max_round is the highest round=N index seen in the last trial.
    # That's "last round number"; total round count is max_round + 1
    # (including round=0). The break round (frontier=0) doesn't get a
    # round= line of its own in the BSP path's trace — actually it does
    # (the (break) line is still tagged round=N). So +1 captures it.
    round_count = max_round + 1 if last_seen >= 0 else 0
    return proc.stdout, round_count


def summarize(csv_path: Path):
    rows = list(csvmod.DictReader(open(csv_path)))
    # group: (family, n, threads, algorithm) -> list[ms]
    buckets: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["family"], int(r["n"]), int(r["parlay_threads"]),
               r["algorithm"])
        buckets.setdefault(key, []).append(float(r["wallclock_ms"]))
    medians = {k: statistics.median(v) for k, v in buckets.items() if v}

    workloads = sorted({(f, n) for (f, n, _, _) in medians})
    threads = sorted({t for (_, _, t, _) in medians})

    print()
    print(f"{'family':<8} {'n':>5} {'thr':>4} | {'bsp_ms':>10} {'async_ms':>10} "
          f"| {'async/bsp':>9} {'bsp_T1/T':>9} {'async_T1/T':>11}")
    print("-" * 88)
    for (fam, n) in workloads:
        bsp_t1 = medians.get((fam, n, threads[0], "par_close"))
        async_t1 = medians.get((fam, n, threads[0], "par_close_async"))
        for t in threads:
            bm = medians.get((fam, n, t, "par_close"))
            am = medians.get((fam, n, t, "par_close_async"))
            bm_s = f"{bm:>10.2f}" if bm is not None else f"{'-':>10}"
            am_s = f"{am:>10.2f}" if am is not None else f"{'-':>10}"
            ratio = f"{am/bm:>8.2f}x" if (bm and am and bm > 0) else f"{'-':>9}"
            bsp_spd = f"{bsp_t1/bm:>7.2f}x" if (bm and bsp_t1) else f"{'-':>9}"
            async_spd = (f"{async_t1/am:>9.2f}x"
                         if (am and async_t1) else f"{'-':>11}")
            print(f"{fam:<8} {n:>5} {t:>4} | {bm_s} {am_s} "
                  f"| {ratio} {bsp_spd} {async_spd}")
        print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8,
                    help="thread count for the comparison (default: 8). "
                         "We pin to a single thread count rather than "
                         "sweeping; the comparison is BSP vs async at "
                         "this fixed core count.")
    ap.add_argument("--families", default=None,
                    help=f"comma-separated subset of "
                         f"{','.join(DEFAULT_FAMILY_NS)} "
                         f"(default: all)")
    ap.add_argument("--ns", default=None,
                    help="comma-separated n values; applied to every "
                         "selected family (default: per-family ranges in "
                         "DEFAULT_FAMILY_NS)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="warmup runs per (family, n, algorithm) "
                         "(default 0)")
    ap.add_argument("--trials", type=int, default=1,
                    help="measured trials per (family, n, algorithm) "
                         "(default 1)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: runs/compare_async_<ts>/)")
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default: <out>/compare.csv)")
    ap.add_argument("--no-numactl", action="store_true",
                    help="don't prepend `numactl -i all` even if available")
    args = ap.parse_args()

    threads = args.threads

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
    out_dir = Path(args.out) if args.out else Path("runs") / f"compare_async_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else out_dir / "compare.csv"

    trace_dir = out_dir / "traces"
    trace_dir.mkdir(exist_ok=True)

    print(f"Output: {out_dir}")
    print(f"  csv:           {csv_path}")
    print(f"  trace logs:    {trace_dir}")
    print(f"  families:      {families}")
    print(f"  threads:       {threads}")
    print(f"  warmup:        {args.warmup}")
    print(f"  trials:        {args.trials}")
    print(f"  numactl:       {' '.join(numactl_prefix) if numactl_prefix else 'none'}")
    print()

    # Live header: one row per (family, n) showing both algorithms.
    print(f"{'family':<8} {'n':>5} | "
          f"{'bsp_ms':>10} {'bsp_rds':>7} | "
          f"{'async_ms':>10} {'async_rds':>9} | "
          f"{'async/bsp':>9}")
    print("-" * 78)

    def parse_wallclocks(csv_text: str, want_header: bool) -> list[float]:
        """Pull the trial-row wallclock_ms values out of synthetic_bench's
        CSV stdout, ignoring a possibly-emitted header row."""
        out: list[float] = []
        for ln in csv_text.splitlines():
            if not ln.strip():
                continue
            if ln.startswith("family,"):
                continue
            parts = ln.split(",")
            try:
                out.append(float(parts[-1]))
            except ValueError:
                continue
        return out

    first = True
    with open(csv_path, "w") as csv_out:
        for fam in families:
            ns = override_ns if override_ns is not None else DEFAULT_FAMILY_NS[fam]
            for n in ns:
                # Run BSP, then async, on the same workload.
                bsp_text, bsp_rounds = run_one(
                    fam, n, threads,
                    use_async=False,
                    warmup=args.warmup, trials=args.trials,
                    emit_header=first,
                    numactl_prefix=numactl_prefix,
                    trace_log_dir=trace_dir)
                csv_out.write(bsp_text); csv_out.flush()
                first = False

                async_text, async_rounds = run_one(
                    fam, n, threads,
                    use_async=True,
                    warmup=args.warmup, trials=args.trials,
                    emit_header=False,
                    numactl_prefix=numactl_prefix,
                    trace_log_dir=trace_dir)
                csv_out.write(async_text); csv_out.flush()

                bsp_ms = statistics.median(parse_wallclocks(bsp_text, first))
                async_ms = statistics.median(
                    parse_wallclocks(async_text, False))
                ratio = async_ms / bsp_ms if bsp_ms > 0 else float("nan")
                print(f"{fam:<8} {n:>5} | "
                      f"{bsp_ms:>10.2f} {bsp_rounds:>7} | "
                      f"{async_ms:>10.2f} {async_rounds:>9} | "
                      f"{ratio:>8.2f}x", flush=True)

    print()
    print(f"Done. CSV: {csv_path}")
    print(f"      Per-config trace logs: {trace_dir}")


if __name__ == "__main__":
    main()
