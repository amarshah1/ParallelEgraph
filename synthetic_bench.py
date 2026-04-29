#!/usr/bin/env python3
"""Drive bench/synthetic_bench across families, n values, and thread counts.

In-process port of the gen_bench.py families: builds DAGs directly into an
EGraph via bulk_init and times only sequential_close_nelson vs
parallel_close — no parse / build / dtor overhead. Use this when you want
clean closure-only numbers; use bench.py against synthetic_benchmarks/
when you want full end-to-end solver time.

Default n ranges focus on the top third of each family — small
configs (where closure runs in microseconds and parallel can't beat
sequential overhead) are dropped:
    chain   5600 .. 7100 step 500
    grid    160  .. 200  step 5
    cube    110  .. 145  step 5
    quartic 38   .. 53   step 3
    quintic 18   .. 23   step 1

Usage:
    python synthetic_bench.py
    python synthetic_bench.py --threads-sweep 1,2,4,8 --csv synth.csv
    python synthetic_bench.py --families grid,cube
    python synthetic_bench.py --families quintic --ns 10,12,14
    python synthetic_bench.py --skip-nelson         # parallel only
"""

import argparse
import csv as csvmod
import datetime
import io
import os
import shutil
import statistics
import subprocess
import sys


FAMILY_NS = {
    "chain":   list(range(5600, 7101, 500)),   # 5600, 6100, 6600, 7100
    "grid":    list(range(160, 201, 5)),        # 160, 165, ..., 200
    "cube":    list(range(110, 146, 5)),        # 110, 115, ..., 145
    "quartic": list(range(38, 54, 3)),          # 38, 41, ..., 53
    "quintic": list(range(18, 24)),             # 18, 19, ..., 23
}

ALL_FAMILIES = list(FAMILY_NS.keys())


def build_synthetic_bench(release: bool) -> str:
    build_dir = "build"
    build_type = "Release" if release else "Debug"
    # Always run cmake configure: idempotent on an already-configured
    # build dir (just regenerates Makefiles), and picks up any
    # CMakeLists.txt edits since the last invocation.
    print(f"Configuring CMake ({build_type})...", flush=True)
    subprocess.run(
        ["cmake", "-B", build_dir, "-S", ".",
         f"-DCMAKE_BUILD_TYPE={build_type}"],
        check=True, capture_output=True,
    )
    print("Building synthetic_bench...", flush=True)
    subprocess.run(
        ["cmake", "--build", build_dir, "-j", "--target", "synthetic_bench"],
        check=True, capture_output=True,
    )
    return os.path.join(build_dir, "synthetic_bench")


def run_one_invocation(
    binary: str,
    families: list[str],
    ns: list[int],
    threads: int | None,
    skip_nelson: bool,
    emit_header: bool,
    numactl_prefix: list[str] | None,
) -> str:
    """Run synthetic_bench once with the given config; return its CSV stdout.

    The bench iterates internally over (family, n) and emits one row per
    (family, n, algorithm, trial). We do NOT pre-multiply across thread
    counts here — caller wraps this in a thread-count loop and passes the
    appropriate PARLAY_NUM_THREADS each time.
    """
    cmd = []
    if numactl_prefix:
        cmd.extend(numactl_prefix)
    cmd.append(binary)

    env = os.environ.copy()
    env["PE_SYNTH_FAMILIES"] = ",".join(families)
    env["PE_SYNTH_NS"] = ",".join(str(n) for n in ns)
    env["PE_BENCH_FORMAT"] = "csv"
    if emit_header:
        env["PE_BENCH_HEADER"] = "1"
    if skip_nelson:
        env["PE_BENCH_SKIP_NELSON"] = "1"
    if threads is not None:
        env["PARLAY_NUM_THREADS"] = str(threads)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"synthetic_bench failed (exit {proc.returncode})")
    # Mirror per-config progress lines from the child to stderr so callers
    # see something happening during long sweeps.
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    return proc.stdout


def print_summary(csv_text: str, has_header: bool):
    """Parse synthetic_bench's CSV stdout and print a per-config summary.

    One line per (family, n, algorithm) with the median wallclock_ms across
    trials. When both nelson_seq and par_close are present we also print
    the speedup. Skipped when the bench produced no rows.
    """
    reader = csvmod.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return
    if has_header:
        rows = rows[1:]
    # Group: (family, n, threads, algorithm) -> [ms, ms, ...]
    grouped: dict[tuple[str, int, str, str], list[float]] = {}
    classes: dict[tuple[str, int], int] = {}
    merges: dict[tuple[str, int], int] = {}
    for r in rows:
        if len(r) < 10:
            continue
        family, n_str, classes_str, merges_str, algorithm, _trial, \
            threads_str, _union, _dnc, ms_str = r[:10]
        try:
            n = int(n_str)
            ms = float(ms_str)
        except ValueError:
            continue
        key = (family, n, threads_str, algorithm)
        grouped.setdefault(key, []).append(ms)
        classes[(family, n)] = int(classes_str)
        merges[(family, n)] = int(merges_str)

    # Index medians by (family, n, threads) so we can pair algorithms.
    medians: dict[tuple[str, int, str], dict[str, float]] = {}
    for (family, n, threads_str, algo), times in grouped.items():
        medians.setdefault((family, n, threads_str), {})[algo] = \
            statistics.median(times)

    # nelson_seq is sequential and thread-count-independent — we only run
    # it on the first thread count in a sweep. Carry that single median
    # across all thread rows in the summary so every par_close row shows
    # a real speedup instead of '-'.
    nelson_per_workload: dict[tuple[str, int], float] = {}
    for key, algos in medians.items():
        if "nelson_seq" in algos:
            nelson_per_workload[(key[0], key[1])] = algos["nelson_seq"]

    # Header
    print(f"  {'family':<8} {'n':>5} {'thr':>4} {'classes':>9} {'merges':>7} "
          f"| {'nelson_ms':>10} {'parclose_ms':>11} {'speedup':>8}")
    # Sort by (family, n, numeric_threads) so the table reads ascending in
    # thread count instead of lexicographically (which would put "10"
    # before "2").
    def sort_key(k: tuple[str, int, str]) -> tuple[str, int, int]:
        try:
            t_int = int(k[2])
        except ValueError:
            t_int = -1   # "None"/"default" sorts before any real count
        return (k[0], k[1], t_int)
    for key in sorted(medians.keys(), key=sort_key):
        family, n, threads_str = key
        algos = medians[key]
        nel = algos.get("nelson_seq") or nelson_per_workload.get((family, n))
        par = algos.get("par_close")
        nel_s = f"{nel:>10.3f}" if nel is not None else f"{'-':>10}"
        par_s = f"{par:>11.3f}" if par is not None else f"{'-':>11}"
        if nel is not None and par is not None and par > 0:
            spd = f"{nel/par:>7.2f}x"
        else:
            spd = f"{'-':>8}"
        c = classes.get((family, n), 0)
        m = merges.get((family, n), 0)
        print(f"  {family:<8} {n:>5} {threads_str:>4} {c:>9} {m:>7} "
              f"| {nel_s} {par_s} {spd}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", default=None,
                        help="comma-separated subset of "
                             f"{','.join(ALL_FAMILIES)} (default: all)")
    parser.add_argument("--ns", default=None,
                        help="comma-separated n values; overrides per-family "
                             "defaults (applied to every selected family)")
    parser.add_argument("--threads", type=int, default=None,
                        help="PARLAY_NUM_THREADS for a single run")
    parser.add_argument("--threads-sweep", default=None,
                        help="comma-separated thread counts (e.g. '1,2,4,8')")
    parser.add_argument("--skip-nelson", action="store_true",
                        help="skip the sequential baseline; parallel only")
    parser.add_argument("--debug", action="store_true",
                        help="build the bench in Debug mode")
    parser.add_argument("--csv", dest="csv_file", default=None,
                        help="append CSV rows here (default: timestamped file)")
    parser.add_argument("--no-numactl", action="store_true",
                        help="don't prepend `numactl -i all` even when "
                             "available (default: use it on Linux when present)")
    args = parser.parse_args()

    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        unknown = [f for f in families if f not in FAMILY_NS]
        if unknown:
            print(f"unknown families: {','.join(unknown)}. "
                  f"valid: {','.join(ALL_FAMILIES)}", file=sys.stderr)
            sys.exit(2)
    else:
        families = list(ALL_FAMILIES)

    if args.ns:
        try:
            override_ns = [int(x) for x in args.ns.split(",") if x.strip()]
        except ValueError:
            print("--ns expects comma-separated integers", file=sys.stderr)
            sys.exit(2)
        if any(n < 1 for n in override_ns):
            print("--ns values must be >= 1", file=sys.stderr)
            sys.exit(2)
    else:
        override_ns = None

    if args.threads_sweep:
        try:
            thread_counts: list[int | None] = sorted(
                int(t) for t in args.threads_sweep.split(",") if t
            )
        except ValueError:
            print("--threads-sweep expects comma-separated integers",
                  file=sys.stderr)
            sys.exit(2)
    else:
        thread_counts = [args.threads]   # may be [None]

    numactl_prefix: list[str] | None = None
    if not args.no_numactl:
        which = shutil.which("numactl")
        if which:
            numactl_prefix = [which, "-i", "all"]

    binary = build_synthetic_bench(release=not args.debug)

    if args.csv_file is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.csv_file = f"synthetic_bench_{ts}.csv"

    # synthetic_bench's CSV columns:
    #   family,n,classes,merges,algorithm,trial,parlay_threads,
    #   union_style,dnc_cutoff,wallclock_ms
    # We append every invocation's rows to the same file. Header is only
    # written when the file doesn't already exist.
    write_header = not os.path.exists(args.csv_file)

    print(f"Synthetic bench → {args.csv_file}")
    print(f"  families: {','.join(families)}")
    print(f"  thread_counts: {thread_counts}")
    if override_ns is not None:
        print(f"  ns (override): {override_ns}")
    if numactl_prefix:
        print(f"  numactl: {' '.join(numactl_prefix[1:])}")
    print(flush=True)

    # Loop order: (family, n) outermost, thread counts innermost. For each
    # workload we run every thread-count back-to-back, then move on. This
    # keeps a workload's data warm in cache across the thread sweep and
    # lets the per-workload summary print as soon as that workload's
    # full thread curve is in.
    #
    # nelson_seq is sequential and thread-count-independent, so we only
    # run it on the first thread count of each workload. Subsequent thread
    # counts skip nelson (par_close only) — saves ~half the work on a
    # multi-thread sweep.
    total_rows = 0
    with open(args.csv_file, "a", buffering=1) as out:
        first_invocation = write_header
        for fam in families:
            ns = override_ns if override_ns is not None else FAMILY_NS[fam]
            for n in ns:
                workload_csv_parts: list[str] = []
                for tidx, t in enumerate(thread_counts):
                    skip_nelson_for_call = args.skip_nelson or tidx > 0
                    thread_label = "default" if t is None else str(t)
                    nel_tag = "" if skip_nelson_for_call else " (+nelson)"
                    print(f"[{fam} n={n}] threads={thread_label}{nel_tag}",
                          flush=True)
                    csv_text = run_one_invocation(
                        binary=binary,
                        families=[fam],
                        ns=[n],
                        threads=t,
                        skip_nelson=skip_nelson_for_call,
                        emit_header=first_invocation,
                        numactl_prefix=numactl_prefix,
                    )
                    out.write(csv_text)
                    total_rows += csv_text.count("\n")
                    # Strip the header (if any) for the merged summary —
                    # we'll print one summary block per workload covering
                    # all thread counts at once.
                    if first_invocation:
                        lines = csv_text.split("\n", 1)
                        body = lines[1] if len(lines) > 1 else ""
                    else:
                        body = csv_text
                    workload_csv_parts.append(body)
                    first_invocation = False
                # One summary table per workload covering every thread count.
                print_summary("".join(workload_csv_parts), has_header=False)

    print()
    print(f"wrote {total_rows} rows to {args.csv_file}")


if __name__ == "__main__":
    main()
