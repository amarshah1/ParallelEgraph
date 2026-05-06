#!/usr/bin/env bash
# Closure benchmark sweep for the 144-core machine. Times the full set of
# closure algorithms across closure_compare, synthetic, and eggcc:
#
#   nelson_topo*       single-pass forward topo  (UNSOUND on cross-depth inits — kept for reference)
#   nelson_topo_iter   topo iterated to fixpoint (sound; ~2 passes on synthetic)
#   nelson_dst         worklist + smaller-into-larger hashcons (sound; correct on arbitrary inputs)
#   par_close          parallel BSP with parents_ frontier (sound)
#   par_async          parallel async-rounds with last_marked_ stamps (sound)
#
# The Nelson baseline (sequential_close_nelson) is skipped via
# PE_BENCH_SKIP_NELSON=1. Re-running overwrites prior results.
#
# Outputs (under bench/results/topo144/):
#   build_info.txt              compiler/git state captured at run time
#   sanity.log                  closure_test pass/fail
#   closure_compare_144T.csv    headline closure_compare at T_max
#   synthetic_144T.csv          headline synthetic_bench at T_max
#   closure_scaling.csv         closure_compare across T = 1..T_max
#   quintic20_scaling.csv       synthetic quintic-20 (12.8M classes) across T
#   eggcc_144T.csv              full 507-file cc-benchmarks/smt-grounded sweep at T_max
#   eggcc_summary_144T.txt      aggregate ratios + win rates by class-bucket
#
# Override T_MAX (default 144) for other thread budgets; the scaling
# sweeps clamp at T_MAX. e.g. T_MAX=192 ./bench/scripts/run_144core_topo.sh
#
# Total wall time ~30-60 min depending on memory bandwidth and disk speed.
# The eggcc sweep dominates (~10-20 min); the scaling sweeps are ~5 min each.

set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=bench/results/topo144
mkdir -p "$OUT"

# Bench drivers honor PE_BENCH_SKIP_NELSON to drop the Nelson baseline
# from the table — leaving only nelson_topo and par_topo timed.
export PE_BENCH_SKIP_NELSON=1

# ------------------- Pre-flight -------------------
# Multi-socket NUMA: interleave allocations across all nodes so atomic
# UF operations and the parlay scheduler's per-thread allocator don't
# pile traffic on one node's memory controller. Override with NUMACTL=
# (empty) to disable, or set NUMACTL='numactl --cpunodebind=0' to pin.
NUMACTL=${NUMACTL:-numactl -i all}
if [ -n "$NUMACTL" ]; then
  command -v numactl >/dev/null 2>&1 || {
    echo "numactl not found. Install with: sudo apt-get install numactl" >&2
    echo "(Or run with NUMACTL= to disable: NUMACTL= ./bench/scripts/run_144core_topo.sh)" >&2
    exit 1
  }
fi

# Always run an incremental build. `cmake --build` is a no-op if every
# target is up-to-date, but catches the case where the user just pulled
# and the on-disk binaries are stale relative to the source — that's
# the easy way to end up running last week's algorithm by accident.
# Configure (the slower step) is skipped when build/ already exists.
TARGETS="closure_compare_bench synthetic_bench smt_bench closure_test"
echo "[build] ensuring bench binaries are up to date..."
[ -d build/CMakeFiles ] || ( mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. )
cmake --build build --target $TARGETS -j

if [ ! -d cc-benchmarks/smt-grounded ] || [ -z "$(ls cc-benchmarks/smt-grounded/*.smt2 2>/dev/null)" ]; then
  echo "cc-benchmarks submodule not populated. Run:" >&2
  echo "  git submodule update --init --recursive cc-benchmarks" >&2
  exit 1
fi

# Capture environment
{
  echo "=== Topo bench run ==="
  date -u
  echo
  echo "git:"; git rev-parse HEAD; git status --short
  echo
  echo "cpu:"
  if [ -r /proc/cpuinfo ]; then
    grep -E 'model name|cpu cores|siblings' /proc/cpuinfo | sort -u
    echo "logical: $(nproc)"
  else
    sysctl -n machdep.cpu.brand_string hw.physicalcpu hw.logicalcpu 2>/dev/null
  fi
  echo
  echo "numa:"; numactl --hardware 2>&1 | head -20 || true
  echo
  echo "numactl prefix: $NUMACTL"
  echo
  echo "compiler:"; cmake --version | head -1
} > "$OUT/build_info.txt" 2>&1

# Sanity check (no NUMACTL — closure_test is sequential and tiny)
echo "[1/6] correctness tests"
./build/closure_test > "$OUT/sanity.log" 2>&1
grep -E "FAIL|tests passed" "$OUT/sanity.log"

T_MAX=${T_MAX:-144}
echo "T_max=$T_MAX"

# ------------------- Headline at T_max -------------------
echo "[2/6] closure_compare headline at T=$T_MAX"
PARLAY_NUM_THREADS=$T_MAX \
  PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  $NUMACTL ./build/closure_compare_bench > "$OUT/closure_compare_144T.csv"

echo "[3/6] synthetic_bench headline at T=$T_MAX"
PARLAY_NUM_THREADS=$T_MAX \
  PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  $NUMACTL ./build/synthetic_bench > "$OUT/synthetic_144T.csv"

# ------------------- Strong scaling: closure_compare across T -------------------
echo "[4/6] closure_compare strong scaling across T"
: > "$OUT/closure_scaling.csv"
HDR=1
for T in 1 2 4 8 16 32 48 64 96 128 144 192; do
  [ $T -gt $T_MAX ] && continue
  echo "  T=$T"
  PARLAY_NUM_THREADS=$T \
    PE_BENCH_FORMAT=csv PE_BENCH_HEADER=$HDR \
    $NUMACTL ./build/closure_compare_bench >> "$OUT/closure_scaling.csv"
  HDR=
done

# ------------------- Quintic-20 scaling (the workload that benefits most) -----
echo "[5/6] synthetic quintic-20 scaling across T (12.8M classes)"
: > "$OUT/quintic20_scaling.csv"
HDR=1
for T in 1 2 4 8 16 32 48 64 96 128 144 192; do
  [ $T -gt $T_MAX ] && continue
  echo "  T=$T"
  PARLAY_NUM_THREADS=$T \
    PE_SYNTH_FAMILIES=quintic PE_SYNTH_NS=20 \
    PE_BENCH_FORMAT=csv PE_BENCH_HEADER=$HDR \
    $NUMACTL ./build/synthetic_bench >> "$OUT/quintic20_scaling.csv"
  HDR=
done

# ------------------- Full eggcc sweep at T_max -------------------
echo "[6/6] full eggcc sweep at T=$T_MAX (507 files; ~10-20 min)"
PARLAY_NUM_THREADS=$T_MAX \
  PE_SMT_TRIALS=1 PE_SMT_WARMUP=0 PE_BENCH_HEADER=1 \
  $NUMACTL ./build/smt_bench cc-benchmarks/smt-grounded > "$OUT/eggcc_144T.csv"

# ------------------- Aggregate eggcc -------------------
echo "[6/6] aggregating eggcc results"
awk -F, '
NR == 1 { next }
{
  ms[$1, $6] = $9
  files[$1] = 1
  classes[$1] = $4 + 0
}
END {
  # Algorithms we care about. nelson_topo is the unsound reference;
  # nelson_topo_iter and nelson_dst are sound sequentials; par_close and
  # par_async are sound parallels. nelson_seq is the original baseline
  # (only present when PE_BENCH_SKIP_NELSON is unset — typically absent).
  algos = "nelson_topo nelson_topo_iter nelson_dst par_close par_async"
  n_algos = split(algos, a, " ")

  for (f in files) {
    have_all = 1
    for (i = 1; i <= n_algos; i++) {
      if (!((f, a[i]) in ms)) { have_all = 0; break }
    }
    if (!have_all) continue
    n_paired++
    for (i = 1; i <= n_algos; i++) {
      sum[a[i]] += ms[f, a[i]]
    }
    # win rates: par_async vs each correct sequential
    if (ms[f, "par_async"] < ms[f, "nelson_topo_iter"]) wins_async_iter++
    if (ms[f, "par_async"] < ms[f, "nelson_dst"])       wins_async_dst++
    if (ms[f, "par_close"] < ms[f, "nelson_topo_iter"]) wins_close_iter++

    if (classes[f] < 1000)         b = "<1K"
    else if (classes[f] < 10000)   b = "1-10K"
    else if (classes[f] < 100000)  b = "10-100K"
    else                           b = ">=100K"
    bn[b]++
    for (i = 1; i <= n_algos; i++) {
      bsum[b, a[i]] += ms[f, a[i]]
    }
    if (ms[f, "par_async"] < ms[f, "nelson_topo_iter"]) bwins_async[b]++
  }

  printf "Files paired (all algos timed): %d / 507\n\n", n_paired
  for (i = 1; i <= n_algos; i++) {
    printf "  Σ %-18s %10.2f ms\n", a[i] ":", sum[a[i]]
  }
  printf "\nspeedups (sum-of-medians):\n"
  printf "  par_async / nelson_topo_iter: %.2fx     (correct seq → correct par)\n", sum["nelson_topo_iter"] / sum["par_async"]
  printf "  par_async / nelson_dst:       %.2fx\n", sum["nelson_dst"] / sum["par_async"]
  printf "  par_close / nelson_topo_iter: %.2fx\n", sum["nelson_topo_iter"] / sum["par_close"]
  printf "  par_async / par_close:        %.2fx     (async vs BSP-frontier)\n", sum["par_close"] / sum["par_async"]
  printf "  nelson_topo / nelson_topo_iter: %.2fx   (cost of correctness — unsound vs sound seq)\n", sum["nelson_topo_iter"] / sum["nelson_topo"]

  printf "\nwin rates over %d files:\n", n_paired
  printf "  par_async  < nelson_topo_iter: %d (%.1f%%)\n", wins_async_iter, 100*wins_async_iter/n_paired
  printf "  par_async  < nelson_dst:       %d (%.1f%%)\n", wins_async_dst, 100*wins_async_dst/n_paired
  printf "  par_close  < nelson_topo_iter: %d (%.1f%%)\n", wins_close_iter, 100*wins_close_iter/n_paired

  printf "\npar_async vs nelson_topo_iter, by classes-per-file:\n"
  printf "%-10s %5s | %12s %12s %8s | %5s\n", "bucket", "files", "Σtopo_iter", "Σpar_async", "ratio", "wins"
  order = "<1K 1-10K 10-100K >=100K"
  split(order, bs, " ")
  for (i = 1; i <= 4; i++) {
    b = bs[i]
    if (bn[b] > 0)
      printf "%-10s %5d | %10.2fms %10.2fms %7.2fx | %d/%d\n",
             b, bn[b], bsum[b, "nelson_topo_iter"], bsum[b, "par_async"],
             bsum[b, "nelson_topo_iter"] / bsum[b, "par_async"], bwins_async[b], bn[b]
  }
}' "$OUT/eggcc_144T.csv" | tee "$OUT/eggcc_summary_144T.txt"

echo
echo "=== done. Results in $OUT/ ==="
ls -la "$OUT/"
