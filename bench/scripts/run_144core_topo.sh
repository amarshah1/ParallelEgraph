#!/usr/bin/env bash
# Closure benchmark sweep for the 144-core machine. Times the full set of
# closure algorithms across closure_compare, synthetic, and eggcc:
#
#   nelson_topo*       single-pass forward topo  (UNSOUND on cross-depth inits — kept for reference)
#   nelson_topo_iter   topo iterated to fixpoint (sound; MIN_ID union — usually 2 passes)
#   nelson_dst         worklist + smaller-into-larger hashcons (sound; correct on arbitrary inputs)
#   par_parents          parallel BSP with parents_ frontier (sound)
#   par_topo_iter      depth-stratified BSP iterated to fixpoint with MIN_ID union (sound)
#   par_filter          parallel async-rounds with last_marked_ stamps + by-rank union (sound)
#   par_filter_min_id   par_filter variant using MIN_ID union (sound; preserves canonical-id invariant)
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

# Sanity check (no NUMACTL — closure_test is sequential and tiny).
# closure_test currently has TWO intentionally-failing cases that document
# known unsoundness in `parallel_topo` and single-pass
# `sequential_close_topo` on cross-depth initial unions; those bugs are
# kept as living regression tests so they don't quietly start "passing"
# from an unrelated change. So a non-zero exit from closure_test is
# *expected*; we only abort if other tests fail.
echo "[1/6] correctness tests"
set +e
./build/closure_test > "$OUT/sanity.log" 2>&1
set -e
grep -E "FAIL|tests passed" "$OUT/sanity.log" || true
expected_fails="par_topo_cross_depth seq_topo_adversarial"
unexpected=$(awk '/^\[FAIL\]/ { print $2 }' "$OUT/sanity.log" | grep -vxF -e par_topo_cross_depth -e seq_topo_adversarial || true)
if [ -n "$unexpected" ]; then
  echo "ERROR: unexpected closure_test failures:" >&2
  echo "$unexpected" >&2
  echo "(expected only: $expected_fails)" >&2
  exit 1
fi

T_MAX=${T_MAX:-144}
echo "T_max=$T_MAX"

# EGGCC_ONLY=1 skips closure_compare, synthetic_bench, and the strong-
# scaling sweeps — runs just the full 507-file eggcc sweep + aggregator.
# Use when you've already run the headlines / scaling sweeps and only
# want to refresh the eggcc data, or when you're debugging an eggcc-
# specific regression and don't want to wait through the synthetic
# stages each time.
EGGCC_ONLY=${EGGCC_ONLY:-0}

# Per-step timeout. Defaults to 30 minutes per bench-binary invocation.
# Override with STEP_TIMEOUT=Xm (e.g. "5m" for 5 minutes, "1h" for 1 hour,
# "0" or empty to disable). Used to bound any single workload-trial that
# misbehaves (e.g., a CAS livelock at high contention) so the rest of the
# sweep still completes.
STEP_TIMEOUT=${STEP_TIMEOUT:-30m}
TIMEOUT=""
if [ -n "$STEP_TIMEOUT" ] && [ "$STEP_TIMEOUT" != "0" ]; then
  if command -v timeout >/dev/null 2>&1; then
    TIMEOUT="timeout --foreground $STEP_TIMEOUT"
  elif command -v gtimeout >/dev/null 2>&1; then
    # macOS: coreutils' gtimeout (brew install coreutils)
    TIMEOUT="gtimeout --foreground $STEP_TIMEOUT"
  else
    echo "[warn] timeout/gtimeout not available; runs are unbounded" >&2
  fi
fi

# Helper: run a bench step with the timeout wrapper. Logs on failure /
# timeout but does not abort the script — `set +e` around the call.
# Caller is responsible for redirecting stdout (the CSV) themselves;
# all status messages from this function go to stderr to keep the
# CSV stream clean.
run_step() {
  local label=$1; shift
  echo "[run_step] $label" >&2
  set +e
  $TIMEOUT "$@"
  local rc=$?
  set -e
  if [ $rc -eq 124 ]; then
    echo "[run_step] $label TIMED OUT after $STEP_TIMEOUT" >&2
  elif [ $rc -ne 0 ]; then
    echo "[run_step] $label exit=$rc" >&2
  fi
  return 0
}

if [ "$EGGCC_ONLY" = "1" ]; then
  echo "[skip] EGGCC_ONLY=1 — skipping headlines + strong-scaling sweeps"
else

# ------------------- Headline at T_max -------------------
echo "[2/6] closure_compare headline at T=$T_MAX"
run_step "closure_compare T=$T_MAX" \
  env PARLAY_NUM_THREADS=$T_MAX PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  $NUMACTL ./build/closure_compare_bench > "$OUT/closure_compare_144T.csv"

echo "[3/6] synthetic_bench headline at T=$T_MAX"
run_step "synthetic_bench T=$T_MAX" \
  env PARLAY_NUM_THREADS=$T_MAX PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  $NUMACTL ./build/synthetic_bench > "$OUT/synthetic_144T.csv"

# ------------------- Strong scaling: closure_compare across T -------------------
echo "[4/6] closure_compare strong scaling across T"
: > "$OUT/closure_scaling.csv"
HDR=1
for T in 1 2 4 8 16 32 48 64 96 128 144 192; do
  [ $T -gt $T_MAX ] && continue
  echo "  T=$T"
  run_step "closure_compare T=$T" \
    env PARLAY_NUM_THREADS=$T PE_BENCH_FORMAT=csv PE_BENCH_HEADER=$HDR \
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
  run_step "quintic-20 T=$T" \
    env PARLAY_NUM_THREADS=$T PE_SYNTH_FAMILIES=quintic PE_SYNTH_NS=20 \
    PE_BENCH_FORMAT=csv PE_BENCH_HEADER=$HDR \
    $NUMACTL ./build/synthetic_bench >> "$OUT/quintic20_scaling.csv"
  HDR=
done

fi  # EGGCC_ONLY guard

# ------------------- Full eggcc sweep at T_max -------------------
# eggcc gets a longer ceiling — it's the longest stage and one slow file
# shouldn't kill the rest. Use STEP_TIMEOUT_EGG to override (default 1h).
EGG_TIMEOUT=${STEP_TIMEOUT_EGG:-1h}
EGG_TIMEOUT_CMD=""
if [ -n "$EGG_TIMEOUT" ] && [ "$EGG_TIMEOUT" != "0" ]; then
  if command -v timeout >/dev/null 2>&1; then
    EGG_TIMEOUT_CMD="timeout --foreground $EGG_TIMEOUT"
  elif command -v gtimeout >/dev/null 2>&1; then
    EGG_TIMEOUT_CMD="gtimeout --foreground $EGG_TIMEOUT"
  fi
fi

echo "[6/6] full eggcc sweep at T=$T_MAX (507 files; ~10-20 min, capped at $EGG_TIMEOUT)"
set +e
$EGG_TIMEOUT_CMD env PARLAY_NUM_THREADS=$T_MAX \
  PE_SMT_TRIALS=1 PE_SMT_WARMUP=0 PE_BENCH_HEADER=1 \
  $NUMACTL ./build/smt_bench cc-benchmarks/smt-grounded > "$OUT/eggcc_144T.csv"
egg_rc=$?
set -e
if [ $egg_rc -eq 124 ]; then
  echo "[run_step] eggcc sweep TIMED OUT after $EGG_TIMEOUT (CSV is partial)" >&2
elif [ $egg_rc -ne 0 ]; then
  echo "[run_step] eggcc sweep exit=$egg_rc" >&2
fi

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
  # nelson_topo_iter and nelson_dst are sound sequentials; par_parents,
  # par_filter, and par_filter_min_id are sound parallels. nelson_seq is
  # the original baseline (only present when PE_BENCH_SKIP_NELSON is
  # unset — typically absent).
  algos = "nelson_topo nelson_topo_iter nelson_dst par_parents par_topo_iter par_filter par_filter_min_id"
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
    # win rates: parallel variants vs the best correct sequential.
    if (ms[f, "par_filter"]        < ms[f, "nelson_topo_iter"]) wins_async_iter++
    if (ms[f, "par_filter"]        < ms[f, "nelson_dst"])       wins_async_dst++
    if (ms[f, "par_parents"]        < ms[f, "nelson_topo_iter"]) wins_close_iter++
    if (ms[f, "par_filter_min_id"] < ms[f, "par_filter"])        wins_min_over_async++
    if (ms[f, "par_topo_iter"]    < ms[f, "nelson_topo_iter"]) wins_topo_iter_iter++
    if (ms[f, "par_topo_iter"]    < ms[f, "par_filter"])        wins_topo_iter_async++

    if (classes[f] < 1000)         b = "<1K"
    else if (classes[f] < 10000)   b = "1-10K"
    else if (classes[f] < 100000)  b = "10-100K"
    else                           b = ">=100K"
    bn[b]++
    for (i = 1; i <= n_algos; i++) {
      bsum[b, a[i]] += ms[f, a[i]]
    }
    # Per-bucket win rate of the best parallel (par_filter_min_id) vs the
    # best correct sequential (whichever is smaller per file).
    best_seq_f = (ms[f, "nelson_topo_iter"] < ms[f, "nelson_dst"]) \
                 ? ms[f, "nelson_topo_iter"] : ms[f, "nelson_dst"]
    if (ms[f, "par_filter_min_id"] < best_seq_f) bwins_min[b]++
  }

  printf "Files paired (all algos timed): %d / 507\n\n", n_paired
  for (i = 1; i <= n_algos; i++) {
    printf "  Σ %-22s %10.2f ms\n", a[i] ":", sum[a[i]]
  }
  printf "\nspeedups (sum-of-medians):\n"
  printf "  par_topo_iter    / nelson_topo_iter:      %.2fx     (depth-strat parallel vs sound seq)\n", sum["nelson_topo_iter"] / sum["par_topo_iter"]
  printf "  par_filter        / nelson_topo_iter:      %.2fx     (async parallel vs sound seq)\n", sum["nelson_topo_iter"] / sum["par_filter"]
  printf "  par_filter_min_id / nelson_topo_iter:      %.2fx     (MIN_ID variant)\n", sum["nelson_topo_iter"] / sum["par_filter_min_id"]
  printf "  par_topo_iter    / nelson_dst:            %.2fx\n", sum["nelson_dst"] / sum["par_topo_iter"]
  printf "  par_filter        / nelson_dst:            %.2fx\n", sum["nelson_dst"] / sum["par_filter"]
  printf "  par_filter_min_id / nelson_dst:            %.2fx\n", sum["nelson_dst"] / sum["par_filter_min_id"]
  printf "  par_parents        / nelson_topo_iter:      %.2fx\n", sum["nelson_topo_iter"] / sum["par_parents"]
  printf "  par_topo_iter    / par_filter:             %.2fx     (depth-strat vs async)\n", sum["par_filter"] / sum["par_topo_iter"]
  printf "  par_filter        / par_parents:             %.2fx     (async vs BSP-frontier)\n", sum["par_parents"] / sum["par_filter"]
  printf "  par_filter_min_id / par_filter:             %.2fx     (MIN_ID gain over by-rank)\n", sum["par_filter"] / sum["par_filter_min_id"]
  printf "  nelson_topo      / nelson_topo_iter:      %.2fx     (cost of correctness — unsound vs sound seq)\n", sum["nelson_topo_iter"] / sum["nelson_topo"]

  printf "\nwin rates over %d files:\n", n_paired
  printf "  par_topo_iter     < nelson_topo_iter: %d (%.1f%%)\n", wins_topo_iter_iter, 100*wins_topo_iter_iter/n_paired
  printf "  par_filter         < nelson_topo_iter: %d (%.1f%%)\n", wins_async_iter, 100*wins_async_iter/n_paired
  printf "  par_filter         < nelson_dst:       %d (%.1f%%)\n", wins_async_dst, 100*wins_async_dst/n_paired
  printf "  par_parents         < nelson_topo_iter: %d (%.1f%%)\n", wins_close_iter, 100*wins_close_iter/n_paired
  printf "  par_filter_min_id  < par_filter:        %d (%.1f%%)\n", wins_min_over_async, 100*wins_min_over_async/n_paired
  printf "  par_topo_iter     < par_filter:        %d (%.1f%%)\n", wins_topo_iter_async, 100*wins_topo_iter_async/n_paired

  printf "\npar_filter_min_id vs best correct sequential, by classes-per-file:\n"
  printf "%-10s %5s | %12s %12s %12s %8s | %5s\n",
         "bucket", "files", "Σnel_iter", "Σnel_dst", "Σpar_filter_m", "ratio", "wins"
  order = "<1K 1-10K 10-100K >=100K"
  split(order, bs, " ")
  for (i = 1; i <= 4; i++) {
    b = bs[i]
    if (bn[b] > 0) {
      best_seq = (bsum[b, "nelson_topo_iter"] < bsum[b, "nelson_dst"]) \
                 ? bsum[b, "nelson_topo_iter"] : bsum[b, "nelson_dst"]
      printf "%-10s %5d | %10.2fms %10.2fms %10.2fms %7.2fx | %d/%d\n",
             b, bn[b], bsum[b, "nelson_topo_iter"], bsum[b, "nelson_dst"],
             bsum[b, "par_filter_min_id"],
             best_seq / bsum[b, "par_filter_min_id"], bwins_min[b], bn[b]
    }
  }
}' "$OUT/eggcc_144T.csv" | tee "$OUT/eggcc_summary_144T.txt"

echo
echo "=== done. Results in $OUT/ ==="
ls -la "$OUT/"
