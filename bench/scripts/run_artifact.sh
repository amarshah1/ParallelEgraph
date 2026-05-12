#!/usr/bin/env bash
# FMCAD 2025 artifact — full reproduction driver.
#
# Reproduces the four headline tables from the paper:
#   T1. closure_compare (synthetic uniform DAGs, 1 + 5 trials)
#   T2. synthetic_bench (chain/grid/cube/quartic/quintic, n ∈ {5,10,20})
#   T3. eggcc smt-grounded (507 .smt2 files, 1 trial)
#   T4. closure_compare strong-scaling across T = {1,2,4,8,16,32,64,96,T_MAX}
#
# Outputs land in bench/results/artifact/ (CSV + summaries).
#
# Wall time:
#   - 12-core laptop: ~30-45 minutes (T_MAX=12 by default; smaller scaling sweep)
#   - 144-core server (numactl -i all): ~25-40 minutes
#   - The eggcc sweep dominates (15-25 minutes); synthetic quintic-20
#     sequential nelson_seq alone is ~10 minutes on a single core.

set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=bench/results/artifact
mkdir -p "$OUT"

T_MAX=${T_MAX:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 12)}
NUMACTL=${NUMACTL:-}
if [ -z "$NUMACTL" ] && command -v numactl >/dev/null 2>&1; then
  # If numactl is available and the box has >1 NUMA node, interleave.
  nodes=$(numactl --hardware 2>/dev/null | awk '/available:/ {print $2}')
  if [ "${nodes:-1}" -gt 1 ] 2>/dev/null; then
    NUMACTL="numactl -i all"
  fi
fi
echo "T_MAX=$T_MAX  NUMACTL='$NUMACTL'"

# ---- preflight ----
for B in closure_compare_bench synthetic_bench smt_bench closure_test; do
  [ -x "build/$B" ] || {
    echo "Missing build/$B. Run:" >&2
    echo "  cmake -B build && cmake --build build -j" >&2
    exit 1
  }
done

# Capture environment
{
  echo "=== Artifact reproduction ==="
  date -u
  echo
  echo "git:"; git rev-parse HEAD 2>/dev/null || echo "(no git)"
  echo
  if [ -r /proc/cpuinfo ]; then
    grep -E 'model name|cpu cores|siblings' /proc/cpuinfo | sort -u
    echo "logical cores: $(nproc)"
  else
    sysctl -n machdep.cpu.brand_string hw.physicalcpu hw.logicalcpu 2>/dev/null || true
  fi
  echo
  command -v numactl >/dev/null && numactl --hardware 2>&1 | head -10 || echo "(numactl absent)"
  echo
  echo "NUMACTL prefix: $NUMACTL"
  echo "T_MAX:           $T_MAX"
} > "$OUT/build_info.txt"

# ---- correctness ----
# 20 cases total. 2 are documented illustrative-unsound variants
# (par_topo_cross_depth, seq_topo_adversarial) that must fail by
# design — they exist to demonstrate the cost of skipping iteration.
# The remaining 18 must pass; any other failure is a regression.
echo "[1/5] correctness tests (expect 18/20; 2 documented illustrative failures)"
./build/closure_test > "$OUT/correctness.log" 2>&1 || true
grep -E "^\[(OK|FAIL)\]|tests passed" "$OUT/correctness.log" | tail -25
unexpected=$(awk '/^\[FAIL\]/ {print $2}' "$OUT/correctness.log" | \
  grep -vE '^(par_topo_cross_depth|seq_topo_adversarial)$' || true)
if [ -n "$unexpected" ]; then
  echo "FAILED: unexpected test failures: $unexpected" >&2; exit 1
fi
grep -q "18/20 closure tests passed" "$OUT/correctness.log" || \
  grep -qE "^[0-9]+/[0-9]+ closure tests passed" "$OUT/correctness.log" || {
  echo "FAILED: no test summary line" >&2; exit 1; }

# ---- T1. closure_compare at T_MAX ----
echo "[2/5] T1 closure_compare at T=$T_MAX (~1 min)"
PARLAY_NUM_THREADS=$T_MAX \
  PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  $NUMACTL ./build/closure_compare_bench > "$OUT/T1_closure_compare.csv"

# ---- T2. synthetic_bench at T_MAX ----
echo "[3/5] T2 synthetic_bench at T=$T_MAX (~10-20 min: quintic-20 is the bulk)"
PARLAY_NUM_THREADS=$T_MAX \
  PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  $NUMACTL ./build/synthetic_bench > "$OUT/T2_synthetic.csv"

# ---- T3. eggcc full sweep at T_MAX ----
echo "[4/5] T3 eggcc sweep at T=$T_MAX (~15-25 min)"
if [ -d cc-benchmarks/smt-grounded ] && [ -n "$(ls cc-benchmarks/smt-grounded/*.smt2 2>/dev/null)" ]; then
  PARLAY_NUM_THREADS=$T_MAX \
    PE_SMT_TRIALS=1 PE_SMT_WARMUP=0 PE_BENCH_HEADER=1 \
    $NUMACTL ./build/smt_bench cc-benchmarks/smt-grounded > "$OUT/T3_eggcc.csv"
else
  echo "  WARN: cc-benchmarks/ submodule not populated; T3 skipped"
  echo "  To enable: git submodule update --init --recursive cc-benchmarks"
fi

# ---- T4. closure_compare strong scaling across T ----
echo "[5/5] T4 closure_compare strong scaling across T (~5 min)"
: > "$OUT/T4_scaling.csv"
HDR=1
for T in 1 2 4 8 16 32 64 96 $T_MAX; do
  [ "$T" -gt "$T_MAX" ] && continue
  echo "  T=$T"
  PARLAY_NUM_THREADS=$T \
    PE_BENCH_FORMAT=csv PE_BENCH_HEADER=$HDR \
    $NUMACTL ./build/closure_compare_bench >> "$OUT/T4_scaling.csv"
  HDR=
done
# de-dup T_MAX if it's already in the canonical list
sort -u "$OUT/T4_scaling.csv" -o "$OUT/T4_scaling.csv.tmp" && \
  mv "$OUT/T4_scaling.csv.tmp" "$OUT/T4_scaling.csv"

# ---- aggregate ----
echo
echo "==> aggregating results into $OUT/summary.txt"
./bench/scripts/aggregate_artifact.sh "$OUT" | tee "$OUT/summary.txt"

echo
echo "=== done. Results in $OUT/ ==="
ls -la "$OUT/"
