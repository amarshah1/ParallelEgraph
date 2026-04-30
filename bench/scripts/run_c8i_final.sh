#!/usr/bin/env bash
# Final benchmark sweep for c8i.metal-48xl (192 vCPUs, 96 phys × 2 SMT,
# single-socket Intel Xeon 6 / Granite Rapids, SNC=3).
#
# Runs:
#   Tier 1.1 — fine-grained strong scaling on 16xl-d3 (T = 1..192, 11 cells)
#   Tier 1.2 — width grid extended to 16xl-d3 at T_max
#   Tier 1.3 — round-level traces at six widths
#   Tier 1.4 — per-phase scaling on 4xl-d3 across T
#   Tier 2.6 — merge-density sweep on xl-d3 at T=192
#
# Outputs CSVs and rendered PNGs under bench/results/. Re-running overwrites
# prior results. Total wall time ~25–35 min on c8i.metal-48xl.

set -euo pipefail
cd "$(dirname "$0")/../.."

# ------------------- Pre-flight -------------------
for B in build/closure_compare_bench build/component_bench; do
  [ -x "$B" ] || { echo "Missing: $B. Run 'cmake --build build' first." >&2; exit 1; }
done

# Confirm closure_compare_bench supports PE_BENCH_CUSTOM and component_bench
# supports PE_COMPONENT_CUSTOM. If older binaries are on disk, the runs below
# will silently fall back to baked-in workloads — flag this clearly.
PE_BENCH_CUSTOM=1,1,1,1,1 PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  ./build/closure_compare_bench 2>&1 >/dev/null | head -1 | \
  grep -q "must be" && { :; } || true
PE_COMPONENT_CUSTOM=1,1,1,1,1 PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  ./build/component_bench 2>&1 >/dev/null | head -1 | \
  grep -q "must be" && { :; } || true

mkdir -p bench/results

T_MAX=${T_MAX:-192}
echo "T_max=$T_MAX"

# ------------------- Tier 1.1 — fine-grained strong scaling on 16xl-d3 -------
echo "[1/5] strong scaling on 16xl-d3 across T"
: > bench/results/strong_scaling.csv
HDR=PE_BENCH_HEADER=1
for T in 1 2 4 8 16 32 48 64 96 144 192; do
  echo "    T=$T"
  env $HDR PARLAY_NUM_THREADS=$T PE_BENCH_FORMAT=csv \
    PE_BENCH_CUSTOM=1600000,16,32000000,3200000,3 \
    PE_BENCH_SKIP_NELSON=1 \
    ./build/closure_compare_bench 2>/dev/null \
    | sed 's/^custom,/16xl-d3,/' \
    >> bench/results/strong_scaling.csv
  HDR=
done

# ------------------- Tier 1.2 — width grid extended -------------------
echo "[2/5] width grid (large, xl-d3, 2xl-d3, 4xl-d3, 8xl-d3, 16xl-d3) at T=$T_MAX"
: > bench/results/width_grid.csv
HDR=PE_BENCH_HEADER=1
for SPEC in "large:50000,16,1000000,100000,3" \
            "xl-d3:100000,16,2000000,200000,3" \
            "2xl-d3:200000,16,4000000,400000,3" \
            "4xl-d3:400000,16,8000000,800000,3" \
            "8xl-d3:800000,16,16000000,1600000,3" \
            "16xl-d3:1600000,16,32000000,3200000,3"; do
  LABEL=${SPEC%%:*}; CUSTOM=${SPEC#*:}
  echo "    $LABEL"
  SKIP=
  case $LABEL in 2xl-d3|4xl-d3|8xl-d3|16xl-d3) SKIP=PE_BENCH_SKIP_NELSON=1 ;; esac
  env $HDR $SKIP PARLAY_NUM_THREADS=$T_MAX PE_BENCH_FORMAT=csv \
    PE_BENCH_CUSTOM=$CUSTOM \
    ./build/closure_compare_bench 2>/dev/null \
    | sed "s/^custom,/$LABEL,/" \
    >> bench/results/width_grid.csv
  HDR=
done

# ------------------- Tier 1.3 — round-level traces -------------------
echo "[3/5] traces at six widths"
echo "workload,parlay_threads,round,work,frontier,next,consolidate_ms,frontier_ms,semisort_ms,keyed_ms,group_by_ms,per_group_ms" \
  > bench/results/trace.csv
for SPEC in "large:50000,16,1000000,100000,3" \
            "xl-d3:100000,16,2000000,200000,3" \
            "2xl-d3:200000,16,4000000,400000,3" \
            "4xl-d3:400000,16,8000000,800000,3" \
            "8xl-d3:800000,16,16000000,1600000,3" \
            "16xl-d3:1600000,16,32000000,3200000,3"; do
  LABEL=${SPEC%%:*}; CUSTOM=${SPEC#*:}
  echo "    $LABEL"
  PE_TRACE=1 PARLAY_NUM_THREADS=$T_MAX PE_BENCH_FORMAT=csv \
    PE_BENCH_CUSTOM=$CUSTOM PE_BENCH_SKIP_NELSON=1 \
    ./build/closure_compare_bench 2>bench/results/trace_${LABEL}.log >/dev/null
  python3 bench/scripts/parse_trace.py "$LABEL" $T_MAX --no-header \
    < bench/results/trace_${LABEL}.log \
    >> bench/results/trace.csv
done

# ------------------- Tier 1.4 — per-phase scaling on 4xl-d3 -------------------
echo "[4/5] components on 4xl-d3 across T"
: > bench/results/components.csv
HDR=PE_BENCH_HEADER=1
for T in 1 16 48 96 144 192; do
  echo "    T=$T"
  env $HDR PARLAY_NUM_THREADS=$T PE_BENCH_FORMAT=csv \
    PE_COMPONENT_CUSTOM=400000,16,8000000,800000,3 \
    ./build/component_bench 2>/dev/null \
    | sed 's/,custom,/,4xl-d3,/' \
    >> bench/results/components.csv
  HDR=
done

# ------------------- Tier 2.6 — merge-density sweep on xl-d3 -------------------
echo "[5/5] merge-density sweep on xl-d3 at T=$T_MAX"
: > bench/results/merge_density.csv
HDR=PE_BENCH_HEADER=1
for FRAC in 0.05 0.1 0.25 0.5 1.0 2.0; do
  MERGES=$(python3 -c "print(int(100000 * $FRAC))")
  echo "    merge_frac=$FRAC (n_merges=$MERGES)"
  env $HDR PARLAY_NUM_THREADS=$T_MAX PE_BENCH_FORMAT=csv \
    PE_BENCH_CUSTOM=100000,16,2000000,$MERGES,3 \
    PE_BENCH_SKIP_NELSON=1 \
    ./build/closure_compare_bench 2>/dev/null \
    | sed 's/^custom,/xl-d3,/' \
    >> bench/results/merge_density.csv
  HDR=
done

# ------------------- Plots, summary, package -------------------
echo "[plot] rendering"
python3 bench/scripts/plot.py all
python3 bench/scripts/summarize.py > bench/results/summary.txt
tar czf bench/results.tgz -C bench results
echo "Done. Tarball: bench/results.tgz"
ls -lh bench/results.tgz
