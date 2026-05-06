#!/usr/bin/env bash
# Topo-closure benchmark sweep for the 144-core machine.
#
# Captures everything we'd want to know about how `parallel_close_topo`
# (depth-stratified semisort BSP) scales relative to `nelson_topo` (the
# optimized one-pass sequential algo) and the older `parallel_close`
# BSP. Re-running overwrites prior results.
#
# Outputs (under bench/results/topo144/):
#   build_info.txt              compiler/git state captured at run time
#   sanity.log                  closure_test pass/fail
#   closure_compare_144T.csv    headline closure_compare at T=144
#   synthetic_144T.csv          headline synthetic_bench at T=144
#   closure_scaling.csv         closure_compare across T = 1..144
#   quintic20_scaling.csv       synthetic quintic-20 (12.8M classes) across T
#   eggcc_144T.csv              full 507-file cc-benchmarks/smt-grounded sweep at T=144
#   eggcc_summary_144T.txt      aggregate (par_topo vs nelson_topo, win rates, by class-bucket)
#
# Total wall time ~30-60 min depending on memory bandwidth and disk speed.
# The eggcc sweep dominates (~10-20 min); the scaling sweeps are ~5 min each.

set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=bench/results/topo144
mkdir -p "$OUT"

# ------------------- Pre-flight -------------------
for B in closure_compare_bench synthetic_bench smt_bench closure_test; do
  [ -x "build/$B" ] || {
    echo "Missing build/$B. Run:" >&2
    echo "  mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. \\" >&2
    echo "    && cmake --build . --target closure_compare_bench synthetic_bench smt_bench closure_test -j" >&2
    exit 1
  }
done

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
  echo "compiler:"; cmake --version | head -1
} > "$OUT/build_info.txt" 2>&1

# Sanity check
echo "[1/6] correctness tests"
./build/closure_test > "$OUT/sanity.log" 2>&1
grep -E "FAIL|tests passed" "$OUT/sanity.log"

T_MAX=${T_MAX:-144}
echo "T_max=$T_MAX"

# ------------------- Headline at T_max -------------------
echo "[2/6] closure_compare headline at T=$T_MAX"
PARLAY_NUM_THREADS=$T_MAX \
  PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  ./build/closure_compare_bench > "$OUT/closure_compare_144T.csv"

echo "[3/6] synthetic_bench headline at T=$T_MAX"
PARLAY_NUM_THREADS=$T_MAX \
  PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  ./build/synthetic_bench > "$OUT/synthetic_144T.csv"

# ------------------- Strong scaling: closure_compare across T -------------------
echo "[4/6] closure_compare strong scaling across T"
: > "$OUT/closure_scaling.csv"
HDR=1
for T in 1 2 4 8 16 32 64 96 144; do
  [ $T -gt $T_MAX ] && continue
  echo "  T=$T"
  PARLAY_NUM_THREADS=$T \
    PE_BENCH_FORMAT=csv PE_BENCH_HEADER=$HDR \
    ./build/closure_compare_bench >> "$OUT/closure_scaling.csv"
  HDR=
done

# ------------------- Quintic-20 scaling (the workload that benefits most) -----
echo "[5/6] synthetic quintic-20 scaling across T (12.8M classes)"
: > "$OUT/quintic20_scaling.csv"
HDR=1
for T in 1 2 4 8 16 32 64 96 144; do
  [ $T -gt $T_MAX ] && continue
  echo "  T=$T"
  PARLAY_NUM_THREADS=$T \
    PE_SYNTH_FAMILIES=quintic PE_SYNTH_NS=20 \
    PE_BENCH_FORMAT=csv PE_BENCH_HEADER=$HDR \
    ./build/synthetic_bench >> "$OUT/quintic20_scaling.csv"
  HDR=
done

# ------------------- Full eggcc sweep at T_max -------------------
echo "[6/6] full eggcc sweep at T=$T_MAX (507 files; ~10-20 min)"
PARLAY_NUM_THREADS=$T_MAX \
  PE_SMT_TRIALS=1 PE_SMT_WARMUP=0 PE_BENCH_HEADER=1 \
  ./build/smt_bench cc-benchmarks/smt-grounded > "$OUT/eggcc_144T.csv"

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
  for (f in files) {
    if (!((f, "nelson_seq") in ms && (f, "nelson_topo") in ms &&
          (f, "par_close") in ms && (f, "par_topo") in ms)) continue
    n_paired++
    sum_nel += ms[f, "nelson_seq"]
    sum_top += ms[f, "nelson_topo"]
    sum_par += ms[f, "par_close"]
    sum_pt  += ms[f, "par_topo"]
    if (ms[f, "par_topo"] < ms[f, "nelson_topo"]) pt_wins_top++
    if (ms[f, "par_topo"] < ms[f, "par_close"])  pt_wins_par++
    if (ms[f, "par_topo"] < ms[f, "nelson_seq"]) pt_wins_nel++
    if (classes[f] < 1000)         { b = "<1K"     }
    else if (classes[f] < 10000)   { b = "1-10K"   }
    else if (classes[f] < 100000)  { b = "10-100K" }
    else                           { b = ">=100K"  }
    bn[b]++
    bs_top[b] += ms[f, "nelson_topo"]
    bs_pt[b]  += ms[f, "par_topo"]
    if (ms[f, "par_topo"] < ms[f, "nelson_topo"]) bw_top[b]++
  }
  printf "Files paired: %d / 507\n\n", n_paired
  printf "Σ nelson_seq:  %10.2f ms\n", sum_nel
  printf "Σ nelson_topo: %10.2f ms\n", sum_top
  printf "Σ par_close:   %10.2f ms\n", sum_par
  printf "Σ par_topo:    %10.2f ms\n\n", sum_pt
  printf "par_topo aggregate (sum):\n"
  printf "  vs nelson_seq:  %.2fx\n", sum_nel/sum_pt
  printf "  vs nelson_topo: %.2fx\n", sum_top/sum_pt
  printf "  vs par_close:   %.2fx\n\n", sum_par/sum_pt
  printf "win rates:\n"
  printf "  par_topo < nelson_seq:  %d / %d (%.1f%%)\n", pt_wins_nel, n_paired, 100*pt_wins_nel/n_paired
  printf "  par_topo < nelson_topo: %d / %d (%.1f%%)\n", pt_wins_top, n_paired, 100*pt_wins_top/n_paired
  printf "  par_topo < par_close:   %d / %d (%.1f%%)\n\n", pt_wins_par, n_paired, 100*pt_wins_par/n_paired
  printf "by classes-per-file (vs nelson_topo):\n"
  printf "%-10s %5s | %12s %12s %8s | %5s\n", "bucket", "files", "Σnel_topo", "Σpar_topo", "ratio", "wins"
  order = "<1K 1-10K 10-100K >=100K"
  split(order, bs, " ")
  for (i = 1; i <= 4; i++) {
    b = bs[i]
    if (bn[b] > 0)
      printf "%-10s %5d | %10.2fms %10.2fms %7.2fx | %d/%d\n",
             b, bn[b], bs_top[b], bs_pt[b], bs_top[b]/bs_pt[b], bw_top[b], bn[b]
  }
}' "$OUT/eggcc_144T.csv" | tee "$OUT/eggcc_summary_144T.txt"

echo
echo "=== done. Results in $OUT/ ==="
ls -la "$OUT/"
