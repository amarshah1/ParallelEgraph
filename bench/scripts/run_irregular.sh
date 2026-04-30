#!/usr/bin/env bash
# Run the irregular workload bench at xl-d3-equivalent size for direct
# comparability with the uniform xl-d3 number in width_grid.csv.
#
# Outputs one CSV file (bench/results/irregular.csv) and prints the
# median speedup table to stdout. Re-running overwrites prior results.
#
# Usage:
#   bash bench/scripts/run_irregular.sh [<n_leaves>,<fns>,<nodes>,<merges>,<depth>]
#
# Default: 100000,16,2000000,200000,3 (matches xl-d3 in scale).

set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=build/irregular_bench
[ -x "$BIN" ] || {
  echo "Missing: $BIN. Run 'cmake --build build' first." >&2
  exit 1
}

mkdir -p bench/results

SPEC="${1:-100000,16,2000000,200000,3}"
T_MAX="${T_MAX:-192}"

echo "[irregular] running PE_BENCH_CUSTOM=$SPEC PARLAY_NUM_THREADS=$T_MAX"

PARLAY_NUM_THREADS=$T_MAX PE_BENCH_FORMAT=csv PE_BENCH_HEADER=1 \
  PE_BENCH_CUSTOM=$SPEC \
  $BIN 2>/dev/null \
  > bench/results/irregular.csv

echo "  → bench/results/irregular.csv"

# Summarize.
python3 - <<'PY'
import pandas as pd
import statistics
df = pd.read_csv("bench/results/irregular.csv")
nel = df[df.algorithm == "nelson_seq"]["wallclock_ms"].median()
par = df[df.algorithm == "par_close"]["wallclock_ms"].median()
n_leaves = int(df.iloc[0]["leaves"])
n_nodes = int(df.iloc[0]["nodes"])
n_merges = int(df.iloc[0]["merges"])
T = int(df.iloc[0]["parlay_threads"])
print()
print(f"=== irregular workload (leaves={n_leaves} nodes={n_nodes} "
      f"merges={n_merges} T={T}) ===")
print(f"  nelson_seq median: {nel:8.2f} ms")
print(f"  par_close  median: {par:8.2f} ms")
print(f"  speedup:           {nel/par:8.2f}x")
PY
