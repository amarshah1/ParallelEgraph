#!/usr/bin/env bash
# FMCAD 2026 artifact — full reproduction driver.
#
# Wraps compare_topo_vs_async.py with the canonical paper invocation:
#   nelson_simple_inline, par_parents, par_filter
# across the random (default + XL ladder), cube_decomp, and gates
# phases. Output lands in runs/topo_vs_async_<ts>/.
#
# Wall time:
#   - 12-core laptop  : ~45-90 minutes
#   - 144-core server : ~25-40 minutes (numactl -i all auto-detected
#                                       when available)
#
# Override knobs:
#   THREADS=1,2,4,...,192   thread-sweep (default 1,2,4,8,16,32,64,96,144,192)
#   TRIALS=N                measured trials per cell (default 5)
#   WARMUP=N                warmup invocations (default 1)
#   EXTRA="--shutdown-after" appended to the python args for unattended runs

set -euo pipefail
cd "$(dirname "$0")/../.."

# Pre-flight
for B in closure_compare_bench synthetic_bench gates_bench closure_test; do
  [ -x "build/$B" ] || {
    echo "Missing build/$B. Run:" >&2
    echo "  cmake -B build && cmake --build build -j" >&2
    exit 1
  }
done

# Sanity check (7/7 closure tests must pass)
echo "[1/2] correctness tests"
./build/closure_test
if ! ./build/closure_test 2>&1 | grep -q "7/7 closure tests passed"; then
  echo "ERROR: closure_test did not pass 7/7 cases" >&2
  exit 1
fi

THREADS=${THREADS:-1,2,4,8,16,32,64,96,144,192}
TRIALS=${TRIALS:-5}
WARMUP=${WARMUP:-1}
EXTRA=${EXTRA:-}

echo
echo "[2/2] compare_topo_vs_async.py sweep"
echo "  threads=$THREADS  trials=$TRIALS  warmup=$WARMUP"
exec python3 compare_topo_vs_async.py \
  --skip synthetic \
  --threads-sweep "$THREADS" \
  --random-modes xl,default \
  --warmup "$WARMUP" \
  --trials "$TRIALS" \
  --algos nelson_simple_inline,par_parents,par_filter \
  $EXTRA
