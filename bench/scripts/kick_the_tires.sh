#!/usr/bin/env bash
# FMCAD 2025 artifact — quick smoke test.
#
# Verifies the build works and reproduces a couple of headline numbers
# at a small scale. Total wall time ~2-3 minutes on a modern laptop.
# Use ./bench/scripts/run_artifact.sh for the full reproduction.

set -euo pipefail
cd "$(dirname "$0")/../.."

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { printf "${GREEN}ok${NC}   %s\n" "$1"; }
warn() { printf "${YELLOW}warn${NC} %s\n" "$1"; }
fail() { printf "${RED}FAIL${NC} %s\n" "$1"; exit 1; }

echo "==> [1/3] correctness suite"
[ -x build/closure_test ] || fail "build/closure_test missing — run: cmake -B build && cmake --build build"

# The test binary exits non-zero if any case fails. Two known-illustrative
# unsound variants (sequential_close_topo single-pass, parallel_topo without
# iteration) fail their adversarial cases *by design* — they exist in the
# source as pedagogical waypoints showing what *requires* iteration to be
# sound. We tolerate the listed failures and reject any others.
out=$(./build/closure_test 2>&1 || true)
echo "$out"
echo
total_fails=$(echo "$out" | grep -c '^\[FAIL\]' || true)
expected_fails="par_topo_cross_depth seq_topo_adversarial"
unexpected=""
for case in $(echo "$out" | awk '/^\[FAIL\]/ {print $2}'); do
  case " $expected_fails " in
    *" $case "*) ;;
    *) unexpected="$unexpected $case" ;;
  esac
done
if [ -n "$unexpected" ]; then
  fail "unexpected test failures:$unexpected"
fi
if echo "$out" | grep -qE '^[0-9]+/[0-9]+ closure tests passed'; then
  summary=$(echo "$out" | grep -E '^[0-9]+/[0-9]+ closure tests passed')
  if [ "$total_fails" -le 2 ]; then
    ok "$summary (2 expected illustrative failures tolerated; see ARTIFACT.md §soundness)"
  else
    fail "more than 2 failures observed"
  fi
else
  fail "no test summary line found"
fi

echo
echo "==> [2/3] closure_compare on 'small' workload"
[ -x build/closure_compare_bench ] || fail "build/closure_compare_bench missing"
PE_BENCH_ONLY=small ./build/closure_compare_bench 2>/dev/null | tee /tmp/kick_cc.out
ok "closure_compare ran. Headline algorithms: nelson_topo_iter (sound seq), par_topo_iter / par_filter* (sound par)."

echo
echo "==> [3/3] eggcc one-file demo (if cc-benchmarks/ is populated)"
demo=cc-benchmarks/smt-grounded/demo.1.ground.smt2
if [ -f "$demo" ]; then
  PE_SMT_TRIALS=1 PE_SMT_WARMUP=0 PE_BENCH_HEADER=1 \
    ./build/smt_bench "$demo" 2>/dev/null
  ok "smt_bench ran on demo.1.ground.smt2"
else
  warn "cc-benchmarks/ submodule not populated — skipping eggcc demo"
  echo "     To enable: git submodule update --init --recursive cc-benchmarks"
fi

echo
ok "kick-the-tires complete. For full reproduction:"
echo "       ./bench/scripts/run_artifact.sh"
