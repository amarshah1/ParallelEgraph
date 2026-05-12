#!/usr/bin/env bash
# FMCAD 2026 artifact — quick smoke test.
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

echo "==> [1/2] correctness suite (expect 7/7 closure tests passed)"
[ -x build/closure_test ] || fail "build/closure_test missing — run: cmake -B build && cmake --build build"

out=$(./build/closure_test 2>&1 || true)
echo "$out"
echo
if ! echo "$out" | grep -q "7/7 closure tests passed"; then
  fail "closure_test did not pass all 7 cases (see output above)"
fi
ok "all 7 closure tests passed"

if [ -x build/unionfind_test ]; then
  echo "==> unionfind_test"
  uf=$(./build/unionfind_test 2>&1 || true)
  echo "$uf"
  echo "$uf" | grep -q "PASS" || fail "unionfind_test did not pass"
fi

echo
echo "==> [2/2] closure_compare on 'small' workload"
[ -x build/closure_compare_bench ] || fail "build/closure_compare_bench missing"
PE_BENCH_ONLY=small ./build/closure_compare_bench 2>/dev/null
ok "closure_compare ran. Headline algorithms: nelson_simple_inline (seq), par_parents (BSP), par_filter (filter)."

echo
ok "kick-the-tires complete. For full reproduction:"
echo "       ./bench/scripts/run_artifact.sh"
