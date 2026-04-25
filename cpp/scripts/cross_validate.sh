#!/usr/bin/env bash
# Cross-validate the Rust and C++ solvers on every .smt2 file in tests/ and
# synthetic_benchmarks/. Runs each file in both sequential and parallel mode
# (and, for parallel, under every PE_REBUILD variant). Fails on any mismatch.
#
# Assumes cargo build --release and cmake --build cpp/build have already been
# run — if not, invoke with `--build` to do both.

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

RUST_BIN="$ROOT/target/release/parallel-egraph"
CPP_BIN="$ROOT/cpp/build/parallel-egraph"

if [[ "${1-}" == "--build" ]]; then
  ( cd "$ROOT" && cargo build --release )
  cmake --build "$ROOT/cpp/build"
fi

if [[ ! -x "$RUST_BIN" ]]; then
  echo "missing Rust binary: $RUST_BIN (try --build)" >&2
  exit 1
fi
if [[ ! -x "$CPP_BIN" ]]; then
  echo "missing C++ binary: $CPP_BIN (try --build)" >&2
  exit 1
fi

files=()
for d in "$ROOT/tests" "$ROOT/synthetic_benchmarks"; do
  [[ -d "$d" ]] || continue
  while IFS= read -r -d '' f; do files+=("$f"); done < <(find "$d" -maxdepth 1 -name '*.smt2' -print0 | sort -z)
done

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no .smt2 files found" >&2
  exit 1
fi

total=0
mismatches=0
for f in "${files[@]}"; do
  # Sequential mode.
  total=$((total + 1))
  rs=$("$RUST_BIN" "$f" 2>/dev/null || echo ERR)
  cp=$("$CPP_BIN" "$f" 2>/dev/null || echo ERR)
  if [[ "$rs" != "$cp" ]]; then
    echo "MISMATCH [seq] $f: rust=$rs cpp=$cp" >&2
    mismatches=$((mismatches + 1))
  fi

  # Parallel mode, each variant.
  for variant in "" sort close; do
    total=$((total + 1))
    rs=$(PE_REBUILD="$variant" "$RUST_BIN" --parallel "$f" 2>/dev/null || echo ERR)
    cp=$(PE_REBUILD="$variant" "$CPP_BIN" --parallel "$f" 2>/dev/null || echo ERR)
    label="${variant:-default}"
    if [[ "$rs" != "$cp" ]]; then
      echo "MISMATCH [par $label] $f: rust=$rs cpp=$cp" >&2
      mismatches=$((mismatches + 1))
    fi
  done
done

echo "cross_validate: $((total - mismatches))/$total agreed ($mismatches mismatches)"
[[ "$mismatches" -eq 0 ]]
