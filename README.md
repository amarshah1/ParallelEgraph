# ParallelEgraph

A parallel e-graph in C++ for the theory of equality and uninterpreted
functions (QF_UF). Lock-free concurrent union-find +
bulk-synchronous-parallel (BSP) congruence closure built on
[parlaylib](https://github.com/cmuparlay/parlaylib). Built as a project
for 15-852 (Parallel Algorithms).

## What is an e-graph?

An e-graph compactly represents a set of terms and known equivalences
between them:

- **E-nodes**: function applications `f(c1, ..., ck)`, each `ci` an
  e-class id.
- **E-classes**: sets of e-nodes known to be equivalent.
- **Union-find**: maps e-class ids to canonical representatives.
- **Hashcons**: maps canonical e-nodes to their e-class for O(1)
  congruence checks.

The core operation is **merge** (assert two e-classes are equal); after
merging, **congruence closure** restores the invariant: if `f(a)` and
`f(b)` are in the graph and `a = b`, then `f(a)` and `f(b)` must be in
the same e-class.

## Build

```
cmake -B build -S .
cmake --build build
```

Parlay is fetched by CMake's `FetchContent`; C++20 required.

## Run

```
./build/egraph-cc examples/regression/03_congruence_unsat.smt2   # → unsat
./build/egraph-cc examples/regression/01_trivial_sat.smt2        # → sat
```

The CLI parses a QF_UF SMT-LIB 2 file, builds the e-graph from the
assertions, runs `parallel_close`, and prints `sat`/`unsat`.

## Test

```
cd build && ctest --output-on-failure
```

Two suites:
- `unionfind_test` — lock-free UF correctness.
- `regression_test` — runs every `examples/regression/*.smt2` and checks
  the result against the filename suffix.

## Generate synthetic inputs

```
python3 gen_bench.py <family> <n>
python3 gen_bench.py sweep 3 11 1
```

Four families (`chain`, `grid`, `cube`, `exp`) producing parameterized
QF_UF inputs of varying complexity. Output goes to `examples/synthetic/`.

## Benchmark

```
./build/closure_compare_bench
```

Compares `EGraph::sequential_close_nelson` (Nelson-style sequential
baseline) against `EGraph::parallel_close` (BSP parallel closure) on six
synthetic workloads (`small`, `medium`, `large`, `deep-s`, `deep-m`,
`deep-l`).

Output:
```
name    leaves    nodes    merges  | nelson_seq |  par_close   par_spd
```

See [DESIGN.md](DESIGN.md) for the algorithm, correctness invariants, env
var tunables (`PARLAY_NUM_THREADS`, `PE_BENCH_ONLY`, `PE_UNION_STYLE`,
`PE_DNC_CUTOFF`, `PE_TRACE`), and a full account of the two parlay-native
hot-path primitives — `parallel_consolidate` and `merge_and_collect_semisort`.
