# ParallelEgraph

A parallel e-graph implementation in Rust for the theory of equality and uninterpreted functions (QF_UF). Built as a project for 15-852 (Parallel Algorithms).

## What is an e-graph?

An e-graph compactly represents a set of terms and known equivalences between them. It maintains:
- **E-nodes**: function applications `f(c1, ..., ck)` where each `ci` is an e-class ID
- **E-classes**: sets of e-nodes known to be equivalent
- **Union-find**: maps e-class IDs to canonical representatives
- **Hashcons**: maps canonical e-nodes to their e-class for O(1) congruence checks

The core operation is **merge**, which asserts two e-classes are equal. After merging, **rebuild** restores the congruence invariant: if `f(a)` and `f(b)` are in the graph and `a = b`, then `f(a)` and `f(b)` must be in the same e-class.

We only support (dis)-equalities over uninterpreted functions. We *do not* currently support any boolean connectives.

## Usage

Solve an SMT-LIB 2 file:
```
cargo run -- tests/03_congruence_unsat.smt2
```

Solve in parallel mode (union-find merges run lock-free across threads via rayon):
```
cargo run -- --parallel tests/03_congruence_unsat.smt2
```

Control thread count with `RAYON_NUM_THREADS` (defaults to number of logical CPUs):
```
RAYON_NUM_THREADS=4 cargo run -- --parallel tests/15_stress_unsat.smt2
```

### Parallel Mode

The `--parallel` (or `-p`) flag enables a lock-free concurrent union-find based on the concurrent DSU algorithm with rank-based union and path compression via CAS (compare-and-swap). In parallel mode:

1. **Phase 1**: All equality assertions are collected, and their union-find operations execute in parallel across rayon threads (lock-free CAS, no mutexes).
2. **Phase 2**: E-class metadata (class contents, parent/use lists) is reconciled sequentially.
3. **Phase 3**: `rebuild()` restores the congruence invariant as usual.

This is most beneficial when many independent merges can be batched (e.g., the `grid` and `cube` synthetic benchmarks).

## Tests

Run all unit tests and regression tests:
```
cargo test
```

The regression suite in `tests/` contains 16 QF_UF instances. Each `.smt2` file's name encodes the expected result (`_sat` or `_unsat`). The test harness runs the solver on each and checks correctness.

## Benchmarks

The `benchmarks/` directory contains QF_UF benchmarks from the [SMT-COMP 2025 Zenodo archive](https://zenodo.org/records/11061097). These are standard SMT-LIB 2 instances used for solver competition evaluation.

### Synthetic Benchmarks

The `gen_bench.py` script generates scalable QF_UF benchmarks parameterized by size `n`, designed to stress-test parallel e-graph operations. Each benchmark produces a single UNSAT disequality that requires processing the entire formula — no individual assertion pair gives an immediate contradiction. A balanced binary tree keeps nesting depth at O(log m) where m is the number of leaf terms.

```
python gen_bench.py <family> <n> [output_dir]
python gen_bench.py all <n> [output_dir]
python gen_bench.py sweep <n1> <n2> <step> [output_dir]
python gen_bench.py sweep:<family> <n1> <n2> <step> [output_dir]
```

Output defaults to `synthetic_benchmarks/`.

Four families are available, spanning O(n) to O(2^n) congruence counts:

| Family | Merges | Congruences | Stress Target |
|--------|--------|-------------|---------------|
| `chain` | 2n+1 | n | Sequential depth |
| `grid` | n | n² | Parallel rebuild |
| `cube` | n | n³ | Heavy rebuild |
| `exp` | 1 | 2^(n+1)−2 | Exponential cascade |

**chain** — Sequential congruence cascade. Asserts `a0 = b0`, then defines `a_{i+1} = f(a_i)` and `b_{i+1} = f(b_i)` for n levels, with a single disequality `a_n != b_n`. Each congruence step depends on the previous one, making this inherently sequential. Useful as a baseline showing the limit of parallelism.

**grid** — Quadratic congruences via binary function. Declares n constant pairs `a_i = b_i` and a binary function `f`. The single disequality compares a balanced tree of all n² terms `f(a_i, a_j)` against `f(b_i, b_j)`. All n merges are independent (parallelizable), and the n² congruences `f(a_i, a_j) = f(b_i, b_j)` are all independent of each other — a good target for parallel rebuild.

**cube** — Cubic congruences via ternary function. Same structure as grid but with a 3-ary function `f(a_i, a_j, a_k)`, producing n³ congruences from n merges. Stresses rebuild throughput more heavily than grid at the same n.

**exp** — Exponential congruence cascade. A single merge `a = b` propagates through n layers, each with 2 unary functions. Layer i has 2^i named terms per side (e.g., `ta1_0 = f0(a)`, `ta2_0 = f2(ta1_0)`). The merge cascades: layer 1 congruences trigger layer 2, which triggers layer 3, etc., producing 2^(n+1)−2 total congruences. Tests how the solver handles cascading congruence discovery where later work depends on earlier results.

**Note on scaling limits:** The number of congruences discovered during congruence closure is bounded by the number of distinct e-nodes, which equals the number of distinct subterms in the formula. This means exponential congruences (exp family) necessarily require exponential formula size — there is no compact encoding that avoids this. For practical scaling, the polynomial families (grid, cube) are more useful. These could be generalized to a `power_k` family using a k-ary function to produce n^k congruences from n merges (e.g., k=4 for n⁴, k=5 for n⁵).

## TODOS

    - Come up with a scalable benchmark set that does not involve boolean connectives ✅
    - Implement parallel (concurrent?) union-find ⏳ (STARTED, BUT INEFFICIENT)
    - Implement parallel (concurrent?) union predecessors ⏳ (STARTED, BUT INEFFICIENT)
    - Build and SMT harness around the egraph to support boolean connectives (note this means that the egraph has to be backtrackable)
