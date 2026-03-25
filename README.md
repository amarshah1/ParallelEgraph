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

## Tests

Run all unit tests and regression tests:
```
cargo test
```

The regression suite in `tests/` contains 16 QF_UF instances. Each `.smt2` file's name encodes the expected result (`_sat` or `_unsat`). The test harness runs the solver on each and checks correctness.

## Benchmarks

The `benchmarks/` directory contains QF_UF benchmarks from the [SMT-COMP 2025 Zenodo archive](https://zenodo.org/records/11061097). These are standard SMT-LIB 2 instances used for solver competition evaluation.

## TODOS

    - Come up with a scalable benchmark set that does not involve boolean connectives
    - Implement parallel (concurrent?) union-find
    - Implement parallel (concurrent?) union predecessors
    - Build and SMT harness around the egraph to support boolean connectives (note this means that the egraph has to be backtrackable)
