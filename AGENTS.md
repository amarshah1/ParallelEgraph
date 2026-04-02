# AGENTS.md

Context for AI agents working on this codebase.

## Project

A parallel e-graph implementation for QF_UF (equality + uninterpreted functions) SMT solving. Built for 15-852 (Parallel Algorithms) at CMU.

## Architecture

```
src/
  lib.rs         -- Public API: solve_with_mode(input, parallel) -> Sat/Unsat
  egraph.rs      -- EGraph struct: hashcons, congruence closure, rebuild
  unionfind.rs   -- ConcurrentUnionFind: lock-free atomic DSU (Listing 3)
  process.rs     -- SMT-LIB term -> EGraph translation
  main.rs        -- CLI: parallel-egraph [--parallel|-p] <file.smt2>
benches/
  uf_ycsb.rs     -- YCSB-style microbenchmark for the concurrent union-find
tests/
  regression.rs  -- 16 QF_UF .smt2 regression tests
```

## Key design decisions

### Concurrent union-find (unionfind.rs)

The union-find uses `Vec<AtomicU32>` where each slot stores either:
- **Rank** (bit 31 set): node is a root, lower 31 bits = rank
- **Parent pointer** (bit 31 clear): node points to parent

Implements the algorithm from Jayanti & Tarjan's "Concurrent Disjoint Set Union" (Listing 3):
- `find(&self, u)` -- recursive with CAS-based path compression. Returns `(root, rank)`. Recursion depth is O(log n) due to rank-based union.
- `union(&self, u, v)` -- CAS retry loop. Higher rank wins; equal ranks broken by node ID (smaller merges into larger), winner's rank incremented best-effort.
- `same_set(&self, u, v)` -- linearizable: finds both roots, then verifies the first is still a root before returning false.

All three take `&self` (not `&mut self`) -- mutation is via atomics. `make_set(&mut self)` is the only mutating method (pushes to the Vec), and must be called single-threaded during the add phase.

### EGraph (egraph.rs)

The EGraph **always** uses `ConcurrentUnionFind` internally (even in sequential mode). This means:
- `find(&self)` and `equiv(&self)` do **not** require `&mut self` -- they use atomic CAS for path compression. This is a change from the typical e-graph API.
- `canonicalize(&self)` is also `&self` for the same reason.
- `add(&mut self)` and `merge(&mut self)` still need exclusive access for hashcons/class metadata.

**Parallel mode** (`EGraph::new_parallel()`):
- `parallel_merge_all(&mut self, pairs)` runs union-find operations in parallel via rayon, then sequentially reconciles class/parent metadata and populates the worklist.
- Rebuild (congruence closure) is always sequential -- it modifies hashcons and triggers recursive merges.

**Solver flow** (lib.rs):
1. Parse SMT-LIB, type-check (yaspar crate)
2. Add all terms to e-graph (sequential)
3. Merge equalities -- sequential (`merge` one at a time) or parallel (`parallel_merge_all` batch)
4. `rebuild()` -- sequential congruence closure
5. Check disequalities against the final equivalence classes

### Parallel rebuild (egraph.rs, as of latest implementation)

**Status: IMPLEMENTED** via batch-parallel round-based congruence closure (ChatGPT document algorithm).

The sequential `rebuild()` worklist algorithm is replaced with a fully-parallel round-based version when `self.parallel == true`. Key changes:

- **Changed from worklist to changed flags**: `Vec<bool> changed` indexed by class ID, updated in parallel after each round's merges.
- **Changed from parent-list drain to flat-array filter**: Build a flat array of all non-leaf enodes; each round, `par_iter().filter()` identifies nodes whose children have changed roots. Eliminates sequential parent-list traversal.
- **Fully parallel round structure**:
  1. `par_iter().filter()` -- frontier gathering
  2. `par_iter().map()` -- canonicalization (lock-free `find()` safe to call concurrently)
  3. `par_sort_unstable()` -- semisort by signature (requires `Ord` on ENode)
  4. [sequential scan] -- merge candidate extraction (O(frontier) proportional to parallel work)
  5. `par_iter().for_each()` -- parallel union application (lock-free CAS)
  6. `par_iter()` + loop -- parallel changed-flag reset and update
- **Deferred metadata reconciliation**: `classes`, `parents`, and `hashcons` are reconciled once at the end when fixpoint is reached, not after each merge.

Benefits:
- Exposes parallelism in the congruence closure phase (the main bottleneck).
- No concurrent HashMaps needed; metadata uses `find()` on stale keys (safe because find is lock-free).
- Parallel frontier scanning eliminates sequential parent-list bookkeeping.

### What is NOT parallelized

- `add()` / term insertion -- inherently sequential (needs hashcons lookup for dedup).
- Term parsing and type-checking (yaspar crate).

### Performance observations

- The atomic overhead makes the concurrent UF slightly slower than a plain `Vec<u32>` sequential UF in single-threaded mode (~5-10x per operation due to memory barriers).
- The concurrent UF scales well with threads (near-linear up to 8 cores on YCSB benchmarks).
- The UF is naturally contention-resistant: under zipfian skew, hot keys merge early, making subsequent operations on them near-free (find returns immediately when roots match).
- For the full SMT solver, the UF is not the bottleneck -- rebuild and hashcons lookups dominate.

## Benchmarks

### YCSB microbenchmark (benches/uf_ycsb.rs)

Run with `cargo bench --bench uf_ycsb`. Experiments:
- `scalability` -- workloads A-E (varying read/write mix) across 1-8 threads
- `contention` -- vary zipfian skew parameter z (0.01 to 0.99) across thread counts
- `sizing` -- vary element count (1K to 1M) across thread counts
- `update-mix` -- vary union/find ratio (0% to 100%) across thread counts

Output is a formatted table to stdout. Constants at the top of the file control `MAX_THREADS`, `OPS_PER_THREAD`, `TRIALS`, and `DEFAULT_SIZE`.

### Synthetic SMT benchmarks (synthetic_benchmarks/)

Generated by `gen_bench.py`. Four families: chain (sequential), grid (n^2 congruences), cube (n^3), exp (2^n cascade). See README.md for details.

### Real benchmarks (benchmarks/)

QF_UF instances from SMT-COMP via Zenodo.

## Dependencies

- `yaspar` / `yaspar-ir` -- SMT-LIB 2 parser and typed IR
- `rayon` -- parallel iteration for batch merges and YCSB benchmark
- `rand` (dev) -- RNG for YCSB benchmark

## Testing

`cargo test` runs 16 unit tests (egraph + unionfind, including parallel mode tests) and 16 SMT regression tests.

## Testing

Unit tests and regression tests now cover both sequential and parallel modes:
- 16 unit tests (egraph + unionfind)
- 16 sequential SMT2 regression tests
- 16 parallel SMT2 regression tests (identical expected results as sequential)

All 32 tests pass, verifying correctness of the parallel rebuild implementation.

## Likely next steps

1. Benchmark parallel rebuild on synthetic benchmarks (cube_n80, exp_n20, etc.) to measure speedup vs sequential
2. Add timing instrumentation to measure per-round frontier size and merge count
3. Add a sequential baseline UF (plain `Vec<u32>`) behind a feature flag for fair benchmarking of atomic overhead
4. Profile on real SMT-COMP instances to identify remaining bottlenecks (parsing, term insertion, etc.)
