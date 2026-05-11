# ParallelEgraph (C++/parlay) — design and correctness

A parallel e-graph: lock-free concurrent union-find + several closure
algorithms (BSP-frontier, depth-stratified BSP, filter-based rounds with
round-stamp tracking) plus three sequential baselines for correctness
checking. Originally a Rust prototype; now C++ with parallel primitives
from [parlaylib](https://github.com/cmuparlay/parlaylib) and
`parlay::sequence` throughout the hot path.

The codebase has been deliberately constrained to what the closure
benchmark needs: a single bulk e-graph constructor, a small set of
closure algorithms, and a benchmark harness comparing them across three
workload corpora (`closure_compare`'s random-uniform DAGs,
`synthetic_bench`'s polynomial cascades, and `smt_bench`'s SMT-LIB
inputs).

## Layout

```
CMakeLists.txt
DESIGN.md                                 — this file
gen_bench.py                              — generate synthetic QF_UF inputs
include/parallel_egraph/
  fxhash.hpp                              — FxHash (rustc-hash style)
  unionfind.hpp                           — ConcurrentUnionFind +
                                            SequentialUnionFind
  smtlib.hpp                              — QF_UF SMT-LIB 2 parser
  smt_to_egraph.hpp                       — SMT-LIB -> e-graph builder
  egraph.hpp                              — EGraph + ENode + sig helpers
                                            + ctor tags (default / filter / topo)
  detail.hpp                              — CanonEntry, SemisortTimings,
                                            internal helper signatures
  dnc_union.hpp                           — divide-and-conquer per-bucket union
                                            (templated on Bucket and
                                            optionally on the union method)
  semisort_common.hpp                     — semisort body
  semisort_sound.hpp                      — structural sigs_equal probe
src/
  unionfind.cpp                           — Concurrent / Sequential UF impl
                                            (incl. union_min_id, union_into)
  smtlib.cpp                              — recursive-descent parser
  egraph.cpp                              — sequential closures + parallel
                                            BSP + parallel topo + topo_iter
  egraph_filter.cpp                        — parallel filter-based closure (by-rank
                                            and MIN_ID variants)
  semisort.cpp                            — TU dispatcher: includes
                                            semisort_sound.hpp
  main.cpp                                — egraph-cc CLI
bench/
  closure_compare.cpp                     — random uniform DAGs (small/medium/
                                            large × depth=1/3)
  synthetic_bench.cpp                     — polynomial-family cascades
                                            (chain, grid, cube, quartic, quintic)
  smt_bench.cpp                           — sweeps a directory of .smt2 files
  component_bench.cpp                     — isolated phase timings
  irregular_bench.cpp                     — variable-arity workloads
  scripts/
    run_144core_topo.sh                   — full sweep driver (closure_compare,
                                            synthetic, scaling, eggcc; per-step
                                            timeouts; aggregator)
tests/
  unionfind_test.cpp                      — UF unit tests
  closure_test.cpp                        — algorithm-vs-oracle correctness
                                            (incl. cross-depth scenarios)
cc-benchmarks/                            — git submodule: 507-file eggcc
                                            (cc-benchmarks) corpus
synthetic_benchmarks/                     — gen_bench.py output (chain/grid/
                                            cube/quartic/quintic .smt2)
```

## Core data structures

```
Id = std::uint32_t

ConcurrentUnionFind   parlay::sequence<std::atomic<u32>>; high bit = rank flag
                      (root stores rank; internal node stores parent ptr).
                      Lock-free find/union (Jayanti & Tarjan listing 3).
                      Three union variants:
                        union_(u, v)        by-rank tie-break (higher id on ties)
                        union_min_id(u, v)  always swing higher-id slot to lower
                        same_set(u, v)      linearizable membership check
                      union_min_id returns bool: true on real merge,
                      false on already-same-class. Used by the
                      MIN_ID-linking variants.
                      Fixed capacity at construction; bulk_init(n) sets
                      size_ = n in one shot.

SequentialUnionFind   plain std::vector<u32>, same encoding. Path
                      compression in find_root. union_(u, v) by-rank,
                      union_into(dying, survivor) splices unconditionally
                      (no rank update — used by smaller-into-larger
                      merging in sequential_close_dst).

EGraph<UF>            templated on the union-find type so the
                      sequential and parallel paths share one ENode +
                      Signature implementation.

  uf_                 UF instance (Concurrent or Sequential).
  nodes_              parlay::sequence<ENode>.
                      Class-id-indexed: nodes_[i] is the ENode for class i.
                      Constructed from caller-supplied DAG-ordered input
                      (every child id < parent id). No hashcons dedup —
                      caller is responsible for uniqueness. Immutable
                      after construction.
  parents_            parlay::sequence<parlay::sequence<Id>>
                      Inverted child→parent index; populated only by the
                      default ctor (BSP flavor). parents_[c] = list of
                      class ids whose ENode has c among its children.
                      Used by parallel_parents (BSP frontier walk),
                      sequential_close_nelson (worklist), and
                      sequential_close_dst (smaller-into-larger).
  last_marked_        parlay::sequence<std::atomic<u64>>
                      Populated only by the filter-tagged ctor. Per-class
                      round-stamps used by parallel_filter
                      to filter "dirty" terms each round.
  depth_buckets_      parlay::sequence<parlay::sequence<Id>>
                      Populated only by the topo-tagged ctor. Class ids
                      grouped by static node depth (longest path from any
                      leaf to the node in the input DAG; child < parent
                      ordering is required). Used by parallel_topo
                      and parallel_topo_iter.
```

Three ctor flavors select which auxiliary state gets populated:

```cpp
EGraph<UF>(nodes)               // default: parents_
EGraph<UF>(nodes, pe::filter)   // filter-based: last_marked_
EGraph<UF>(nodes, pe::topo)     // depth-stratified: depth_buckets_
```

Each closure path uses only its own auxiliary state, so the bench can
A/B compare without paying for the wrong-flavor setup.

## Closure algorithms

Nine algorithms, summarized in [src/egraph.cpp](src/egraph.cpp) and
[src/egraph_filter.cpp](src/egraph_filter.cpp). The table below is the
quickest way to navigate which one to use.

| algorithm | seq/par | sound? | notes |
|---|---|---|---|
| `sequential_close_nelson` | seq | sound | Original Nelson worklist baseline (signature table + parents_ frontier). Drains parents_ on first visit; **has its own latent bug on cross-depth inits** because re-pushed classes find their parents_ already empty. Kept as historical baseline. |
| `sequential_close_topo` | seq | **unsound** | Single forward pass over `nodes_` in DAG order. Per-class signature → hashmap; collisions trigger union. Order-dependent: only correct when input ordering puts canonical (lowest-id) members structurally first. Even with MIN_ID linking, fails on adversarial DAG orderings. Documented as failing test `seq_topo_adversarial`. |
| `sequential_close_topo_iter` | seq | sound | Wraps `sequential_close_topo` in a fixpoint loop with MIN_ID linking. Re-walks `nodes_` until a full pass yields no new unions. On Family C polynomial cascades converges in 2 passes (1 work + 1 verification); on irregular inputs needs more passes. |
| `sequential_close_dst` | seq | sound | Nelson-Oppen-style worklist + structural hashcons + smaller-into-larger merging (`union_into`). Seeds the hashcons with every node up front; each merge re-pends `parents_` of the dying class. Stale hashcons entries are tolerated (still semantically valid by congruence monotonicity). Correct on arbitrary inputs; expensive seed sweep on workloads with deep-cascade structure (`O(N log N)` due to parents migration). |
| `parallel_parents` | par | sound | BSP-frontier algorithm. Each round: parallel-apply pending unions, parallel-consolidate `parents_` from dead-roots into surviving-roots, semisort frontier by 96-bit signature hash, dnc_union per bucket, emit next-round work. Loops until frontier is empty. The reference for cross-depth correctness. |
| `parallel_topo` | par | **unsound** | Depth-stratified BSP: round d processes every class at depth d in parallel. Same intra-batch staleness bug as `sequential_close_topo` — in-round unions can change find_root of children referenced by other depth-d sigs that were already computed. Documented as failing test `par_topo_cross_depth`. Kept in the codebase as a reference/baseline; **not wired into any bench driver**. |
| `parallel_topo_iter` | par | sound | Sound iterated variant of `parallel_topo`. Same depth-stratified structure, but every union is MIN_ID and the entire depth walk is wrapped in a fixpoint loop. On Family C cascades converges in 2 rounds (1 work + 1 verification); the depth-stratified scheduling batches the entire cascade into one big parallel pass per depth, which beats the filter path on regular cascades. |
| `parallel_filter` | par | sound | Round-stamp dirty-filter algorithm using `last_marked_`. Each round: filter terms with at least one child whose root was recently rerooted (mark in {R-1, R}), semisort dirty by sig, dnc_union per bucket, stamp surviving roots with R. Loops until dirty is empty. By-rank linking. |
| `parallel_filter_min_id` | par | sound | Same as parallel_filter but every union uses `union_min_id`. Modest 5–15% improvement on regular cascades from the canonical-id-stable invariant; essentially neutral on irregular inputs. |

## Soundness story: cross-depth initial unions

The two unsound algorithms share a structural failure mode worth
documenting because the fix shaped the rest of the algorithm catalog.

**Setup.** Consider an e-graph with leaves `a, b, ta1, ta2, t` plus
`f(a), f(b), g(ta1, t), g(ta2, t)`, and initial unions
`{a = b, ta1 = f(a), ta2 = f(b)}`. The middle two are "cross-depth":
they equate a leaf class with a depth-1 (function-application) class.

The mathematical closure: `a = b ⇒ f(a) ≡ f(b)` by congruence, then
through the cross-depth inits `ta1 ≡ ta2`, then `g(ta1, t) ≡ g(ta2, t)`
by congruence on `t`.

**Why single-pass topo misses it.** Forward-walk processes `f(a)` and
`f(b)` first; they match and union into one class. But the g's were
processed earlier (or in parallel for `parallel_topo`) with sigs
referencing distinct find_roots for `ta1` and `ta2`. After the f-union,
those find_roots become equal — but the g sigs were already inserted
into the hashmap with stale values. No re-check fires.

**The fixpoint fix.** Wrap the depth walk in a loop that repeats until a
full pass yields zero unions. Caught congruences from pass `k` enable
new sig matches in pass `k+1`. Bounded by total class count.

**The MIN_ID structural fix.** With rank-based union, the canonical
representative of an equivalence class is unpredictable; sigs computed
across in-pass merges reference different ids. With MIN_ID linking, the
lowest-id member of every class is always the root, so on
DAG-ordered inputs where the structurally-canonical sub-DAG is emitted
first (Family C, by construction), single-pass topo is correct.

**Combined**: `sequential_close_topo_iter` and
`parallel_topo_iter` use both — MIN_ID linking + fixpoint loop —
so they are correct on arbitrary inputs (the loop catches anything MIN_ID
misses) and fast on regular cascades (MIN_ID makes 1 work pass suffice).

Empirical impact: on synthetic quintic-n=20 (12.8M classes, g-tree
depth 22), pre-MIN_ID `topo_iter` needed 18 rounds and ~107 s (one full
N-walk per cascade level). Post-MIN_ID: 2 rounds, ~3 s. Same workload at
144 cores with `parallel_topo_iter`: ~500 ms in 2 rounds.

## §1 `parallel_consolidate` ([src/egraph.cpp](src/egraph.cpp))

Used only by `parallel_parents` (the BSP-frontier path). The other
parallel algorithms either don't use `parents_` at all (topo, filter) or
use it differently (dst's smaller-into-larger).

### Why it exists

`parents_[c]` is keyed by the *add-time root* of the child class. After
`union(a, b)`, one of `a` or `b` becomes a non-root; whichever loses the
tie-break still owns its add-time `parents_` entries, but the new
root's slot does not. If a future round only iterates the *current*
root's slot, it misses every parent registered under the old, now-stale
id — exactly the soundness bug the `exp_n11` family of inputs exposed
in earlier implementations.

The fix migrates entries from non-root slots into the current root at
the start of every round.

### What it does

For every `c` in this round's work list whose current root differs (`c
!= root_of(c)`), append `parents_[c]` to `parents_[root_of_c]` and clear
`parents_[c]`. Subsequent rounds only need to read the root's slot.

### How it parallelizes

Three nested levels, all parlay-native:

```
1. (root_of_c, c) pairs for non-root c           parlay::map_maybe
2. groups[g] = (root, [cs sharing this root])    parlay::group_by_key
3. for each group (parallel — distinct root):
     parents[rep] = parlay::flatten(
         parlay::map(sources, [&](Id c) { return std::move(parents[c]); })
     );
```

### Race-freedom argument

* **Across groups**: each group's destination is a distinct root, so
  writes to `parents_[r]` never collide.
* **Within a group**: the cs are distinct (group_by_key keys are
  unique), and `parlay::flatten` allocates the destination buffer once
  and scatters in parallel internally — no contention.

Span: `O(log N)` per round.

## §2 `merge_and_collect_semisort` ([src/semisort.cpp](src/semisort.cpp))

The grouping primitive shared by every parallel BSP variant.

### What it does

Given a `canon` sequence (one entry per class in the round's frontier
— `(hash, root, secondary_hash | class_id)` where `hash =
sig_hash(node)` and `root = find_root(class_of_node)`), it merges every
set of class roots whose entries share a signature and returns the set
of touched class ids for the next round.

### Equality

Semisort groups by the 64-bit primary hash; within a same-hash run we
verify equality structurally with `sigs_equal` — the children-walk
comparison resolved through the UF. Sound under any hash function.
`CanonEntry` is 16 bytes: primary hash, root, and `class_id` (the index
into `nodes_` used to recover the ENode for the structural compare).

### Two output flavors

| function                       | output                                     | used by |
|--------------------------------|--------------------------------------------|---------|
| `merge_and_collect_semisort`   | returns next-round work (touched roots)    | `parallel_parents`, `parallel_filter*` |
| `apply_congruence_semisort`    | unions only; no return                     | `parallel_topo`, `parallel_topo_iter` |

The lean variant skips the per-group `parlay::map(values, .root)` and
the outer `parlay::flatten` — the topo path drives rounds from
precomputed `depth_buckets_` rather than reading next-round work, so
that work is wasted.

The lean variant uses a faster bucketing path: `parlay::integer_sort`
in place by primary hash, find run boundaries via
`parlay::tabulate + pack_index`, then `parlay::parallel_for` over the
runs. No keyed-pair allocation, no per-group `parlay::sequence<value>`
allocation.

### `dnc_union` and `dnc_union_with`

Per-bucket merging is divide-and-conquer:

```cpp
template <typename Bucket>
void dnc_union(Bucket& bucket, std::size_t lo, std::size_t hi,
               ConcurrentUnionFind& uf) {
  if (hi - lo <= 1) return;
  if (hi - lo <= dnc_cutoff()) {
    for (std::size_t i = lo + 1; i < hi; ++i)
      uf.union_(as_root(bucket[lo]), as_root(bucket[i]));
    return;
  }
  std::size_t mid = lo + (hi - lo) / 2;
  parlay::par_do([&] { dnc_union(bucket, lo, mid, uf); },
                 [&] { dnc_union(bucket, mid, hi, uf); });
  uf.union_(as_root(bucket[lo]), as_root(bucket[mid]));
}
```

`PE_DNC_CUTOFF` (default 16) tunes the leaf-call sequential threshold.

`dnc_union_with` is a templated variant that takes a caller-supplied
union method (lambda). Used by the MIN_ID-linking variants
(`parallel_filter_min_id`, `parallel_topo_iter`) to
substitute `union_min_id` for `union_` without duplicating the dnc
structure.

## Correctness invariants

For `parallel_parents` (BSP-frontier):

1. **`parents_[c]` always reflects "nodes whose direct child was `c` at
   add time."** Never overwritten in a way that loses entries; only
   moved (consolidation) or appended-to. After consolidation,
   `parents_[c]` may be empty, but its entries live in
   `parents_[root_of_c]`, so iterating roots covers everything.

2. **Every class id whose root changes in round R is in the work list of
   round R+1.** True because `merge_and_collect_semisort` returns every
   `.root` of every same-sig group entry — both endpoints of every
   union it issued. Lock-free union also covers transitive merges.

For `parallel_topo_iter` (depth-stratified iterated):

3. **Within each round of the depth walk, sigs are computed against a
   snapshot of the UF state.** `parlay::map` join-barrier separates the
   read phase from the write phase. Phase 1 only does path-compression
   CAS (idempotent w.r.t. find_root return values). Phase 2 does the
   unions.

4. **Outer fixpoint loop terminates.** Each iteration that fires any
   real union strictly reduces the number of distinct equivalence
   classes. So the loop runs at most `N` times in the worst case.

For `sequential_close_dst`:

5. **Stale hashcons entries are harmless.** A `(sig → pidx)` entry can
   become stale if `pidx`'s child roots change. A future structurally-
   matching insertion that hits the stale entry yields a valid merge by
   monotonicity of congruence (once two terms were congruent, they stay
   congruent). The new entry is also inserted, so no information is
   lost — just possibly redundant work.

## How to build and run

### Build

```
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Parlay and ankerl::unordered_dense are pulled by CMake's `FetchContent`.
C++20 required. `libjemalloc-dev` is required by default (Ubuntu/Debian:
`sudo apt-get install libjemalloc-dev`); see `USE_JEMALLOC` to opt out.

### CMake options

| option                | default | effect |
|-----------------------|---------|--------|
| `USE_JEMALLOC`        | `ON`    | Link against jemalloc to bypass glibc per-arena lock contention. ~1.2× speedup on closure_compare 16xl-d3 at T=192, ~1.8× on wide-shallow synthetic workloads. |
| `PE_SEMISORT_SOUND`   | `OFF`   | Switch the semisort equality check from secondary-hash to structural `sigs_equal`. Deterministic under any hash function but ~1.2× slower. `CanonEntry` stays at 16 B in either build. |

### Test

```
cd build && ctest --output-on-failure
```

Two suites:
- `unionfind_test` — lock-free concurrent UF (basic ops, transitive
  chain, parallel_for stress).
- `closure_test` — algorithm correctness against `parallel_parents` as
  oracle. **14 of 16 tests pass; two are intentional regression markers
  documenting known unsound algorithms** (`par_topo_cross_depth` and
  `seq_topo_adversarial`). Those two cases stay failing on purpose; the
  bench-driver script tolerates them and aborts only on unexpected
  failures.

### CLI

```
./build/egraph-cc <file.smt2>
```

Reads a QF_UF SMT-LIB 2 instance, builds the e-graph from the asserted
equalities and disequalities, runs `parallel_parents` on the equality
list, and prints `sat` or `unsat`.

### Generating synthetic inputs

```
python3 gen_bench.py <family> <n> [output_dir]
python3 gen_bench.py all <n>
python3 gen_bench.py sweep <n1> <n2> <step>
```

Five families: `chain` (O(n) sequential cascade), `grid` (n²), `cube`
(n³), `quartic` (n⁴), `quintic` (n⁵). Output defaults to
`synthetic_benchmarks/`.

## Benchmarks

Three drivers; each emits human-readable text by default and CSV under
`PE_BENCH_FORMAT=csv`. They share `PE_BENCH_HEADER=1` (emit CSV header)
and `PE_BENCH_SKIP_NELSON=1` (skip the original `nelson_seq` baseline).

### `closure_compare_bench`

Six random uniform-DAG workloads at depth 1 or 3:

| name    | leaves | nodes     | merges  | depth |
|---------|--------|-----------|---------|-------|
| small   | 1 000  | 10 000    | 2 000   | 1     |
| medium  | 10 000 | 200 000   | 20 000  | 1     |
| large   | 50 000 | 1 000 000 | 100 000 | 1     |
| deep-s  | 1 000  | 10 000    | 2 000   | 3     |
| deep-m  | 10 000 | 200 000   | 20 000  | 3     |
| deep-l  | 50 000 | 1 000 000 | 100 000 | 3     |

Initial unions are leaf-leaf only (half x↔x, half y↔y) — *no
cross-depth inits*, so the unsound algorithms happen to produce correct
results on this corpus.

Output columns:
```
nelson_seq | nelson_topo* | topo_iter | nelson_dst | par_parents | par_topo_iter | par_filter | par_filter_min
```
(`*` = unsound; kept for reference)

### `synthetic_bench`

Five polynomial families (`chain`, `grid`, `cube`, `quartic`, `quintic`)
at default sizes `n ∈ {5, 10, 20}`. Override via `PE_SYNTH_FAMILIES`,
`PE_SYNTH_NS`, `PE_SYNTH_D` (g-tree fan-in).

The polynomial families construct:
- 2n leaves (a-side and b-side)
- 2 × n^arity f-applications (one per multi-index, both sides)
- A balanced binary g-tree wrapper on each side (~m / (d-1) g-nodes
  where d = g-tree fan-in, default 2).

Initial unions: `a_i = b_i` for `i ∈ [0, n)` (leaf-leaf only).

The g-tree depth is `⌈log_d(n^arity)⌉`. For quintic n=20 with d=2,
that's 22 levels — the largest synthetic cascade we run, ~12.8M classes.

### `smt_bench`

Sweeps a directory of `.smt2` files (one bench per file). Filename
convention `<family>_n<N>_(sat|unsat).smt2` (matching `gen_bench.py`)
gets `family` and `n` columns; other names get blank `family`. Used for
the eggcc corpus at `cc-benchmarks/smt-grounded`.

### Tunables (env vars)

| var                       | values            | effect                                                       |
|---------------------------|-------------------|--------------------------------------------------------------|
| `PARLAY_NUM_THREADS`      | `1, 2, 4, …`      | parlay scheduler thread count (default = logical CPUs).      |
| `PE_BENCH_ONLY`           | `large` etc.      | closure_compare: run only the named workload.                |
| `PE_BENCH_SKIP_NELSON`    | `1`               | skip the `nelson_seq` baseline.                              |
| `PE_BENCH_FORMAT`         | `csv`             | emit one row per (workload, algorithm, trial).               |
| `PE_BENCH_HEADER`         | `1`               | emit the CSV header row (gated for appended runs).           |
| `PE_BENCH_TRIALS`         | `1, 2, …`         | override default trials (5 for closure_compare/synthetic, 11 for smt_bench). |
| `PE_BENCH_WARMUP`         | `0, 1, …`         | override default warmup count.                               |
| `PE_SYNTH_FAMILIES`       | `chain,quintic`   | synthetic_bench: comma-separated subset of families.         |
| `PE_SYNTH_NS`             | `5,10,20`         | synthetic_bench: comma-separated n values.                   |
| `PE_SYNTH_D`              | `2, 3, …`         | g-tree fan-in for polynomial families (default 2).           |
| `PE_SMT_TRIALS`           | `1, …`            | smt_bench: trials per file (default 11).                     |
| `PE_SMT_WARMUP`           | `0, …`            | smt_bench: warmup per file (default 3).                      |
| `PE_DNC_CUTOFF`           | `16` (default)    | sequential cutoff inside `dnc_union`.                        |
| `PE_TRACE`                | `1`               | per-round timing trace from `parallel_parents*` to stderr.      |
| `PE_TRACE_ITER`           | `1`               | per-pass union counts for `*_topo_iter` to stderr.           |

## The 144-core benchmark script

[bench/scripts/run_144core_topo.sh](bench/scripts/run_144core_topo.sh)
drives the full sweep on a many-core machine: closure_compare and
synthetic at T=144, strong-scaling sweeps across T ∈ {1, 2, 4, 8, 16,
32, 48, 64, 96, 128, 144, 192}, and the full 507-file eggcc corpus.

```bash
./bench/scripts/run_144core_topo.sh
```

Outputs land in `bench/results/topo144/`:

```
build_info.txt              git SHA, CPU info, NUMA topo, compiler
sanity.log                  closure_test results
closure_compare_144T.csv    closure_compare at T_max
synthetic_144T.csv          synthetic_bench at T_max
closure_scaling.csv         closure_compare across T
quintic20_scaling.csv       synthetic quintic-20 (12.8M classes) across T
eggcc_144T.csv              full 507-file cc-benchmarks/smt-grounded sweep
eggcc_summary_144T.txt      aggregate ratios + win rates by class-bucket
```

### Knobs

| env var               | default | effect |
|-----------------------|---------|--------|
| `T_MAX`               | `144`   | thread budget for headlines + scaling ceiling. |
| `NUMACTL`             | `numactl -i all` | wraps each parallel binary. Override with empty string to disable, or `'numactl --cpunodebind=0'` to pin. |
| `STEP_TIMEOUT`        | `30m`   | per-step timeout for non-eggcc bench invocations. Set `0` to disable. |
| `STEP_TIMEOUT_EGG`    | `1h`    | timeout for the full eggcc sweep. |
| `EGGCC_ONLY`          | `0`     | skip closure_compare + synthetic + scaling sweeps; run just eggcc + aggregator. Useful for debugging eggcc-specific regressions. |

The script uses `set -euo pipefail` but tolerates the two
intentionally-failing closure_test cases (it parses the test output and
aborts only on *unexpected* failures). The per-step `timeout` wrapper
prevents any single misbehaving workload from hanging the whole sweep.

### Aggregator output

The eggcc summary surfaces:
- Σ wallclock per algorithm.
- Speedup ratios for each parallel algorithm vs the best correct
  sequential (`nelson_topo_iter` vs `nelson_dst`).
- Per-file win rates (par_filter vs nelson_topo_iter, par_topo_iter vs
  par_filter, par_filter_min_id vs par_filter, etc.).
- Per-class-bucket table (<1K, 1-10K, 10-100K, ≥100K classes) showing
  par_filter_min_id ratio against the better of the two correct
  sequentials per file.

## Practical recommendations (current state)

| input shape | recommended sound algorithm |
|---|---|
| <10K classes, any structure | `nelson_topo_iter` (with MIN_ID) — sequential overhead-free. |
| 10K–500K, regular cascade (Family C, leaf-only inits) | `par_topo_iter` — depth-stratified scheduling beats filter on regular shapes. |
| 10K–500K, irregular (eggcc) | `par_filter_min_id` — dirty-filter handles imbalanced fanout. |
| ≥500K, regular | `par_topo_iter` decisively (`5–10×` over best sequential). |
| ≥500K, irregular | `par_filter` or `par_filter_min_id` (~5% spread between them). |
| arbitrary correctness with no parallel infrastructure | `nelson_dst` — sound, correct on any input, `O(N log N)` per merge with smaller-into-larger. Slower on Family C (deep cascade) than `topo_iter` but avoids the `O(N × depth)` worst case. |

The crossover for parallel-vs-sequential at 144 cores sits around
**30K–100K classes** depending on workload shape. Below that, sequential
wins on overhead alone; above, parallel scales but framework cost is
non-trivial and the choice between filter and depth-stratified depends
on cascade regularity.
