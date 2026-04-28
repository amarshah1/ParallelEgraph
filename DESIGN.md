# ParallelEgraph (C++/parlay) — design and correctness

A parallel e-graph: lock-free concurrent union-find + bulk-synchronous-
parallel (BSP) congruence closure. Originally a Rust prototype; now C++
with parallel primitives from [parlaylib](https://github.com/cmuparlay/parlaylib)
and `parlay::sequence` throughout the hot path. The codebase is trimmed to
exactly what the closure algorithm benchmark needs.

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
  egraph.hpp                              — EGraph + ENode + sig helpers
src/
  unionfind.cpp
  smtlib.cpp                              — recursive-descent parser
  egraph.cpp                              — add, parallel_close,
                                            sequential_close_nelson,
                                            parallel_consolidate,
                                            merge_and_collect_semisort,
                                            dnc_union
  main.cpp                                — egraph-cc CLI driver
bench/
  closure_compare.cpp                     — closure-vs-baseline bench
tests/
  unionfind_test.cpp                      — UF unit tests
  regression_test.cpp                     — every examples/regression/*.smt2
                                            in parser→close mode
examples/
  regression/                             — 17 hand-written QF_UF tests
  synthetic/                              — gen_bench.py output (gitignored)
```

## Core data structures

```
ConcurrentUnionFind   std::vector<std::atomic<u32>>; high bit = rank flag.
                      find/union/same_set are *non-const* (CAS writes).
                      Fixed capacity at construction. make_set() bumps
                      a single-threaded counter; bulk_init(n) sets the
                      counter to n in one shot.

EGraph
  nodes_              parlay::sequence<pair<ENode, Id>>
                      Flat array of every non-leaf e-node, in add order.
                      Append-only during build, immutable during closure.

  parent_index_       parlay::sequence<parlay::sequence<Id>>
                      parent_index_[c] = list of indices into nodes_ whose
                      direct child was c at add time. Mutated only by
                      parallel_consolidate.

  uf_                 ConcurrentUnionFind, fixed capacity = #classes.

  hashcons_           std::unordered_map<ENode, Id, ENodeHash>
                      Sequential dedup during add(); empty after bulk_init.
```

`nodes_`, `parent_index_`, and `uf_` are sized to a known upper bound at
`EGraph` construction. No dynamic growth in the hot path.

### Two ways to populate the e-graph

1. **`add(node)` (sequential)** — canonicalize, dedupe via hashcons,
   `make_set` on miss, append to `nodes_`, push the node-idx into each
   child's `parent_index_` slot. Used by the CLI driver
   ([src/main.cpp](src/main.cpp)) where one node arrives at a time from
   the parser.
2. **`bulk_init(parlay::sequence<ENode>)` (parallel)** — when every node
   is known up-front and uniqueness is guaranteed by the caller. Skips
   the hashcons map; all of `uf_` / `nodes_` / `parent_index_` are
   constructed in one parallel pass:
   - `uf_.bulk_init(n)` sets `size_ = n`.
   - `parlay::scan` over `is_nonleaf[]` assigns each non-leaf its
     `node_idx` in `nodes_`.
   - `parlay::flatten(parlay::tabulate(...))` emits the
     `(child_class, node_idx)` pairs for every child of every non-leaf.
   - `parlay::group_by_index` groups those pairs by child class →
     `parent_index_` directly.
   - `parlay::parallel_for` scatters non-leaf ENodes into `nodes_`.

   Used by the closure benchmark ([bench/closure_compare.cpp](bench/closure_compare.cpp))
   to construct ~1M-node workloads without paying for a sequential
   `add()` loop.

## Closure algorithms

Two paths exposed:

| API                        | description                                                                  |
|----------------------------|------------------------------------------------------------------------------|
| `EGraph::sequential_close_nelson(initial_unions)` | Nelson-style sequential baseline (signature table + worklist), used by the bench's reference column. |
| `EGraph::parallel_close(initial_unions)`          | BSP round-based parallel closure. The vehicle for everything that follows.   |

`parallel_close`'s outer skeleton:

```
apply initial_unions in parallel
work := flatten initial_unions
loop {
  work := dedup(work)
  roots := map(c -> uf.find_root(c)) over work
  parallel_consolidate(parent_index, work, roots)        // §1
  frontier := flatten(parent_index[r] for r in dedup(roots))
  canon := map(idx -> (sig_hash, idx, find_root(class))) over frontier
  next  := merge_and_collect_semisort(canon)             // §2
  work  := dedup(next)
}
```

---

## §1 `parallel_consolidate`  ([src/egraph.cpp](src/egraph.cpp))

### Why it exists

`parent_index_[c]` is keyed by the *add-time root* of the child class. After
`union(a, b)`, one of `a` or `b` becomes a non-root; whichever loses the
rank tiebreak still owns its add-time `parent_index` entries, but the new
root's slot does not. If a future round only iterates the *current* root's
slot, it misses every parent registered under the old, now-stale id —
exactly the soundness bug the `exp_n11` family of inputs exposed in
earlier implementations.

The fix migrates entries from non-root slots into the current root at the
start of every round.

### What it does

For every `c` in this round's work list whose current root differs (`c !=
root_of(c)`), append `parent_index[c]` to `parent_index[root_of_c]` and
clear `parent_index[c]`. Subsequent rounds only need to read the root's
slot.

### How it parallelizes

Three nested levels, all parlay-native:

```
1. (root_of_c, c) pairs for non-root c           parlay::map_maybe
2. groups[g] = (root, [cs sharing this root])    parlay::group_by_key
3. for each group (parallel — distinct root):
     sizes[i]   = parent_index[cs[i]].size()     parlay::tabulate
     offsets, total = scan(sizes)                parlay::scan
     dst.resize(old + total)                     one-shot, exclusive
     for each c in group (parallel):
       for each j in parent_index[c]:            parlay::parallel_for
                                                 (or sequential below
                                                  COPY_SEQ_CUTOFF=1024)
         dst[old + offsets[i] + j] = src[j]
       src.clear()
```

### Race-freedom argument

* **Across groups**: each group's destination is a distinct root, so writes
  to `parent_index[r]` never collide.
* **Within a group**: the cs are distinct (group_by_key keys are unique),
  source slots don't collide either, and the prefix-scan partitions the
  destination into disjoint offset ranges.
* **Resize**: happens exactly once per group, *before* any concurrent
  writes — the writes only touch already-allocated tail slots.

Span: `O(log N)` per round (dominated by `parlay::scan`).

---

## §2 `merge_and_collect_semisort`  ([src/egraph.cpp](src/egraph.cpp))

### What it does

Given a `canon` sequence (one entry per parent node in the frontier — `(h,
idx, root)` where `h = sig_hash(node)` and `root = find_root(class_of_node)`),
it merges every set of class roots whose entries share a sig and returns
the set of touched class ids for the next round.

### Why it's a semisort, not a sort

A *full* sort by `(hash, sig_cmp)` would also work, but parlay's semisort
primitive (`group_by_key` with custom hash + equal) is a closer fit: items
with equal key go to the same group, and we don't care about the order
within a group. Sample sort does extra work to maintain a global ordering
we'd immediately discard.

### Implementation

```cpp
auto hash_fn  = [](const CanonEntry& e) { return e.h; };
auto equal_fn = [&](const CanonEntry& a, const CanonEntry& b) {
  return a.h == b.h && sigs_equal(a.idx, b.idx, uf, nodes);
};
auto groups = parlay::group_by_key(keyed, hash_fn, equal_fn);

auto per_group = parlay::map(groups, [&](auto& kv) {
  auto& values = kv.second;
  if (values.size() < 2) return parlay::sequence<Id>{};
  dnc_union(values, 0, values.size(), uf);
  return parlay::tabulate(values.size(), [&](size_t i) {
    return values[i].root;
  });
});
return parlay::flatten(per_group);
```

#### 2a. Custom hash + equality

`parlay::group_by_key(R, hash, equal)` is parlay's hash-distribute-then-
equality-resolve grouping primitive. We feed it:

* `hash` = `sig_hash` (already cached in `canon[i].h`, so this is a memory
  read).
* `equal` = `sigs_equal` (`op` match plus `find_root(children)` match
  componentwise).

Hash collisions at 64 bits are filtered by `equal`, so groups are *exact*:
every entry in a group has the same signature. No probabilistic
unsoundness, no in-bucket sort needed.

#### 2b. Within-group union

Two strategies, env-selectable via `PE_UNION_STYLE`:

* **D&C** (default; `dnc_union` recursive `parlay::par_do`): the union
  tree has depth `O(log n)` and `n - 1` total unions. Sibling unions at
  the bottom of the tree target *disjoint* class ids — no CAS contention.
  Below `PE_DNC_CUTOFF` (default 16), falls through to a sequential
  pairwise sweep.
* **`adjacent`** (`PE_UNION_STYLE=adjacent`): a flat
  `parlay::parallel_for(i, ...) { union(bucket[i], bucket[i+1]); }`.
  Trivially simple; every adjacent pair shares an endpoint with its
  neighbour, so the lock-free CAS retries more under load. Empirically
  wins on `large` (~10–15% faster) because group sizes are small enough
  that the contention is bounded.

### Touched ids

Every entry's `.root` (the *pre-round* root, captured at canon
construction) goes into the touched output. The next round picks them up,
the consolidation step migrates their `parent_index` slots into whatever
root absorbed them, and the cycle continues.

---

## Correctness invariants

1. **`parent_index[c]` always reflects "nodes whose direct child was `c`
   at add time."** Never overwritten in a way that loses entries; only
   moved (consolidation) or appended-to. After consolidation,
   `parent_index[c]` may be empty, but its entries live in
   `parent_index[root_of_c]`, so iterating roots covers everything.

2. **Every class id whose root changes in round R is in the work list of
   round R+1.** True because `merge_and_collect_semisort` returns *every
   `.root` of every same-sig group entry* — both endpoints of every union
   it issued. The lock-free UF's `union` also covers any chain that
   transitively merges classes, so after a round all touched classes'
   new roots show up.

(1) and (2) together rule out the soundness bug in which a non-root `c`
whose root just changed gets orphaned: it ends up in the work list, so
consolidation migrates `parent_index[c]` into the new root, so the root
iteration finds every parent that needs re-canonicalization.

---

## How to build and run

### Build

```
cmake -B build -S .
cmake --build build
```

Parlay is pulled by CMake's `FetchContent`. C++20 required.

### Test

```
cd build && ctest --output-on-failure
```

Two suites:
- `unionfind_test` — lock-free concurrent UF (basic single-thread ops,
  transitive chain, a 1000-element `parlay::parallel_for` union check).
- `regression_test` — runs every `examples/regression/*.smt2` through
  the parser → e-graph → `parallel_close` pipeline; verifies sat/unsat
  against the `_sat`/`_unsat` filename suffix.

### CLI

```
./build/egraph-cc <file.smt2>
```

Reads a QF_UF SMT-LIB 2 instance, builds the e-graph from the asserted
equalities and disequalities, runs `parallel_close` on the equality list,
and prints `sat` or `unsat`. The parser handles `set-logic`,
`declare-sort/fun/const`, `assert (= a b)`, `assert (not (= a b))`, and
`check-sat`. Boolean connectives (`and`/`or`/`=>`/`xor`/`ite`/`distinct`)
are not modeled — easy to add back to `Term::Kind` and `parse_term` when
a downstream consumer needs them.

### Generating synthetic inputs

```
python3 gen_bench.py <family> <n> [output_dir]
python3 gen_bench.py all <n>
python3 gen_bench.py sweep <n1> <n2> <step>
```

Four families: `chain` (O(n) sequential cascade), `grid` (n²), `cube`
(n³), `exp` (2ⁿ exponential cascade through layered functions). Default
output is `examples/synthetic/`. Each file has a known UNSAT disequality
requiring the entire formula to derive.

### Benchmark

```
./build/closure_compare_bench
```

Six workloads, comparing `sequential_close_nelson` vs `parallel_close`:

| name    | leaves | nodes     | merges  | depth |
|---------|--------|-----------|---------|-------|
| small   | 1 000  | 10 000    | 2 000   | 1     |
| medium  | 10 000 | 200 000   | 20 000  | 1     |
| large   | 50 000 | 1 000 000 | 100 000 | 1     |
| deep-s  | 1 000  | 10 000    | 2 000   | 3     |
| deep-m  | 10 000 | 200 000   | 20 000  | 3     |
| deep-l  | 50 000 | 1 000 000 | 100 000 | 3     |

Output columns:
```
name    leaves    nodes    merges  | nelson_seq |  par_close   par_spd
```

`nelson_seq` is the median of 11 trials (3 warmup) of
`EGraph::sequential_close_nelson`; `par_close` the same for
`EGraph::parallel_close`; `par_spd = nelson_seq / par_close`.

Workload construction itself is also fully parallel: a single
`parlay::tabulate` over `total = 2*n_leaves + n_nodes` produces every
ENode (leaves and per-level function nodes), then `EGraph::bulk_init`
builds `uf_` / `nodes_` / `parent_index_` in one parallel pass. Random
draws use `parlay::random_generator` with `gen[i]` to fork an
independent sub-generator per tabulate index — no sequential RNG state.
The 14 builds per workload (3 warmup + 11 trials) are why this matters:
without the parallel build, build dominated wall time on `large` and
`deep-l`.

#### Tunables (env vars)

| var                       | values            | effect                                                       |
|---------------------------|-------------------|--------------------------------------------------------------|
| `PARLAY_NUM_THREADS`      | `1, 2, 4, 8, 12…` | parlay scheduler thread count (default = logical CPUs).      |
| `PE_BENCH_ONLY`           | `large` etc.      | Run only the named workload (faster iteration during sweeps).|
| `PE_BENCH_SKIP_NELSON`    | `1`               | Skip the `sequential_close_nelson` baseline.                 |
| `PE_UNION_STYLE`          | `adjacent`        | Use flat `parallel_for` inside groups instead of D&C.        |
| `PE_DNC_CUTOFF`           | `16` (default)    | Group-size threshold below which `dnc_union` falls through to a sequential pairwise sweep. |
| `PE_TRACE`                | `1`               | Per-round timing trace from `parallel_close` to stderr.      |

Examples:

```
# Scaling sweep on `large` only:
for T in 1 2 4 8 12; do
  PARLAY_NUM_THREADS=$T PE_BENCH_ONLY=large ./build/closure_compare_bench \
    | grep '^large' | sed "s/^/T=$T  /"
done

# DNC vs adjacent head-to-head:
for s in '' adjacent; do
  PE_UNION_STYLE=$s PE_BENCH_ONLY=large PE_BENCH_SKIP_NELSON=1 \
    ./build/closure_compare_bench | grep '^large'
done
```

## Eval (reproducing the plots)

The bench infra under [bench/scripts/](bench/scripts/) drives sweeps and
emits plots. CSV output from `closure_compare_bench` is opt-in
(`PE_BENCH_FORMAT=csv`); the human-readable text format is unchanged
when the env var is unset.

```
cmake --build build
pip install -r bench/requirements.txt
python3 bench/scripts/run.py all
python3 bench/scripts/plot.py all
```

Outputs land in `bench/results/` (gitignored). `run.py` has
subcommands you can run individually — `strong-scaling`,
`workload-sweep`, `trace`, `components`, `smt`, `all` — each writes one
CSV; `plot.py` consumes them.

| plot                           | what it shows                                                                              |
|--------------------------------|--------------------------------------------------------------------------------------------|
| `fig_strong_scaling.png`       | speedup vs PARLAY_NUM_THREADS, one line per workload, with an ideal `y=x` reference        |
| `fig_wallclock.png`            | par_close wallclock vs threads, log-log; companion to the speedup plot                     |
| `fig_trace_rounds_<w>_T<n>.png`| per-round time broken into consolidate / frontier / semisort (stacked bar)                 |
| `fig_components.png`           | round-0 `parallel_consolidate` and `merge_and_collect_semisort` wallclock vs threads       |
| `fig_workload_depth.png`       | par_close ms vs depth, one line per merge_frac                                              |
| `fig_workload_merge_frac.png`  | par_close ms vs merge_frac, one line per depth                                              |
| `fig_smt_scaling.png`          | par_close ms vs `n` for the four `gen_bench.py` families (`chain`, `grid`, `cube`, `exp`)   |

Two new bench binaries:

- `./build/component_bench` — times `parallel_consolidate` and
  `merge_and_collect_semisort` in isolation on round-0 mid-state.
  CSV mode under `PE_BENCH_FORMAT=csv`; workload via `PE_COMPONENT_WORKLOAD`.
- `./build/smt_bench <dir-or-file> ...` — runs `parallel_close` over
  every `.smt2` under the given paths, emitting CSV. Filename pattern
  `<family>_n<N>_(sat|unsat).smt2` (matching `gen_bench.py`) gets
  `family` and `n` columns; other names get blank `family`.

Both share `PE_BENCH_HEADER=1` (emit CSV header) and
`PE_BENCH_SKIP_NELSON=1` (skip the sequential baseline).
