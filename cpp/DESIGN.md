# ParallelEgraph C++/parlay port — design and correctness

A 1:1 port of the Rust e-graph in [src/](../src/) to C++/parlay, intended as
the scaling vehicle going forward. The algorithm is the same — lock-free
union-find + bulk-synchronous-parallel (BSP) congruence closure — but the
parallel primitives come from [parlaylib](https://github.com/cmuparlay/parlaylib)
instead of rayon, and the data structures are `parlay::sequence` throughout
the hot path.

## Layout

```
cpp/
  CMakeLists.txt
  include/parallel_egraph/
    fxhash.hpp     — FxHash (rustc-hash style) for ENode / sig hashing
    unionfind.hpp  — ConcurrentUnionFind (Jayanti & Tarjan Listing 3) +
                     SequentialUnionFind
    smtlib.hpp     — recursive-descent QF_UF parser
    egraph.hpp     — EGraph + ENode + signature helpers
    solve.hpp      — solve_with_mode / solve_timed / SolveResult
  src/
    unionfind.cpp
    smtlib.cpp
    egraph.cpp     — add, merge, all closure variants, helpers
    solve.cpp
    main.cpp
  bench/
    rebuild_compare.cpp       — port of benches/rebuild_compare.rs
  tests/
    unionfind_test.cpp        — UF unit tests
    regression_test.cpp       — every tests/*.smt2 in seq + par
    parallel_close_test.cpp   — exp_n11 via parallel_close directly
  scripts/
    cross_validate.sh         — Rust vs C++ diff across all variants
```

## Core data structures

```
ConcurrentUnionFind   std::vector<std::atomic<u32>>; high bit = rank flag.
                      find/union/same_set are *non-const* (CAS writes).
                      Fixed capacity at construction (no make_set growth).

EGraph
  nodes_              parlay::sequence<pair<ENode, Id>>
                      Flat array of every non-leaf e-node, in add order.
                      Append-only during build, immutable during closure.

  parent_index_       parlay::sequence<parlay::sequence<Id>>
                      parent_index_[c] = list of indices into nodes_ whose
                      direct child was c at add time. Mutated only by
                      parallel_consolidate.

  changed_            parlay::sequence<atomic<bool>>
                      Per-class "needs visiting" flag for the rebuild
                      variants. Drained at the start of each round.

  uf_                 ConcurrentUnionFind, fixed capacity = #classes.
```

`nodes_`, `parent_index_`, `changed_`, and `uf_` are all sized to a known
upper bound at `EGraph` construction (the solver pre-walks the AST to count
total subterms; the bench knows leaves+depth ahead of time). No dynamic
growth in the hot path.

## Closure algorithms

Three BSP variants, all sharing the same outer skeleton:

```
loop {
  collect changed classes / take work list
  parallel_consolidate(parent_index, work, roots)         // see §1
  frontier = flatten(map(unique_roots, &parent_index[r]))
  canon    = map(frontier, |i| (sig_hash, i, find_root(class_of i)))
  next     = merge_and_collect_semisort(canon)            // see §2
  apply: changed[id] = true / work = next
}
```

| variant                       | how `next` is computed                                                                                                                |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `parallel_rebuild_semisort`   | `merge_and_collect_semisort` (default; `PE_REBUILD` unset)                                                                            |
| `parallel_rebuild_sort`       | global `parlay::sort_inplace` with `(hash, sig_cmp)` comparator + adjacent-pair scan applying unions (`PE_REBUILD=sort`)              |
| `parallel_close`              | takes explicit `initial_unions: parlay::sequence<pair<Id,Id>>`, drives the work list directly; uses `merge_and_collect_semisort` too. |

`sequential_close_nelson` is the Nelson-style sequential baseline used by
the benchmark: a `SequentialUnionFind`, a `unordered_map<u64, vector<u32>>`
signature table, and a stack-based worklist. It's not on the parallel hot
path.

---

## §1 `parallel_consolidate`  ([cpp/src/egraph.cpp:257](src/egraph.cpp#L257))

### Why it exists

`parent_index_[c]` is keyed by the *add-time root* of the child class. After
`union(a, b)`, one of `a` or `b` becomes a non-root; whichever loses the
rank tiebreak still owns its add-time `parent_index` entries, but the new
root's slot does not. If a future round only iterates the *current* root's
slot, it misses every parent registered under the old, now-stale id —
exactly the soundness bug `exp_n11_unsat.smt2` exposed in earlier
implementations.

The fix migrates entries from non-root slots into the current root at the
start of every round.

### What it does

For every `c` in this round's work list whose current root differs (`c !=
root_of(c)`), append `parent_index[c]` to `parent_index[root_of_c]` and
clear `parent_index[c]`. Subsequent rounds only need to read the root's
slot.

### How it parallelizes

Three nested levels, all parlay-native, no user-authored sequential loop:

```
1. (root_of_c, c) pairs for non-root c           parlay::map_maybe
2. groups[g] = (root, [cs sharing this root])    parlay::group_by_key
3. for each group (parallel — distinct root):
     sizes[i] = parent_index[cs[i]].size()       parlay::tabulate
     offsets, total = scan(sizes)                parlay::scan
     dst.resize(old + total)                     one-shot, exclusive
     for each c in group (parallel):
       for each j in parent_index[c] (parallel): parlay::parallel_for
         dst[old + offsets[i] + j] = src[j]
       src.clear()
```

### Race-freedom argument

* **Across groups**: each group's destination is a distinct root, so writes
  to `parent_index[r]` never collide.
* **Within a group**: the cs are distinct (group_by_key keys are unique),
  so source slots don't collide either, and the prefix-scan partitions the
  destination into disjoint offset ranges.
* **Resize**: happens exactly once per group, *before* any concurrent
  writes — the writes only touch already-allocated tail slots.

Span: `O(log N)` per round (dominated by `parlay::scan`).

### Why we still need it with fixed nodes

Even if we drop incremental `add()` and build the e-graph in one bulk pass,
closure-time merges still change roots. Consolidation is about
"merges-after-build" not "build itself," so the simplification doesn't
eliminate it. (An alternative — scan all `[0, n)` each round filtering by
`find_root(c) ∈ touched_roots` — works, costs `O(n)` per round vs
`O(|work|)`. Worth it only when `|work|` ≈ `n`.)

---

## §2 `merge_and_collect_semisort`  ([cpp/src/egraph.cpp:333](src/egraph.cpp#L333))

### What it does

Given a `canon` sequence (one entry per parent node in the frontier — `(h,
idx, root)` where `h = sig_hash(node)` and `root = find_root(class_of_node)`),
it merges every set of class roots whose entries share a sig and returns
the set of touched class ids for the next round.

### Why it's a semisort, not a sort

A *full* sort by `(hash, sig_cmp)` would also work — that's what
`parallel_rebuild_sort` does — but parlay's semisort primitive is a closer
fit for what we actually need: items with equal key go to the same group,
and we don't care about the order within a group. Sample sort does extra
work to maintain a global ordering we'd immediately discard.

### Implementation

Two parlay primitives + the divide-and-conquer union helper:

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

Two design points worth flagging:

#### 2a. Custom hash + equality

`parlay::group_by_key(R, hash, equal)` is parlay's hash-distribute-then-
equality-resolve grouping primitive. We feed it:

* `hash` = `sig_hash` (already cached in `canon[i].h` so this is a memory
  read, not a recompute).
* `equal` = `sigs_equal` (`op` match plus `find_root(children)` match
  componentwise).

Hash collisions at 64 bits are filtered by `equal`, so groups are *exact*:
every entry in a group has the same signature. No probabilistic
unsoundness, no in-bucket sort needed (compare with the older Rust
`parallel_close`, which sorts by `(hash, idx)` lex and accepts the rare
collision-induced miss).

#### 2b. D&C union per group

```cpp
template <typename Bucket>
void dnc_union(Bucket& bucket, size_t lo, size_t hi, ConcurrentUnionFind& uf) {
  if (hi - lo <= 1) return;
  if (hi - lo == 2) { uf.union_(bucket[lo].root, bucket[lo+1].root); return; }
  size_t mid = lo + (hi - lo) / 2;
  parlay::par_do([&] { dnc_union(bucket, lo, mid, uf); },
                 [&] { dnc_union(bucket, mid, hi, uf); });
  uf.union_(bucket[lo].root, bucket[mid].root);
}
```

Within a same-sig group of `n` entries, the union tree has depth `O(log n)`
and `n - 1` total unions. Crucially, sibling unions at the bottom of the
tree target *disjoint* class ids — no CAS contention. An adjacent-pair scan
applying unions in parallel (`union(e0,e1)`, `union(e1,e2)`, …) does the
same number of unions but every pair shares an endpoint, so the lock-free
UF spends most of its time on CAS retries.

### Touched ids

Every entry's `.root` (which is the *pre-round* root, captured at canon
construction) goes into the touched output. The next round picks them up,
the consolidation step migrates their `parent_index` slots into whatever
root absorbed them, and the cycle continues.

---

## Correctness invariants

The two non-obvious invariants the parallel pipeline maintains:

1. **`parent_index[c]` always reflects "nodes whose direct child was `c`
   at add time."** Never overwritten in a way that loses entries; only
   moved (consolidation) or appended-to. After consolidation,
   `parent_index[c]` may be empty, but its entries live in
   `parent_index[root_of_c]`, so iterating roots covers everything.

2. **Every class id whose root changes in round R is in the work list of
   round R+1.** True because `merge_and_collect_semisort` returns *every
   `.root` of every same-sig group entry* — i.e., both endpoints of every
   union it issued. The lock-free UF's union also covers any chain that
   transitively merges classes (it acts on current roots), so after a
   round all touched classes' new roots show up.

The combination of (1) and (2) is what rules out the `exp_n11_unsat` class
of soundness bugs: a non-root `c` whose root just changed will be in `work`
next round, so consolidation migrates `parent_index[c]` into the new root,
so the root iteration finds every parent that needs re-canonicalization.

A regression test pinning this is at
[tests/17_aliased_named_unsat.smt2](../tests/17_aliased_named_unsat.smt2),
backed by the standalone [parallel_close_test](tests/parallel_close_test.cpp)
on `synthetic_benchmarks/exp_n11_unsat.smt2`.

## Cross-validation

[`cpp/scripts/cross_validate.sh`](scripts/cross_validate.sh) runs both the
Rust and C++ binaries on every `.smt2` file in `tests/` and
`synthetic_benchmarks/`, in sequential mode and every parallel variant,
and diffs the sat/unsat output. Last clean run: 144/144 agreed.

---

## How to build and run

### Build

```
cmake -B cpp/build cpp
cmake --build cpp/build
```

Parlay is pulled by CMake's `FetchContent` (no submodules to init). C++20
required.

### Tests

```
cd cpp/build && ctest --output-on-failure
```

Runs three suites: `unionfind`, `regression` (every `tests/*.smt2` × {seq,
par}), and `parallel_close` (the `exp_n11` direct-API test).

### Cross-validate against the Rust oracle

```
./cpp/scripts/cross_validate.sh           # assumes binaries already built
./cpp/scripts/cross_validate.sh --build   # build both first
```

Must report `0 mismatches`. Prints details on any diverging file/variant.

### Benchmark

```
./cpp/build/rebuild_compare_bench
```

Same six workloads as `cargo bench --bench rebuild_compare`:

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

`nelson_seq` is `EGraph::sequential_close_nelson` (median of 11 trials, 3
warmup builds). `par_close` is `EGraph::parallel_close`. `par_spd` is the
ratio.

Thread count is parlay's default (one per logical CPU). Override with
`PARLAY_NUM_THREADS=N` for a scaling sweep, e.g.:

```
for T in 1 2 4 8 12; do
  PARLAY_NUM_THREADS=$T ./cpp/build/rebuild_compare_bench 2>/dev/null \
    | grep -E '^large|^deep-l' \
    | sed "s/^/T=$T  /"
done
```

### CLI

```
./cpp/build/parallel-egraph [--parallel|-p] [--timing|-t] <file.smt2>
PE_REBUILD=semisort|sort|close       # selects the parallel variant
PE_TRACE=1                           # per-round timing trace from parallel_close
```

Sample usage:
```
./cpp/build/parallel-egraph --parallel synthetic_benchmarks/exp_n11_unsat.smt2
PE_REBUILD=close PE_TRACE=1 ./cpp/build/parallel-egraph --parallel \
    synthetic_benchmarks/exp_n11_unsat.smt2 2>trace.log
```
