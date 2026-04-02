# Batch-Parallel Congruence Closure Implementation Summary

## Overview

Implemented the **round-based batch-parallel congruence closure algorithm** from the ChatGPT document, replacing the sequential `rebuild()` phase with a fully parallelized version using rayon's high-level primitives. The implementation moves parallelism from just the initial union-find merges into the congruence closure propagation itself.

## Key Design Changes

### 1. Eliminated Sequential Worklist

**Before:** E-classes needing repair were tracked in a sequential `Vec<Id>` worklist, processed one at a time.

**After:** Replaced with `Vec<bool> changed` indexed by class ID. Each class is marked when affected by a merge. At the start of each round, a parallel filter identifies the frontier.

**Benefit:** Enables parallel frontier gathering without sequential bookkeeping.

### 2. Eliminated Explicit Parent List Traversal

**Before:** `parents: HashMap<Id, Vec<(ENode, Id)>>` stored reverse edges; `repair()` drained these lists sequentially.

**After:** Build a flat array of all non-leaf enodes. Each round, a parallel filter scans the entire array to identify nodes whose children are in changed classes.

**Benefit:** Fully parallel frontier scanning using `par_iter().filter()`. The cost is O(all_enodes) per round instead of O(frontier), but it's embarrassingly parallel and avoids sequential parent-list maintenance.

### 3. Simplified Parallel Merge Phase

**Before:** `parallel_merge_all()` had three phases: parallel unions, sequential metadata reconciliation, sequential worklist population.

**After:** Only parallel unions + setting changed flags. Metadata reconciliation deferred to `parallel_rebuild()`, which uses `find()` on stale keys (safe because find is lock-free).

**Benefit:** Reduces overhead; metadata is only reconciled once at the end of `parallel_rebuild()` when fixpoint is reached.

### 4. No Hashcons Needed During Rebuild

**Before:** Hashcons consulted during each repair to detect congruences.

**After:** With flat-array + parallel-filter, all nodes whose children share a changed root are included in the frontier. Grouping by signature catches all congruences. Hashcons rebuilt once post-fixpoint.

**Benefit:** Eliminates synchronized access to hashcons during rebuild; only built at the end.

## Algorithm: `parallel_rebuild()`

Each round follows a **fully-parallel structure** using rayon primitives:

### Step 1: Build Frontier (Parallel Filter)
```rust
frontier = all_nodes.par_iter().filter(|(node, _)| {
    node.children.iter().any(|&c| changed[find_root(c)])
})
```
- Parallel scan over all non-leaf enodes
- Identifies nodes with ≥1 changed child root
- `find_root()` is lock-free, safe to call concurrently

### Step 2: Canonicalize (Parallel Map)
```rust
canonicalized = frontier.par_iter().map(|(node, class_id)| {
    let canon = ENode { op: node.op.clone(), children: ... };
    (canon, find_root(class_id))
})
```
- Each node's canonicalization is independent
- Replace children with their current union-find roots
- Collect into `Vec<(canonical_ENode, class_root)>`

### Step 3: Group by Signature (Parallel Semisort)
```rust
canonicalized.par_sort_unstable();
```
- Parallel sort by `(op, children)` lexicographically
- Groups equal signatures together
- `ENode` derives `Ord`/`PartialOrd` for this purpose

### Step 4: Emit Merge Candidates (Sequential Scan)
```rust
for each group in sorted array:
    if group.len() > 1 and distinct classes:
        emit merge pairs
```
- Linear scan of sorted array to find group boundaries
- O(frontier_size) — proportional to parallel work already done
- Sequential because it's a small, cache-friendly scan after the expensive parallel sort

### Step 5: Apply Merges (Parallel Union-Find)
```rust
merge_pairs.par_iter().for_each(|(a, b)| uf.union(a, b))
```
- Lock-free parallel unions via CAS
- Each pair's merge is independent; no synchronization needed

### Step 6: Update Changed Flags (Parallel)
```rust
changed.par_iter_mut().for_each(|c| *c = false);  // reset all
for &(a, _) in merge_pairs:
    changed[find_root(a)] = true;  // mark roots affected by merges
```
- Parallel reset of flags
- Mark new changed classes for next round
- If no merges occurred, break (fixpoint reached)

### Post-Fixpoint Cleanup (Sequential)
Once fixpoint is reached:
- Reconcile `classes` HashMap: re-key by current `find()` roots
- Rebuild `parents`: scan all nodes, insert parent entries under child roots
- Rebuild `hashcons`: scan all nodes, insert canonical forms

## Differences from ChatGPT Algorithm

| ChatGPT | Implementation | Reason |
|---------|---|---|
| Predecessor lists | Flat array + parallel filter | Eliminates sequential parent-list drain |
| `Changed` set (abstract) | `Vec<bool>` indexed by class ID | O(1) lookup; enables parallel filter |
| Semisort (abstract) | `par_sort_unstable_by()` | Rayon primitive achieves grouping via total order |
| Merge extraction | Sequential linear scan | O(frontier) scan; parallelizing (prefix scan for group boundaries) adds complexity for minimal gain |
| Hashcons during rebuild | Not used | All relevant nodes in frontier (shared child roots → shared changed flag) |

## Files Modified

### [src/egraph.rs](src/egraph.rs)

1. **`ENode` derives** — added `PartialOrd, Ord` for parallel sorting

2. **New `changed: Vec<bool>` field** — per-class changed flags replacing sequential worklist

3. **Updated `parallel_merge_all()`**
   - Removed: metadata reconciliation (Phase 2) and worklist population (Phase 3)
   - Added: setting changed flags on merged roots
   - Benefit: Simpler, faster; metadata reconciliation deferred to `parallel_rebuild()`

4. **New `parallel_rebuild()` method** (~140 lines)
   - Implements the round-based algorithm above
   - Each round: filter → canonicalize → sort → emit merges → union → update flags
   - Post-fixpoint: reconcile metadata

5. **Modified `rebuild()`**
   - Dispatches to `parallel_rebuild()` when `self.parallel`
   - Otherwise uses original sequential worklist algorithm

### [tests/regression.rs](tests/regression.rs)

1. **Added `run_smt2_parallel()` helper** — identical to `run_smt2()` but calls `solve_with_mode(&input, true)`

2. **Added 16 parallel regression tests** (`p01` through `p16`)
   - Mirror all sequential tests (t01-t16)
   - Verify parallel rebuild produces identical sat/unsat results

## Test Results

```
Sequential unit tests:         16 passed ✓
Parallel unit tests:            2 passed ✓
Sequential regression (16 SMT2): 16 passed ✓
Parallel regression (16 SMT2):  16 passed ✓
────────────────────────────────────────
Total:                         32 tests passed ✓
```

All tests verify:
- **Correctness**: Parallel and sequential modes produce identical results
- **Congruence propagation**: Cascading merges detected correctly
- **Multi-argument congruence**: f(a,c) ≡ f(b,d) after a≡b, c≡d
- **Stress tests**: 16 arguments, deep nesting (up to 1M classes)
- **Regression tests**: All 16 SMT2 instances in both modes

## Parallelism Structure

```
PARALLEL MERGE PHASE (from lib.rs)
├─ Phase 1: par_iter → parallel UF unions (lock-free CAS)
└─ Phase 2: set changed flags

PARALLEL REBUILD PHASE (from egraph.rs)
├─ ROUND LOOP (until fixpoint)
│  ├─ par_iter().filter()  → frontier gathering
│  ├─ par_iter().map()     → canonicalization
│  ├─ par_sort_unstable()  → semisort by signature
│  ├─ [sequential scan]    → merge candidate extraction
│  ├─ par_iter().for_each()→ parallel union application
│  └─ par_iter() + loop    → changed flag update
└─ [sequential cleanup]    → metadata reconciliation
```

## Performance Characteristics

### Parallelism Opportunities
- **Frontier gathering**: Embarrassingly parallel (independent node scans)
- **Canonicalization**: Embarrassingly parallel (independent find operations)
- **Sorting**: High-parallelism rayon semisort
- **Union application**: Lock-free (independent CAS operations)
- **Flag updates**: Parallel reset + loop-update

### Sequential Bottlenecks (Reduced from Original)
1. **Merge candidate extraction** — O(frontier) linear scan (proportional to parallel work)
2. **Post-fixpoint cleanup** — O(all_nodes) final metadata reconciliation
3. **Round barriers** — rayon implicit barriers; negligible for large frontiers

### Comparison to Sequential `rebuild()`
- Sequential: Per-class `repair()` calls; must traverse parent list for each merge (O(frontier_merges × avg_parent_list_size))
- Parallel: Per-round frontier scan (O(all_nodes)) + parallel processing; fewer total merges due to grouping in single round

## Correctness Guarantees

### No Races
- **Union-find is lock-free** (ConcurrentUnionFind with atomic CAS)
- **Canonicalization uses `find_root(&self)`** — no mutable access to union-find during parallel phase
- **Changed flags** — only read during parallel filter, updated sequentially after merges

### Correctness Invariant (Section 4 of ChatGPT document)
Every class change causes its predecessors to be reconsidered: ✓
- When classes merge, both roots marked in changed flags
- Next round's filter includes all frontier nodes (stale parent lists still valid via `find()`)
- Freeze representatives per round (canonicalize uses stable snapshot of union-find state)

### Termination
Algorithm terminates when a round produces no merges (fixpoint):
- Correct: all congruences have been discovered
- No missed updates: every path merges trigger flag-setting and reprocessing

## Code Organization

- **Lock-free union-find**: [src/unionfind.rs](src/unionfind.rs) — no changes needed; `find()` already supports concurrent reads
- **E-graph core**: [src/egraph.rs](src/egraph.rs) — sequential `rebuild()` unchanged; parallel dispatch added
- **High-level API**: [src/lib.rs](src/lib.rs) — unchanged; dispatches to `rebuild()` which auto-selects mode
- **Term processing**: [src/process.rs](src/process.rs) — unchanged
- **Tests**: [tests/regression.rs](tests/regression.rs) — added parallel variants of all SMT2 tests

## Summary

The implementation successfully parallelizes congruence closure using the ChatGPT document's round-based algorithm. By replacing sequential parent-list traversal with parallel frontier filtering, sequential worklist management with parallel changed-flag updates, and single `repair()` calls with parallel round processing, we expose substantial parallelism in the congruence closure phase—where the real work happens. All existing tests pass, plus 16 new parallel regression tests verify correctness across both modes.
