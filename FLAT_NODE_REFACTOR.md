# Flat Node Array + Parent Index Refactor

## Motivation

The previous implementation used three `HashMap`s (`classes`, `parents`, `hashcons`) as the primary data structures in both sequential and parallel modes. During `parallel_rebuild()`, these HashMaps were:

1. **Read at startup** to build temporary flat arrays (`all_nodes`, `parent_index`)
2. **Never touched** during the round loop itself
3. **Fully rebuilt** in a sequential post-fixpoint cleanup phase

This meant the parallel hot path paid for HashMap indirection it didn't need, and the sequential post-fixpoint cleanup (re-keying `classes`, rebuilding `parents`, rebuilding `hashcons`) was an O(all_nodes) bottleneck that ran after every `parallel_rebuild()` call.

## Design Principle

The parallel rebuild loop needs exactly three things:

| Structure | Type | Role |
|-----------|------|------|
| `uf` | `ConcurrentUnionFind` | Class membership (only shared mutable state during rounds) |
| `nodes` | `Vec<(ENode, Id)>` | Flat array of all non-leaf enodes (immutable after build phase) |
| `parent_index` | `Vec<Vec<usize>>` | child_class -> indices into `nodes` (maintained across rounds) |

Plus `changed: Vec<bool>` for per-class dirty flags. No HashMaps are accessed during the rebuild loop.

## Changes

### New fields on `EGraph`

```rust
// Flat array of all non-leaf enodes: (enode, class_id).
// Append-only during build phase, immutable during rebuild.
nodes: Vec<(ENode, Id)>,

// child_class -> [indices into nodes].
// Built during add(), maintained (consolidated) during parallel_rebuild().
parent_index: Vec<Vec<usize>>,
```

### `add()` — populate flat structures during build phase

When a non-leaf enode is added, it is pushed to `nodes` and registered in `parent_index` under each child's current root:

```rust
if !canon.children.is_empty() {
    let node_idx = self.nodes.len();
    self.nodes.push((canon.clone(), id));
    for &child in &canon.children {
        self.parent_index[self.find(child) as usize].push(node_idx);
    }
}
```

In parallel mode, the `classes` and `parents` HashMaps are **not populated** during `add()`. Only `hashcons` is maintained (needed for dedup). In sequential mode, both old and new structures are populated for backward compatibility.

### `parallel_merge_all()` — consolidate parent_index after unions

After parallel union-find operations, `parent_index` entries may be keyed under old (non-root) class IDs. Phase 2 consolidates them:

```rust
for &(a, b) in pairs {
    let root = self.find(a) as usize;
    if (a as usize) != root {
        let entries = std::mem::take(&mut self.parent_index[a as usize]);
        self.parent_index[root].extend(entries);
    }
    if (b as usize) != root {
        let entries = std::mem::take(&mut self.parent_index[b as usize]);
        self.parent_index[root].extend(entries);
    }
    self.changed[root] = true;
}
```

`mem::take` drains the old slot, so subsequent operations on already-drained slots are no-ops. This correctly handles transitive merges across multiple pairs.

### `parallel_rebuild()` — no construction, no post-fixpoint cleanup

**Before:** Built `all_nodes` and `parent_index` from `self.classes` at the top. After fixpoint, rebuilt `classes`, `parents`, and `hashcons` sequentially.

**After:** Uses `self.nodes` and `self.parent_index` directly. After fixpoint, clears `changed` flags and returns. No HashMap reconstruction.

The round loop structure is unchanged:

```
ROUND LOOP (until fixpoint)
  1. Gather frontier from parent_index of changed classes  [O(frontier)]
  2. Parallel canonicalize frontier nodes                   [par_iter().map()]
  3. Parallel sort by canonical signature                   [par_sort_unstable()]
  4. Emit merge candidates from equal-signature groups      [sequential scan]
  5. Parallel union application                             [par_iter().for_each()]
  6. Consolidate parent_index + update changed flags        [sequential]
```

Step 6 uses the same `mem::take` + `extend` consolidation as `parallel_merge_all()`.

### Sequential mode — unchanged

The `merge()`, `repair()`, and sequential `rebuild()` methods are untouched. They continue to use `classes`, `parents`, `hashcons`, and `worklist`. The `nodes`/`parent_index` fields are populated during `add()` but not read by the sequential path.

## Data structure usage by mode

| Structure | Sequential mode | Parallel build phase | Parallel rebuild |
|-----------|----------------|---------------------|-----------------|
| `uf` | read/write | read/write | read/write |
| `nodes` | written (unused) | written | read only |
| `parent_index` | written (unused) | written | read/write (consolidation) |
| `changed` | unused | written | read/write |
| `classes` | read/write | unused | unused |
| `parents` | read/write | unused | unused |
| `hashcons` | read/write | read/write (dedup) | unused |
| `worklist` | read/write | unused | unused |

## What was eliminated

### From `parallel_rebuild()` startup
- Construction of `all_nodes: Vec<(ENode, Id)>` by iterating `self.classes` — O(all_nodes) with HashMap iteration + cloning
- Construction of `parent_index: Vec<Vec<usize>>` by iterating `all_nodes` — O(all_nodes)

### From post-fixpoint cleanup (entirely removed)
- Re-keying `classes` HashMap under current roots — O(all_nodes) with HashMap remove/insert
- Rebuilding `parents` HashMap by scanning all classes — O(all_nodes) with HashMap inserts + ENode cloning
- Rebuilding `hashcons` HashMap by canonicalizing all nodes — O(all_nodes) with HashMap inserts + ENode cloning

### From `add()` in parallel mode
- `parents` HashMap inserts (ENode cloning + HashMap entry operations)
- `classes` HashMap inserts

## Correctness

### Stale class IDs in `nodes`

Each entry in `nodes` stores the class ID assigned at `add()` time. After merges, this ID may no longer be a root. This is safe because `find(class_id)` resolves to the current root — the union-find is the source of truth for class membership, not the stored ID.

### Parent index consolidation

When `union(a, b)` makes `root` the new representative, `parent_index[a]` and `parent_index[b]` may both contain relevant entries. The consolidation step moves non-root entries to `parent_index[root]` via `mem::take`, ensuring:
- All parents of both merged classes are found when `root` is in `changed`
- Drained slots produce empty vecs on subsequent access (idempotent)
- No entries are lost or duplicated across rounds

### Frontier gathering

The push-based frontier (looking up `parent_index[changed_class]`) produces the same set of affected nodes as the previous pull-based full scan (checking every node's children against `changed`), because `parent_index` is the exact inverse of the children relationship. Deduplication via `sort_unstable()` + `dedup()` prevents processing the same node multiple times per round.

## Limitations

- `num_classes()`, `num_enodes()`, and `print()` rely on the `classes` HashMap, which is not populated in parallel mode. These diagnostic methods return 0 / empty in parallel mode.
- `hashcons` is not updated after `parallel_rebuild()`. Calling `add()` after a parallel rebuild may create duplicate nodes. The current solver usage pattern (build, merge, rebuild, check) does not require post-rebuild `add()` calls.
