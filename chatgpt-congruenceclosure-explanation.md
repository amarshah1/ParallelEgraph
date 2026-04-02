# Batch-Parallel Nelson-Style E-Graphs / Congruence Closure

## Scope and framing

This note summarizes the main considerations behind a **batch-parallel implementation of a Nelson-style e-graph / congruence-closure engine**, intentionally avoiding the additional complexity of modern equality-saturation systems.

The target setting is the original style of congruence closure:

- a fixed or mostly append-only set of ground terms,
- equivalence classes maintained with union-find,
- **predecessor lists** (modern: parent lists / use lists), and
- repeated propagation of congruence consequences until a fixpoint is reached.

A key simplifying choice is to use a **bulk-synchronous batch algorithm** rather than a fully concurrent online data structure.

---

## Background terminology

In Nelson's original terminology, each equivalence class stores its **predecessors**: the nodes that mention that class as a child. In more modern e-graph language, these are usually called **parent lists** or **use lists**.

These reverse edges are crucial because when two classes merge, only their predecessors need to be reconsidered for newly enabled congruence. Without predecessor lists, one would need to rescan all terms after every merge.

So in this note:

- **predecessor list** = **parent list** = reverse incidence list from a class to the enodes that use it as a child.

---

## 1. Potential sequential bottlenecks

Even in a batch-parallel design, several parts of the computation may remain partially sequential or become effective bottlenecks.

### 1.1 Closure depth / number of rounds

The main sequential bottleneck is the possibility that congruence consequences propagate only one "layer" per round.

A simple adversarial example is:

- `x_{i+1} = f(x_i)`
- `y_{i+1} = f(y_i)`
- initially assert only `x_0 ≡ y_0`

Then a round-based algorithm may discover:

- round 1: `x_1 ≡ y_1`
- round 2: `x_2 ≡ y_2`
- ...
- round `n`: `x_n ≡ y_n`

So the number of rounds can be **linear** in the height of the dependency chain.

This does **not** mean there is little parallelism within a round, but it does imply a worst-case sequential dependency across rounds.

### 1.2 Union / representative maintenance

Even if merge candidates are found in parallel, the actual application of unions may become a bottleneck.

Reasons:

- union-find roots are hot spots,
- many candidate merges may collapse to the same few classes,
- duplicate or already-satisfied merges must be filtered or tolerated.

The union phase may still be parallelized to some extent, but it is the least embarrassingly parallel part of the pipeline.

### 1.3 Frontier construction from changed classes

After a round of unions, the next round needs the predecessors of the newly changed classes. If these predecessor lists are very skewed, a small number of classes may dominate the work.

This can create:

- load imbalance,
- poor locality,
- repeated processing of very large predecessor sets.

### 1.4 Duplicate work and repeated reconsideration

A bad batch implementation may repeatedly revisit the same predecessors or rediscover the same merge candidates across many rounds.

This is the main route to poor total work. Even if the number of rounds is only linear, repeated rescanning can push total work toward quadratic or worse on pathological examples.

### 1.5 Barrier synchronization between rounds

A bulk-synchronous algorithm naturally introduces a barrier between rounds:

1. process current frontier,
2. produce candidate merges,
3. apply merges,
4. determine next frontier,
5. repeat.

If frontiers are small, barrier overhead may dominate.

---

## 2. Opportunities for parallelism

Despite the possible sequential bottlenecks above, the available parallelism inside each round can still be large in practice.

### 2.1 Parallel processing of predecessor frontiers

The most important source of parallel work is the set of predecessor nodes affected by the classes changed in the previous round.

Each affected predecessor can be processed largely independently:

- read its operator,
- find the current representative of each child,
- build its canonical signature.

This is a natural data-parallel map.

### 2.2 Parallel canonicalization

For each affected predecessor node, canonicalization consists of replacing each child class by its current representative. This is usually independent across nodes and can be done in parallel.

### 2.3 Parallel grouping of canonical signatures

Once canonical signatures are computed, the algorithm must bring equal signatures together.

This can be done with:

- a parallel sort, or
- a **semisort / group-by-key** primitive, which is often conceptually closer to what is needed.

This step can expose substantial parallelism when the frontier is large.

### 2.4 Parallel scan over equal-signature groups

After grouping, each group of equal signatures can be scanned independently to emit merge candidates.

### 2.5 Potentially parallel union application

The union phase is less cleanly parallel than the grouping phase, but it may still admit some parallelism if candidate merges are deduplicated first and applied with a parallel or lightly synchronized union-find.

### 2.6 Practical breadth even when worst-case depth is poor

Worst-case examples show that the number of rounds can be linear, but this does not imply low parallelism in practice.

Real workloads may have:

- large predecessor frontiers,
- many terms at similar depths,
- wide waves of newly exposed congruence,
- enough per-round work to amortize synchronization.

Thus the algorithm can have poor theoretical depth but still good practical speedup.

---

## 3. The proposed batch-parallel algorithm

This is the core proposal.

The goal is to preserve the spirit of the original Nelson-style congruence-closure engine while exposing parallelism through **bulk predecessor processing**.

### 3.1 Data structures

Assume the following structures.

#### Enodes

Each enode stores:

- an operator / function symbol,
- an ordered list of child class IDs,
- a unique node ID.

Example:

```text
node 17 = f(c1, c2, c3)
```

#### Union-find over equivalence classes

Maintain equivalence classes of enodes using union-find.

Operations needed:

- `find(c)` = current representative of class `c`
- `union(c, d)` = merge classes `c` and `d`

#### Predecessor lists

For each class `c`, store:

```text
preds(c) = { n | node n mentions c as a child }
```

These are Nelson's **predecessors**.

#### Changed-class frontier

Maintain a set or vector of classes whose merges in the previous round may have enabled new congruence among predecessors.

---

### 3.2 High-level round structure

The algorithm proceeds in rounds until no new merges are produced.

#### Round input

A set `Changed` of classes affected by the previous batch of unions.

#### Round output

A new set `Changed'` induced by the unions performed in this round.

---

### 3.3 Batch-parallel round algorithm

```text
while Changed is not empty:
    1. Gather the affected predecessor nodes:
         Frontier := union of preds(c) for c in Changed

    2. Canonicalize each node in Frontier in parallel:
         sig(n) := (op(n), find(child1(n)), ..., find(childk(n)))

    3. Group nodes by canonical signature:
         groups := GroupByKey((sig(n), n) for n in Frontier)
         (via semisort or sort)

    4. For each equal-signature group, emit merge candidates:
         if group = {n1, n2, ..., nm},
         choose one representative node r in the group,
         emit candidate merges (r, ni) for all i > 1

    5. Deduplicate / filter merge candidates:
         remove trivial pairs already in the same class
         optionally canonicalize pair order

    6. Apply the candidate merges:
         perform unions for all remaining pairs

    7. Build the next changed frontier:
         Changed := classes whose union-find structure changed
```

Terminate when a round emits no successful new merges.

---

### 3.4 Important implementation notes

#### Grouping primitive

Step 3 is the key indexing step.

It can be implemented by:

- **parallel sort** on canonical signatures, or
- **parallel semisort** / bucket-by-key.

A semisort is especially attractive because we do not need a total order; we only need equal signatures to be grouped together.

#### Why not fully online hashconsing?

A fully online hashcons table is not necessary here.

Instead of maintaining a globally up-to-date unique table at all times, the batch algorithm may:

- freeze representatives for a round,
- compute canonical signatures from that snapshot,
- group equal signatures in bulk,
- then perform unions afterward.

This avoids fine-grained synchronization and is much simpler to implement.

#### Frontier granularity

The frontier should ideally include **only predecessors of newly changed classes**, not all predecessors globally.

This is important for work efficiency.

---

## 4. Correctness considerations: how to avoid missing updates

This is the most important correctness issue in the batch formulation.

Because the algorithm does not maintain a perfectly up-to-date congruence closure after every local merge, one must ensure that delayed processing does not cause updates to be missed.

### 4.1 Key idea: it is safe to delay, but not to forget

The batch algorithm relies on the following principle:

> A newly enabled congruence only needs to be discovered eventually, not necessarily immediately.

If a pair of nodes becomes congruent because some child classes were merged, then those merges will place the affected classes into `Changed`, and their predecessors will eventually re-enter the frontier.

So the central rule is:

- **every merge that may enable new congruence must cause the relevant predecessors to be reconsidered in a later round**.

### 4.2 Freeze representatives within a round

A simple correctness discipline is:

- within a round, treat the current union-find representatives as a fixed snapshot,
- do not interleave new unions with signature computation.

This prevents races where one thread canonicalizes a node using one representative while another thread changes it mid-round.

Staleness is acceptable: at worst, some congruence is discovered one round later.

### 4.3 Reconsider predecessors of changed classes

This is the main correctness invariant.

Whenever classes are merged, the next round must include the predecessors of the changed classes.

Why this is enough:

- the only way a new congruence can arise is that some children of two nodes have become equivalent,
- that means at least one relevant child class changed due to a union,
- therefore the predecessor nodes that mention those classes must be reconsidered.

So correctness depends on maintaining the invariant:

> If a class changes, all nodes in its predecessor list are eventually reprocessed.

### 4.4 Duplicate processing is harmless

It is fine if the same predecessor node appears multiple times in the frontier, or if the same merge candidate is produced more than once.

For correctness, redundancy is harmless as long as:

- `union` is idempotent on already-equal classes,
- duplicate merge candidates do not cause unsoundness.

Thus it is better to tolerate some duplication than to risk dropping necessary work.

### 4.5 Exact signature equality after hashing

If grouping is implemented with hashes or semisort by hash, collisions must be handled carefully.

Correctness requires:

- hash only for preliminary bucketing,
- exact equality check on the full canonical signature before emitting a merge.

Otherwise distinct nodes could be merged unsoundly.

### 4.6 Termination condition

The algorithm should terminate only when a round produces **no successful new merges**.

It is not enough that no new signatures were seen in some intermediate data structure; the real fixpoint condition is that the partition of classes no longer changes.

### 4.7 Avoiding missed updates in practice

A safe implementation strategy is:

1. record all successful unions in the current round,
2. collect all classes affected by those unions,
3. form the next frontier from the predecessor lists of those affected classes,
4. repeat until no successful unions occur.

This ensures that every change to the equivalence relation is eventually propagated through the predecessor graph.

---

## 5. Expected behavior and pathological cases

### 5.1 Worst-case rounds can be linear

As discussed above, dependency-chain examples can force the number of rounds to be linear.

This is a true limitation of the bulk-synchronous formulation.

### 5.2 The more serious risk is repeated rescanning

Even if the round count is only linear, total work can become poor if large predecessor sets are reconsidered repeatedly.

Thus the major practical challenge is not just the number of rounds, but the amount of repeated work per round.

### 5.3 Practical parallelism may still be high

In realistic workloads, each round may still process a large frontier, giving substantial data parallelism.

So the algorithm may have:

- poor worst-case depth,
- but strong practical speedup on broad-frontier instances.

---

## 6. Recommended project positioning

For a course project, the right framing is probably:

> We implement and evaluate a batch-parallel version of Nelson-style congruence closure. The algorithm preserves the original predecessor-based propagation structure, but replaces online incremental updates with bulk rounds of predecessor collection, canonical signature grouping, and union application.

This has several advantages:

- faithful to the original algorithmic setting,
- simpler than a modern concurrent e-graph,
- rich enough to discuss both theory and systems issues,
- realistic to implement and evaluate.

A particularly good simplification is to assume a fixed set of ground terms up front, so that the project focuses on **parallel congruence closure**, not dynamic insertion.

---

## 7. Short summary

### Potential sequential bottlenecks

- the number of closure rounds may be linear in the worst case,
- union application may become a hot spot,
- large predecessor lists may be skewed,
- barriers between rounds may dominate on small frontiers,
- repeated rescanning may lead to poor total work.

### Opportunities for parallelism

- predecessor frontier expansion,
- canonicalization of affected nodes,
- grouping / semisort by canonical signature,
- per-group merge extraction,
- some parallelism in union application.

### Core batch-parallel algorithm

1. maintain union-find, enodes, and predecessor lists,
2. keep a frontier of changed classes,
3. gather predecessors of changed classes,
4. canonicalize them in parallel,
5. group equal canonical signatures,
6. emit and apply candidate merges,
7. form the next frontier from changed classes,
8. repeat to fixpoint.

### Correctness principle

The algorithm is correct as long as:

- every class change causes its predecessors to be reconsidered,
- representatives are treated as a stable snapshot within each round,
- grouping uses exact signature equality before merging,
- the algorithm terminates only when no new merges occur.

