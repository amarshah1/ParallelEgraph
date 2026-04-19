use dashmap::DashSet;
use std::cmp::Ordering as CmpOrdering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{collections::HashMap, sync::atomic::Ordering};
use std::sync::atomic::AtomicBool;

use crate::unionfind::{ConcurrentUnionFind, SequentialUnionFind};

#[inline]
fn sig_hash(node: &ENode, uf: &ConcurrentUnionFind) -> u64 {
    let mut h = DefaultHasher::new();
    node.op.hash(&mut h);
    for &c in &node.children {
        uf.find_root(c).hash(&mut h);
    }
    h.finish()
}

#[inline]
fn sig_hash_seq(node: &ENode, uf: &mut SequentialUnionFind) -> u64 {
    let mut h = DefaultHasher::new();
    node.op.hash(&mut h);
    for &c in &node.children {
        uf.find_root(c).hash(&mut h);
    }
    h.finish()
}

#[inline]
fn sig_hash2_seq(node: &ENode, uf: &mut SequentialUnionFind) -> (u64, u64) {
    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    h2.write_u64(0x9E3779B97F4A7C15);
    node.op.hash(&mut h1);
    node.op.hash(&mut h2);
    for &c in &node.children {
        let r = uf.find_root(c);
        r.hash(&mut h1);
        r.hash(&mut h2);
    }
    (h1.finish(), h2.finish())
}

#[inline]
fn sig_hash2(node: &ENode, uf: &ConcurrentUnionFind) -> (u64, u64) {
    let mut h1 = DefaultHasher::new();
    let mut h2 = DefaultHasher::new();
    h2.write_u64(0x9E3779B97F4A7C15);
    node.op.hash(&mut h1);
    node.op.hash(&mut h2);
    for &c in &node.children {
        let r = uf.find_root(c);
        r.hash(&mut h1);
        r.hash(&mut h2);
    }
    (h1.finish(), h2.finish())
}

#[inline]
fn sig_cmp(
    ia: u32,
    ib: u32,
    uf: &ConcurrentUnionFind,
    nodes: &[(ENode, Id)],
) -> CmpOrdering {
    let (na, _) = &nodes[ia as usize];
    let (nb, _) = &nodes[ib as usize];
    match na.op.cmp(&nb.op) {
        CmpOrdering::Equal => {}
        o => return o,
    }
    match na.children.len().cmp(&nb.children.len()) {
        CmpOrdering::Equal => {}
        o => return o,
    }
    for (&ca, &cb) in na.children.iter().zip(&nb.children) {
        let ra = uf.find_root(ca);
        let rb = uf.find_root(cb);
        match ra.cmp(&rb) {
            CmpOrdering::Equal => continue,
            o => return o,
        }
    }
    CmpOrdering::Equal
}

#[inline]
fn sigs_equal(
    ia: u32,
    ib: u32,
    uf: &ConcurrentUnionFind,
    nodes: &[(ENode, Id)],
) -> bool {
    let (na, _) = &nodes[ia as usize];
    let (nb, _) = &nodes[ib as usize];
    if na.op != nb.op {
        return false;
    }
    if na.children.len() != nb.children.len() {
        return false;
    }
    na.children
        .iter()
        .zip(&nb.children)
        .all(|(&ca, &cb)| uf.find_root(ca) == uf.find_root(cb))
}

#[inline]
fn sigs_equal_seq(
    ia: u32,
    ib: u32,
    uf: &mut SequentialUnionFind,
    nodes: &[(ENode, Id)],
) -> bool {
    let (na, _) = &nodes[ia as usize];
    let (nb, _) = &nodes[ib as usize];
    if na.op != nb.op {
        return false;
    }
    if na.children.len() != nb.children.len() {
        return false;
    }
    for (&ca, &cb) in na.children.iter().zip(&nb.children) {
        if uf.find_root(ca) != uf.find_root(cb) {
            return false;
        }
    }
    true
}

pub type Id = u32;

/// An e-node: an operator applied to e-class IDs.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ENode {
    pub op: String,
    pub children: Vec<Id>,
}

impl ENode {
    pub fn leaf(op: impl Into<String>) -> Self {
        ENode { op: op.into(), children: vec![] }
    }

    pub fn new(op: impl Into<String>, children: Vec<Id>) -> Self {
        ENode { op: op.into(), children }
    }
}

impl std::fmt::Display for ENode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.children.is_empty() {
            write!(f, "{}", self.op)
        } else {
            write!(f, "{}(", self.op)?;
            for (i, c) in self.children.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", c)?;
            }
            write!(f, ")")
        }
    }
}

/// E-graph: maintains e-classes, a union-find, a hashcons, and supports
/// merge with congruence closure.
///
/// The union-find is lock-free (based on concurrent DSU with ranks and
/// path compression via CAS), so `find` and `equiv` do not require `&mut self`.
/// When constructed with `new_parallel()`, batch merge operations use rayon
/// to run union-find operations across multiple threads.
///
/// In parallel mode, the primary data structures are:
/// - `uf`: lock-free union-find (the only shared mutable state during rebuild)
/// - `nodes`: flat array of non-leaf enodes (immutable after build phase)
/// - `parent_index`: child_class -> [node indices] (maintained during rebuild)
/// - `changed`: per-class flags
///
/// The `classes`/`parents`/`hashcons`/`worklist` structures are only used in
/// sequential mode. In parallel mode, `hashcons` is used during `add()` for
/// dedup but not during rebuild.
pub struct EGraph {
    // Lock-free concurrent union-find (works in both sequential and parallel modes)
    uf: ConcurrentUnionFind,

    // --- Parallel mode primary structures ---

    // Flat array of all non-leaf enodes: (enode, class_id).
    // Append-only during build phase, immutable during rebuild.
    // Class membership resolved via find(class_id).
    nodes: Vec<(ENode, Id)>,

    // child_class -> [indices into nodes].
    // Built during add(), maintained (consolidated) during parallel_rebuild().
    parent_index: Vec<Vec<Id>>,

    // Per-class changed flags for parallel rebuild (indexed by class id)
    changed: Vec<AtomicBool>,

    // --- Sequential mode structures ---

    // E-class id -> e-nodes in that class
    classes: HashMap<Id, Vec<ENode>>,

    // For each e-class, the list of (enode, eclass_of_enode) where enode
    // has this class as a child (the "use list" / "parent list")
    parents: HashMap<Id, Vec<(ENode, Id)>>,

    // E-classes needing congruence repair (sequential mode)
    worklist: Vec<Id>,

    predecessors_modified: DashSet<Id>,

    // --- Shared ---

    // Canonical e-node -> e-class id (used by add() for dedup in both modes)
    hashcons: HashMap<ENode, Id>,

    // Whether to use parallel (rayon) operations for batch merges
    parallel: bool,

    size: usize,
}

impl EGraph {
    /// Create a new e-graph in sequential mode (growable via `add`).
    pub fn new() -> Self {
        Self::new_with_size(0, false)
    }

    /// Create a new e-graph in sequential mode with a pre-allocated union-find.
    pub fn new_sized(size: usize) -> Self {
        Self::new_with_size(size, false)
    }

    fn new_with_size(size: usize, parallel: bool) -> Self {
        EGraph {
            size,
            uf: if size > 0 { ConcurrentUnionFind::with_size(size) } else { ConcurrentUnionFind::new() },
            nodes: Vec::new(),
            parent_index: Vec::new(),
            changed: Vec::new(),
            classes: HashMap::new(),
            parents: HashMap::new(),
            worklist: Vec::new(),
            predecessors_modified: DashSet::new(),
            hashcons: HashMap::new(),
            parallel,
        }
    }

    /// Create a new e-graph in parallel mode (growable via `add`).
    pub fn new_parallel() -> Self {
        Self::new_with_size(0, true)
    }

    /// Create a new e-graph in parallel mode with a pre-allocated union-find.
    pub fn new_parallel_sized(size: usize) -> Self {
        Self::new_with_size(size, true)
    }

    pub fn is_parallel(&self) -> bool {
        self.parallel
    }

    fn make_id(&mut self) -> Id {
        self.uf.make_set()
    }

    /// Find the canonical representative. Lock-free, does not require `&mut self`.
    /// Path compression happens via CAS on the internal atomic array.
    pub fn find(&self, id: Id) -> Id {
        self.uf.find_root(id)
    }

    /// Canonicalize an e-node: replace each child with its find root.
    fn canonicalize(&self, node: &ENode) -> ENode {
        let children = node.children.iter().map(|&c| self.find(c)).collect();
        ENode { op: node.op.clone(), children }
    }

    /// Add a single e-node. Returns the e-class id it belongs to.
    pub fn add(&mut self, node: ENode) -> Id {
        // Canonicalize Enode to add
        let canon = self.canonicalize(&node);

        // Check hashcons for an existing congruent node
        if let Some(&id) = self.hashcons.get(&canon) {
            // If already exists, just return that node
            return self.find(id);
        }

        // Fresh e-class
        let id = self.make_id();

        // Register non-leaf nodes in flat array + parent index (for parallel rebuild)
        if !canon.children.is_empty() {
            let node_idx = self.nodes.len();
            self.nodes.push((canon.clone(), id));
            // Grow parent_index to cover all class IDs
            if self.parent_index.len() < self.uf.len() {
                self.parent_index.resize_with(self.uf.len(), Vec::new);
            }
            for &child in &canon.children {
                let child_root = self.find(child) as usize;
                self.parent_index[child_root].push(node_idx as u32);
            }
        }

        // Sequential mode: also maintain parents and classes
        if !self.parallel {
            for &child in &canon.children {
                self.parents.entry(child).or_default().push((canon.clone(), id));
            }
            self.classes.entry(id).or_default().push(canon.clone());
        }

        self.hashcons.insert(canon, id);
        id
    }

    /// Merge two e-classes. Returns the new canonical id.
    pub fn merge(&mut self, a: Id, b: Id) -> Id {
        let a = self.find(a);
        let b = self.find(b);
        if a == b {
            return a;
        }

        // Perform the lock-free union
        self.uf.union(a, b);
        let root = self.find(a);
        let merged = if root == a { b } else { a };

        // Merge class contents
        if let Some(nodes) = self.classes.remove(&merged) {
            self.classes.entry(root).or_default().extend(nodes);
        }

        // Merge parent (use) lists
        if let Some(parent_list) = self.parents.remove(&merged) {
            self.parents.entry(root).or_default().extend(parent_list);
        }

        self.worklist.push(root);
        root
    }

    /// Batch-merge many pairs in parallel using rayon.
    ///
    /// Phase 1: all union-find operations run lock-free across threads.
    /// Phase 2: mark changed classes for parallel rebuild.
    ///
    /// Metadata reconciliation (classes, parents, hashcons) is deferred to
    /// `parallel_rebuild`, which uses `find()` on stale keys.
    pub fn parallel_merge_all(&mut self, pairs: &[(Id, Id)]) {
        use rayon::prelude::*;

        if pairs.is_empty() {
            return;
        }

        let n_classes = self.uf.len();
        self.grow_changed_to(n_classes);
        if self.parent_index.len() < n_classes {
            self.parent_index.resize_with(n_classes, Vec::new);
        }

        pairs.par_iter().for_each(|&(a, b)| {
            self.parallel_merge(a, b);
        });
    }

    fn parallel_merge(&self, a: Id, b: Id) {
        self.uf.union(a, b);
        self.changed[a as usize].store(true, Ordering::Release);
        self.changed[b as usize].store(true, Ordering::Release);
    }

    fn grow_changed_to(&mut self, n_classes: usize) {
        use rayon::prelude::*;
        let need = n_classes.saturating_sub(self.changed.len());
        if need > 0 {
            self.changed.par_extend(
                (0..need).into_par_iter().map(|_| AtomicBool::new(false)),
            );
        }
    }

    /// Sequential Nelson-style congruence closure with a signature table.
    ///
    /// Each round pops a changed class `c` from the worklist, drains
    /// `parent_index[c]`, and for each parent `p` computes `sig_hash(p)` with
    /// current roots. A `HashMap<u64, Vec<node_idx>>` table is kept across
    /// rounds: lookups walk the bucket with `sigs_equal` to filter hash
    /// collisions; on match, merge the two classes and push both endpoints;
    /// otherwise append `p` to the bucket.
    ///
    /// Runs against a local `SequentialUnionFind` (no atomic CAS) to isolate
    /// algorithmic cost from concurrency overhead. The final partition is
    /// replayed into `self.uf` so `EGraph::find()` reflects the closure.
    pub fn sequential_close_nelson(&mut self, initial_unions: &[(Id, Id)]) {
        let n_classes = self.uf.len();
        if self.parent_index.len() < n_classes {
            self.parent_index.resize_with(n_classes, Vec::new);
        }

        let mut uf = SequentialUnionFind::with_size(n_classes);
        for c in 0..n_classes as Id {
            let r = self.uf.find_root(c);
            if r != c {
                uf.union(c, r);
            }
        }

        let mut work: Vec<Id> = Vec::with_capacity(initial_unions.len() * 2);
        for &(a, b) in initial_unions {
            let ra = uf.find_root(a);
            let rb = uf.find_root(b);
            if ra != rb {
                uf.union(ra, rb);
                work.push(ra);
                work.push(rb);
            }
        }

        let mut sig_table: HashMap<u64, Vec<u32>> = HashMap::new();

        while let Some(c) = work.pop() {
            if (c as usize) >= self.parent_index.len() {
                continue;
            }
            let frontier: Vec<u32> = std::mem::take(&mut self.parent_index[c as usize]);
            if frontier.is_empty() {
                continue;
            }
            for ni in frontier {
                let (node, my_class_raw) = &self.nodes[ni as usize];
                let h = sig_hash_seq(node, &mut uf);
                let my_class = uf.find_root(*my_class_raw);

                let bucket = sig_table.entry(h).or_default();
                let mut match_ni: Option<u32> = None;
                for &o in bucket.iter() {
                    if sigs_equal_seq(ni, o, &mut uf, &self.nodes) {
                        match_ni = Some(o);
                        break;
                    }
                }

                match match_ni {
                    Some(o) => {
                        let other_class_raw = self.nodes[o as usize].1;
                        let other_class = uf.find_root(other_class_raw);
                        if my_class != other_class {
                            uf.union(my_class, other_class);
                            work.push(my_class);
                            work.push(other_class);
                        }
                    }
                    None => {
                        bucket.push(ni);
                    }
                }
            }
        }

        for c in 0..n_classes as Id {
            let r = uf.find_root(c);
            if r != c {
                self.uf.union(c, r);
            }
        }
    }

    /// Downey-Sethi-Tarjan 1980 sequential congruence closure.
    ///
    /// From "Variations on the Common Subexpression Problem" (JACM, 1980).
    /// Maintains a signature table `sig -> node_idx` that is kept
    /// *consistent with current roots* by explicitly removing stale entries
    /// before each union and re-inserting the affected predecessors with
    /// their updated signatures. Each re-insertion either matches an
    /// existing canonical node (→ schedule a merge) or takes its place.
    ///
    /// Per-representative predecessor lists are concatenated on union, as in
    /// Nelson-Oppen, but merge candidates are discovered via the sig table
    /// instead of a Pu × Pv cross-scan, giving near-linear amortized cost.
    ///
    /// Signatures are 128-bit hashes `(h1, h2)` of `(op, find_roots(children))`,
    /// keyed in the `HashMap` directly. Collisions at 128 bits are
    /// astronomically unlikely on realistic workloads; a collision would
    /// cause an unsound merge, not a miss.
    pub fn sequential_close_dst(&mut self, initial_unions: &[(Id, Id)]) {
        let n_classes = self.uf.len();
        if self.parent_index.len() < n_classes {
            self.parent_index.resize_with(n_classes, Vec::new);
        }

        let mut uf = SequentialUnionFind::with_size(n_classes);
        for c in 0..n_classes as Id {
            let r = self.uf.find_root(c);
            if r != c {
                uf.union(c, r);
            }
        }

        // Seed preds-per-root.
        let mut preds: HashMap<Id, Vec<u32>> = HashMap::new();
        for c in 0..n_classes as Id {
            let slot = &self.parent_index[c as usize];
            if slot.is_empty() {
                continue;
            }
            let root = uf.find_root(c);
            preds.entry(root).or_default().extend_from_slice(slot);
        }

        // sig -> canonical node idx. sig = sig_hash2_seq(node, uf).
        let mut sig_table: HashMap<(u64, u64), u32> =
            HashMap::with_capacity(self.nodes.len());
        let mut pending: Vec<(Id, Id)> = initial_unions.to_vec();

        // Populate sig_table; schedule merges for any pre-existing duplicates.
        for ni in 0..self.nodes.len() as u32 {
            let sig = sig_hash2_seq(&self.nodes[ni as usize].0, &mut uf);
            match sig_table.get(&sig).copied() {
                Some(q) => {
                    let cp = uf.find_root(self.nodes[ni as usize].1);
                    let cq = uf.find_root(self.nodes[q as usize].1);
                    if cp != cq {
                        pending.push((cp, cq));
                    }
                }
                None => {
                    sig_table.insert(sig, ni);
                }
            }
        }

        while let Some((u, v)) = pending.pop() {
            let ru = uf.find_root(u);
            let rv = uf.find_root(v);
            if ru == rv {
                continue;
            }

            let pu = preds.remove(&ru).unwrap_or_default();
            let pv = preds.remove(&rv).unwrap_or_default();

            // Weighted union: only the LOSER's find_root changes, so only
            // its preds need their sigs re-keyed. Winner preds keep the
            // same sigs and stay in sig_table untouched.
            let loser_root = uf.predict_loser(ru, rv);
            let (loser_preds, winner_preds) = if loser_root == ru {
                (pu, pv)
            } else {
                (pv, pu)
            };

            // Evict old sigs for loser preds (pre-union).
            for &p in &loser_preds {
                let old_sig = sig_hash2_seq(&self.nodes[p as usize].0, &mut uf);
                if sig_table.get(&old_sig).copied() == Some(p) {
                    sig_table.remove(&old_sig);
                }
            }

            uf.union(ru, rv);
            let new_root = uf.find_root(ru);

            // Re-insert loser preds with new sigs.
            for &p in &loser_preds {
                let new_sig = sig_hash2_seq(&self.nodes[p as usize].0, &mut uf);
                match sig_table.get(&new_sig).copied() {
                    Some(q) if q != p => {
                        let cp = uf.find_root(self.nodes[p as usize].1);
                        let cq = uf.find_root(self.nodes[q as usize].1);
                        if cp != cq {
                            pending.push((cp, cq));
                        }
                    }
                    _ => {
                        sig_table.insert(new_sig, p);
                    }
                }
            }

            let mut merged = winner_preds;
            merged.extend(loser_preds);
            preds.insert(new_root, merged);
        }

        for c in 0..n_classes as Id {
            let r = uf.find_root(c);
            if r != c {
                self.uf.union(c, r);
            }
        }
    }

    /// Batch-parallel congruence closure.
    ///
    /// Applies `initial_unions`, then iterates rounds to fixpoint. Each
    /// round runs 3 parallel phases: (1) canonicalize + compute 128-bit
    /// snapshot signature over the frontier, (2) global `par_sort_unstable`
    /// on `(h1, h2, idx)`, (3) `par_windows(2)` scan that applies unions on
    /// adjacent matching signatures and accumulates next-round endpoints.
    ///
    /// No AtomicBool change-flag scan: next-round endpoints are collected
    /// directly from the scan via `filter_map().flat_map_iter(...)`. The
    /// frontier is gathered by draining `parent_index[c]` for each endpoint
    /// in `work` via `mem::take` (serial, O(|work|) small Vec moves).
    ///
    /// Signatures use two independent 64-bit hashes (h1/h2) computed once per
    /// round. Per §4.1-4.3 of the algorithm spec, any merge missed due to
    /// stale `find_root` reads during concurrent union application is
    /// recovered in a later round.
    pub fn parallel_close(&mut self, initial_unions: &[(Id, Id)]) {
        use rayon::prelude::*;
        use rayon::slice::ParallelSlice;

        if self.parent_index.len() < self.uf.len() {
            self.parent_index.resize_with(self.uf.len(), Vec::new);
        }

        {
            let uf = &self.uf;
            initial_unions.par_iter().for_each(|&(a, b)| {
                uf.union(a, b);
            });
        }
        let mut work: Vec<u32> = initial_unions
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .collect();

        while !work.is_empty() {
            let parent_index = &self.parent_index;
            let pi_len = parent_index.len();
            let frontier: Vec<u32> = work
                .par_iter()
                .filter(|&&c| (c as usize) < pi_len)
                .flat_map_iter(|&c| parent_index[c as usize].iter().copied())
                .collect();
            if frontier.is_empty() {
                break;
            }

            let uf = &self.uf;
            let nodes = &self.nodes;

            let mut canon: Vec<(u64, u64, u32)> = frontier
                .par_iter()
                .map(|&idx| {
                    let (node, _) = &nodes[idx as usize];
                    let (h1, h2) = sig_hash2(node, uf);
                    (h1, h2, idx)
                })
                .collect();

            canon.par_sort_unstable();

            let mut next_work: Vec<u32> = canon
                .par_windows(2)
                .filter_map(|w| {
                    let (h1a, h2a, ia) = w[0];
                    let (h1b, h2b, ib) = w[1];
                    if h1a != h1b || h2a != h2b {
                        return None;
                    }
                    // Exact equality check (§4.5): guard against hash collisions.
                    if !sigs_equal(ia, ib, uf, nodes) {
                        return None;
                    }
                    let ca = nodes[ia as usize].1;
                    let cb = nodes[ib as usize].1;
                    let ra = uf.find_root(ca);
                    let rb = uf.find_root(cb);
                    if ra != rb {
                        uf.union(ra, rb);
                        Some((ra, rb))
                    } else {
                        None
                    }
                })
                .flat_map_iter(|(a, b)| [a, b].into_iter())
                .collect();
            next_work.par_sort_unstable();
            next_work.dedup();
            work = next_work;
        }
    }

    /// Default parallel rebuild: semisort-based grouping.
    pub fn parallel_rebuild(&mut self) {
        self.parallel_rebuild_semisort();
    }

    /// Batch-parallel congruence closure using the round-based algorithm.
    ///
    /// Uses only `uf`, `nodes`, `parent_index`, and `changed` — no HashMap
    /// access during the hot loop. The `nodes` array and `parent_index` are
    /// built during `add()` and maintained across rounds.
    ///
    /// Each round: (1) frontier from parent_index, (2) parallel canonicalization,
    /// (3) parallel semisort by signature, (4) merge candidate extraction,
    /// (5) parallel merge application, (6) changed-flag + parent_index update.
    pub fn parallel_rebuild_semisort(&mut self) {
        use rayon::prelude::*;

        let n_classes = self.uf.len();
        self.grow_changed_to(n_classes);
        if self.parent_index.len() < n_classes {
            self.parent_index.resize_with(n_classes, Vec::new);
        }

        loop {
            let changed_ref = &self.changed;
            let changed_classes: Vec<usize> = changed_ref
                .par_iter()
                .enumerate()
                .filter_map(|(idx, flag)| {
                    if flag.swap(false, Ordering::AcqRel) { Some(idx) } else { None }
                })
                .collect();

            if changed_classes.is_empty() {
                break;
            }

            let parent_index = &self.parent_index;
            let frontier: Vec<u32> = changed_classes
                .par_iter()
                .flat_map(|&c| parent_index[c].par_iter().copied())
                .collect();

            if frontier.is_empty() {
                break;
            }

            let uf = &self.uf;
            let nodes = &self.nodes;
            let canon: Vec<(u64, u32, Id)> = frontier
                .par_iter()
                .map(|&idx| {
                    let (node, class_id) = &nodes[idx as usize];
                    (sig_hash(node, uf), idx, uf.find_root(*class_id))
                })
                .collect();

            let n = canon.len();
            let num_threads = rayon::current_num_threads();
            let num_buckets = ((n / 64).max(num_threads * 4))
                .next_power_of_two()
                .max(1);
            let mask = (num_buckets - 1) as u64;
            let keys: Vec<usize> = canon
                .par_iter()
                .map(|&(h, _, _)| (h & mask) as usize)
                .collect();

            let mut bucketed: Vec<(u64, u32, Id)> = Vec::new();
            let mut bucket_offsets: Vec<usize> = Vec::new();
            crate::collect_reduce::parallel_semisort(
                &canon,
                &keys,
                num_buckets,
                &mut bucketed,
                &mut bucket_offsets,
            );

            let bucketed_ref = &bucketed;
            let offsets_ref = &bucket_offsets;
            let merge_pairs: Vec<(Id, Id)> = (0..num_buckets)
                .into_par_iter()
                .flat_map_iter(|b| {
                    let lo = offsets_ref[b];
                    let hi = offsets_ref[b + 1];
                    let mut local: Vec<(u64, u32, Id)> = bucketed_ref[lo..hi].to_vec();
                    local.sort_unstable_by(|&(ha, ia, _), &(hb, ib, _)| {
                        match ha.cmp(&hb) {
                            CmpOrdering::Equal => sig_cmp(ia, ib, uf, nodes),
                            other => other,
                        }
                    });
                    let mut merges: Vec<(Id, Id)> = Vec::new();
                    for i in 1..local.len() {
                        let (h0, i0, a) = local[i - 1];
                        let (h1, i1, b) = local[i];
                        if h0 != h1 {
                            continue;
                        }
                        if !sigs_equal(i0, i1, uf, nodes) {
                            continue;
                        }
                        if a != b {
                            merges.push((a, b));
                        }
                    }
                    merges.into_iter()
                })
                .collect();

            if merge_pairs.is_empty() {
                break;
            }

            merge_pairs.par_iter().for_each(|&(a, b)| {
                uf.union(a, b);
            });

            let changed = &self.changed;
            merge_pairs.par_iter().for_each(|&(a, b)| {
                changed[a as usize].store(true, Ordering::Release);
                changed[b as usize].store(true, Ordering::Release);
            });
        }
    }

    /// Alternative parallel rebuild using a single global parallel sort
    /// on hash-keyed signatures. Same correctness as the semisort variant,
    /// but no bucket-partitioned grouping step.
    pub fn parallel_rebuild_sort(&mut self) {
        use rayon::prelude::*;

        let n_classes = self.uf.len();
        self.grow_changed_to(n_classes);
        if self.parent_index.len() < n_classes {
            self.parent_index.resize_with(n_classes, Vec::new);
        }

        loop {
            let changed_ref = &self.changed;
            let changed_classes: Vec<usize> = changed_ref
                .par_iter()
                .enumerate()
                .filter_map(|(idx, flag)| {
                    if flag.swap(false, Ordering::AcqRel) { Some(idx) } else { None }
                })
                .collect();

            if changed_classes.is_empty() {
                break;
            }

            let parent_index = &self.parent_index;
            let frontier: Vec<u32> = changed_classes
                .par_iter()
                .flat_map(|&c| parent_index[c].par_iter().copied())
                .collect();

            if frontier.is_empty() {
                break;
            }

            let uf = &self.uf;
            let nodes = &self.nodes;
            let mut canon: Vec<(u64, u32, Id)> = frontier
                .par_iter()
                .map(|&idx| {
                    let (node, class_id) = &nodes[idx as usize];
                    (sig_hash(node, uf), idx, uf.find_root(*class_id))
                })
                .collect();

            canon.par_sort_unstable_by(|&(ha, ia, _), &(hb, ib, _)| {
                match ha.cmp(&hb) {
                    CmpOrdering::Equal => sig_cmp(ia, ib, uf, nodes),
                    other => other,
                }
            });

            let n = canon.len();
            let merge_pairs: Vec<(Id, Id)> = (1..n)
                .into_par_iter()
                .filter_map(|i| {
                    let (h0, i0, a) = canon[i - 1];
                    let (h1, i1, b) = canon[i];
                    if h0 != h1 {
                        return None;
                    }
                    if !sigs_equal(i0, i1, uf, nodes) {
                        return None;
                    }
                    if a != b { Some((a, b)) } else { None }
                })
                .collect();

            if merge_pairs.is_empty() {
                break;
            }

            merge_pairs.par_iter().for_each(|&(a, b)| {
                uf.union(a, b);
            });

            let changed = &self.changed;
            merge_pairs.par_iter().for_each(|&(a, b)| {
                changed[a as usize].store(true, Ordering::Release);
                changed[b as usize].store(true, Ordering::Release);
            });
        }
    }

    /// Restore the congruence invariant after merges.
    /// Dispatches to `parallel_rebuild` in parallel mode.
    pub fn rebuild(&mut self) {
        if self.parallel {
            self.parallel_rebuild();
            return;
        }
        while !self.worklist.is_empty() {
            let todo: Vec<Id> = std::mem::take(&mut self.worklist);
            for id in todo {
                let id_root = self.find(id);
                self.repair(id_root);
            }
        }
    }

    fn repair(&mut self, id: Id) {
        let id = self.find(id);

        // Drain the parent list for this class
        let old_parents = self.parents.remove(&id).unwrap_or_default();

        for (p_node, p_class) in old_parents {
            // Remove the (possibly stale) hashcons entry
            self.hashcons.remove(&p_node);

            // Re-canonicalize
            let p_canon = self.canonicalize(&p_node);
            let mut p_id = self.find(p_class);

            // Check for congruence
            if let Some(&existing) = self.hashcons.get(&p_canon) {
                let existing = self.find(existing);
                p_id = self.merge(p_id, existing);
            }

            let p_id = self.find(p_id);
            self.hashcons.insert(p_canon.clone(), p_id);
            let root = self.find(id);
            self.parents.entry(root).or_default().push((p_canon, p_id));
        }
    }

    /// Check whether two e-class ids are equivalent. Lock-free.
    pub fn equiv(&self, a: Id, b: Id) -> bool {
        self.find(a) == self.find(b)
    }

    /// Number of distinct e-classes.
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Total number of e-nodes.
    pub fn num_enodes(&self) -> usize {
        self.classes.values().map(|v| v.len()).sum()
    }

    /// Print the e-graph state for debugging.
    pub fn print(&self) {
        println!("EGraph: {} classes, {} e-nodes", self.num_classes(), self.num_enodes());
        let mut ids: Vec<_> = self.classes.keys().copied().collect();
        ids.sort();
        for id in ids {
            let nodes = &self.classes[&id];
            let node_strs: Vec<_> = nodes.iter().map(|n| n.to_string()).collect();
            println!("  class {}: {{{}}}", id, node_strs.join(", "));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_add_and_merge() {
        let mut eg = EGraph::new();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        assert!(!eg.equiv(a, b));

        eg.merge(a, b);
        eg.rebuild();
        assert!(eg.equiv(a, b));
    }

    #[test]
    fn congruence_closure() {
        let mut eg = EGraph::new();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let fa = eg.add(ENode::new("f", vec![a]));
        let fb = eg.add(ENode::new("f", vec![b]));
        assert!(!eg.equiv(fa, fb));

        eg.merge(a, b);
        eg.rebuild();
        assert!(eg.equiv(fa, fb), "f(a) == f(b) after a=b (congruence)");
    }

    #[test]
    fn cascading_congruence() {
        let mut eg = EGraph::new();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let fa = eg.add(ENode::new("f", vec![a]));
        let fb = eg.add(ENode::new("f", vec![b]));
        let gfa = eg.add(ENode::new("g", vec![fa]));
        let gfb = eg.add(ENode::new("g", vec![fb]));

        eg.merge(a, b);
        eg.rebuild();
        assert!(eg.equiv(fa, fb));
        assert!(eg.equiv(gfa, gfb), "g(f(a)) == g(f(b)) cascading");
    }

    #[test]
    fn multi_arg_congruence() {
        let mut eg = EGraph::new();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let c = eg.add(ENode::leaf("c"));
        let d = eg.add(ENode::leaf("d"));
        let fac = eg.add(ENode::new("f", vec![a, c]));
        let fbd = eg.add(ENode::new("f", vec![b, d]));

        eg.merge(a, b);
        eg.merge(c, d);
        eg.rebuild();
        assert!(eg.equiv(fac, fbd), "f(a,c) == f(b,d) after a=b, c=d");
    }

    #[test]
    fn different_ops_not_congruent() {
        let mut eg = EGraph::new();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let fa = eg.add(ENode::new("f", vec![a]));
        let gb = eg.add(ENode::new("g", vec![b]));

        eg.merge(a, b);
        eg.rebuild();
        assert!(!eg.equiv(fa, gb), "f(a) != g(b) even after a=b");
    }

    #[test]
    fn hashcons_dedup() {
        let mut eg = EGraph::new();
        let _a = eg.add(ENode::leaf("a"));
        let fa1 = eg.add(ENode::new("f", vec![_a]));
        let fa2 = eg.add(ENode::new("f", vec![_a]));
        assert!(eg.equiv(fa1, fa2), "duplicate terms share same e-class");
        assert_eq!(eg.num_classes(), 2);
    }

    #[test]
    fn rebuild_idempotent() {
        let mut eg = EGraph::new();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let _fa = eg.add(ENode::new("f", vec![a]));
        let _fb = eg.add(ENode::new("f", vec![b]));
        eg.merge(a, b);
        eg.rebuild();
        let n = eg.num_classes();
        eg.rebuild();
        assert_eq!(eg.num_classes(), n, "rebuild is idempotent");
    }

    #[test]
    fn deep_congruence() {
        // a = b  =>  f(a) = f(b)  =>  g(f(a), a) = g(f(b), b)
        let mut eg = EGraph::new();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let fa = eg.add(ENode::new("f", vec![a]));
        let fb = eg.add(ENode::new("f", vec![b]));
        let gfaa = eg.add(ENode::new("g", vec![fa, a]));
        let gfbb = eg.add(ENode::new("g", vec![fb, b]));

        eg.merge(a, b);
        eg.rebuild();
        assert!(eg.equiv(gfaa, gfbb), "g(f(a),a) == g(f(b),b) after a=b");
    }

    // ---- Parallel mode tests ----

    #[test]
    fn parallel_basic_merge() {
        let mut eg = EGraph::new_parallel();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let c = eg.add(ENode::leaf("c"));

        eg.parallel_merge_all(&[(a, b), (b, c)]);
        eg.rebuild();

        assert!(eg.equiv(a, b));
        assert!(eg.equiv(b, c));
        assert!(eg.equiv(a, c));
    }

    #[test]
    fn parallel_congruence() {
        let mut eg = EGraph::new_parallel();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let fa = eg.add(ENode::new("f", vec![a]));
        let fb = eg.add(ENode::new("f", vec![b]));

        eg.parallel_merge_all(&[(a, b)]);
        eg.rebuild();

        assert!(eg.equiv(fa, fb), "f(a) == f(b) after parallel merge a=b");
    }

    #[test]
    fn parallel_cascading_congruence() {
        let mut eg = EGraph::new_parallel();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let fa = eg.add(ENode::new("f", vec![a]));
        let fb = eg.add(ENode::new("f", vec![b]));
        let gfa = eg.add(ENode::new("g", vec![fa]));
        let gfb = eg.add(ENode::new("g", vec![fb]));

        eg.parallel_merge_all(&[(a, b)]);
        eg.rebuild();

        assert!(eg.equiv(fa, fb));
        assert!(eg.equiv(gfa, gfb), "g(f(a)) == g(f(b)) cascading (parallel)");
    }

    #[test]
    fn parallel_multi_arg_congruence() {
        let mut eg = EGraph::new_parallel();
        let a = eg.add(ENode::leaf("a"));
        let b = eg.add(ENode::leaf("b"));
        let c = eg.add(ENode::leaf("c"));
        let d = eg.add(ENode::leaf("d"));
        let fac = eg.add(ENode::new("f", vec![a, c]));
        let fbd = eg.add(ENode::new("f", vec![b, d]));

        eg.parallel_merge_all(&[(a, b), (c, d)]);
        eg.rebuild();

        assert!(eg.equiv(fac, fbd), "f(a,c) == f(b,d) after parallel a=b, c=d");
    }

    #[test]
    fn parallel_many_merges() {
        let mut eg = EGraph::new_parallel();
        let n = 100;
        let ids: Vec<Id> = (0..n).map(|i| eg.add(ENode::leaf(format!("x{i}")))).collect();

        // Merge all into one equivalence class
        let pairs: Vec<(Id, Id)> = (0..n - 1).map(|i| (ids[i as usize], ids[(i + 1) as usize])).collect();
        eg.parallel_merge_all(&pairs);
        eg.rebuild();

        for i in 0..n {
            assert!(eg.equiv(ids[0], ids[i as usize]), "all should be equivalent");
        }
    }
}
