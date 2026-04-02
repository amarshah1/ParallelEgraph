use dashmap::DashSet;
use std::collections::HashSet;
use std::{collections::HashMap, sync::atomic::Ordering};
use std::sync::atomic::AtomicBool;

use crate::unionfind::ConcurrentUnionFind;

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
    /// Create a new e-graph in sequential mode.
    pub fn new(size: usize) -> Self {
        EGraph {
            size,
            uf: ConcurrentUnionFind::with_size(size),
            nodes: Vec::new(),
            parent_index: Vec::new(),
            changed: Vec::new(),
            classes: HashMap::new(),
            parents: HashMap::new(),
            worklist: Vec::new(),
            predecessors_modified: DashSet::new(),
            hashcons: HashMap::new(),
            parallel: false,
        }
    }

    /// Create a new e-graph in parallel mode.
    /// Batch merges via `parallel_merge_all` will use rayon for parallelism.
    pub fn new_parallel(size: usize) -> Self {
        EGraph {
            size,
            uf: ConcurrentUnionFind::with_size(size),
            nodes: Vec::new(),
            parent_index: Vec::new(),
            changed: Vec::new(),
            classes: HashMap::new(),
            parents: HashMap::new(),
            worklist: Vec::new(),
            predecessors_modified: DashSet::new(),
            hashcons: HashMap::new(),
            parallel: true,
        }
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

        pairs.par_iter().for_each(|&(a, b)| {
            self.parallel_merge(a, b);
        });
    }

    fn parallel_merge(&self, a: Id, b: Id) {
        use rayon::prelude::*;
        self.uf.union(a, b);
        self.changed[a as usize].store(true, Ordering::Release);
        self.changed[b as usize].store(true, Ordering::Release);
        // self.parent_index[a as usize].par_iter().for_each(|&idx| {
        //     self.predecessors_modified.insert(idx);
        // });
        // self.parent_index[a as usize].par_iter().for_each(|&idx| {
        //     self.predecessors_modified.insert(idx);
        // });
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
    pub fn parallel_rebuild(&mut self) {
        use rayon::prelude::*;

        
        loop {
            // self.predecessors_modified.iter().for_each(|idx| {
            //     self.changed[idx as usize].store(true, Ordering::Release);
            // });
            let predecessors = 
                self.changed.par_iter().enumerate().filter_map(|(idx, b)| {
                    if b.load(Ordering::Acquire) {
                        self.changed[idx].store(false, Ordering::Release);
                        Some(idx)
                    } else {
                        None
                    }
                }).collect::<Vec<_>>();



            

        }



        // Ensure parent_index and changed flags cover all class ids
        // self.parent_index.resize_with(self.uf.len(), Vec::new);
        // self.changed.resize(self.uf.len(), false);

        // loop {
        //     // 1. GATHER FRONTIER [push from changed classes via parent_index]
        //     //    O(frontier) — only visit parents of changed classes.
        //     let mut frontier_indices: Vec<usize> = Vec::new();
        //     for (class_id, &is_changed) in self.changed.iter().enumerate() {
        //         if is_changed {
        //             frontier_indices.extend_from_slice(&self.parent_index[class_id]);
        //         }
        //     }
        //     frontier_indices.sort_unstable();
        //     frontier_indices.dedup();

        //     if frontier_indices.is_empty() {
        //         break;
        //     }

        //     // 2. CANONICALIZE [par_iter + map]
        //     //    Parallel map: compute (canonical_signature, class_root).
        //     //    Safe because find() is lock-free.
        //     let uf = &self.uf;
        //     let nodes = &self.nodes;
        //     let mut canonicalized: Vec<(ENode, Id)> = frontier_indices
        //         .par_iter()
        //         .map(|&idx| {
        //             let (node, class_id) = &nodes[idx];
        //             let canon = ENode {
        //                 op: node.op.clone(),
        //                 children: node.children.iter().map(|&c| uf.find_root(c)).collect(),
        //             };
        //             (canon, uf.find_root(*class_id))
        //         })
        //         .collect();

        //     // 3. GROUP BY SIGNATURE — SEMISORT [par_sort_unstable]
        //     canonicalized.par_sort_unstable();

        //     // 4. EMIT MERGE CANDIDATES [sequential scan over sorted groups]
        //     //    O(frontier_size) — proportional to the parallel work already done.
        //     let mut merge_pairs: Vec<(Id, Id)> = Vec::new();
        //     let mut i = 0;
        //     while i < canonicalized.len() {
        //         // Find end of this group (consecutive equal signatures)
        //         let mut j = i + 1;
        //         while j < canonicalized.len() && canonicalized[j].0 == canonicalized[i].0 {
        //             j += 1;
        //         }
        //         // Emit merge pairs: chain the distinct classes in this group
        //         let first_root = self.find(canonicalized[i].1);
        //         for k in (i + 1)..j {
        //             let other_root = self.find(canonicalized[k].1);
        //             if other_root != first_root {
        //                 merge_pairs.push((first_root, other_root));
        //             }
        //         }
        //         i = j;
        //     }

        //     if merge_pairs.is_empty() {
        //         break;
        //     }

        //     // 5. APPLY MERGES [par_iter + for_each]
        //     //    Parallel union via lock-free CAS union-find.
        //     let uf = &self.uf;
        //     merge_pairs.par_iter().for_each(|&(a, b)| {
        //         uf.union(a, b);
        //     });

        //     // 6. UPDATE CHANGED FLAGS + MERGE PARENT INDEX
        //     //    Consolidate parent_index entries under the new root so
        //     //    future rounds find all parents. Mark both sides changed.
        //     self.changed.par_iter_mut().for_each(|c| *c = false);
        //     for &(a, b) in &merge_pairs {
        //         let root = self.uf.find_root(a) as usize;
        //         if (a as usize) != root {
        //             let entries = std::mem::take(&mut self.parent_index[a as usize]);
        //             self.parent_index[root].extend(entries);
        //         }
        //         if (b as usize) != root {
        //             let entries = std::mem::take(&mut self.parent_index[b as usize]);
        //             self.parent_index[root].extend(entries);
        //         }
        //         self.changed[root] = true;
        //     }
        // }

        // // Clear changed flags — no other cleanup needed.
        // // The union-find is the source of truth for class membership.
        // // nodes/parent_index remain valid (stale class_ids resolved via find()).
        // self.changed.iter_mut().for_each(|c| *c = false);
    }

    /// Restore the congruence invariant after merges.
    /// Dispatches to `parallel_rebuild` in parallel mode.
    pub fn rebuild(&mut self) {
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
