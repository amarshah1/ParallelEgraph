use std::collections::HashMap;

pub type Id = u32;

/// An e-node: an operator applied to e-class IDs.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
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
pub struct EGraph {
    // Union-find
    parent: Vec<Id>,
    size: Vec<u32>,

    // E-class id -> e-nodes in that class
    classes: HashMap<Id, Vec<ENode>>,

    // For each e-class, the list of (enode, eclass_of_enode) where enode
    // has this class as a child (the "use list" / "parent list")
    parents: HashMap<Id, Vec<(ENode, Id)>>,

    // Canonical e-node -> e-class id
    hashcons: HashMap<ENode, Id>,

    // E-classes needing congruence repair
    worklist: Vec<Id>,

    next_id: Id,
}

impl EGraph {
    pub fn new() -> Self {
        EGraph {
            parent: Vec::new(),
            size: Vec::new(),
            classes: HashMap::new(),
            parents: HashMap::new(),
            hashcons: HashMap::new(),
            worklist: Vec::new(),
            next_id: 0,
        }
    }

    fn make_id(&mut self) -> Id {
        let id = self.next_id;
        self.next_id += 1;
        self.parent.push(id);
        self.size.push(1);
        id
    }

    /// Find the canonical representative with path halving.
    pub fn find(&self, mut id: Id) -> Id {
        while self.parent[id as usize] != id {
            // self.parent[id as usize] = self.parent[self.parent[id as usize] as usize];
            id = self.parent[id as usize];
        }
        id
    }

    /// Canonicalize an e-node: replace each child with its find root.
    fn canonicalize(&mut self, node: &ENode) -> ENode {
        let children = node.children.iter().map(|&c| self.find(c)).collect();
        ENode { op: node.op.clone(), children }
    }

    /// Add a single e-node. Returns the e-class id it belongs to.
    pub fn add(&mut self, node: ENode) -> Id {
        let canon = self.canonicalize(&node);

        // Check hashcons for an existing congruent node
        if let Some(&id) = self.hashcons.get(&canon) {
            return self.find(id);
        }

        // Fresh e-class
        let id = self.make_id();
        // Register as parent of each child
        for &child in &canon.children {
            self.parents.entry(child).or_default().push((canon.clone(), id));
        }
        self.classes.entry(id).or_default().push(canon.clone());
        self.hashcons.insert(canon, id);
        id
    }

    /// Merge two e-classes. Returns the new canonical id.
    pub fn merge(&mut self, a: Id, b: Id) -> Id {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            return a;
        }

        // Union by size: smaller merges into larger
        if self.size[a as usize] < self.size[b as usize] {
            std::mem::swap(&mut a, &mut b);
        }
        // b merges into a
        self.parent[b as usize] = a;
        self.size[a as usize] += self.size[b as usize];

        // Merge class contents
        if let Some(nodes_b) = self.classes.remove(&b) {
            self.classes.entry(a).or_default().extend(nodes_b);
        }

        // Merge parent (use) lists
        if let Some(parents_b) = self.parents.remove(&b) {
            self.parents.entry(a).or_default().extend(parents_b);
        }

        self.worklist.push(a);
        a
    }

    /// Restore the congruence invariant after merges.
    pub fn rebuild(&mut self) {
        while !self.worklist.is_empty() {
            let todo: Vec<Id> = std::mem::take(&mut self.worklist);
            for id in todo {
                self.repair(self.find(id));
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

    /// Check whether two e-class ids are equivalent.
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
}
