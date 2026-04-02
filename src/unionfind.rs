use std::sync::atomic::{AtomicU32, Ordering};

/// Flag bit: if set, the value represents a rank (node is a root);
/// if clear, the value is a parent pointer (node is not a root).
const RANK_FLAG: u32 = 0x8000_0000;

#[inline]
fn is_rank(val: u32) -> bool {
    val & RANK_FLAG != 0
}

#[inline]
fn rank_value(val: u32) -> u32 {
    val & !RANK_FLAG
}

#[inline]
fn make_rank(r: u32) -> u32 {
    r | RANK_FLAG
}

/// Lock-free concurrent union-find using rank-based union and path compression,
/// based on "Concurrent Disjoint Set Union" (Listing 3).
///
/// Each slot in the array stores either:
/// - A rank value with the high bit set (the node is a root with that rank)
/// - A parent pointer with the high bit clear (the node points to its parent)
///
/// All query and union operations use CAS and are safe to call from multiple
/// threads without external synchronization.
pub struct ConcurrentUnionFind {
    data: Vec<AtomicU32>,
}

impl ConcurrentUnionFind {
    pub fn with_size(size: usize) -> Self {
        ConcurrentUnionFind { data: (0..size).map(|_| AtomicU32::new(make_rank(0))).collect() }
    }
    // pub fn new() -> Self {
    //     ConcurrentUnionFind { data: Vec::new() }
    // }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    // /// Create a new singleton set with rank 0. Returns its id.
    // /// Must be called single-threaded (during the "add" phase).
    // pub fn make_set(&mut self) -> u32 {
    //     let id = self.data.len() as u32;
    //     self.data.push(AtomicU32::new(make_rank(0)));
    //     id
    // }

    /// Find the root of `u` and its rank, with path compression via CAS.
    ///
    /// Lock-free, safe to call concurrently. Recursion depth is O(log n)
    /// due to rank-based union.
    ///
    /// Corresponds to Listing 3, lines 16-23.
    pub fn find(&self, u: u32) -> (u32, u32) {
        let p = self.data[u as usize].load(Ordering::Acquire);
        if is_rank(p) {
            return (u, rank_value(p));
        }
        // p is a parent pointer; recurse to find the root
        let (root, rank) = self.find(p);
        // Path compression: try to point u directly to root
        if p != root {
            let _ = self.data[u as usize].compare_exchange(
                p,
                root,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
        (root, rank)
    }

    /// Find the root of `u`, discarding the rank.
    #[inline]
    pub fn find_root(&self, u: u32) -> u32 {
        self.find(u).0
    }

    /// Union the sets containing `u` and `v`.
    ///
    /// Lock-free, safe to call concurrently. Uses rank-based priority
    /// with tie-breaking by node id (smaller merges into larger).
    ///
    /// Corresponds to Listing 3, lines 1-15.
    pub fn union(&self, u: u32, v: u32) {
        loop {
            let (u_root, ru) = self.find(u);
            let (v_root, rv) = self.find(v);

            if u_root == v_root {
                return;
            }

            if ru < rv {
                // Merge u into v (v has higher rank)
                if self.data[u_root as usize]
                    .compare_exchange(make_rank(ru), v_root, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    return;
                }
            } else if ru > rv {
                // Merge v into u (u has higher rank)
                if self.data[v_root as usize]
                    .compare_exchange(make_rank(rv), u_root, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    return;
                }
            } else {
                // Equal ranks: smaller id merges into larger id
                let (lo, hi) = if u_root < v_root {
                    (u_root, v_root)
                } else {
                    (v_root, u_root)
                };
                let r = ru; // == rv
                if self.data[lo as usize]
                    .compare_exchange(make_rank(r), hi, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    // Try to increment the winner's rank (best-effort)
                    let _ = self.data[hi as usize].compare_exchange(
                        make_rank(r),
                        make_rank(r + 1),
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                    return;
                }
            }
            // A CAS failed; retry from the top
        }
    }

    /// Check whether `u` and `v` are in the same set.
    ///
    /// Lock-free, safe to call concurrently.
    ///
    /// Corresponds to Listing 3, lines 25-30.
    pub fn same_set(&self, u: u32, v: u32) -> bool {
        loop {
            let (u_root, _) = self.find(u);
            let (v_root, _) = self.find(v);
            if u_root == v_root {
                return true;
            }
            // Verify u_root is still a root (linearizability check)
            let p = self.data[u_root as usize].load(Ordering::Acquire);
            if is_rank(p) {
                return false;
            }
            // u_root was concurrently merged; retry
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_union_find() {
        let mut uf = ConcurrentUnionFind::new();
        let a = uf.make_set();
        let b = uf.make_set();
        let c = uf.make_set();

        assert!(!uf.same_set(a, b));
        assert!(!uf.same_set(b, c));

        uf.union(a, b);
        assert!(uf.same_set(a, b));
        assert!(!uf.same_set(a, c));

        uf.union(b, c);
        assert!(uf.same_set(a, c));
    }

    #[test]
    fn find_returns_consistent_root() {
        let mut uf = ConcurrentUnionFind::new();
        let ids: Vec<u32> = (0..10).map(|_| uf.make_set()).collect();

        // Chain: 0-1, 1-2, ..., 8-9
        for i in 0..9 {
            uf.union(ids[i], ids[i + 1]);
        }

        let root = uf.find_root(ids[0]);
        for &id in &ids {
            assert_eq!(uf.find_root(id), root);
        }
    }

    #[test]
    fn concurrent_unions() {
        use rayon::prelude::*;

        let mut uf = ConcurrentUnionFind::new();
        let n = 1000u32;
        for _ in 0..n {
            uf.make_set();
        }

        // Union all even numbers together in parallel: (0,2), (2,4), ...
        let pairs: Vec<(u32, u32)> = (0..n - 2).step_by(2).map(|i| (i, i + 2)).collect();
        pairs.par_iter().for_each(|&(a, b)| {
            uf.union(a, b);
        });

        // All even numbers should share a root
        for i in (0..n).step_by(2) {
            assert!(uf.same_set(0, i), "0 and {i} should be in the same set");
        }
        // Odd numbers should be separate from evens
        assert!(!uf.same_set(0, 1));
    }
}
