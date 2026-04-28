#pragma once
// E-graph with hashcons, congruence closure, BSP parallel rebuild.
// 1:1 port of src/egraph.rs.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <parlay/sequence.h>

#include "parallel_egraph/concurrent_append_vec.hpp"
#include "parallel_egraph/fxhash.hpp"
#include "parallel_egraph/unionfind.hpp"

namespace pe {

struct ENode {
  std::string op;
  std::vector<Id> children;

  bool operator==(const ENode& other) const {
    return op == other.op && children == other.children;
  }
};

// Hasher for hashcons keyed on fully-canonicalized ENodes.
struct ENodeHash {
  std::size_t operator()(const ENode& n) const noexcept;
};

// ---- Signature helpers (used during BSP rounds) ---------------------------

std::uint64_t sig_hash(const ENode& node, ConcurrentUnionFind& uf);
std::uint64_t sig_hash_seq(const ENode& node, SequentialUnionFind& uf);

bool sigs_equal(std::uint32_t ia, std::uint32_t ib,
                ConcurrentUnionFind& uf,
                const parlay::sequence<std::pair<ENode, Id>>& nodes);

bool sigs_equal_seq(std::uint32_t ia, std::uint32_t ib,
                    SequentialUnionFind& uf,
                    const parlay::sequence<std::pair<ENode, Id>>& nodes);

int sig_cmp(std::uint32_t ia, std::uint32_t ib,
            ConcurrentUnionFind& uf,
            const parlay::sequence<std::pair<ENode, Id>>& nodes);

// ---- EGraph ---------------------------------------------------------------

class EGraph {
 public:
  EGraph(std::size_t capacity, bool parallel);
  static EGraph sequential(std::size_t capacity) { return EGraph(capacity, false); }
  static EGraph parallel(std::size_t capacity) { return EGraph(capacity, true); }

  bool is_parallel() const { return parallel_; }

  // Non-const: path compression is a CAS write. Safe to call concurrently.
  Id find(Id id) { return uf_.find_root(id); }

  Id add(ENode node);
  Id merge(Id a, Id b);

  // ---- Proof tracking ----------------------------------------------------
  // Opt-in: enable by calling merge_with_reason / rebuild_with_reasons /
  // parallel_rebuild_with_reasons instead of merge / rebuild /
  // parallel_rebuild. The default code paths (merge/rebuild/parallel_rebuild)
  // are unchanged and pay zero cost.
  //
  // ProofReason records why two classes were merged. Two kinds:
  //   Asserted   — came from a SAT-side literal (`sat_lit`).
  //   Congruence — produced by rebuild_with_reasons because two e-nodes
  //                f(c1..ck) and f(c'1..c'k) had pairwise-congruent
  //                children. The reason stores the *raw* child class
  //                ids (NOT canonicalized — explain() does the find()s
  //                itself), so explain() can recurse with
  //                explain(children_a[i], children_b[i]) for each i.
  struct ProofReason {
    enum class Kind : std::uint8_t { Asserted = 1, Congruence = 2 };
    Kind kind = Kind::Asserted;
    int sat_lit = 0;                       // when kind == Asserted
    std::vector<Id> children_a;            // when kind == Congruence
    std::vector<Id> children_b;            // when kind == Congruence
    static ProofReason asserted(int lit) {
      ProofReason r; r.kind = Kind::Asserted; r.sat_lit = lit; return r;
    }
    static ProofReason congruence(std::vector<Id> ca, std::vector<Id> cb) {
      ProofReason r; r.kind = Kind::Congruence;
      r.children_a = std::move(ca);
      r.children_b = std::move(cb);
      return r;
    }
  };

  // Like merge(a, b) but records the merge in the lock-free append-only
  // merge_log_ and updates the per-class explain_adj_. Safe to call
  // concurrently with itself for distinct (a, b) pairs (the union-find
  // is lock-free and the proof log uses ConcurrentAppendVec).
  Id merge_with_reason(Id a, Id b, ProofReason r);

  // Sequential rebuild that records every congruence merge via
  // merge_with_reason. Mirrors rebuild() except it calls
  // merge_with_reason instead of merge.
  void rebuild_with_reasons();

  // Parallel rebuild that records every congruence merge via
  // merge_with_reason. Mirrors parallel_rebuild_semisort: same BSP
  // round structure, same parallel primitives. Each merge in the apply
  // step records a Congruence reason carrying both witness e-nodes'
  // raw children.
  void parallel_rebuild_with_reasons();

  // Returns a small set of SAT literals whose conjunction forces a == b.
  // BFS over the merge_log_ adjacency from a until reaching b, walks
  // back to collect events, recursively expands each Congruence reason
  // into proofs of its child equalities, then deduplicates the
  // resulting flat list of asserted reasons via a fresh union-find:
  // keep a reason iff its endpoints aren't yet equivalent. The kept
  // set is a spanning forest of the conflict's relevant component.
  // Precondition: a and b are in the same e-class.
  std::vector<int> explain(Id a, Id b);

  // Deep clone of the e-graph state for snapshot/restore in DPLL(T).
  // Replays the recorded add/merge history into a fresh EGraph of the
  // same capacity. EGraph contains atomics (non-copyable, non-movable),
  // so this is the only way to duplicate state.
  //
  // Cost: O(history size). Acceptable for v1 of the SAT integration; the
  // design doc (cpp/SAT_INTEGRATION.md) lays out the trail-based scheme
  // that replaces this in a later iteration.
  std::unique_ptr<EGraph> clone() const;

  // Each entry is an asserted equality (a == b), tagged with the SAT
  // literal that caused it (lit; 0 if no literal — the legacy non-SAT
  // path). Used as the input type for both parallel_merge_all (which
  // ignores `lit`) and parallel_merge_all_reason (which logs `lit` as
  // an Asserted reason). One shared shape so the propagator can pass
  // its `pos_eqs` to either method without conversion.
  struct EqLit { int lit; Id a; Id b; };

  // Batch-parallel merge. Ignores eqs[i].lit; for proof tracking, use
  // parallel_merge_all_reason instead.
  void parallel_merge_all(const parlay::sequence<EqLit>& eqs);

  // Like parallel_merge_all but logs an Asserted(eqs[i].lit) reason for
  // each merge.
  void parallel_merge_all_reason(const parlay::sequence<EqLit>& eqs);

  // Sequential congruence closure (worklist).
  void rebuild();

  // Parallel rebuild. Dispatcher reads env var PE_REBUILD.
  void parallel_rebuild();
  void parallel_rebuild_semisort();
  void parallel_rebuild_sort();

  // BSP closure invoked directly with explicit initial unions. Fully parallel
  // within each round (see cpp/src/egraph.cpp for the single flagged
  // exception: per-group concat inside consolidation).
  void parallel_close(parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Sequential Nelson-style closure baseline (rebuild_compare harness).
  void sequential_close_nelson(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  bool equiv(Id a, Id b) { return uf_.find_root(a) == uf_.find_root(b); }
  std::size_t num_classes() const { return classes_.size(); }
  std::size_t num_enodes() const;
  void print() const;

  // Internal but exposed for tests / benches that need raw union-find access.
  ConcurrentUnionFind& uf() { return uf_; }
  const parlay::sequence<std::pair<ENode, Id>>& nodes() const { return nodes_; }

 private:
  Id make_id() { return uf_.make_set(); }
  ENode canonicalize(const ENode& node);

  void repair(Id id);
  void parallel_merge(Id a, Id b);
  void parallel_merge_reason(Id a, Id b, ProofReason r);
  void grow_changed_to(std::size_t n_classes);

  ConcurrentUnionFind uf_;

  // Parallel-mode primary structures. parlay::sequence throughout to match
  // parlay's pipeline and stay out of std::vector anywhere near the hot path.
  parlay::sequence<std::pair<ENode, Id>> nodes_;
  parlay::sequence<parlay::sequence<Id>> parent_index_;
  // Per-class changed flags; atomic<bool> is non-movable so we size the
  // sequence up-front.
  parlay::sequence<std::atomic<bool>> changed_;

  // Sequential-mode aux structures (not used on the parallel hot path).
  std::unordered_map<Id, std::vector<ENode>> classes_;
  std::unordered_map<Id, std::vector<std::pair<ENode, Id>>> parents_;
  std::vector<Id> worklist_;

  // Shared: hashcons for dedup during add(). Sequential-only access.
  std::unordered_map<ENode, Id, ENodeHash> hashcons_;

  // ---- Proof-tracking state (only populated via merge_with_reason) -------
  // Append-only log of merges. Each entry is a (a, b, reason) triple,
  // where (a, b) are the *pre-find* class ids being unioned. Lock-free
  // concurrent append; index returned by append() is stable.
  struct MergeEvent {
    Id          a, b;
    ProofReason reason;
  };
  ConcurrentAppendVec<MergeEvent> merge_log_;

  // Per-class adjacency for explain BFS. explain_adj_[c] is a list of
  // (neighbor class id, log-event-index) pairs. Concurrent appends
  // come from merge_with_reason: each merge appends to the two
  // endpoints. The outer vector is only resized in add() (sequential),
  // so we don't need ConcurrentAppendVec at the outer level.
  std::vector<ConcurrentAppendVec<std::pair<Id, std::uint64_t>>> explain_adj_;

  bool parallel_;
};

}  // namespace pe
