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

  // ---- Explanations via adjacency log ------------------------------------
  // Opt-in proof-tracking: enable via `merge_with_reason` instead of
  // `merge`. The default code path (merge / rebuild / parallel_rebuild)
  // is unchanged and pays zero cost.
  //
  // ProofReason records why two classes were merged. Two kinds:
  //   Asserted   — came from a SAT-side literal (`sat_lit`).
  //   Congruence — produced by rebuild_with_reasons because two e-nodes
  //                f(c1..ck) and f(c'1..c'k) had pairwise-congruent
  //                children. The reason stores the *raw* child class
  //                ids (NOT canonicalized — explain() does the
  //                find()s itself), so explain() can recurse with
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

  // Sequential-only: like merge(a, b) but records the merge in the
  // append-only `merge_log_` and updates `explain_adj_`. Each merge is
  // O(1) extra work; no path-flipping needed.
  Id merge_with_reason(Id a, Id b, ProofReason r);

  // Sequential rebuild that records congruence reasons via
  // merge_with_reason as it discovers them. Pair with merge_with_reason
  // for asserted equalities to get a complete proof log.
  void rebuild_with_reasons();

  // Returns a small set of SAT literals whose conjunction forces a == b.
  // Algorithm:
  //   1. BFS over the adjacency log from `a` until reaching `b`.
  //   2. Recursively expand each Congruence reason on the path into
  //      proofs of its children's equalities.
  //   3. Deduplicate the resulting flat list of asserted reasons by
  //      replaying them through a fresh union-find: keep a reason iff
  //      its two endpoints are not yet equivalent. The kept set is a
  //      spanning forest of the conflict's relevant component.
  // Precondition: a and b are in the same e-class. Output may be empty
  // if a == b.
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

  // Batch-parallel merge. Takes parlay::sequence to keep hot-path code free
  // of std::vector.
  void parallel_merge_all(const parlay::sequence<std::pair<Id, Id>>& pairs);

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

  // ---- Explanation log (adjacency-based) ---------------------------------
  // Append-only log of merges. Populated by merge_with_reason; untouched
  // by the existing default merge / rebuild / parallel_rebuild paths.
  struct MergeEvent {
    Id           a, b;        // class ids at time of merge (pre-find)
    ProofReason  reason;
  };
  std::vector<MergeEvent> merge_log_;

  // Per-class adjacency for explain BFS. explain_adj_[c] is a list of
  // (neighbor class id, log event index) pairs. Maintained incrementally
  // by merge_with_reason: each merge appends to the two endpoints.
  // Sized lazily via add() (resized on each make_id).
  std::vector<std::vector<std::pair<Id, std::uint32_t>>> explain_adj_;

  bool parallel_;
};

}  // namespace pe
