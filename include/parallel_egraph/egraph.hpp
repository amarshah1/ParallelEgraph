#pragma once
// E-graph trimmed to exactly what the closure benchmark needs: bulk_init
// to construct from a flat sequence of ENodes, then parallel_close /
// sequential_close_nelson on a list of equalities. No incremental add()
// — all consumers go through bulk_init, so nodes_ is sparsely indexed by
// class id (the array index *is* the class id).

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
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

// ---- Signature helpers (used during BSP rounds) ---------------------------

std::uint64_t sig_hash(const ENode& node, ConcurrentUnionFind& uf);
std::uint64_t sig_hash_seq(const ENode& node, SequentialUnionFind& uf);

// Returns a 64-bit primary hash and a 32-bit secondary hash computed in a
// single pass over the node's children (one set of UF lookups). The
// secondary uses a different FxHasher seed; combined 96-bit entropy makes
// hash collisions across distinct signatures effectively impossible
// (≈10⁻¹⁴ across a 64M-element batch on the largest synthetic workload).
std::pair<std::uint64_t, std::uint32_t> sig_hashes(
    const ENode& node, ConcurrentUnionFind& uf);

bool sigs_equal(std::uint32_t ia, std::uint32_t ib,
                ConcurrentUnionFind& uf,
                const parlay::sequence<ENode>& nodes);

bool sigs_equal_seq(std::uint32_t ia, std::uint32_t ib,
                    SequentialUnionFind& uf,
                    const parlay::sequence<ENode>& nodes);

int sig_cmp(std::uint32_t ia, std::uint32_t ib,
            ConcurrentUnionFind& uf,
            const parlay::sequence<ENode>& nodes);

// ---- EGraph ---------------------------------------------------------------

class EGraph {
 public:
  explicit EGraph(std::size_t capacity);

  // Lock-free; safe to call concurrently. Path compression CAS is a write,
  // so this is non-const.
  Id find(Id id) { return uf_.find_root(id); }

  // Bulk-construct an EGraph from a sequence of ENodes given in DAG order:
  // the i-th node receives class id i, and every child id in node i must
  // be < i. The caller is responsible for ensuring no two nodes are
  // structurally identical (no hashcons dedup is performed). nodes_ ends
  // up class-id-indexed (sparse), so leaves occupy slots in nodes_ but
  // contribute nothing to parent_index_. Returns a unique_ptr because
  // EGraph holds atomics (non-movable).
  static std::unique_ptr<EGraph> bulk_init(parlay::sequence<ENode> nodes);

  // BSP closure on explicit initial unions; fully parallel within rounds.
  // See DESIGN.md §1 (parallel_consolidate) and §2 (merge_and_collect_semisort)
  // for the algorithm.
  void parallel_close(parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Sequential Nelson-style closure baseline used by the bench.
  void sequential_close_nelson(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  bool equiv(Id a, Id b) { return uf_.find_root(a) == uf_.find_root(b); }

  // Internal but exposed for tests / benches that need raw access.
  ConcurrentUnionFind& uf() { return uf_; }
  const parlay::sequence<ENode>& nodes() const { return nodes_; }
  parlay::sequence<parlay::sequence<Id>>& parent_index() { return parent_index_; }

 private:
  ConcurrentUnionFind uf_;
  parlay::sequence<ENode> nodes_;
  parlay::sequence<parlay::sequence<Id>> parent_index_;
};

}  // namespace pe
