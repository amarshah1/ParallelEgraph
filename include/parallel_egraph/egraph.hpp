#pragma once
// E-graph trimmed to exactly what the bench needs: build via add(), then
// run sequential_close_nelson or parallel_close on a list of equalities.
// Everything else (the multiple parallel_rebuild variants, sequential
// merge/repair/rebuild worklist, parallel_merge_all, clone, num_classes,
// the SAT integration, the SMT-LIB solver, the CLI) was removed.

#include <atomic>
#include <cstdint>
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
  explicit EGraph(std::size_t capacity);

  // Lock-free; safe to call concurrently. Path compression CAS is a write,
  // so this is non-const.
  Id find(Id id) { return uf_.find_root(id); }

  // Insert (and dedupe via hashcons) a single e-node. Sequential — runs
  // during the build phase before any closure call.
  Id add(ENode node);

  // BSP closure on explicit initial unions; fully parallel within rounds.
  // See cpp/DESIGN.md §1 (parallel_consolidate) and §2
  // (merge_and_collect_semisort) for the algorithm.
  void parallel_close(parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Sequential Nelson-style closure baseline used by the bench.
  void sequential_close_nelson(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  bool equiv(Id a, Id b) { return uf_.find_root(a) == uf_.find_root(b); }

  // Internal but exposed for tests / benches that need raw access.
  ConcurrentUnionFind& uf() { return uf_; }
  const parlay::sequence<std::pair<ENode, Id>>& nodes() const { return nodes_; }

 private:
  Id make_id() { return uf_.make_set(); }
  ENode canonicalize(const ENode& node);

  ConcurrentUnionFind uf_;
  parlay::sequence<std::pair<ENode, Id>> nodes_;
  parlay::sequence<parlay::sequence<Id>> parent_index_;

  // Hashcons for dedup during add(). Sequential-only access.
  std::unordered_map<ENode, Id, ENodeHash> hashcons_;
};

}  // namespace pe
