#pragma once
// E-graph trimmed to exactly what the closure benchmark needs: a single
// ctor that takes a flat sequence of ENodes, then parallel_close /
// sequential_close_nelson on a list of equalities. No incremental add() —
// all consumers go through the bulk ctor, so nodes_ is sparsely indexed
// by class id (the array index *is* the class id).
//
// Templated on the union-find type. `EGraph<ConcurrentUnionFind>` carries
// `parallel_close` and is used from the parallel BSP path;
// `EGraph<SequentialUnionFind>` carries `sequential_close_nelson` and is
// used from the sequential Nelson baseline. Each instance constructs only
// the UF it needs — no bridging between them.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>

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
//
// Templated on the union-find type so the parallel path
// (`ConcurrentUnionFind`) and sequential Nelson baseline
// (`SequentialUnionFind`) share one implementation. Both UF types expose
// `find_root(Id) -> Id`; that's the only requirement on `UF`.

template <typename UF>
std::uint64_t sig_hash(const ENode& node, UF& uf) {
  FxHasher h;
  h.write_str(node.op);
  for (Id c : node.children) h.write_u32(uf.find_root(c));
  return h.finish();
}

// Returns a 64-bit primary hash and a 32-bit secondary hash computed in a
// single pass over the node's children (one set of UF lookups). The
// secondary uses a different FxHasher seed; combined 96-bit entropy makes
// hash collisions across distinct signatures effectively impossible
// (≈10⁻¹⁴ across a 64M-element batch on the largest synthetic workload).
template <typename UF>
std::pair<std::uint64_t, std::uint32_t> sig_hashes(
    const ENode& node, UF& uf) {
  // Golden-ratio-derived seed; structurally unrelated to FxHasher's mixing
  // constant, keeping h1 and h2 effectively independent.
  static constexpr std::uint64_t kSigHash2Seed = 0x9E37'79B9'7F4A'7C15ULL;
  FxHasher h1;
  FxHasher h2(kSigHash2Seed);
  h1.write_str(node.op);
  h2.write_str(node.op);
  for (Id c : node.children) {
    Id r = uf.find_root(c);
    h1.write_u32(r);
    h2.write_u32(r);
  }
  return {h1.finish(), static_cast<std::uint32_t>(h2.finish())};
}

template <typename UF>
bool sigs_equal(std::uint32_t ia, std::uint32_t ib, UF& uf,
                const parlay::sequence<ENode>& nodes) {
  const auto& na = nodes[ia];
  const auto& nb = nodes[ib];
  if (na.op != nb.op) return false;
  if (na.children.size() != nb.children.size()) return false;
  for (std::size_t i = 0; i < na.children.size(); ++i) {
    if (uf.find_root(na.children[i]) != uf.find_root(nb.children[i])) {
      return false;
    }
  }
  return true;
}

// ---- EGraph ---------------------------------------------------------------

template <typename UF>
class EGraph {
 public:
  // Bulk-construct from a sequence of ENodes given in DAG order: the i-th
  // node receives class id i, and every child id in node i must be < i.
  // The caller is responsible for ensuring no two nodes are structurally
  // identical (no hashcons dedup). nodes_ ends up class-id-indexed
  // (sparse), so leaves occupy slots in nodes_ but contribute nothing to
  // parents_. EGraph<ConcurrentUnionFind> holds atomics (non-movable),
  // so callers heap-allocate via `std::make_unique<EGraph<UF>>(...)`.
  explicit EGraph(parlay::sequence<ENode> nodes) : uf_(nodes.size()) {
    const std::size_t n = nodes.size();
    uf_.bulk_init(n);

    auto child_pairs = parlay::flatten(parlay::tabulate(
        n, [&](std::size_t i) -> parlay::sequence<std::pair<Id, Id>> {
          const auto& cs = nodes[i].children;
          parlay::sequence<std::pair<Id, Id>> result;
          result.reserve(cs.size());
          Id parent = static_cast<Id>(i);
          for (Id c : cs) result.emplace_back(c, parent);
          return result;
        }));

    parents_ = parlay::group_by_index(
        std::move(child_pairs), static_cast<Id>(n));
    nodes_ = std::move(nodes);
  }

  // Lock-free for ConcurrentUnionFind; safe to call concurrently. Path
  // compression CAS is a write, so this is non-const.
  Id find(Id id) { return uf_.find_root(id); }

  // BSP closure on explicit initial unions; fully parallel within rounds.
  // See DESIGN.md §1 (parallel_consolidate) and §2 (merge_and_collect_semisort)
  // for the algorithm. Defined out-of-line in src/egraph.cpp only as an
  // explicit specialization on `ConcurrentUnionFind` — calling it on the
  // sequential flavor is a link error.
  void parallel_close(parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Sequential Nelson-style closure baseline. Defined out-of-line only as
  // an explicit specialization on `SequentialUnionFind`.
  void sequential_close_nelson(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  bool equiv(Id a, Id b) { return uf_.find_root(a) == uf_.find_root(b); }

  // Internal but exposed for tests / benches that need raw access.
  UF& uf() { return uf_; }
  const parlay::sequence<ENode>& nodes() const { return nodes_; }
  parlay::sequence<parlay::sequence<Id>>& parents() { return parents_; }

 private:
  UF uf_;
  parlay::sequence<ENode> nodes_;
  parlay::sequence<parlay::sequence<Id>> parents_;
};

using ConcurrentEGraph = EGraph<ConcurrentUnionFind>;
using SequentialEGraph = EGraph<SequentialUnionFind>;

}  // namespace pe
