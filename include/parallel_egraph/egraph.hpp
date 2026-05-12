#pragma once
// E-graph trimmed to exactly what the closure benchmark needs: a single
// ctor that takes a flat sequence of ENodes, then parallel_parents /
// sequential_close_nelson on a list of equalities. No incremental add() —
// all consumers go through the bulk ctor, so nodes_ is sparsely indexed
// by class id (the array index *is* the class id).
//
// Templated on the union-find type. `EGraph<ConcurrentUnionFind>` carries
// `parallel_parents` and is used from the parallel BSP path;
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

// Tag type selecting the auxiliary state populated by the EGraph ctor.
// The default (no tag) is the BSP layout — `parents_` populated, used by
// `parallel_parents`. The filter layout populates `last_marked_` instead
// (per-class round-stamp tracking when each class was most recently
// rerooted) and is used by `parallel_filter`. Each closure
// path uses only its own auxiliary state, so the bench can A/B compare
// without paying for the wrong-flavor setup.
struct filter_t { explicit filter_t() = default; };
inline constexpr filter_t filter{};

// Round-stamp slot used by the async closure path. Padded to 64 bytes
// (one cache line) so that concurrent CAS-max updates on adjacent
// class ids don't false-share. Without padding, eight slots fit in
// one cache line: at high core counts the MESI traffic from
// updating slot[3] invalidates slot[0..7] on every other core,
// turning an O(unions) atomic workload into an O(unions × cores)
// cache-coherence storm. With padding, distinct slots are on
// distinct cache lines, so threads can update unrelated classes in
// parallel without trashing each other's caches. Cost is 8× memory
// (64B per slot vs 8B), which is fine on the supercomputer at
// quintic-n=18 scale (~240MB).
struct alignas(64) PaddedMark {
  std::atomic<std::uint64_t> v;
  // Trailing padding fills the rest of the cache line; sized to keep
  // the struct exactly 64 bytes regardless of std::atomic<uint64_t>'s
  // alignment requirements on this target.
  char _pad[64 - sizeof(std::atomic<std::uint64_t>)];
};
static_assert(sizeof(PaddedMark) == 64,
              "PaddedMark must be exactly one cache line");
static_assert(alignof(PaddedMark) == 64,
              "PaddedMark must be cache-line aligned");

template <typename UF>
class EGraph {
 public:
  // BSP-flavor ctor: populate `parents_` (the inverted child→parent
  // index used by the BSP frontier walk). Leaves `last_marked_`
  // empty; calling `parallel_filter` on an instance
  // built this way is undefined.
  //
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

  // Async-flavor ctor: skip `parents_` setup entirely; allocate
  // `last_marked_` with one slot per class, default-initialized to 0.
  // The closure uses these as round-stamps: when class c gets unioned
  // into a new root r, last_marked_[r] is set to the current global
  // round R via a monotone CAS-max. A term `t` is dirty in round R iff
  // some child c of t has last_marked_[find_root(c)] ∈ {R-1, R}.
  // Calling `parallel_parents` on an instance built this way is
  // undefined: `parents_` is empty.
  EGraph(parlay::sequence<ENode> nodes, filter_t) : uf_(nodes.size()) {
    const std::size_t n = nodes.size();
    uf_.bulk_init(n);
    nodes_ = std::move(nodes);
    // parlay::sequence<PaddedMark> supports in-place construction via
    // uninitialized + parallel placement; we then zero-init each
    // slot's atomic. PaddedMark's default ctor would leave .v
    // unspecified, so the explicit store is required. PaddedMark is
    // 64B so distinct slots live on distinct cache lines — eliminates
    // false sharing on `last_marked_` at high core counts.
    last_marked_ = parlay::sequence<PaddedMark>::uninitialized(n);
    parlay::parallel_for(0, n, [&](std::size_t i) {
      new (&last_marked_[i]) PaddedMark{std::atomic<std::uint64_t>(0), {}};
    });
  }

  // Lock-free for ConcurrentUnionFind; safe to call concurrently. Path
  // compression CAS is a write, so this is non-const.
  Id find(Id id) { return uf_.find_root(id); }

  // BSP closure on explicit initial unions; fully parallel within rounds.
  // See DESIGN.md §1 (parallel_consolidate) and §2 (merge_and_collect_semisort)
  // for the algorithm. Defined out-of-line in src/egraph.cpp only as an
  // explicit specialization on `ConcurrentUnionFind` — calling it on the
  // sequential flavor is a link error.
  void parallel_parents(parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Async-style filter closure. Drops the `parents_` machinery in
  // favor of round-stamp tracking on each class: `last_marked_[r]` is
  // the round number when class r last gained a new member (or was
  // demoted into another root). Each round filters to "dirty" terms
  // (any child whose root has last_marked ∈ {R-1, R}), semisorts them
  // by current canonical sig, dnc_unions per bucket, and re-stamps
  // surviving roots. Loops until dirty is empty. Defined out-of-line
  // as an explicit specialization on `ConcurrentUnionFind`.
  void parallel_filter(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Most general sequential CC: one inline-storage hashtable per
  // arity bucket (`SigK<K>` for arities 1..K=4), plus a bump-allocated
  // `SigBump` fallback for arity > K. Each per-arity table's
  // `operator==` and hash loops are fully unrolled at compile time;
  // every signature lives entirely in its own struct (no heap traffic
  // on the fast path). Same deterministic structural correctness as
  // the other simple-family variants. Defined out-of-line as an
  // explicit specialization on `SequentialUnionFind`.
  void sequential_close_simple_inline(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  bool equiv(Id a, Id b) { return uf_.find_root(a) == uf_.find_root(b); }

  // Internal but exposed for tests / benches that need raw access.
  UF& uf() { return uf_; }
  const parlay::sequence<ENode>& nodes() const { return nodes_; }
  parlay::sequence<parlay::sequence<Id>>& parents() { return parents_; }
  parlay::sequence<PaddedMark>& last_marked() {
    return last_marked_;
  }

 private:
  UF uf_;
  parlay::sequence<ENode> nodes_;
  // BSP-only state. Populated by the default ctor; left empty by the
  // async ctor.
  parlay::sequence<parlay::sequence<Id>> parents_;
  // Async-only state. Populated by the `filter_t`-tagged ctor; left
  // empty by the default ctor. last_marked_[r].v = highest round
  // number for which class r was unioned-into-as-the-new-root.
  // Updated via parlay::write_max (monotone CAS-max), so a tail
  // worker holding a stale R can never overwrite a fresher mark.
  // Each entry is cache-line padded (PaddedMark = 64B) to prevent
  // false sharing between adjacent class ids' marks. Indexed by
  // class id.
  parlay::sequence<PaddedMark> last_marked_;
};

using ConcurrentEGraph = EGraph<ConcurrentUnionFind>;
using SequentialEGraph = EGraph<SequentialUnionFind>;

}  // namespace pe
