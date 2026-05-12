#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/detail.hpp"
#include "parallel_egraph/dnc_union.hpp"
#include "parallel_egraph/ring_buffer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>

namespace pe {

// `sig_hash`, `sig_hashes`, and `sigs_equal` are templated on the UF
// type and live in egraph.hpp — one definition serves both the parallel
// (ConcurrentUnionFind) and sequential (SequentialUnionFind) paths.
//
// `EGraph<UF>::EGraph` is defined inline in egraph.hpp; only the two
// close methods are out-of-line here, each as an explicit specialization
// on the UF type that algorithm requires.

// ---- parallel consolidation helper ---------------------------------------
//
// For each c in `cs` where c != root_of_c, migrate parents[c] into
// parents[root_of_c]. Outer parallel: group_by_key partitions the (root,
// non_root) pairs so each root r is written by exactly one task; the
// per-group body then concatenates parents[r] with each parents[c_i] via
// a single parlay::flatten that allocates the destination buffer once
// and scatters in parallel internally.

namespace detail {

void parallel_consolidate(
    parlay::sequence<parlay::sequence<Id>>& parents,
    const parlay::sequence<Id>& cs,
    const parlay::sequence<Id>& roots) {
  auto non_root_pairs = parlay::map_maybe(
      parlay::iota(cs.size()),
      [&](std::size_t i) -> std::optional<std::pair<Id, Id>> {
        if (cs[i] == roots[i]) return std::nullopt;
        return std::make_pair(roots[i], cs[i]);
      });
  if (non_root_pairs.empty()) return;

  auto groups = parlay::group_by_key(std::move(non_root_pairs));

  parlay::parallel_for(0, groups.size(), [&](std::size_t g) {
    Id rep = groups[g].first;
    const auto& members = groups[g].second;

    // Build the inputs sequence directly via tabulate: slot 0 = parents[rep],
    // slots 1..k = parents[m_i]. Each inner is moved (O(1) pointer transfer);
    // parlay::flatten then uninitialized-relocates them into one freshly-
    // allocated destination buffer (the sequence<sequence<T>>&& overload).
    // The previous implementation built a `sources` index list and a
    // `parlay::map(sources, …)` — both are gratuitous; tabulate fuses them.
    auto inputs = parlay::tabulate(members.size() + 1, [&](std::size_t i) {
      return std::move(i == 0 ? parents[rep] : parents[members[i - 1]]);
    });
    parents[rep] = parlay::flatten(std::move(inputs));
  });
}

}  // namespace detail

// `merge_and_collect_semisort` lives in semisort_sound.hpp. The shared
// dnc helpers — dnc_cutoff() and dnc_union() — are in dnc_union.hpp.

// ---- parallel_parents ------------------------------------------------------

template <>
void EGraph<ConcurrentUnionFind>::parallel_parents(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf_.union_(initial_unions[i].first, initial_unions[i].second);
  });

  auto work = parlay::flatten(parlay::map(initial_unions, [](auto p) {
    return parlay::sequence<Id>{p.first, p.second};
  }));

  std::size_t round = 0;
  while (!work.empty()) {
    work = parlay::remove_duplicates(std::move(work));

    auto t_consolidate = clk::now();
    auto roots = parlay::map(work, [&](Id c) { return uf_.find_root(c); });

    // Tail-round skip: peek at every relevant parents_ slot before
    // paying the consolidate cost. If both parents_[work[i]]
    // (pre-consolidate content of losers) and parents_[roots[i]]
    // (winner's pre-existing content) are empty for every i, the
    // post-consolidate frontier would be empty and we'd break anyway.
    // Saves the consolidate work (~9 ms on large round 1) at the
    // cost of one O(work.size()) parallel any_of (~3 ms on the same
    // workload). Net win ~6 ms. parlay::any_of short-circuits on
    // the first non-empty slot, so productive rounds pay only the
    // cost of scanning until the first hit.
    bool has_any_parents = parlay::any_of(
        parlay::iota(work.size()), [&](std::size_t i) {
          return !parents_[work[i]].empty() || !parents_[roots[i]].empty();
        });
    if (!has_any_parents) {
      if (trace) {
        std::fprintf(stderr,
                     "[pe] round=%3zu work=%9zu (tail-skip: all parents empty)\n",
                     round, work.size());
      }
      break;
    }

    detail::parallel_consolidate(parents_, work, roots);
    double consolidate_ms = trace ? ms_since(t_consolidate) : 0.0;

    auto t_frontier = clk::now();
    auto unique_roots = parlay::remove_duplicates(roots);
    auto frontier = parlay::flatten(parlay::map(unique_roots, [&](Id r) {
      return parlay::make_slice(parents_[r]);
    }));
    // Dedup the frontier when it's big enough that duplicates dominate.
    // The raw frontier counts a parent once per child slot it occupies
    // in a merged class; on flat high-fanout workloads the
    // duplicate ratio runs ~1.7-1.8× (matches par_filter's dirty
    // count once deduped). Skipping dedup below the threshold avoids
    // paying its ~5 ms parlay::remove_duplicates cost on workloads
    // where the raw frontier is small enough that the saved semisort
    // work doesn't pay it back. Cutoff is configurable via
    // PE_FRONTIER_DEDUP_CUTOFF for sweeps.
    static const std::size_t kFrontierDedupCutoff = [] {
      const char* s = std::getenv("PE_FRONTIER_DEDUP_CUTOFF");
      return s ? static_cast<std::size_t>(std::atoll(s)) : std::size_t{500'000};
    }();
    if (frontier.size() >= kFrontierDedupCutoff) {
      frontier = parlay::remove_duplicates(std::move(frontier));
    }
    double frontier_ms = trace ? ms_since(t_frontier) : 0.0;

    if (frontier.empty()) {
      if (trace) {
        std::fprintf(stderr,
                     "[pe] round=%3zu work=%9zu frontier=        0 next=        0 "
                     "consolidate=%7.3fms frontier=%7.3fms semisort=%7.3fms (break)\n",
                     round, work.size(), consolidate_ms, frontier_ms, 0.0);
      }
      break;
    }

    auto t_semisort = clk::now();
    auto canon = parlay::map(frontier, [&](std::uint32_t idx) {
      // `idx` is a frontier element — under sparse `nodes_` storage it
      // is the class id of a parent node and the index into `nodes_`.
      const auto& node = nodes_[idx];
      return detail::CanonEntry{sig_hash(node, uf_), uf_.find_root(idx), idx};
    });

    detail::SemisortTimings semi_t{};
    auto next_work = detail::merge_and_collect_semisort(
        std::move(canon), uf_, nodes_, trace ? &semi_t : nullptr);
    next_work = parlay::remove_duplicates(std::move(next_work));
    double semisort_ms = trace ? ms_since(t_semisort) : 0.0;

    if (trace) {
      std::fprintf(stderr,
                   "[pe] round=%3zu work=%9zu frontier=%9zu next=%9zu "
                   "consolidate=%7.3fms frontier=%7.3fms semisort=%7.3fms "
                   "(keyed=%6.3fms group_by=%7.3fms per_group=%6.3fms)\n",
                   round, work.size(), frontier.size(), next_work.size(),
                   consolidate_ms, frontier_ms, semisort_ms,
                   semi_t.keyed_ms, semi_t.group_by_ms, semi_t.per_group_ms);
    }

    work = std::move(next_work);
    ++round;
  }
}

// ---- sequential_close_simple_inline -------------------------------------
//
// One inline-storage hashtable per arity bucket (`SigK<1>` through
// `SigK<kMaxK>`), with `SigBump` as the arena-allocated fallback
// for arity > kMaxK. Each per-arity table's `operator==` and hash
// loops are fully unrolled at compile time, and every signature lives
// entirely in its own struct (no heap traffic on the fast path).
//
// `kMaxK` is hardcoded at 4 here, which covers all of our measured
// workloads (gates are arity ≤ 2; cube benchmarks parameterize over d
// up to 5 but typical d ≤ 4; SMT operations mostly arity ≤ 4). The
// bump fallback handles anything above kMaxK.

namespace {

// Arity-specialized signature template. `SigK<K>` holds exactly K
// canonicalized children inline (no heap allocation), with the arity
// implicit in the type — each `SigK<K>` instance is exclusively for
// arity-K nodes. Used by `sequential_close_simple_inline` to keep one
// hashtable per arity bucket, so each table's `operator==` and hash
// loops are fully unrolled at compile time.
//
// We also cache the 64-bit FxHash of (op, children) directly in the
// struct. `parlay::group_by_key` calls its `Hash` functor twice per
// element (once for bucket assignment, once for per-bucket hashtable
// probing), so caching the hash collapses two ~20 ns FxHasher passes
// into two 8 B reads. The 8 extra bytes per signature cost ~1 ns each
// in count_sort movement — net ~20 ns saved per element. Equality
// (`operator==`) intentionally ignores the cached hash: it's
// redundant with the structural compare, and parlay only ever calls
// `equal` after a hash match.
template <std::size_t K>
struct SigK {
  std::uint64_t hash;
  const std::string* op;
  std::array<Id, K> children;

  bool operator==(const SigK& o) const noexcept {
    if (*op != *o.op) return false;
    // K is a compile-time constant; the loop is fully unrolled.
    for (std::size_t i = 0; i < K; ++i) {
      if (children[i] != o.children[i]) return false;
    }
    return true;
  }
};

template <std::size_t K>
struct SigKHash {
  using is_avalanching = void;
  std::size_t operator()(const SigK<K>& s) const noexcept {
    return static_cast<std::size_t>(s.hash);
  }
};

template <std::size_t K, typename UF>
SigK<K> canonical_sig_k(const ENode& node, UF& uf) {
  SigK<K> sig;
  sig.op = &node.op;
  FxHasher h;
  h.write_str(node.op);
  for (std::size_t i = 0; i < K; ++i) {
    sig.children[i] = uf.find_root(node.children[i]);
    h.write_u32(sig.children[i]);
  }
  sig.hash = h.finish();
  return sig;
}

// Bump-allocator-backed signature: identical content to `Signature` but
// the children array lives in a caller-owned arena rather than a
// per-Signature heap allocation. The arena hands out monotonically
// increasing slots from a chunked backing store; once allocated, slots
// are stable for the lifetime of the arena.
//
// On collision the local `SigBump` is discarded but its children slots
// remain claimed in the arena (no per-call free). This leaks
// transient slots, which is fine because the arena's lifetime is the
// closure procedure — it's deallocated wholesale when the algorithm
// returns.
struct SigBump {
  const std::string* op;
  Id* data;
  std::uint32_t size;

  bool operator==(const SigBump& o) const noexcept {
    if (*op != *o.op) return false;
    if (size != o.size) return false;
    return std::equal(data, data + size, o.data);
  }
};

struct SigBumpHash {
  using is_avalanching = void;
  std::size_t operator()(const SigBump& s) const noexcept {
    FxHasher h;
    h.write_str(*s.op);
    for (std::uint32_t i = 0; i < s.size; ++i) h.write_u32(s.data[i]);
    return static_cast<std::size_t>(h.finish());
  }
};

// Chunked bump allocator: 4 MB per chunk (1M Id slots). New chunks
// are appended without invalidating prior pointers — critical because
// the hashtable holds pointers to children that must remain valid
// across the entire closure. Designed for max-arity ≤ chunk_size,
// which holds for every workload we measure.
class BumpArena {
 public:
  Id* allocate(std::size_t n) {
    if (offset_ + n > kChunkSize) {
      chunks_.emplace_back(std::make_unique<Id[]>(kChunkSize));
      offset_ = 0;
    }
    Id* p = chunks_.back().get() + offset_;
    offset_ += n;
    return p;
  }
 private:
  static constexpr std::size_t kChunkSize = 1 << 20;  // 4 MB / chunk
  std::vector<std::unique_ptr<Id[]>> chunks_;
  std::size_t offset_ = kChunkSize;  // force first chunk on first alloc
};

template <typename UF>
SigBump canonical_sig_bump(const ENode& node, UF& uf, BumpArena& arena) {
  SigBump sig;
  sig.op = &node.op;
  sig.size = static_cast<std::uint32_t>(node.children.size());
  sig.data = arena.allocate(sig.size);
  for (std::uint32_t i = 0; i < sig.size; ++i) {
    sig.data[i] = uf.find_root(node.children[i]);
  }
  return sig;
}

}  // namespace

template <>
void EGraph<SequentialUnionFind>::sequential_close_simple_inline(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  const std::size_t n = nodes_.size();

  std::vector<std::vector<Id>> class_preds(n);
  for (Id u = 0; u < n; ++u) {
    for (Id c : nodes_[u].children) {
      class_preds[uf_.find_root(c)].push_back(u);
    }
  }

  // Per-arity inline-storage tables; one bump-arena fallback table for
  // arity > kMaxK. Each `SigK<K>` table holds only arity-K signatures;
  // dispatch ensures arity-K nodes only ever consult their own table.
  constexpr std::size_t kMaxK = 4;
  ankerl::unordered_dense::map<SigK<1>, Id, SigKHash<1>> t1;
  ankerl::unordered_dense::map<SigK<2>, Id, SigKHash<2>> t2;
  ankerl::unordered_dense::map<SigK<3>, Id, SigKHash<3>> t3;
  ankerl::unordered_dense::map<SigK<4>, Id, SigKHash<4>> t4;
  ankerl::unordered_dense::map<SigBump, Id, SigBumpHash> tN;
  // Most workloads have arity 2 dominating; reserve only t2 upfront.
  t2.reserve(n);
  BumpArena arena;

  std::vector<std::uint8_t> in_queue(n, 0);
  RingBuffer<Id> pending(n);

  for (Id v = 0; v < static_cast<Id>(n); ++v) {
    if (!nodes_[v].children.empty()) {
      pending.push(v);
      in_queue[v] = 1;
    }
  }

  // Local helper: try_emplace into `table` with `sig`; return
  // (representative, collision) where collision is true iff the sig
  // was already present and `representative` is the previously-stored
  // node id.
  auto probe = [&](auto& table, auto&& sig, Id v) -> std::pair<Id, bool> {
    auto [it, inserted] = table.try_emplace(std::forward<decltype(sig)>(sig), v);
    return inserted ? std::pair<Id, bool>{v, false}
                    : std::pair<Id, bool>{it->second, true};
  };

  while (!pending.empty()) {
    Id v = pending.pop();
    in_queue[v] = 0;

    const auto& node = nodes_[v];
    std::pair<Id, bool> r;
    switch (node.children.size()) {
      case 1: r = probe(t1, canonical_sig_k<1>(node, uf_), v); break;
      case 2: r = probe(t2, canonical_sig_k<2>(node, uf_), v); break;
      case 3: r = probe(t3, canonical_sig_k<3>(node, uf_), v); break;
      case 4: r = probe(t4, canonical_sig_k<4>(node, uf_), v); break;
      default:
        r = probe(tN, canonical_sig_bump(node, uf_, arena), v);
        break;
    }
    if (!r.second) continue;  // no collision: just recorded the sig

    const Id w  = r.first;
    const Id rv = uf_.find_root(v);
    const Id rw = uf_.find_root(w);
    if (rv == rw) continue;

    const Id new_root = uf_.union_(rv, rw);
    const Id loser    = (new_root == rv) ? rw : rv;

    auto& dst = class_preds[new_root];
    auto& src = class_preds[loser];
    for (Id p : src) {
      dst.push_back(p);
      if (!in_queue[p]) {
        pending.push(p);
        in_queue[p] = 1;
      }
    }
    src.clear();
    src.shrink_to_fit();
  }
  // Silence unused-variable warning when kMaxK changes the case list.
  (void)kMaxK;
}

}  // namespace pe
