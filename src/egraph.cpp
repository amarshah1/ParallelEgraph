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

// ---- parallel_parents_groupby_sigk ----------------------------------------
//
// Defined at the bottom of this file because it references the
// per-arity helpers (`apply_unions_groupby_sigk<K>`) which live in the
// anonymous namespace lower down (alongside SigK<K>, SigBump, etc.).
//
// ---- parallel_topo ------------------------------------------------
//
// Depth-stratified BSP. Round d is a two-phase parallel step:
//   Phase 1 (parlay::map):   compute CanonEntry for every depth-d class.
//                            Reads UF (find_root); the only writes are
//                            path-compression CAS, which preserves
//                            find_root return values.
//   parlay::map JOINS — implicit barrier between phases.
//   Phase 2 (apply_congruence_semisort): semisort canon by hash, dnc_union
//                            each non-singleton run. The only UF writes
//                            in this round happen here.
// The barrier is the actual safety property: every sig read in phase 1
// observes the same UF snapshot — the post-round-(d-1) state — so the
// semisort catches every intra-round congruence reachable from that
// snapshot. Note that this safety does NOT come from depth stratification
// per se: a class's representative can be a node at any depth (cross-depth
// initial unions like `assert(leaf_ta1_0 = f0(a))` make the leaf's class
// have a depth-1 root), and round-d unions can mutate those representatives.
// What guarantees correctness is that phase 1 doesn't observe phase-2
// mutations because the barrier orders them.
//
// Total rounds = max depth. Correct on arbitrary DAGs (including
// cross-depth initial unions); unlike `sequential_close_topo`, no
// level-d node ever misses a congruence with another level-d node — the
// semisort sees them all in one frozen-snapshot bucketing pass.

template <>
void EGraph<ConcurrentUnionFind>::parallel_topo(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf_.union_(initial_unions[i].first, initial_unions[i].second);
  });

  // Depth 0 = leaves; no signatures to canonicalize. Walk d = 1..max.
  for (std::size_t d = 1; d < depth_buckets_.size(); ++d) {
    const auto& bucket = depth_buckets_[d];
    if (bucket.empty()) continue;

    auto t_round = clk::now();
    auto canon = parlay::map(bucket, [&](Id idx) {
      const auto& node = nodes_[idx];
      return detail::CanonEntry{sig_hash(node, uf_), uf_.find_root(idx), idx};
    });

    detail::apply_congruence_semisort(std::move(canon), uf_, nodes_);

    if (trace) {
      std::fprintf(stderr,
                   "[pe-topo] depth=%3zu bucket=%9zu round=%7.3fms\n",
                   d, bucket.size(), ms_since(t_round));
    }
  }
}

// ---- parallel_topo_iter -------------------------------------------
//
// Sound parallel topo: depth-stratified BSP wrapped in a fixpoint loop,
// with MIN_ID linking. The single-pass `parallel_topo` is unsound
// on cross-depth initial unions because in-round unions can change
// find_root values for sibling sigs that were already computed; iterating
// catches the missed congruences across passes, and MIN_ID makes the
// fixpoint converge in 1-2 passes on regular cascades.

template <>
void EGraph<ConcurrentUnionFind>::parallel_topo_iter(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE_ITER") != nullptr;
  using clk = std::chrono::steady_clock;

  // ---- step 1: apply initial unions in parallel (MIN_ID) -----------------
  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf_.union_min_id(initial_unions[i].first, initial_unions[i].second);
  });

  // Lambda dispatched into dnc_union_with — returns the surviving root
  // (so dnc_union_with can chain), and sets `*changed` when an actual
  // merge commits.
  std::atomic<bool> changed{false};
  auto union_track = [&changed](ConcurrentUnionFind& u, Id a, Id b) -> Id {
    auto [survivor, did_merge] = u.union_min_id(a, b);
    if (did_merge) changed.store(true, std::memory_order_relaxed);
    return survivor;
  };

  std::size_t round = 0;
  do {
    changed.store(false, std::memory_order_relaxed);
    auto t_round = trace ? clk::now() : clk::time_point{};

    // Walk d = 1..max_depth. depth 0 = leaves; nothing to canonicalize.
    for (std::size_t d = 1; d < depth_buckets_.size(); ++d) {
      const auto& bucket = depth_buckets_[d];
      if (bucket.empty()) continue;

      // Build CanonEntry for every class in the bucket.
      auto canon = parlay::map(bucket, [&](Id idx) {
        const auto& node = nodes_[idx];
        return detail::CanonEntry{sig_hash(node, uf_), uf_.find_root(idx), idx};
      });

      if (canon.empty()) continue;

      // Sort by primary hash (in-place) and walk runs.
      parlay::integer_sort_inplace(
          parlay::make_slice(canon),
          [](const detail::CanonEntry& e) -> std::uint64_t { return e.hash; });

      auto starts_flag = parlay::tabulate(canon.size(), [&](std::size_t i) {
        return i == 0 || canon[i].hash != canon[i - 1].hash;
      });
      auto run_starts = parlay::pack_index<std::uint32_t>(starts_flag);

      parlay::parallel_for(0, run_starts.size(), [&](std::size_t r) {
        const std::size_t lo = run_starts[r];
        const std::size_t hi = (r + 1 < run_starts.size())
                                   ? static_cast<std::size_t>(run_starts[r + 1])
                                   : canon.size();
        if (hi - lo < 2) return;
        // Same primary-hash run; split by full equality (secondary or sound).
        std::size_t i = lo;
        while (i < hi) {
          std::size_t j = i + 1;
          while (j < hi) {
            if (!sigs_equal(canon[i].class_id, canon[j].class_id, uf_, nodes_))
              break;
            ++j;
          }
          if (j - i >= 2) {
            detail::dnc_union_with(canon, i, j, uf_, union_track);
          }
          i = j;
        }
      });
    }

    ++round;
    if (trace) {
      double ms = std::chrono::duration<double, std::milli>(
                      clk::now() - t_round).count();
      std::fprintf(stderr,
                   "[par_topo_iter] round=%2zu changed=%d  %7.2fms\n",
                   round, int(changed.load()), ms);
    }
  } while (changed.load(std::memory_order_relaxed));

  if (trace) {
    std::fprintf(stderr,
                 "[par_topo_iter] converged after %zu round(s)\n", round);
  }
}

// ---- sequential_close_nelson --------------------------------------------

template <>
void EGraph<SequentialUnionFind>::sequential_close_nelson(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  const std::size_t n_classes = uf_.len();

  std::vector<Id> worklist;
  worklist.reserve(initial_unions.size() * 2);
  for (auto [a, b] : initial_unions) {
    Id ra = uf_.find_root(a);
    Id rb = uf_.find_root(b);
    if (ra != rb) {
      uf_.union_(ra, rb);
      worklist.push_back(ra);
      worklist.push_back(rb);
    }
  }

  // Flat hashmap (robin-hood) replaces std::unordered_map's chained
  // buckets. ~3x faster on this insert-heavy workload. Reserve up-front
  // to skip rehash spikes — sig_table holds at most one entry per visited
  // parent class; n_classes is a safe over-estimate of the steady state.
  ankerl::unordered_dense::map<std::uint64_t, std::vector<std::uint32_t>> sig_table;
  sig_table.reserve(n_classes / 4);

  while (!worklist.empty()) {
    Id c = worklist.back();
    worklist.pop_back();

    parlay::sequence<std::uint32_t> frontier = std::move(parents_[c]);
    parents_[c].clear();
    if (frontier.empty()) continue;

    for (std::uint32_t pidx : frontier) {
      // class_id == pidx since nodes_ is class-id-indexed.
      const auto& node = nodes_[pidx];
      std::uint64_t h = sig_hash(node, uf_);
      Id my_class = uf_.find_root(static_cast<Id>(pidx));

      auto& bucket = sig_table[h];
      std::int64_t match_o = -1;
      for (std::uint32_t o : bucket) {
        if (sigs_equal(pidx, o, uf_, nodes_)) {
          match_o = static_cast<std::int64_t>(o);
          break;
        }
      }
      if (match_o >= 0) {
        Id other_class = uf_.find_root(static_cast<Id>(match_o));
        if (my_class != other_class) {
          uf_.union_(my_class, other_class);
          worklist.push_back(my_class);
          worklist.push_back(other_class);
        }
      } else {
        bucket.push_back(pidx);
      }
    }
  }
}

// ---- sequential_close_topo ----------------------------------------------
//
// One-shot upward propagation: walk every class in reverse topological
// order (children before parents). For each node, build its canonical
// signature (op, find_root(c) for c in children) and try_emplace into a
// hashmap keyed by that signature. Repeated signature ⇒ union the current
// class with the prior holder. Because we visit children before parents,
// every child's root is already its final-for-this-pass value when its
// parent's signature is computed; no worklist, no fixed-point loop, one
// linear pass through nodes_.
//
// Caveat: this is a *single pass*. If two nodes A and B at lower indices
// were inserted with non-matching signatures and later become congruent
// because of a union deduced *after* both were visited, this pass will
// not retroactively merge them. The forward DAG walk catches all
// cascading congruences that flow upward from the initial unions, which
// is sufficient for the synthetic workloads in closure_compare_bench;
// callers needing the full Nelson fixed-point should use
// sequential_close_nelson.

namespace {

// Canonical signature key: op (pointer into the node's op string —
// nodes_ is stable for the closure's lifetime) plus the find_root'd
// children. Two signatures are equal iff op strings match and
// canonicalized child sequences match element-wise.
struct Signature {
  const std::string* op;
  std::vector<Id> children;

  bool operator==(const Signature& o) const noexcept {
    if (*op != *o.op) return false;
    if (children.size() != o.children.size()) return false;
    return std::equal(children.begin(), children.end(), o.children.begin());
  }
};

struct SignatureHash {
  // FxHasher avalanches well; tell ankerl to skip its own mixing.
  using is_avalanching = void;

  std::size_t operator()(const Signature& s) const noexcept {
    FxHasher h;
    h.write_str(*s.op);
    for (Id c : s.children) h.write_u32(c);
    return static_cast<std::size_t>(h.finish());
  }
};

template <typename UF>
Signature canonical_sig(const ENode& node, UF& uf) {
  Signature sig;
  sig.op = &node.op;
  sig.children.reserve(node.children.size());
  for (Id c : node.children) sig.children.push_back(uf.find_root(c));
  return sig;
}

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

// Backwards-compatible Sig2 alias for `sequential_close_simple_arity`
// and `sequential_close_simple_arity_bump`. These variants discriminate
// arities 0..2 within a single struct (with an explicit `arity` field),
// rather than using one SigK<K> table per arity. Both styles coexist
// for benchmarking; the SigK<K> approach is the cleaner generalization.
struct Sig2 {
  const std::string* op;
  std::uint8_t arity;
  Id c0;
  Id c1;

  bool operator==(const Sig2& o) const noexcept {
    if (*op != *o.op) return false;
    if (arity != o.arity) return false;
    return c0 == o.c0 && c1 == o.c1;
  }
};

struct Sig2Hash {
  using is_avalanching = void;
  std::size_t operator()(const Sig2& s) const noexcept {
    FxHasher h;
    h.write_str(*s.op);
    h.write_u32(static_cast<std::uint32_t>(s.arity));
    h.write_u32(s.c0);
    h.write_u32(s.c1);
    return static_cast<std::size_t>(h.finish());
  }
};

template <typename UF>
Sig2 canonical_sig2(const ENode& node, UF& uf) {
  Sig2 sig;
  sig.op = &node.op;
  sig.arity = static_cast<std::uint8_t>(node.children.size());
  sig.c0 = sig.arity >= 1 ? uf.find_root(node.children[0]) : 0;
  sig.c1 = sig.arity >= 2 ? uf.find_root(node.children[1]) : 0;
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

// Probabilistic 96-bit signature, matching the parallel CanonEntry's
// equality model: 64-bit primary hash for table bucketing, 32-bit
// secondary hash for in-bucket equality. No allocation per term.
// Used by sequential_close_simple_hash to provide an apples-to-apples
// baseline against the parallel algorithms' correctness model.
struct HashSignature {
  std::uint64_t primary;
  std::uint32_t secondary;

  bool operator==(const HashSignature& o) const noexcept {
    return primary == o.primary && secondary == o.secondary;
  }
};

struct HashSignatureHash {
  using is_avalanching = void;
  std::size_t operator()(const HashSignature& s) const noexcept {
    // FxHasher output is already avalanched; use the 64-bit primary
    // directly as the bucket key. Secondary participates only in
    // operator== to break primary-hash collisions.
    return static_cast<std::size_t>(s.primary);
  }
};

// ---- Per-arity group_by_key helpers (used by parallel_parents_groupby_sigk)

// For each arity K in 1..kMaxK, group the arity-K subset of the frontier
// by structural `SigK<K>` and dnc_union each multi-element group's
// members. Returns the loser ids (members whose post-merge root is not
// the survivor) so the caller can build the next round's worklist.
//
// The structural `SigK<K>` equality is fully unrolled at compile time
// (fixed-size inline children array). Per-bucket allocation is parlay's
// `sequence<Id>` of group members, whose SBO covers singleton and small
// buckets — the dominant case in late rounds.
template <std::size_t K>
parlay::sequence<Id> apply_unions_groupby_sigk(
    const parlay::sequence<Id>& arity_k_frontier,
    ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes) {
  if (arity_k_frontier.empty()) return {};

  // Build (SigK<K>, e-class-id) pairs.
  auto pairs = parlay::map(arity_k_frontier, [&](Id idx) {
    return std::pair<SigK<K>, Id>{canonical_sig_k<K>(nodes[idx], uf), idx};
  });

  // Semisort by structural signature. parlay::group_by_key internally
  // count-sorts by `SigKHash<K> % num_buckets` then runs a sequential
  // per-bucket open-addressed dedup that uses `std::equal_to` on the
  // full structural key — so the resulting grouping is deterministic
  // and sound, no hash-collision fallback needed.
  auto groups = parlay::group_by_key(
      std::move(pairs), SigKHash<K>{}, std::equal_to<SigK<K>>{});

  // Per bucket: union the members and emit the loser roots (pre-merge
  // roots that aren't the surviving root). Important: capture pre-merge
  // roots BEFORE dnc_union runs — afterwards find_root returns the
  // survivor for every bucket member and there'd be nothing to emit.
  auto per_group = parlay::map(groups,
      [&](auto& kv) -> parlay::sequence<Id> {
        auto& bucket = kv.second;
        if (bucket.size() < 2) return {};

        // Snapshot pre-merge roots. Doubles as the all_same_root
        // precheck input.
        auto pre_roots = parlay::map(bucket, [&](Id b) {
          return uf.find_root(b);
        });
        const Id ref = pre_roots[0];
        bool all_same = true;
        for (Id r : pre_roots) {
          if (r != ref) { all_same = false; break; }
        }
        if (all_same) return {};

        const Id survivor = detail::dnc_union(
            bucket, 0, bucket.size(), uf);

        return parlay::map_maybe(pre_roots, [&](Id r) -> std::optional<Id> {
          if (r == survivor) return std::nullopt;
          return r;
        });
      });
  return parlay::flatten(per_group);
}

// Parallel analog of `sequential_close_simple_bump`: for arity > kMaxK
// we group via parlay::group_by_key on full-arity `Signature` (which
// keeps children in a `std::vector<Id>`). This path is hit rarely on
// our workloads (gates are arity ≤ 2; cube with d > 4 is uncommon),
// so the per-canon-entry malloc is acceptable here.
inline parlay::sequence<Id> apply_unions_groupby_bigsig(
    const parlay::sequence<Id>& bump_frontier,
    ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes) {
  if (bump_frontier.empty()) return {};

  auto pairs = parlay::map(bump_frontier, [&](Id idx) {
    return std::pair<Signature, Id>{canonical_sig(nodes[idx], uf), idx};
  });
  auto groups = parlay::group_by_key(
      std::move(pairs), SignatureHash{}, std::equal_to<Signature>{});

  auto per_group = parlay::map(groups,
      [&](auto& kv) -> parlay::sequence<Id> {
        auto& bucket = kv.second;
        if (bucket.size() < 2) return {};
        const Id ref = uf.find_root(bucket[0]);
        bool all_same = true;
        for (Id b : bucket) {
          if (uf.find_root(b) != ref) { all_same = false; break; }
        }
        if (all_same) return {};
        const Id survivor = detail::dnc_union(bucket, 0, bucket.size(), uf);
        return parlay::map_maybe(bucket, [&](Id b) -> std::optional<Id> {
          if (uf.find_root(b) == survivor) return std::nullopt;
          return b;
        });
      });
  return parlay::flatten(per_group);
}

}  // namespace

template <>
void EGraph<SequentialUnionFind>::sequential_close_topo(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  ankerl::unordered_dense::map<Signature, std::uint32_t, SignatureHash>
      bucket;
  bucket.reserve(nodes_.size() * 2);

  const std::size_t n = nodes_.size();
  for (std::uint32_t pidx = 0; pidx < n; ++pidx) {
    auto [it, inserted] = bucket.try_emplace(
        canonical_sig(nodes_[pidx], uf_), pidx);
    if (!inserted) {
      Id ra = uf_.find_root(pidx);
      Id rb = uf_.find_root(it->second);
      if (ra != rb) {
        // MIN_ID union: lower class id wins. Preserves the invariant
        // that on a DAG-ordered input, the canonical (earliest-emitted)
        // representative of any equivalence class is its current root —
        // which lets a forward-pass collapse the entire cascade in a
        // single sweep, since downstream sigs reference roots that were
        // valid at insertion time AND remain valid after subsequent
        // intra-pass merges.
        uf_.union_into(std::max(ra, rb), std::min(ra, rb));
      }
    }
  }
}

// ---- sequential_close_topo_iter -----------------------------------------
//
// Drive `sequential_close_topo` to a fixpoint. Each iteration is a fresh
// forward walk through `nodes_` with a fresh hashcons; we stop when a
// full pass produces zero new unions. Recovers correctness from the
// known unsoundness of single-pass topo on cross-depth / adversarial
// inputs (see test_seq_topo_adversarial_order). Number of iterations =
// (longest cross-depth chain in the augmented DAG) + 1.

template <>
void EGraph<SequentialUnionFind>::sequential_close_topo_iter(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  const bool trace = std::getenv("PE_TRACE_ITER") != nullptr;
  using clk = std::chrono::steady_clock;

  const std::size_t n = nodes_.size();
  std::size_t round = 0;
  std::size_t total_unions = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    auto t0 = trace ? clk::now() : clk::time_point{};
    std::size_t round_unions = 0;

    ankerl::unordered_dense::map<Signature, std::uint32_t, SignatureHash>
        bucket;
    bucket.reserve(n);
    for (std::uint32_t pidx = 0; pidx < n; ++pidx) {
      auto [it, inserted] = bucket.try_emplace(
          canonical_sig(nodes_[pidx], uf_), pidx);
      if (!inserted) {
        Id ra = uf_.find_root(pidx);
        Id rb = uf_.find_root(it->second);
        if (ra != rb) {
          // MIN_ID union: see comment in sequential_close_topo.
          // For DAG-ordered emission with canonical members at lower
          // ids, this collapses the entire cascade in 2 rounds (1 work
          // pass + 1 verification pass). Without it, each round
          // advances exactly one g-tree level and total cost is
          // O(N × depth) instead of O(N).
          uf_.union_into(std::max(ra, rb), std::min(ra, rb));
          changed = true;
          ++round_unions;
        }
      }
    }
    ++round;
    total_unions += round_unions;
    if (trace) {
      double ms = std::chrono::duration<double, std::milli>(
                      clk::now() - t0).count();
      std::fprintf(stderr,
                   "[topo_iter] round=%3zu unions=%9zu cumulative=%9zu  %7.2fms\n",
                   round, round_unions, total_unions, ms);
    }
  }
  if (trace) {
    std::fprintf(stderr,
                 "[topo_iter] converged after %zu rounds, %zu total unions\n",
                 round, total_unions);
  }
}

// ---- sequential_close_dst -----------------------------------------------
//
// Worklist closure with smaller-into-larger merging and a structural
// hashcons (Signature -> class id). Correct on arbitrary initial unions.
//
// Invariants:
//   * `parents_[r]` (for any current UF root r) holds every node id that
//     has a class member of r among its children. When a root dies, its
//     parents migrate to the survivor's parents list.
//   * The hashcons may contain *stale* entries (sig -> pidx where pidx's
//     current canonical sig differs from the stored key). Stale entries
//     are harmless: a later structurally-matching insertion either hits
//     the stale entry — yielding a valid merge by congruence at the time
//     the stale sig was canonical, which is monotonically still valid
//     now — or misses and inserts a fresh entry.
//
// Two queues drive the loop:
//   * pending_merges: pairs (a, b) of class ids that need to be unioned.
//     Seeded from `initial_unions` and from duplicate sigs detected on
//     the initial seed sweep; refilled by phase 1 when re-canonicalized
//     parents collide with existing hashcons entries.
//   * pending_nodes: node ids whose canonical sig may have changed
//     (because one of their children's roots moved). Refilled by
//     phase 2 from `parents_[ra]` whenever ra dies.

template <>
void EGraph<SequentialUnionFind>::sequential_close_dst(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  ankerl::unordered_dense::map<Signature, Id, SignatureHash> hashcons;
  hashcons.reserve(nodes_.size());

  std::vector<std::pair<Id, Id>> pending_merges;
  pending_merges.reserve(initial_unions.size() + nodes_.size() / 4);
  for (auto [a, b] : initial_unions) pending_merges.emplace_back(a, b);

  // Initial seed: every node's signature goes into the hashcons.
  // Pre-existing congruences (two nodes with identical structure)
  // surface as duplicates and become initial pending_merges.
  for (std::uint32_t pidx = 0; pidx < nodes_.size(); ++pidx) {
    auto [it, inserted] = hashcons.try_emplace(
        canonical_sig(nodes_[pidx], uf_), pidx);
    if (!inserted) pending_merges.emplace_back(pidx, it->second);
  }

  std::vector<std::uint32_t> pending_nodes;

  while (!pending_merges.empty() || !pending_nodes.empty()) {
    // Phase 1: re-canonicalize each pending node. A collision with an
    // existing hashcons entry queues a fresh merge.
    for (std::uint32_t pidx : pending_nodes) {
      auto [it, inserted] = hashcons.try_emplace(
          canonical_sig(nodes_[pidx], uf_), pidx);
      if (!inserted && it->second != pidx) {
        pending_merges.emplace_back(pidx, it->second);
      }
    }
    pending_nodes.clear();

    // Phase 2: drain the merge queue, possibly adding to pending_nodes.
    auto merges = std::move(pending_merges);
    pending_merges.clear();
    for (auto [a, b] : merges) {
      Id ra = uf_.find_root(a);
      Id rb = uf_.find_root(b);
      if (ra == rb) continue;

      // Smaller-half: the class with fewer parents dies. O(N log N)
      // total parent-list migration over the lifetime of the closure.
      if (parents_[ra].size() > parents_[rb].size()) std::swap(ra, rb);

      uf_.union_into(ra, rb);

      // Migrate ra's parents into rb's list and queue them for
      // re-canonicalization (their child-root just moved). One pass
      // over src does both.
      auto& src = parents_[ra];
      auto& dst = parents_[rb];
      for (Id pidx : src) {
        pending_nodes.push_back(pidx);
        dst.push_back(pidx);
      }
      src.clear();
    }
  }
}

// ---- sequential_close_simple ----------------------------------------------
//
// Worklist closure with all non-leaves seeded upfront. Distinguishing
// feature vs `sequential_close_dst`: that one drives the loop from
// pre-existing congruences detected during initial hashcons seeding;
// this one drives it from a deque containing every non-leaf node.
// Simpler shape, more like a textbook Nelson-Oppen worklist.
//
// `class_preds` is a self-contained scratch structure (vector<vector>),
// rebuilt fresh from `nodes_` after applying initial unions. It plays
// the role of `parents_` but isn't that field — keeps the function
// independent of the EGraph's BSP-flavor `parents_` and avoids
// mutating it. The original sketch this came from named the helper
// `build_class_preds`; same idea here.

template <>
void EGraph<SequentialUnionFind>::sequential_close_simple(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  const std::size_t n = nodes_.size();

  // Build per-class predecessor lists post-initial-unions: each parent
  // u goes into the slot of its child's CURRENT root, not the literal
  // child id. Captures the right "which classes have u as a successor"
  // mapping for the worklist's union-splice step below.
  std::vector<std::vector<Id>> class_preds(n);
  for (Id u = 0; u < n; ++u) {
    for (Id c : nodes_[u].children) {
      class_preds[uf_.find_root(c)].push_back(u);
    }
  }

  ankerl::unordered_dense::map<Signature, Id, SignatureHash> sig_table;
  sig_table.reserve(n);

  // FIFO worklist as a fixed-capacity ring buffer. `in_queue` ensures
  // each node is enqueued at most once at any time, so live entries
  // never exceed `n` and the ring can't overflow. Replaces an earlier
  // `std::deque<Id>` whose lazily-grown chunks didn't change wall-
  // clock here, but the ring's contiguous storage keeps the
  // allocation pattern aligned with the function's other pre-sized
  // structures (sig_table, class_preds, in_queue).
  std::vector<std::uint8_t> in_queue(n, 0);
  RingBuffer<Id> pending(n);

  for (Id v = 0; v < static_cast<Id>(n); ++v) {
    if (!nodes_[v].children.empty()) {
      pending.push(v);
      in_queue[v] = 1;
    }
  }

  while (!pending.empty()) {
    Id v = pending.pop();
    in_queue[v] = 0;

    auto sig = canonical_sig(nodes_[v], uf_);
    auto [it, inserted] = sig_table.try_emplace(std::move(sig), v);
    if (inserted) continue;

    const Id w = it->second;
    const Id rv = uf_.find_root(v);
    const Id rw = uf_.find_root(w);
    if (rv == rw) continue;

    // Collision on a different class → merge. union_ returns the
    // surviving root (post-refactor), so we can identify the loser
    // without an extra find_root call. Splice loser's predecessors
    // into the winner's list and re-queue them: their canonical sigs
    // each contain a child whose root just moved.
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
}

// ---- sequential_close_simple_hash ----------------------------------------
//
// Variant of `sequential_close_simple` that matches the parallel
// algorithms' probabilistic equality model: the hashcons is keyed by
// the same 96-bit canonical signature (`sig_hashes`) used in
// `merge_and_collect_semisort`, with equality decided by hash compare
// rather than by structural recheck. The two algorithms thus share the
// same correctness guarantee (deterministic agreement with the true
// congruence closure with probability at least 1 − N²/2⁹⁷); the
// comparison between sequential and parallel is then a comparison of
// the same algorithm at different parallelism levels, not of two
// algorithms with different correctness models.
//
// Aside from the signature type, the body is identical to
// sequential_close_simple. Kept as a separate template specialization
// rather than parameterized over signature type to keep the inner loop
// fully inlined for each variant.

template <>
void EGraph<SequentialUnionFind>::sequential_close_simple_hash(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  const std::size_t n = nodes_.size();

  std::vector<std::vector<Id>> class_preds(n);
  for (Id u = 0; u < n; ++u) {
    for (Id c : nodes_[u].children) {
      class_preds[uf_.find_root(c)].push_back(u);
    }
  }

  ankerl::unordered_dense::map<HashSignature, Id, HashSignatureHash>
      sig_table;
  sig_table.reserve(n);

  std::vector<std::uint8_t> in_queue(n, 0);
  RingBuffer<Id> pending(n);

  for (Id v = 0; v < static_cast<Id>(n); ++v) {
    if (!nodes_[v].children.empty()) {
      pending.push(v);
      in_queue[v] = 1;
    }
  }

  while (!pending.empty()) {
    Id v = pending.pop();
    in_queue[v] = 0;

    auto [h1, h2] = sig_hashes(nodes_[v], uf_);
    auto [it, inserted] = sig_table.try_emplace(HashSignature{h1, h2}, v);
    if (inserted) continue;

    const Id w = it->second;
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
}

// ---- sequential_close_simple_arity --------------------------------------
//
// Variant of `sequential_close_simple` that splits the signature table
// by arity, with an inline-only `Sig2` for the arity-≤2 fast path. The
// allocation cost that dominates the structural-compare baseline lives
// in `canonical_sig`'s `std::vector<Id>` construction; the fast path
// avoids it entirely. Arity > 2 falls back to the original
// `Signature` + `std::vector` path so correctness is preserved on
// arbitrary inputs.
//
// On gate-style workloads (~all arity ≤ 2) the fast path covers every
// term; on SMT/cube workloads with mixed arities each table sees a
// disjoint subset, so the worklist's dispatch branch is well-
// predicted.

template <>
void EGraph<SequentialUnionFind>::sequential_close_simple_arity(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  const std::size_t n = nodes_.size();

  std::vector<std::vector<Id>> class_preds(n);
  for (Id u = 0; u < n; ++u) {
    for (Id c : nodes_[u].children) {
      class_preds[uf_.find_root(c)].push_back(u);
    }
  }

  // Two physical hashtables: one keyed by `Sig2` for arity ≤ 2, one
  // keyed by `Signature` for arity > 2. A term of arity k consults
  // only the table for its arity, so the two tables hold disjoint
  // term sets.
  ankerl::unordered_dense::map<Sig2, Id, Sig2Hash> small_table;
  ankerl::unordered_dense::map<Signature, Id, SignatureHash> large_table;
  small_table.reserve(n);
  // large_table starts empty; reserves on first insert. Most workloads
  // never touch it.

  std::vector<std::uint8_t> in_queue(n, 0);
  RingBuffer<Id> pending(n);

  for (Id v = 0; v < static_cast<Id>(n); ++v) {
    if (!nodes_[v].children.empty()) {
      pending.push(v);
      in_queue[v] = 1;
    }
  }

  while (!pending.empty()) {
    Id v = pending.pop();
    in_queue[v] = 0;

    const auto& node = nodes_[v];
    Id w;  // representative on collision, set in either branch below
    bool collision;
    if (node.children.size() <= 2) {
      auto sig = canonical_sig2(node, uf_);
      auto [it, inserted] = small_table.try_emplace(sig, v);
      collision = !inserted;
      w = collision ? it->second : v;
    } else {
      auto sig = canonical_sig(node, uf_);
      auto [it, inserted] = large_table.try_emplace(std::move(sig), v);
      collision = !inserted;
      w = collision ? it->second : v;
    }
    if (!collision) continue;

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
}

// ---- sequential_close_simple_bump ---------------------------------------
//
// Variant of `sequential_close_simple` that uses a bump allocator
// backed by a chunked heap arena for the children of each canonical
// signature. Eliminates the per-term `malloc/free` pair from the
// baseline's `std::vector<Id>`-backed `Signature`. Same deterministic
// structural correctness, single signature type (no arity dispatch).
//
// The bump arena is owned by the procedure and freed wholesale on
// return. Collision-discarded signatures leak their slots but the
// leak is bounded by the closure's total work.

template <>
void EGraph<SequentialUnionFind>::sequential_close_simple_bump(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  const std::size_t n = nodes_.size();

  std::vector<std::vector<Id>> class_preds(n);
  for (Id u = 0; u < n; ++u) {
    for (Id c : nodes_[u].children) {
      class_preds[uf_.find_root(c)].push_back(u);
    }
  }

  ankerl::unordered_dense::map<SigBump, Id, SigBumpHash> sig_table;
  sig_table.reserve(n);
  BumpArena arena;

  std::vector<std::uint8_t> in_queue(n, 0);
  RingBuffer<Id> pending(n);

  for (Id v = 0; v < static_cast<Id>(n); ++v) {
    if (!nodes_[v].children.empty()) {
      pending.push(v);
      in_queue[v] = 1;
    }
  }

  while (!pending.empty()) {
    Id v = pending.pop();
    in_queue[v] = 0;

    auto sig = canonical_sig_bump(nodes_[v], uf_, arena);
    auto [it, inserted] = sig_table.try_emplace(sig, v);
    if (inserted) continue;

    const Id w = it->second;
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
}

// ---- sequential_close_simple_arity_bump ----------------------------------
//
// Combination of `sequential_close_simple_arity` and
// `sequential_close_simple_bump`: arity ≤ 2 uses the inline `Sig2`
// (no allocation at all), arity > 2 uses `SigBump` (bump-allocated
// children, no per-call malloc). Same deterministic structural
// correctness; eliminates allocation on both fast and slow paths.
//
// For workloads where every term has arity ≤ 2 (gates), only the
// Sig2 table sees activity. For mixed-arity workloads (SMT), the
// bump arena absorbs the >2-arity tail without ever calling malloc.

template <>
void EGraph<SequentialUnionFind>::sequential_close_simple_arity_bump(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  const std::size_t n = nodes_.size();

  std::vector<std::vector<Id>> class_preds(n);
  for (Id u = 0; u < n; ++u) {
    for (Id c : nodes_[u].children) {
      class_preds[uf_.find_root(c)].push_back(u);
    }
  }

  ankerl::unordered_dense::map<Sig2,     Id, Sig2Hash>      small_table;
  ankerl::unordered_dense::map<SigBump,  Id, SigBumpHash>   large_table;
  small_table.reserve(n);
  BumpArena arena;

  std::vector<std::uint8_t> in_queue(n, 0);
  RingBuffer<Id> pending(n);

  for (Id v = 0; v < static_cast<Id>(n); ++v) {
    if (!nodes_[v].children.empty()) {
      pending.push(v);
      in_queue[v] = 1;
    }
  }

  while (!pending.empty()) {
    Id v = pending.pop();
    in_queue[v] = 0;

    const auto& node = nodes_[v];
    Id w;
    bool collision;
    if (node.children.size() <= 2) {
      auto sig = canonical_sig2(node, uf_);
      auto [it, inserted] = small_table.try_emplace(sig, v);
      collision = !inserted;
      w = collision ? it->second : v;
    } else {
      auto sig = canonical_sig_bump(node, uf_, arena);
      auto [it, inserted] = large_table.try_emplace(sig, v);
      collision = !inserted;
      w = collision ? it->second : v;
    }
    if (!collision) continue;

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
}

// ---- sequential_close_simple_inline -------------------------------------
//
// Most general variant: one inline-storage hashtable per arity bucket
// (`SigK<1>` through `SigK<kMaxK>`), with `SigBump` as the
// arena-allocated fallback for arity > kMaxK. Each per-arity table's
// `operator==` and hash loops are fully unrolled at compile time, and
// every signature lives entirely in its own struct (no heap traffic
// on the fast path). Same deterministic structural correctness as
// `sequential_close_simple`.
//
// `kMaxK` is hardcoded at 4 here, which covers all of our measured
// workloads (gates are arity ≤ 2; cube benchmarks parameterize over d
// up to 5 but typical d ≤ 4; SMT operations mostly arity ≤ 4). Higher
// K is a one-line code change: add a `SigK<5>`/etc. table and one
// switch arm. The bump fallback handles anything above kMaxK.

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

// ---- parallel_parents_groupby_sigk ----------------------------------------
//
// BSP closure that replaces the integer_sort+run-walk semisort with
// `parlay::group_by_key` over per-arity `SigK<K>` keys (K = 1..4) plus
// a `Signature` fallback for arity > 4. Same `parents_`-driven outer
// round structure as `parallel_parents`; differs only in the per-round
// canonicalize+merge step.
//
// Structurally sound by construction: `parlay::group_by_key` calls
// `SigK<K>::operator==` (a compile-time-unrolled inline-children
// compare) on every potential bucket member, so two entries collapse
// into the same group iff their canonical signatures are structurally
// equal. No hash-collision fallback path needed.
//
// Fast because `parlay::group_by_key` is itself a semisort under the
// hood: count_sort by `hash % num_buckets` into cache-resident
// buckets, then sequential per-bucket open-addressed dedup. Per-arity
// dispatch keeps each `SigK<K>` instance small (16–32 bytes) so the
// count_sort moves less data, and the sequential dedup loop is fully
// unrolled at compile time.
//
// `kMaxK = 4` covers virtually all of our workloads (gates arity ≤ 2,
// cube with d ≤ 4, SMT mostly arity ≤ 4). Arity > kMaxK falls back to
// a vector-backed `Signature` + `group_by_key`; same correctness,
// per-canon-entry malloc on the rare path.

template <>
void EGraph<ConcurrentUnionFind>::parallel_parents_groupby_sigk(
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

  constexpr std::size_t kMaxK = 4;
  static const std::size_t kFrontierDedupCutoff = [] {
    const char* s = std::getenv("PE_FRONTIER_DEDUP_CUTOFF");
    return s ? static_cast<std::size_t>(std::atoll(s)) : std::size_t{500'000};
  }();

  std::size_t round = 0;
  while (!work.empty()) {
    work = parlay::remove_duplicates(std::move(work));

    auto t_consolidate = clk::now();
    auto roots = parlay::map(work, [&](Id c) { return uf_.find_root(c); });

    bool has_any_parents = parlay::any_of(
        parlay::iota(work.size()), [&](std::size_t i) {
          return !parents_[work[i]].empty() || !parents_[roots[i]].empty();
        });
    if (!has_any_parents) {
      if (trace) {
        std::fprintf(stderr,
                     "[pe-gbk] round=%3zu work=%9zu (tail-skip: all parents empty)\n",
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
    if (frontier.size() >= kFrontierDedupCutoff) {
      frontier = parlay::remove_duplicates(std::move(frontier));
    }
    double frontier_ms = trace ? ms_since(t_frontier) : 0.0;

    if (frontier.empty()) {
      if (trace) {
        std::fprintf(stderr,
                     "[pe-gbk] round=%3zu work=%9zu frontier=        0 next=        0 "
                     "consolidate=%7.3fms frontier=%7.3fms semisort=%7.3fms (break)\n",
                     round, work.size(), consolidate_ms, frontier_ms, 0.0);
      }
      break;
    }

    auto t_semisort = clk::now();

    // Bucket the frontier by arity (clamped to kMaxK+1 = overflow).
    // Bucket K = arity K terms; bucket kMaxK+1 = arity > kMaxK terms.
    auto by_arity = parlay::group_by_index(
        parlay::map(frontier, [&](Id idx) {
          std::size_t a = nodes_[idx].children.size();
          if (a > kMaxK) a = kMaxK + 1;
          return std::pair<std::size_t, Id>{a, idx};
        }),
        kMaxK + 2);

    // Per arity, run structural group_by_key + dnc_union, collect
    // losers. We fan out the five arity buckets as five independent
    // parlay calls; on workloads where one arity dominates (e.g.
    // gates → arity 2) most parlay workers end up inside that
    // arity's call.
    parlay::sequence<parlay::sequence<Id>> losers_per_arity(kMaxK + 2);
    losers_per_arity[1] = apply_unions_groupby_sigk<1>(by_arity[1], uf_, nodes_);
    losers_per_arity[2] = apply_unions_groupby_sigk<2>(by_arity[2], uf_, nodes_);
    losers_per_arity[3] = apply_unions_groupby_sigk<3>(by_arity[3], uf_, nodes_);
    losers_per_arity[4] = apply_unions_groupby_sigk<4>(by_arity[4], uf_, nodes_);
    losers_per_arity[kMaxK + 1] =
        apply_unions_groupby_bigsig(by_arity[kMaxK + 1], uf_, nodes_);
    // by_arity[0] is empty (leaves don't have parents, frontier
    // elements always have arity ≥ 1) so we skip it.

    auto next_work = parlay::flatten(std::move(losers_per_arity));
    next_work = parlay::remove_duplicates(std::move(next_work));
    double semisort_ms = trace ? ms_since(t_semisort) : 0.0;

    if (trace) {
      std::fprintf(stderr,
                   "[pe-gbk] round=%3zu work=%9zu frontier=%9zu next=%9zu "
                   "consolidate=%7.3fms frontier=%7.3fms semisort=%7.3fms "
                   "(arity1=%zu arity2=%zu arity3=%zu arity4=%zu arity>%zu=%zu)\n",
                   round, work.size(), frontier.size(), next_work.size(),
                   consolidate_ms, frontier_ms, semisort_ms,
                   by_arity[1].size(), by_arity[2].size(),
                   by_arity[3].size(), by_arity[4].size(),
                   kMaxK, by_arity[kMaxK + 1].size());
    }

    work = std::move(next_work);
    ++round;
  }
}

}  // namespace pe
