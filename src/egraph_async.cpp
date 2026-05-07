// Async-style congruence closure (still rounds-based, BSP-shaped). The
// algorithm replaces the BSP path's `parents_`-driven frontier walk
// with per-class round-stamp tracking via `last_marked_`:
//
//   * Each class r records the highest round R for which it was the
//     new-root after a union. The mark is updated via parlay::write_max
//     (monotone CAS-max) so a stale-R tail worker cannot overwrite a
//     fresher mark.
//   * Each round R: filter "dirty" = terms whose at least one child
//     has find_root.last_marked ∈ {R-1, R}. The double-check covers
//     the race where a tail union from round R-1 lands AFTER we've
//     bumped to R but is logically part of "what just happened."
//   * Semisort dirty by current canonical sig. For each multi-root
//     bucket, dnc_union the members onto bucket[0]; the new root for
//     each bucket gets last_marked stamped with R.
//   * Loop until dirty is empty.
//
// Correctness sketch: a bucket-mate pair t, t' with current_canon(t)
// == current_canon(t') can only become bucket-mates *now* if at least
// one child of t (or t') had its root change since the last semisort.
// That child's new root carries last_marked == R-of-recent. So both
// t and t' end up in dirty in round R+1. dnc_union merges them.
//
// On flat workloads (random with depth=1) most rounds have small
// dirty sets and the filter is fast. On wide-fanout workloads (cube
// round 0) the f-layer is entirely dirty in round 0, but each
// subsequent round the dirty set shrinks rapidly: only terms whose
// child's root is among the fresh marks get included.

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/detail.hpp"
#include "parallel_egraph/dnc_union.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/utilities.h>
#include <parlay/internal/group_by.h>

namespace pe {

namespace {

// Monotone CAS-max via parlay's helper. Used to stamp the surviving
// root of every union with the current round number; if two threads
// race and one is writing R-1 while the other writes R, only the
// larger value survives.
inline void mark_round(std::atomic<std::uint64_t>& slot, std::uint64_t r) {
  parlay::write_max(&slot, r, std::less<std::uint64_t>{});
}

}  // namespace

template <>
void EGraph<ConcurrentUnionFind>::parallel_close_async_rounds(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  const std::size_t n = nodes_.size();

  // Round numbering: R starts at 1 so that "is_marked == R-1" with R=1
  // catches the round-0 mark (initial unions in step 1). All
  // last_marked_[i] start at 0 (never touched) per the async ctor.
  std::uint64_t R = 1;

  // ---- step 1: apply initial unions in parallel; mark new roots ----------
  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    auto [a, b] = initial_unions[i];
    uf_.union_(a, b);
    // Whichever side became the new root, find_root(a) returns it.
    mark_round(last_marked_[uf_.find_root(a)], R);
  });

  // Build the list of non-leaf indices once. Leaves never need to be
  // sorted — their canonical sig is just their op string and can never
  // share a bucket with a non-leaf.
  auto all_idx = parlay::iota<std::uint32_t>(static_cast<std::uint32_t>(n));
  auto non_leaf = parlay::filter(all_idx, [&](std::uint32_t i) {
    return !nodes_[i].children.empty();
  });

  std::size_t round = 0;
  while (true) {
    // Bump R for the upcoming dirty filter. The tail-write race is
    // handled by the {R-1, R} filter and the monotone mark update.
    ++R;

    // ---- step 2: filter to dirty -----------------------------------
    auto t_filter = clk::now();
    auto dirty = parlay::filter(non_leaf, [&](std::uint32_t i) {
      const auto& cs = nodes_[i].children;
      for (Id c : cs) {
        const std::uint64_t m =
            last_marked_[uf_.find_root(c)].load(std::memory_order_relaxed);
        if (m == R - 1 || m == R) return true;
      }
      return false;
    });
    double filter_ms = trace ? ms_since(t_filter) : 0.0;

    if (dirty.empty()) {
      // No term has a child whose root was recently rerooted → no
      // new congruences possible → fixpoint.
      if (trace) {
        std::fprintf(stderr,
                     "[pe-async] round=%3zu R=%llu dirty=        0 "
                     "filter=%7.3fms (clean break)\n",
                     round, (unsigned long long)R, filter_ms);
      }
      break;
    }

    // ---- step 3: semisort dirty by current canonical sig -----------
    auto t_semisort = clk::now();
    auto canon_entries = parlay::map(dirty, [&](std::uint32_t i) {
#ifdef PE_SEMISORT_SOUND
      return detail::CanonEntry{sig_hash(nodes_[i], uf_), uf_.find_root(i),
                                static_cast<Id>(i)};
#else
      auto [h1, h2] = sig_hashes(nodes_[i], uf_);
      return detail::CanonEntry{h1, uf_.find_root(i), h2};
#endif
    });

    auto keyed = parlay::map(canon_entries, [](const detail::CanonEntry& e) {
      return std::pair<detail::CanonEntry, detail::CanonEntry>{e, e};
    });
    auto hash_fn = [](const detail::CanonEntry& e) -> std::size_t {
      return e.hash;
    };
#ifdef PE_SEMISORT_SOUND
    auto equal_fn = [&](const detail::CanonEntry& a,
                        const detail::CanonEntry& b) {
      return a.hash == b.hash &&
             sigs_equal(a.class_id, b.class_id, uf_, nodes_);
    };
#else
    auto equal_fn = [](const detail::CanonEntry& a,
                       const detail::CanonEntry& b) {
      return a.hash == b.hash && a.secondary_hash == b.secondary_hash;
    };
#endif
    auto groups = parlay::group_by_key(keyed, hash_fn, equal_fn);

    // PE_TRACE_VERBOSE=1: dump every dirty term + bucket assignment.
    // Only useful at small scale.
    if (std::getenv("PE_TRACE_VERBOSE")) {
      std::fprintf(stderr, "[pe-async-v] round=%zu R=%llu dirty terms:\n",
                   round, (unsigned long long)R);
      for (std::size_t k = 0; k < dirty.size(); ++k) {
        const std::uint32_t i = dirty[k];
        std::fprintf(stderr, "  [%u] op='%s' children=[", i, nodes_[i].op.c_str());
        for (std::size_t cc = 0; cc < nodes_[i].children.size(); ++cc) {
          if (cc) std::fputs(",", stderr);
          const Id child = nodes_[i].children[cc];
          const Id child_root = uf_.find_root(child);
          std::fprintf(stderr, "%u(root=%u,marked=%llu)", child, child_root,
                       (unsigned long long)last_marked_[child_root].load());
        }
        std::fprintf(stderr, "] self_root=%u\n", uf_.find_root(i));
      }
      std::fprintf(stderr, "[pe-async-v] round=%zu groups (%zu):\n",
                   round, groups.size());
      for (std::size_t g = 0; g < groups.size(); ++g) {
        std::fprintf(stderr, "  group hash=0x%llx members=[",
                     (unsigned long long)groups[g].first.hash);
        for (std::size_t m = 0; m < groups[g].second.size(); ++m) {
          if (m) std::fputs(",", stderr);
          std::fprintf(stderr, "root=%u", groups[g].second[m].root);
        }
        std::fputs("]\n", stderr);
      }
    }

    // For every multi-member bucket whose elements don't already
    // share a root, dnc_union them. Stamp the surviving root with R
    // so next round's dirty filter picks up parents-of-rerooted-class.
    parlay::parallel_for(0, groups.size(), [&](std::size_t g) {
      auto& bucket = groups[g].second;
      if (bucket.size() < 2) return;
      const Id ref = bucket[0].root;
      bool all_same = parlay::all_of(
          bucket, [ref](const detail::CanonEntry& e) { return e.root == ref; });
      if (all_same) return;
      detail::dnc_union(bucket, 0, bucket.size(), uf_);
      // Bucket merged; whichever root survived is now the canonical
      // root for every member. Stamp it.
      mark_round(last_marked_[uf_.find_root(bucket[0].root)], R);
    });
    double semisort_ms = trace ? ms_since(t_semisort) : 0.0;

    if (trace) {
      std::fprintf(stderr,
                   "[pe-async] round=%3zu R=%llu dirty=%9zu groups=%9zu "
                   "filter=%7.3fms semisort=%7.3fms\n",
                   round, (unsigned long long)R, dirty.size(),
                   groups.size(), filter_ms, semisort_ms);
    }
    ++round;
  }
}

// ---- MIN_ID variant -----------------------------------------------------
//
// Same algorithm as `parallel_close_async_rounds`; every union (initial
// unions and dnc_union) goes through `ConcurrentUnionFind::union_min_id`
// so the lower-id root always survives. This preserves the
// canonical-id-stable invariant on DAG-ordered inputs: lookups and
// insertions across in-pass merges agree on which class id is the root,
// so a regular cascade (Family C) collapses in a single BSP round
// instead of one round per level.

template <>
void EGraph<ConcurrentUnionFind>::parallel_close_async_rounds_min_id(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  const std::size_t n = nodes_.size();
  std::uint64_t R = 1;

  // ---- step 1: apply initial unions in parallel; mark new roots --------
  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    auto [a, b] = initial_unions[i];
    uf_.union_min_id(a, b);
    mark_round(last_marked_[uf_.find_root(a)], R);
  });

  auto all_idx = parlay::iota<std::uint32_t>(static_cast<std::uint32_t>(n));
  auto non_leaf = parlay::filter(all_idx, [&](std::uint32_t i) {
    return !nodes_[i].children.empty();
  });

  // Lambda dispatching dnc_union_with onto the MIN_ID merge.
  auto union_min_fn = [](ConcurrentUnionFind& uf, Id a, Id b) {
    uf.union_min_id(a, b);
  };

  std::size_t round = 0;
  while (true) {
    ++R;

    auto t_filter = clk::now();
    auto dirty = parlay::filter(non_leaf, [&](std::uint32_t i) {
      const auto& cs = nodes_[i].children;
      for (Id c : cs) {
        const std::uint64_t m =
            last_marked_[uf_.find_root(c)].load(std::memory_order_relaxed);
        if (m == R - 1 || m == R) return true;
      }
      return false;
    });
    double filter_ms = trace ? ms_since(t_filter) : 0.0;

    if (dirty.empty()) {
      if (trace) {
        std::fprintf(stderr,
                     "[pe-async-min] round=%3zu R=%llu dirty=        0 "
                     "filter=%7.3fms (clean break)\n",
                     round, (unsigned long long)R, filter_ms);
      }
      break;
    }

    auto t_semisort = clk::now();
    auto canon_entries = parlay::map(dirty, [&](std::uint32_t i) {
#ifdef PE_SEMISORT_SOUND
      return detail::CanonEntry{sig_hash(nodes_[i], uf_), uf_.find_root(i),
                                static_cast<Id>(i)};
#else
      auto [h1, h2] = sig_hashes(nodes_[i], uf_);
      return detail::CanonEntry{h1, uf_.find_root(i), h2};
#endif
    });

    auto keyed = parlay::map(canon_entries, [](const detail::CanonEntry& e) {
      return std::pair<detail::CanonEntry, detail::CanonEntry>{e, e};
    });
    auto hash_fn = [](const detail::CanonEntry& e) -> std::size_t {
      return e.hash;
    };
#ifdef PE_SEMISORT_SOUND
    auto equal_fn = [&](const detail::CanonEntry& a,
                        const detail::CanonEntry& b) {
      return a.hash == b.hash &&
             sigs_equal(a.class_id, b.class_id, uf_, nodes_);
    };
#else
    auto equal_fn = [](const detail::CanonEntry& a,
                       const detail::CanonEntry& b) {
      return a.hash == b.hash && a.secondary_hash == b.secondary_hash;
    };
#endif
    auto groups = parlay::group_by_key(keyed, hash_fn, equal_fn);

    parlay::parallel_for(0, groups.size(), [&](std::size_t g) {
      auto& bucket = groups[g].second;
      if (bucket.size() < 2) return;
      const Id ref = bucket[0].root;
      bool all_same = parlay::all_of(
          bucket, [ref](const detail::CanonEntry& e) { return e.root == ref; });
      if (all_same) return;
      detail::dnc_union_with(bucket, 0, bucket.size(), uf_, union_min_fn);
      // After MIN_ID dnc_union, the surviving root is the lowest-id
      // member of the bucket's pre-merge roots. Stamp it.
      mark_round(last_marked_[uf_.find_root(bucket[0].root)], R);
    });
    double semisort_ms = trace ? ms_since(t_semisort) : 0.0;

    if (trace) {
      std::fprintf(stderr,
                   "[pe-async-min] round=%3zu R=%llu dirty=%9zu groups=%9zu "
                   "filter=%7.3fms semisort=%7.3fms\n",
                   round, (unsigned long long)R, dirty.size(),
                   groups.size(), filter_ms, semisort_ms);
    }
    ++round;
  }
}

}  // namespace pe
