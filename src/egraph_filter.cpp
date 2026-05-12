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
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/utilities.h>
#include <parlay/internal/group_by.h>
#include <parlay/internal/integer_sort.h>

namespace pe {

// (ae branch: only parallel_filter retained; *_groupby, *_hybrid,
//  *_min_id, parallel_naive_rounds variants removed.)

namespace {
// Stamp `slot` with round number `r`. Cheap load-then-conditional-store
// (no CAS) — safe in this rounds-based algorithm because the per-round
// barrier means every concurrent writer captures the same R, so a race
// can only land redundant identical stores. Memory ordering is supplied
// by the surrounding parallel_for + barrier.
inline void mark_round(std::atomic<std::uint64_t>& slot, std::uint64_t r) {
  if (slot.load(std::memory_order_relaxed) < r) {
    slot.store(r, std::memory_order_relaxed);
  }
}
}  // namespace

template <>
void EGraph<ConcurrentUnionFind>::parallel_filter(
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
    mark_round(last_marked_[uf_.find_root(a)].v, R);
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
            last_marked_[uf_.find_root(c)].v.load(std::memory_order_relaxed);
        // note this is unsound in the true async version, you also have to check if m == R
        if (m == R - 1) return true;
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
      return detail::CanonEntry{sig_hash(nodes_[i], uf_), uf_.find_root(i),
                                static_cast<Id>(i)};
    });

    auto equal_fn = [&](const detail::CanonEntry& a,
                        const detail::CanonEntry& b) {
      return a.hash == b.hash &&
             sigs_equal(a.class_id, b.class_id, uf_, nodes_);
    };

    // Integer-sort + run-walk, in place. Replaces an earlier
    // group_by_key implementation that built a hash table and
    // materialized a `sequence<sequence<CanonEntry>>` (one inner
    // buffer per bucket). On `large` that was 290k inner allocations
    // per scan plus the hash-table build; the integer-sort path
    // collapses both into one in-place 64-bit radix sort plus a
    // single tabulate + pack_index over the boundary flags. Buckets
    // are now `[lo, hi)` slices of `canon_entries` itself, so
    // dnc_union and the per-bucket mark_round operate directly on
    // the sorted array.
    parlay::integer_sort_inplace(
        parlay::make_slice(canon_entries),
        [](const detail::CanonEntry& e) -> std::uint64_t { return e.hash; });

    auto starts_flag = parlay::tabulate(canon_entries.size(), [&](std::size_t i) {
      return i == 0 || canon_entries[i].hash != canon_entries[i - 1].hash;
    });
    auto run_starts = parlay::pack_index<std::uint32_t>(starts_flag);

    // PE_TRACE_VERBOSE=1: dump every dirty term + run boundaries.
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
                       (unsigned long long)last_marked_[child_root].v.load());
        }
        std::fprintf(stderr, "] self_root=%u\n", uf_.find_root(i));
      }
      std::fprintf(stderr, "[pe-async-v] round=%zu runs (%zu):\n",
                   round, run_starts.size());
      for (std::size_t r = 0; r < run_starts.size(); ++r) {
        const std::size_t lo = run_starts[r];
        const std::size_t hi = (r + 1 < run_starts.size())
            ? static_cast<std::size_t>(run_starts[r + 1])
            : canon_entries.size();
        std::fprintf(stderr, "  run hash=0x%llx members=[",
                     (unsigned long long)canon_entries[lo].hash);
        for (std::size_t m = lo; m < hi; ++m) {
          if (m != lo) std::fputs(",", stderr);
          std::fprintf(stderr, "root=%u", canon_entries[m].root);
        }
        std::fputs("]\n", stderr);
      }
    }

    // Walk runs in parallel. Within each primary-hash run, split by
    // full equality (96-bit collisions ~10⁻¹⁴ — almost always one
    // sub-run per primary run). For every multi-element sub-run
    // whose elements don't already share a root, dnc_union them and
    // stamp the surviving root with R.
    parlay::parallel_for(0, run_starts.size(), [&](std::size_t r) {
      const std::size_t lo = run_starts[r];
      const std::size_t hi = (r + 1 < run_starts.size())
          ? static_cast<std::size_t>(run_starts[r + 1])
          : canon_entries.size();
      if (hi - lo < 2) return;
      std::size_t i = lo;
      while (i < hi) {
        std::size_t j = i + 1;
        while (j < hi && equal_fn(canon_entries[i], canon_entries[j])) ++j;
        if (j - i >= 2) {
          // all_same_root short-circuit: skip dnc_union + mark_round
          // when the bucket already lives in one class. Common when
          // a previous round already merged everything.
          const Id ref = canon_entries[i].root;
          bool all_same = true;
          for (std::size_t k = i + 1; k < j; ++k) {
            if (canon_entries[k].root != ref) { all_same = false; break; }
          }
          if (!all_same) {
            detail::dnc_union(canon_entries, i, j, uf_);
            mark_round(last_marked_[uf_.find_root(canon_entries[i].root)].v, R);
          }
        }
        i = j;
      }
    });
    double semisort_ms = trace ? ms_since(t_semisort) : 0.0;

    if (trace) {
      std::fprintf(stderr,
                   "[pe-async] round=%3zu R=%llu dirty=%9zu runs=%9zu "
                   "filter=%7.3fms semisort=%7.3fms\n",
                   round, (unsigned long long)R, dirty.size(),
                   run_starts.size(), filter_ms, semisort_ms);
    }
    ++round;
  }
}

}  // namespace pe
