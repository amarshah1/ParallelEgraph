// Async-style congruence closure (still rounds-based, BSP-shaped). The
// algorithm replaces the BSP path's `parents_`-driven frontier walk with
// a per-term cached canonical signature `last_canon_[i]`:
//
//   * Each round recomputes every non-leaf term's current canonical sig.
//   * Filter to the "dirty" set: terms whose current sig differs from
//     last_canon_. A clean term either (a) has no merged children, or
//     (b) had its sig already published into a bucket last round, so it
//     can't trigger new congruences this round without first becoming
//     dirty again.
//   * Semisort dirty by sig; for each bucket of size > 1, union all
//     members onto bucket[0] in parallel.
//   * Update last_canon_ to the just-sorted sig for every dirty term.
//
// Termination: the loop exits when a round (a) finds zero dirty terms,
// or (b) finds dirty terms but every bucket already shares a root —
// i.e. signatures are out of date but the unions they imply are no-ops.
// In either case, the previous semisort proved no new congruences fire.
//
// This trades the BSP path's O(touched_classes × avg_parent_count) per
// round for O(N) per round (we filter all N terms). The win on deep
// workloads is that filtering is contiguous-array-friendly while the
// BSP path's parents_ traversal is pointer-chasing. The win on async
// (when we eventually drop the round boundary) is that step (3)'s
// per-bucket unions can run concurrently with step (2)'s next-round
// canonicalization — no global barrier is needed for correctness, only
// for benchmarking the "rounds" decomposition.

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
#include <parlay/internal/group_by.h>

namespace pe {

template <>
void EGraph<ConcurrentUnionFind>::parallel_close_async_rounds(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  // last_canon_ was populated by the async ctor under the *initial* UF
  // (every class is its own root). After we apply the initial unions
  // below, terms whose children's roots moved will mismatch and enter
  // the first round's dirty set.
  const std::size_t n = nodes_.size();

  // ---- step 1: apply initial unions in parallel --------------------------
  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf_.union_(initial_unions[i].first, initial_unions[i].second);
  });

  std::size_t round = 0;
  while (true) {
    // ---- step 2: recompute current sigs and pick dirty terms -----------
    auto t_canon = clk::now();
    auto current_canon = parlay::tabulate(n, [&](std::size_t i) -> std::uint64_t {
      // Leaves' signatures are constant under any UF state (no children
      // means find_root is never called); they'll always match
      // last_canon_ and be filtered out, so we can skip the work.
      if (nodes_[i].children.empty()) return last_canon_[i];
      return sig_hash(nodes_[i], uf_);
    });
    double canon_ms = trace ? ms_since(t_canon) : 0.0;

    auto t_filter = clk::now();
    auto all_idx = parlay::iota<std::uint32_t>(static_cast<std::uint32_t>(n));
    auto dirty = parlay::filter(all_idx, [&](std::uint32_t i) {
      return current_canon[i] != last_canon_[i];
    });
    double filter_ms = trace ? ms_since(t_filter) : 0.0;

    if (dirty.empty()) {
      // Quiescence: every term's signature still matches the one we
      // last published into a bucket. No new congruences possible.
      if (trace) {
        std::fprintf(stderr,
                     "[pe-async] round=%3zu dirty=        0 canon=%7.3fms "
                     "filter=%7.3fms (clean break)\n",
                     round, canon_ms, filter_ms);
      }
      break;
    }

    // ---- step 3: semisort dirty by sig; union per bucket --------------
    auto t_semisort = clk::now();
    auto canon_entries = parlay::map(dirty, [&](std::uint32_t i) {
      // CanonEntry's trailing 4-byte field differs by build flag; both
      // shapes are populated correctly via the existing helper because
      // sig_hashes() returns the same primary-hash that sig_hash() does
      // (single-pass over the node's children).
#ifdef PE_SEMISORT_SOUND
      return detail::CanonEntry{current_canon[i], uf_.find_root(i),
                                static_cast<Id>(i)};
#else
      auto [h1, h2] = sig_hashes(nodes_[i], uf_);
      (void)h1;  // h1 == current_canon[i]; we already paid for it above.
      return detail::CanonEntry{current_canon[i], uf_.find_root(i), h2};
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

    // For every multi-member bucket whose elements don't already share
    // a single root, union all members. dnc_union handles the actual
    // CAS pattern (same as the BSP path uses). Track whether any
    // bucket was non-trivial — if zero, the round was a no-op even
    // though dirty was non-empty (signatures shifted but the unions
    // they imply are already in place). That counts as quiescence.
    std::atomic<std::size_t> nontrivial_buckets{0};
    parlay::parallel_for(0, groups.size(), [&](std::size_t g) {
      auto& bucket = groups[g].second;
      if (bucket.size() < 2) return;
      const Id ref = bucket[0].root;
      bool all_same = parlay::all_of(
          bucket, [ref](const detail::CanonEntry& e) { return e.root == ref; });
      if (all_same) return;
      detail::dnc_union(bucket, 0, bucket.size(), uf_);
      nontrivial_buckets.fetch_add(1, std::memory_order_relaxed);
    });
    double semisort_ms = trace ? ms_since(t_semisort) : 0.0;

    // ---- step 4: publish new last_canon_ for every dirty term ---------
    auto t_publish = clk::now();
    parlay::parallel_for(0, dirty.size(), [&](std::size_t k) {
      const std::uint32_t i = dirty[k];
      last_canon_[i] = current_canon[i];
    });
    double publish_ms = trace ? ms_since(t_publish) : 0.0;

    const std::size_t nb = nontrivial_buckets.load(std::memory_order_relaxed);
    if (trace) {
      std::fprintf(stderr,
                   "[pe-async] round=%3zu dirty=%9zu groups=%9zu "
                   "merged=%9zu canon=%7.3fms filter=%7.3fms "
                   "semisort=%7.3fms publish=%7.3fms\n",
                   round, dirty.size(), groups.size(), nb,
                   canon_ms, filter_ms, semisort_ms, publish_ms);
    }
    if (nb == 0) {
      // Sigs shifted but every bucket was already merged. The dirty
      // filter caught up to a state where no new unions are produced.
      // Per the contract that we "must end on a clean semisort", this
      // round was the clean one — terminate.
      break;
    }
    ++round;
  }
}

}  // namespace pe
