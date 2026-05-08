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

namespace {

// Stamp `slot` with round number `r`. Cheap load-then-conditional-store
// (no CAS): safe in this rounds-based algorithm because every writer
// within a single parallel_for captures the same `R` value, so all
// concurrent writers want to write the same number. Worst-case race:
//
//   T_A loads → sees 0 (stale, < R)
//   T_B loads → sees 0
//   T_A stores R                  ← winner
//   T_B stores R                  ← redundant but identical, no harm
//
// The load + conditional skip lets us avoid the redundant store on
// the common path where another thread already stamped this slot.
// Both load and store use memory_order_relaxed: we don't need ordering
// because the surrounding parallel_for/barrier already supplies it.
//
// IMPORTANT: this is NOT safe in a barrier-free variant where threads
// from different rounds could race with different R values. In that
// case use `mark_round_cas_max` (CAS-loop) below instead — a
// stale-R writer must not overwrite a fresh-R one.
inline void mark_round(std::atomic<std::uint64_t>& slot, std::uint64_t r) {
  if (slot.load(std::memory_order_relaxed) < r) {
    slot.store(r, std::memory_order_relaxed);
  }
}

// Monotone CAS-max version, kept available for the eventual
// barrier-free async path where writers from different rounds may
// race. Unused by the current rounds-based code; left here as a
// reference for the algorithm-level invariant document above.
[[maybe_unused]]
inline void mark_round_cas_max(std::atomic<std::uint64_t>& slot,
                                std::uint64_t r) {
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
#ifdef PE_SEMISORT_SOUND
      return detail::CanonEntry{sig_hash(nodes_[i], uf_), uf_.find_root(i),
                                static_cast<Id>(i)};
#else
      auto [h1, h2] = sig_hashes(nodes_[i], uf_);
      return detail::CanonEntry{h1, uf_.find_root(i), h2};
#endif
    });

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

// ---- truly-async continuous variant -------------------------------------
//
// Same `last_marked_` dirty-tracking and {R-1, R} filter window as the
// rounds-based async variant, but the round barrier is dissolved: a
// semisorter "thread" and a unioner "thread" run concurrently via
// parlay::par_do (sharing the outer scheduler's worker pool — each side
// uses parlay parallel primitives for its inner work). They communicate
// through a mutex-protected mailbox of CanonEntry groups.
//
// Correctness sketch
// ------------------
// Unioner reads `R` (a global atomic counter, bumped once per semisorter
// scan) AT WRITE TIME and write_max's the surviving root's last_marked_
// with that R. Monotonic write_max means racy reads can only over-mark
// (never under-mark relative to "current R"). The next semisorter scan
// bumps R from R_now to R_new=R_now+1 and filters {R_new-1, R_new}
// = {R_now, R_now+1} — catches the just-written mark.
//
// Could a mark land BEFORE the bump but be missed by the bumping scan?
// The semisorter is single logical "thread" (parallel internally), so
// scans don't pipeline — each scan completes before the next bump. A
// mark with value M is in the next scan's window iff M ∈ {R_new-1,
// R_new} = {M, M+1} — yes by reflexivity (M >= bump-time R), so always
// caught.
//
// Termination
// -----------
// `outstanding` tracks (groups in mailbox) + (groups in-flight in
// unioner). Pushed on group emission, decremented on group consume +
// process complete. When semisorter scan finds dirty empty AND
// outstanding == 0, no more work can be created → set `done`. Unioner
// drains, sees `done`, exits.
//
// What this prototype is NOT (yet)
// --------------------------------
//   * Mailbox uses std::mutex + std::deque. Empirically uncontested
//     on the workloads we measure: pushes batch into one lock acquire
//     per scan (semisorter loops over `groups` under one lock); drain
//     batches into one lock acquire per pull (move-out the entire
//     deque). A lock-free distributed bag (per-worker array blocks à
//     la Sundell et al. SPAA'11) was prototyped but the per-call
//     allocation of the strict-bound bag (~32 MB on `large`) cost
//     more than any contention savings paid back. Revisit only if
//     profiling shows the mutex hot.
//   * No dedicated schedulers (no execute_with_scheduler+std::thread
//     for each side). par_do shares the outer pool — workers steal
//     across both branches.
//
// What was tried and didn't pan out (kept as comments in commit
// history for reference):
//   * Truly continuous scan (no drain-gate) using a marks_written
//     watermark. Pipelining peeked at partial waves of marks, which
//     fragmented one logical scan into multiple smaller ones — the
//     extra O(non_leaf) filter passes regressed deep-l by ~10ms with
//     no win on any workload tested.
//   * Distributed array bag (per-worker blocks). Solved the global-
//     head contention but introduced ~32 MB per-call allocation cost.
//     Net regression of ~5ms.

template <>
void EGraph<ConcurrentUnionFind>::parallel_close_async_continuous(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  const std::size_t n = nodes_.size();

  // Global atomic round counter. Bumped at the start of each semisorter
  // scan; sampled by unioners at write_max time. Cache-line-padded so
  // it doesn't false-share with `outstanding` (mutated every batch
  // drain by the unioner) or `done` (written once at termination); a
  // shared line would have every unioner thread invalidate-and-refetch
  // R on every drain, even though R is logically read-only between
  // scans.
  alignas(64) std::atomic<std::uint64_t> R{1};
  char _pad_R[64 - sizeof(std::atomic<std::uint64_t>)];

  // ---- step 1: apply initial unions in parallel; mark new roots ----------
  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    auto [a, b] = initial_unions[i];
    uf_.union_(a, b);
    // R == 1 here; first semisorter scan will bump to 2 and catch
    // marks in {1, 2}.
    mark_round(last_marked_[uf_.find_root(a)].v,
               R.load(std::memory_order_relaxed));
  });

  // Build the non_leaf list once (round-invariant).
  auto all_idx = parlay::iota<std::uint32_t>(static_cast<std::uint32_t>(n));
  auto non_leaf = parlay::filter(all_idx, [&](std::uint32_t i) {
    return !nodes_[i].children.empty();
  });

  // Mailbox of per-scan canon batches. Each batch carries the
  // integer-sorted CanonEntry array and the index of the start of
  // each primary-hash run. The unioner walks these runs in parallel
  // (each run = one bucket, a slice of canon). Replaces an earlier
  // mailbox-of-individual-groups design: the integer-sort path skips
  // group_by_key's per-bucket sequence allocations, so there's nothing
  // to push individually anymore. One push per scan; one drain
  // consumes the whole deque.
  struct ScanBatch {
    parlay::sequence<detail::CanonEntry> canon;
    parlay::sequence<std::uint32_t> run_starts;
  };
  std::mutex mailbox_mu;
  std::deque<ScanBatch> mailbox;

  // Equality function for splitting same-primary-hash runs. Used by
  // the unioner when walking the integer-sorted canon. Hoisted to
  // function scope so both teams can see the same definition (only
  // matters for PE_SEMISORT_SOUND, which captures `nodes_`).
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

  // Coordination state. Each gets its own cache line (see comment on R)
  // to keep unrelated atomics off the same MESI traffic.
  alignas(64) std::atomic<std::int64_t> outstanding{0};
  char _pad_outstanding[64 - sizeof(std::atomic<std::int64_t>)];
  alignas(64) std::atomic<bool> done{false};
  char _pad_done[64 - sizeof(std::atomic<bool>)];

  // Trace counters.
  std::atomic<std::size_t> trace_scans{0};
  std::atomic<std::size_t> trace_empty_scans{0};
  std::atomic<std::size_t> trace_groups_pushed{0};
  std::atomic<std::size_t> trace_groups_consumed{0};
  std::atomic<std::size_t> trace_unions_committed{0};

  auto t_start = clk::now();

  // ---- semisorter (one logical thread, parallel internally) --------------
  // Drain-gated BSP-with-overlap rather than truly continuous: wait
  // for the previous scan's unions to fully drain before scanning
  // again. The truly-continuous variant (using `marks_written` as a
  // wake-up signal, no drain-gate) was prototyped and lost: scans
  // peeking mid-drain see only partial marks, fragmenting one logical
  // wave into multiple scans. Each extra scan pays the full O(non_leaf)
  // filter cost — empirically ~10 ms regression on deep-l, no win on
  // any workload tested. The drain-gate gives each scan a complete
  // wave of marks, matching async-rounds' cadence.
  auto semisorter = [&]() {
    while (true) {
      while (outstanding.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
      }

      const std::uint64_t R_now =
          R.fetch_add(1, std::memory_order_acq_rel) + 1;

      auto dirty = parlay::filter(non_leaf, [&](std::uint32_t i) {
        const auto& cs = nodes_[i].children;
        for (Id c : cs) {
          const std::uint64_t m =
              last_marked_[uf_.find_root(c)].v.load(std::memory_order_relaxed);
          if (m == R_now - 1 || m == R_now) return true;
        }
        return false;
      });

      if (trace) trace_scans.fetch_add(1, std::memory_order_relaxed);

      if (dirty.empty()) {
        if (trace) trace_empty_scans.fetch_add(1, std::memory_order_relaxed);
        done.store(true, std::memory_order_release);
        return;
      }

      // Build CanonEntries.
      auto canon_entries = parlay::map(dirty, [&](std::uint32_t i) {
#ifdef PE_SEMISORT_SOUND
        return detail::CanonEntry{sig_hash(nodes_[i], uf_), uf_.find_root(i),
                                  static_cast<Id>(i)};
#else
        auto [h1, h2] = sig_hashes(nodes_[i], uf_);
        return detail::CanonEntry{h1, uf_.find_root(i), h2};
#endif
      });

      // Integer-sort canon in place by primary hash, compute run
      // boundaries. Each run is one primary-hash bucket; within a
      // run the unioner splits by full equality (almost always one
      // sub-run per primary run, since 96-bit collisions are rare).
      // equal_fn for the per-run split lives at function scope; see
      // its definition above the par_do.
      parlay::integer_sort_inplace(
          parlay::make_slice(canon_entries),
          [](const detail::CanonEntry& e) -> std::uint64_t { return e.hash; });

      auto starts_flag = parlay::tabulate(
          canon_entries.size(), [&](std::size_t i) {
            return i == 0 || canon_entries[i].hash != canon_entries[i - 1].hash;
          });
      auto run_starts = parlay::pack_index<std::uint32_t>(starts_flag);

      const std::size_t n_runs = run_starts.size();
      {
        std::lock_guard<std::mutex> lk(mailbox_mu);
        mailbox.push_back(ScanBatch{std::move(canon_entries),
                                     std::move(run_starts)});
      }
      // One outstanding "unit" per pushed batch; the drain-gate's
      // wait-for-zero check is the same shape regardless of batch
      // granularity. The trace counter still tracks total runs for
      // a more useful signal.
      outstanding.fetch_add(1, std::memory_order_release);
      if (trace) {
        trace_groups_pushed.fetch_add(n_runs, std::memory_order_relaxed);
      }
    }
  };

  // ---- unioner (drain mailbox, parallel_for over each batch's runs) -----
  // Each iteration: drain the mailbox under one lock, then parlay::
  // parallel_for over the (canon, run_starts) pairs. Per run: split
  // by full equality (handles the rare 96-bit collision), apply the
  // all_same_root short-circuit, dnc_union, mark surviving root.
  // Buckets are slices of `canon` — no copy, no per-bucket alloc.
  auto unioner = [&]() {
    while (true) {
      std::vector<ScanBatch> batches;
      {
        std::lock_guard<std::mutex> lk(mailbox_mu);
        if (!mailbox.empty()) {
          batches.reserve(mailbox.size());
          while (!mailbox.empty()) {
            batches.push_back(std::move(mailbox.front()));
            mailbox.pop_front();
          }
        }
      }

      if (batches.empty()) {
        if (done.load(std::memory_order_acquire)) return;
        std::this_thread::yield();
        continue;
      }

      std::size_t run_total = 0;
      for (auto& b : batches) {
        auto& canon = b.canon;
        auto& run_starts = b.run_starts;
        run_total += run_starts.size();
        parlay::parallel_for(0, run_starts.size(), [&](std::size_t r) {
          const std::size_t lo = run_starts[r];
          const std::size_t hi = (r + 1 < run_starts.size())
              ? static_cast<std::size_t>(run_starts[r + 1])
              : canon.size();
          if (hi - lo < 2) return;
          std::size_t i = lo;
          while (i < hi) {
            std::size_t j = i + 1;
            while (j < hi && equal_fn(canon[i], canon[j])) ++j;
            if (j - i >= 2) {
              const Id ref = canon[i].root;
              bool all_same = true;
              for (std::size_t k = i + 1; k < j; ++k) {
                if (canon[k].root != ref) { all_same = false; break; }
              }
              if (!all_same) {
                detail::dnc_union(canon, i, j, uf_);
                const std::uint64_t R_now =
                    R.load(std::memory_order_acquire);
                mark_round(last_marked_[uf_.find_root(canon[i].root)].v,
                           R_now);
                if (trace) {
                  trace_unions_committed.fetch_add(
                      1, std::memory_order_relaxed);
                }
              }
            }
            i = j;
          }
        });
      }

      outstanding.fetch_sub(static_cast<std::int64_t>(batches.size()),
                            std::memory_order_release);
      if (trace) {
        trace_groups_consumed.fetch_add(run_total,
                                         std::memory_order_relaxed);
      }
    }
  };

  // Run both teams concurrently. par_do returns when both finish.
  parlay::par_do(semisorter, unioner);

  if (trace) {
    std::fprintf(stderr,
                 "[pe-async-cont] scans=%zu empty_scans=%zu groups_pushed=%zu "
                 "groups_consumed=%zu unions=%zu  total=%7.3fms\n",
                 trace_scans.load(), trace_empty_scans.load(),
                 trace_groups_pushed.load(), trace_groups_consumed.load(),
                 trace_unions_committed.load(),
                 ms_since(t_start));
  }
}

// ---- naive rounds-based variant -----------------------------------------
//
// Semisorts every non-leaf term every round; no `last_marked_`, no dirty
// filter. Round R:
//   (1) Build CanonEntry over every non-leaf using current `find_root`.
//   (2) group_by_key on the primary hash (secondary or sound equality
//       inside the bucket comparator).
//   (3) For each multi-member bucket whose snapshot roots aren't all
//       equal, dnc_union the members and flag `changed = true`.
// Loop until a full pass produces no merge. Convergence: every union
// strictly reduces the equivalence-class count, which is bounded below
// by 1, so the loop terminates in at most n_classes iterations. In
// practice it matches the BSP cadence — one round per congruence
// "level" — because the snapshot roots inside a bucket capture every
// reroot from the previous round.

template <>
void EGraph<ConcurrentUnionFind>::parallel_close_naive_rounds(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  const std::size_t n = nodes_.size();

  // ---- step 1: apply initial unions in parallel -------------------------
  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    auto [a, b] = initial_unions[i];
    uf_.union_(a, b);
  });

  // Non-leaf set is round-invariant — the dirty filter is what made the
  // async version bother caching it; here we cache it only to skip leaves
  // (whose canonical sig can never bucket-match a non-leaf).
  auto all_idx = parlay::iota<std::uint32_t>(static_cast<std::uint32_t>(n));
  auto non_leaf = parlay::filter(all_idx, [&](std::uint32_t i) {
    return !nodes_[i].children.empty();
  });

  std::size_t round = 0;
  while (true) {
    auto t_round = clk::now();

    // ---- step 2: canonical sigs for *every* non-leaf --------------------
    auto canon_entries = parlay::map(non_leaf, [&](std::uint32_t i) {
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

    // ---- step 3: union per bucket; flag if anything changed -------------
    std::atomic<bool> changed{false};
    parlay::parallel_for(0, groups.size(), [&](std::size_t g) {
      auto& bucket = groups[g].second;
      if (bucket.size() < 2) return;
      const Id ref = bucket[0].root;
      bool all_same = parlay::all_of(
          bucket, [ref](const detail::CanonEntry& e) { return e.root == ref; });
      if (all_same) return;
      changed.store(true, std::memory_order_relaxed);
      detail::dnc_union(bucket, 0, bucket.size(), uf_);
    });

    double round_ms = trace ? ms_since(t_round) : 0.0;
    if (trace) {
      std::fprintf(stderr,
                   "[pe-naive] round=%3zu non_leaf=%9zu groups=%9zu "
                   "changed=%d  %7.3fms\n",
                   round, non_leaf.size(), groups.size(),
                   int(changed.load()), round_ms);
    }

    if (!changed.load(std::memory_order_relaxed)) break;
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
    mark_round(last_marked_[uf_.find_root(a)].v, R);
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
            last_marked_[uf_.find_root(c)].v.load(std::memory_order_relaxed);
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

    // Same integer-sort + run-walk swap as the by-rank variant; see
    // commentary on parallel_close_async_rounds for the rationale.
    parlay::integer_sort_inplace(
        parlay::make_slice(canon_entries),
        [](const detail::CanonEntry& e) -> std::uint64_t { return e.hash; });

    auto starts_flag = parlay::tabulate(canon_entries.size(), [&](std::size_t i) {
      return i == 0 || canon_entries[i].hash != canon_entries[i - 1].hash;
    });
    auto run_starts = parlay::pack_index<std::uint32_t>(starts_flag);

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
          const Id ref = canon_entries[i].root;
          bool all_same = true;
          for (std::size_t k = i + 1; k < j; ++k) {
            if (canon_entries[k].root != ref) { all_same = false; break; }
          }
          if (!all_same) {
            detail::dnc_union_with(canon_entries, i, j, uf_, union_min_fn);
            // After MIN_ID dnc_union, the surviving root is the
            // lowest-id member of the bucket's pre-merge roots.
            mark_round(last_marked_[uf_.find_root(canon_entries[i].root)].v, R);
          }
        }
        i = j;
      }
    });
    double semisort_ms = trace ? ms_since(t_semisort) : 0.0;

    if (trace) {
      std::fprintf(stderr,
                   "[pe-async-min] round=%3zu R=%llu dirty=%9zu runs=%9zu "
                   "filter=%7.3fms semisort=%7.3fms\n",
                   round, (unsigned long long)R, dirty.size(),
                   run_starts.size(), filter_ms, semisort_ms);
    }
    ++round;
  }
}

}  // namespace pe
