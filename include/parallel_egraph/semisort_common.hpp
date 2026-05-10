#pragma once
// Shared body of `merge_and_collect_semisort` for both equality variants.
// The two variants (semisort_secondary / semisort_sound) differ in just
// the equality predicate; everything else — keying, group_by_key, the
// per-group dnc_union, and the optional sub-phase timing — is identical.
//
// This header is included from exactly one TU per build (the variant
// header that lives next to it).

#include "parallel_egraph/detail.hpp"
#include "parallel_egraph/dnc_union.hpp"
#include "parallel_egraph/egraph.hpp"

#include <chrono>
#include <utility>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>
#include <parlay/internal/integer_sort.h>

namespace pe::detail {

// Stripped-down variant of `semisort_with_equality` for
// `parallel_close_topo`: semisort by signature, dnc_union each
// non-singleton bucket, done. No next-round frontier, no
// all_same_root short-circuit (that check existed only to keep BSP's
// `parents_[root]`-self-reference from re-entering next_work; since
// the topo path drives rounds from precomputed depth buckets, it
// can't loop, and dnc_union is a UF no-op when all members already
// share a root). Skipping the per-group `parlay::map(values, .root)`
// and the outer `parlay::flatten` is the whole point.
//
// Two implementations:
//   * `apply_unions_semisort`        — uses `parlay::group_by_key`. Builds a
//                                     keyed-pair array + sequence-of-sequences.
//   * `apply_unions_integer_sort`    — sorts in-place by primary hash, walks
//                                     runs in parallel. Avoids the keyed-pair
//                                     map and the per-group sequence build.
//                                     Used by `parallel_close_topo`.

template <typename EqualFn>
void apply_unions_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    EqualFn equal_fn) {
  if (canon.empty()) return;
  auto keyed = parlay::map(canon, [](const CanonEntry& e) {
    return std::pair<CanonEntry, CanonEntry>{e, e};
  });
  auto hash_fn = [](const CanonEntry& e) -> std::size_t { return e.hash; };
  auto groups = parlay::group_by_key(keyed, hash_fn, equal_fn);
  parlay::parallel_for(0, groups.size(), [&](std::size_t g) {
    auto& values = groups[g].second;
    if (values.size() < 2) return;
    dnc_union(values, 0, values.size(), uf);
  });
}

// Optimization A: in-place integer-sort by primary hash + parallel
// run-walk. One sort pass, one boundary-flag tabulate, one pack_index,
// one per-run parallel_for. No keyed-pair allocation, no per-group
// `parlay::sequence<value>` allocation.
//
// Within each primary-hash run we split by full equality (`equal_fn`).
// The expected case is "every element in the run matches" — true
// 96-bit collisions are ~10^-14 per 64M-element batch — so the inner
// while-loop almost always iterates once and does a single dnc_union.
template <typename EqualFn>
void apply_unions_integer_sort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    EqualFn equal_fn) {
  if (canon.empty()) return;

  parlay::integer_sort_inplace(
      parlay::make_slice(canon),
      [](const CanonEntry& e) -> std::uint64_t { return e.hash; });

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
    // Same-primary-hash run: split by full equality. Expected fast
    // path: single subgroup spanning [lo, hi).
    std::size_t i = lo;
    while (i < hi) {
      std::size_t j = i + 1;
      while (j < hi && equal_fn(canon[i], canon[j])) ++j;
      if (j - i >= 2) dnc_union(canon, i, j, uf);
      i = j;
    }
  });
}

template <typename EqualFn>
parlay::sequence<Id> semisort_with_equality(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    SemisortTimings* timings, EqualFn equal_fn) {
  if (canon.empty()) {
    if (timings) *timings = {0.0, 0.0, 0.0};
    return {};
  }

  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  // parlay::group_by_key wants a sequence of (key, value); we pair each
  // CanonEntry with itself (we want the full struct on both sides).
  auto t0 = timings ? clk::now() : clk::time_point{};
  auto keyed = parlay::map(canon, [](const CanonEntry& e) {
    return std::pair<CanonEntry, CanonEntry>{e, e};
  });
  if (timings) timings->keyed_ms = ms_since(t0);

  auto hash_fn = [](const CanonEntry& e) -> std::size_t { return e.hash; };
  auto t1 = timings ? clk::now() : clk::time_point{};
  auto groups = parlay::group_by_key(keyed, hash_fn, equal_fn);
  if (timings) timings->group_by_ms = ms_since(t1);

  // For each non-singleton group: precheck all_same_root (a cheap
  // O(n) read scan) to skip buckets where every member already shares
  // a root — those buckets would do dnc_union with all-no-op union_
  // calls and emit nothing. Otherwise dnc_union returns the surviving
  // root directly (no separate find_root needed), and we emit losers
  // (members whose pre-union .root differs from the survivor) into
  // next_work. Skipping the survivor saves one entry per merged
  // bucket vs the older "emit every member" pattern.
  // For each non-singleton group: first scan for all_same_root and
  // skip those buckets — they'd produce O(bucket_size) `union_`
  // no-ops in dnc_union (each with a find()-and-CAS chain), and
  // empirically they dominate late-round work on big workloads
  // (~30% on 8XL). The precheck is purely a perf optimization
  // post-loser-only emission: with loser-only, an all_same_root
  // bucket would naturally produce empty output, so correctness is
  // preserved either way (see test_self_merge_no_loop).
  auto t2 = timings ? clk::now() : clk::time_point{};
  auto per_group = parlay::map(groups, [&](auto& kv) -> parlay::sequence<Id> {
    auto& values = kv.second;
    if (values.size() < 2) return {};
    const Id root_ref = values[0].root;
    bool all_same_root = parlay::all_of(
        values, [root_ref](const CanonEntry& e) { return e.root == root_ref; });
    if (all_same_root) return {};
    const Id survivor = dnc_union(values, 0, values.size(), uf);
    return parlay::map_maybe(values, [survivor](const detail::CanonEntry& e)
                                    -> std::optional<Id> {
      if (e.root == survivor) return std::nullopt;  // winner
      return e.root;  // loser
    });
  });
  auto out = parlay::flatten(per_group);
  if (timings) timings->per_group_ms = ms_since(t2);
  return out;
}

}  // namespace pe::detail
