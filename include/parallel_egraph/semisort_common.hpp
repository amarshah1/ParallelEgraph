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

namespace pe::detail {

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

  // For each non-singleton group, union all members and emit their
  // pre-round roots into next_work. Skip groups whose members all
  // already share a single root: the unions would be no-ops and the
  // shared root would re-enter next_work with parents_[root] unchanged
  // — looping the BSP forever. Reachable any time a class is merged
  // with one of its own descendants (e.g. an asserted f(a,b) = a).
  auto t2 = timings ? clk::now() : clk::time_point{};
  auto per_group = parlay::map(groups, [&](auto& kv) -> parlay::sequence<Id> {
    auto& values = kv.second;
    if (values.size() < 2) return {};
    const Id root_ref = values[0].root;
    bool all_same_root = parlay::all_of(
        values, [root_ref](const CanonEntry& e) { return e.root == root_ref; });
    if (all_same_root) return {};
    dnc_union(values, 0, values.size(), uf);
    return parlay::map(values, [](const CanonEntry& e) { return e.root; });
  });
  auto out = parlay::flatten(per_group);
  if (timings) timings->per_group_ms = ms_since(t2);
  return out;
}

}  // namespace pe::detail
