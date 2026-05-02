#pragma once
// Sound `merge_and_collect_semisort`: open-addressing `parlay::group_by_key`
// keyed on the cached 64-bit primary hash, with equality verified by the
// structural `sigs_equal` walk over each candidate's children (resolving
// every child id through the UF). Slower than the secondary-hash default
// but correct under any hash function — picks up unions that the
// probabilistic variant would miss on hash collisions.
//
// Selected at build time by `src/semisort.cpp` when `PE_SEMISORT_SOUND`
// is defined. Despite the .hpp extension, this header is included from
// exactly one translation unit.
//
// CanonEntry's trailing 4-byte slot is `class_id` here (selected by the
// PE_SEMISORT_SOUND #ifdef in detail.hpp); we use it only to recover the
// ENode via `nodes[e.class_id]` for the equality probe.

#include "parallel_egraph/detail.hpp"
#include "parallel_egraph/dnc_union.hpp"
#include "parallel_egraph/egraph.hpp"

#include <chrono>
#include <utility>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>

namespace pe::detail {

parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes,
    SemisortTimings* timings) {
  const std::size_t n = canon.size();
  if (n == 0) {
    if (timings) *timings = {0.0, 0.0, 0.0};
    return {};
  }

  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  // Pair each CanonEntry with itself: simplest possible input shape for
  // parlay::group_by_key. Both halves of the pair are just `e`.
  auto t0 = timings ? clk::now() : clk::time_point{};
  auto keyed = parlay::map(canon, [](const CanonEntry& e) {
    return std::pair<CanonEntry, CanonEntry>{e, e};
  });
  if (timings) timings->keyed_ms = ms_since(t0);

  // Recompute the signature hash from the node every call (the cached
  // `e.hash` is ignored). Equality is the full sigs_equal — no fast path,
  // no h2 short-circuit.
  auto hash_fn = [&](const CanonEntry& e) -> std::size_t {
    return e.hash;
  };
  auto equal_fn = [&](const CanonEntry& a, const CanonEntry& b) -> bool {
    return sigs_equal(a.class_id, b.class_id, uf, nodes);
  };
  auto t1 = timings ? clk::now() : clk::time_point{};
  auto groups = parlay::group_by_key(keyed, hash_fn, equal_fn);
  if (timings) timings->group_by_ms = ms_since(t1);

  // For each non-singleton group, union all members and emit their
  // pre-round roots into next_work.
  auto t2 = timings ? clk::now() : clk::time_point{};
  auto per_group = parlay::map(groups, [&](auto& kv) -> parlay::sequence<Id> {
    auto& values = kv.second;
    if (values.size() < 2) return {};
    dnc_union(values, 0, values.size(), uf);
    return parlay::map(values, [](const CanonEntry& e) { return e.root; });
  });
  auto out = parlay::flatten(per_group);
  if (timings) timings->per_group_ms = ms_since(t2);
  return out;
}

}  // namespace pe::detail
