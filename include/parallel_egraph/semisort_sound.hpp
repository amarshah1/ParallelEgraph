#pragma once
// `merge_and_collect_semisort`: equality verified by the structural
// `sigs_equal` walk over each candidate's children (resolving every child
// id through the UF). Sound under any hash function — picks up unions
// that a probabilistic variant would miss on hash collisions.
//
// Included from `src/semisort.cpp`. CanonEntry carries `class_id` so we
// can recover the ENode via `nodes[e.class_id]` for the equality probe.
// Despite the .hpp extension, this header is included from exactly one
// translation unit.

#include "parallel_egraph/semisort_common.hpp"

namespace pe::detail {

parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes,
    SemisortTimings* timings) {
  return semisort_with_equality(std::move(canon), uf, timings,
      [&](const CanonEntry& a, const CanonEntry& b) {
        return sigs_equal(a.class_id, b.class_id, uf, nodes);
      });
}

}  // namespace pe::detail
