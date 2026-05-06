#pragma once
// Sound `merge_and_collect_semisort`: equality verified by the structural
// `sigs_equal` walk over each candidate's children (resolving every child
// id through the UF). Slower than the secondary-hash default but correct
// under any hash function — picks up unions that the probabilistic
// variant would miss on hash collisions.
//
// Selected at build time by `src/semisort.cpp` when `PE_SEMISORT_SOUND`
// is defined. CanonEntry's trailing 4-byte slot is `class_id` here
// (selected by the PE_SEMISORT_SOUND #ifdef in detail.hpp); we use it
// only to recover the ENode via `nodes[e.class_id]` for the equality
// probe. Despite the .hpp extension, this header is included from
// exactly one translation unit.

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

void apply_congruence_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes) {
  apply_unions_semisort(std::move(canon), uf,
      [&](const CanonEntry& a, const CanonEntry& b) {
        return sigs_equal(a.class_id, b.class_id, uf, nodes);
      });
}

}  // namespace pe::detail
