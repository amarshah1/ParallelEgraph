#pragma once
// Default `merge_and_collect_semisort`: equality verified by a 32-bit
// secondary hash. Combined 96-bit (h1+h2) entropy makes collisions
// effectively impossible (≈10⁻¹⁴ across a 64M-element batch); we accept
// that probabilistic guarantee in exchange for a near-free equality
// probe (one 32-bit compare per candidate match) versus the structural
// `sigs_equal` walk used by the sound variant.
//
// Selected at build time by `src/semisort.cpp` when `PE_SEMISORT_SOUND`
// is *not* defined. Despite the .hpp extension, this header is included
// from exactly one translation unit.

#include "parallel_egraph/semisort_common.hpp"

namespace pe::detail {

parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    [[maybe_unused]] const parlay::sequence<ENode>& nodes,
    SemisortTimings* timings) {
  return semisort_with_equality(std::move(canon), uf, timings,
      [](const CanonEntry& a, const CanonEntry& b) {
        return a.hash == b.hash && a.secondary_hash == b.secondary_hash;
      });
}

void apply_congruence_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    [[maybe_unused]] const parlay::sequence<ENode>& nodes) {
  apply_unions_semisort(std::move(canon), uf,
      [](const CanonEntry& a, const CanonEntry& b) {
        return a.hash == b.hash && a.secondary_hash == b.secondary_hash;
      });
}

}  // namespace pe::detail
