#pragma once
// Internal phase helpers from parallel_parents, exposed only for the
// component microbench. Not part of the stable public API.

#include <cstdint>
#include <utility>

#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/unionfind.hpp"

namespace pe::detail {

// `hash` is the 64-bit primary signature hash, used as the group_by key.
// `class_id` is the e-class id (also the index into `nodes_`), used by
// sigs_equal to recover the ENode for a structural compare. 16 bytes.
struct CanonEntry {
  std::uint64_t hash;
  Id root;
  Id class_id;
};

// Optional sub-phase wallclock attribution for merge_and_collect_semisort.
// Caller passes nullptr to skip the chrono::steady_clock samples entirely.
struct SemisortTimings {
  double keyed_ms;       // parlay::tabulate of keyed pairs
  double group_by_ms;    // parlay::group_by_key (hash distribute + equal probe)
  double per_group_ms;   // parlay::map + dnc_union + flatten
};

void parallel_consolidate(
    parlay::sequence<parlay::sequence<Id>>& parents,
    const parlay::sequence<Id>& cs,
    const parlay::sequence<Id>& roots);

parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon,
    ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes,
    SemisortTimings* timings = nullptr);

// Lean variant: semisort by signature, dnc_union each non-singleton
// bucket, return nothing. Used by `parallel_topo`, which has no
// next-round frontier to build.
void apply_congruence_semisort(
    parlay::sequence<CanonEntry> canon,
    ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes);

}  // namespace pe::detail
