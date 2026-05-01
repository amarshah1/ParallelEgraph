#pragma once
// Internal phase helpers from parallel_close, exposed only for the
// component microbench. Not part of the stable public API.

#include <cstdint>
#include <utility>

#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/unionfind.hpp"

namespace pe::detail {

// `hash` is the 64-bit primary signature hash, used as the group_by key.
// The trailing 4-byte slot is either `secondary_hash` (default ordered +
// dual-hash path) or `class_id` (legacy hash path, the e-class id of the
// node being canonicalized; under sparse `nodes_` storage that is also
// the index into `nodes_`, used by sigs_equal to recover the ENode).
// Selected at build time via the PE_GROUPBY_HASH compile-time switch
// (CMake option PE_GROUPBY_HASH). The struct is 16 bytes either way.
struct CanonEntry {
  std::uint64_t hash;
  Id root;
#ifdef PE_GROUPBY_HASH
  Id class_id;
#else
  std::uint32_t secondary_hash;
#endif
};

// Optional sub-phase wallclock attribution for merge_and_collect_semisort.
// Caller passes nullptr to skip the chrono::steady_clock samples entirely.
struct SemisortTimings {
  double keyed_ms;       // parlay::tabulate of keyed pairs
  double group_by_ms;    // parlay::group_by_key (hash distribute + equal probe)
  double per_group_ms;   // parlay::map + dnc_union + flatten
};

void parallel_consolidate(
    parlay::sequence<parlay::sequence<Id>>& parent_index,
    const parlay::sequence<Id>& cs,
    const parlay::sequence<Id>& roots);

parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon,
    ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes,
    SemisortTimings* timings = nullptr);

}  // namespace pe::detail
