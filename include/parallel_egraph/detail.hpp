#pragma once
// Internal phase helpers from parallel_close, exposed only for the
// component microbench. Not part of the stable public API.

#include <cstdint>
#include <utility>

#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/unionfind.hpp"

namespace pe::detail {

struct CanonEntry {
  std::uint64_t h;
  std::uint32_t idx;
  Id root;
};

void parallel_consolidate(
    parlay::sequence<parlay::sequence<Id>>& parent_index,
    const parlay::sequence<Id>& cs,
    const parlay::sequence<Id>& roots);

parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon,
    ConcurrentUnionFind& uf,
    const parlay::sequence<std::pair<ENode, Id>>& nodes);

}  // namespace pe::detail
