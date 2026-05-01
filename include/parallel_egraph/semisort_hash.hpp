#pragma once
// Kernel `merge_and_collect_semisort` implementation: parlay's
// open-addressing `group_by_key` with sig_hash as the hash and sigs_equal
// as the equality predicate. Hash is recomputed from the ENode on every
// call — no caching, no h2 fast-path, no in-bucket short-circuits. The
// reference / fallback implementation; deterministically correct under
// any hash function and slower than the integer-sort default (~1.2× on
// `large` at T=192) because each `hash_fn` and `equal_fn` invocation
// re-walks the children with UF lookups.
//
// Selected at build time when the dispatcher (`src/semisort.cpp`) sees
// `PE_GROUPBY_HASH` defined. Despite the .hpp extension, this header is
// included from exactly one translation unit — never from anywhere else.
//
// CanonEntry's trailing 4-byte slot stores a class id (`class_id`)
// instead of a secondary hash (selected by the `PE_GROUPBY_HASH` define
// in detail.hpp); the kernel uses it only to recover the ENode (via
// `nodes[e.class_id]`) for the hash and equality probes — the cached
// `e.hash` field is intentionally ignored for clarity.

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
    return sig_hash(nodes[e.class_id], uf);
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
