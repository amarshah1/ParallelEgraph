#pragma once
// Default `merge_and_collect_semisort` implementation: parlay's
// integer-sort-based `group_by_key_ordered` keyed on the 64-bit primary
// hash, with a 32-bit secondary hash carried per-element. Within each
// hash-equal group, a fast path checks that all secondary hashes match
// (one 32-bit compare per element) — combined 96-bit entropy makes
// collisions effectively impossible (≈10⁻¹⁴ across a 64M-element batch).
// A defensive slow path buckets-by-h2 for the (statistically unreachable)
// adversarial case. Both paths flow through `dnc_union` to apply the
// per-group merges in parallel.
//
// Despite the .hpp extension, this header is included from exactly one
// translation unit (`src/semisort.cpp`) — never from anywhere else. The
// dispatcher there `#include`s this file or `semisort_hash.hpp` based
// on the `PE_GROUPBY_HASH` macro.

#include "parallel_egraph/detail.hpp"
#include "parallel_egraph/dnc_union.hpp"
#include "parallel_egraph/egraph.hpp"

#include <chrono>
#include <cstdint>
#include <utility>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>

namespace pe::detail {

namespace {

// Per-element payload during integer-sort grouping: root for the eventual
// union, h2 for the in-group "all equal" check.
struct RootH2 {
  Id root;
  std::uint32_t secondary_hash;
};
static_assert(sizeof(RootH2) == 8);

// `as_root` overload paired with the type in the same anonymous namespace
// so dnc_union<>'s ADL lookup finds it at instantiation time.
inline Id as_root(const RootH2& v) { return v.root; }

}  // namespace

parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    const parlay::sequence<ENode>& nodes,
    SemisortTimings* timings) {
  (void)nodes;  // sigs_equal not invoked from this path

  const std::size_t n = canon.size();
  if (n == 0) {
    if (timings) *timings = {0.0, 0.0, 0.0};
    return {};
  }

  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  auto t0 = timings ? clk::now() : clk::time_point{};
  auto keyed = parlay::delayed_tabulate(n, [&](std::size_t i) {
    return std::pair<std::uint64_t, RootH2>{
        canon[i].hash, RootH2{canon[i].root, canon[i].secondary_hash}};
  });
  if (timings) timings->keyed_ms = ms_since(t0);

  auto t1 = timings ? clk::now() : clk::time_point{};
  auto groups = parlay::group_by_key_ordered(keyed);
  if (timings) timings->group_by_ms = ms_since(t1);

  auto t2 = timings ? clk::now() : clk::time_point{};
  auto per_group = parlay::map(groups, [&](auto& kv) -> parlay::sequence<Id> {
    auto& values = kv.second;
    const std::size_t k = values.size();
    if (k < 2) return {};

    // Fast path: every element shares secondary_hash with the first → same signature.
    bool same = true;
    const std::uint32_t secondary_hash_ref = values[0].secondary_hash;
    for (std::size_t i = 1; i < k; ++i) {
      if (values[i].secondary_hash != secondary_hash_ref) { same = false; break; }
    }
    if (same) {
      // Skip groups whose members all already share the same root — the
      // unions would be no-ops and the touched ids would re-enter
      // next_work, looping the BSP forever.
      const Id root_ref = values[0].root;
      bool all_same_root = true;
      for (std::size_t i = 1; i < k; ++i) {
        if (values[i].root != root_ref) { all_same_root = false; break; }
      }
      if (all_same_root) return {};
      dnc_union(values, 0, k, uf);
      return parlay::tabulate(k,
                              [&](std::size_t i) { return values[i].root; });
    }

    // Slow path: bucket by secondary_hash within this primary-hash-equal
    // group. With 96-bit combined entropy this should never fire on real
    // inputs.
    parlay::sequence<bool> consumed(k, false);
    parlay::sequence<Id> touched;
    for (std::size_t i = 0; i < k; ++i) {
      if (consumed[i]) continue;
      consumed[i] = true;
      parlay::sequence<RootH2> bucket;
      bucket.push_back(values[i]);
      for (std::size_t j = i + 1; j < k; ++j) {
        if (!consumed[j] && values[j].secondary_hash == values[i].secondary_hash) {
          consumed[j] = true;
          bucket.push_back(values[j]);
        }
      }
      if (bucket.size() < 2) continue;
      dnc_union(bucket, 0, bucket.size(), uf);
      for (auto& v : bucket) touched.push_back(v.root);
    }
    return touched;
  });
  auto out = parlay::flatten(per_group);
  if (timings) timings->per_group_ms = ms_since(t2);
  return out;
}

}  // namespace pe::detail
