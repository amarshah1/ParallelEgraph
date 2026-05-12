#pragma once
// Shared body of `merge_and_collect_semisort`. The equality predicate is
// supplied by semisort_sound.hpp (structural sigs_equal); everything
// else — keying, group_by_key, the per-group dnc_union, and the optional
// sub-phase timing — lives here.

#include "parallel_egraph/detail.hpp"
#include "parallel_egraph/dnc_union.hpp"
#include "parallel_egraph/egraph.hpp"

#include <chrono>
#include <utility>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>
#include <parlay/internal/integer_sort.h>

namespace pe::detail {

// Crossover point between group_by_key and integer_sort+run-walk for
// the semisort step. Below this canon.size(), the hash-table path is
// faster (its fixed-cost setup amortizes better on small inputs);
// above it, the in-place radix sort wins (better scaling per element).
// Empirically measured on the closure_compare workloads at T=8 ─
// large/deep-l (frontier > 500k) gain 12-20%; deep-m (frontier ~100k)
// regresses 40%+ with integer_sort alone, so we route small frontiers
// to group_by_key. Overridable at runtime via PE_SEMISORT_INT_CUTOFF
// for sweeps without recompiling.
inline std::size_t semisort_integer_sort_cutoff() {
  static const std::size_t v = [] {
    const char* s = std::getenv("PE_SEMISORT_INT_CUTOFF");
    return s ? static_cast<std::size_t>(std::atoll(s)) : std::size_t{250'000};
  }();
  return v;
}

// group_by_key path: hash table + per-bucket sequence<CanonEntry>.
// Better fixed-cost behavior at small canon sizes.
template <typename EqualFn>
parlay::sequence<Id> semisort_group_by_key(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    SemisortTimings* timings, EqualFn equal_fn) {
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };
  auto t0 = timings ? clk::now() : clk::time_point{};
  auto keyed = parlay::map(canon, [](const CanonEntry& e) {
    return std::pair<CanonEntry, CanonEntry>{e, e};
  });
  if (timings) timings->keyed_ms = ms_since(t0);

  auto hash_fn = [](const CanonEntry& e) -> std::size_t { return e.hash; };
  auto t1 = timings ? clk::now() : clk::time_point{};
  auto groups = parlay::group_by_key(keyed, hash_fn, equal_fn);
  if (timings) timings->group_by_ms = ms_since(t1);

  auto t2 = timings ? clk::now() : clk::time_point{};
  auto per_group = parlay::map(groups, [&](auto& kv) -> parlay::sequence<Id> {
    auto& values = kv.second;
    if (values.size() < 2) return {};
    const Id root_ref = values[0].root;
    bool all_same_root = parlay::all_of(
        values, [root_ref](const CanonEntry& e) { return e.root == root_ref; });
    if (all_same_root) return {};
    const Id survivor = dnc_union(values, 0, values.size(), uf);
    return parlay::map_maybe(values, [survivor](const detail::CanonEntry& e)
                                    -> std::optional<Id> {
      if (e.root == survivor) return std::nullopt;
      return e.root;
    });
  });
  auto out = parlay::flatten(per_group);
  if (timings) timings->per_group_ms = ms_since(t2);
  return out;
}

// integer_sort + run-walk path: in-place radix sort on primary hash,
// then walk runs. Avoids the keyed-pair allocation and the per-bucket
// sequence build. Wins on large canon sizes where the radix passes
// amortize.
template <typename EqualFn>
parlay::sequence<Id> semisort_integer_sort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    SemisortTimings* timings, EqualFn equal_fn) {
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };
  auto t0 = timings ? clk::now() : clk::time_point{};
  parlay::integer_sort_inplace(
      parlay::make_slice(canon),
      [](const CanonEntry& e) -> std::uint64_t { return e.hash; });
  if (timings) timings->keyed_ms = ms_since(t0);

  auto t1 = timings ? clk::now() : clk::time_point{};
  auto starts_flag = parlay::tabulate(canon.size(), [&](std::size_t i) {
    return i == 0 || canon[i].hash != canon[i - 1].hash;
  });
  auto run_starts = parlay::pack_index<std::uint32_t>(starts_flag);
  if (timings) timings->group_by_ms = ms_since(t1);

  auto t2 = timings ? clk::now() : clk::time_point{};
  auto per_run = parlay::map(run_starts, [&, n = canon.size()](std::uint32_t lo32)
                                          -> parlay::sequence<Id> {
    const std::size_t lo = lo32;
    std::size_t hi = lo + 1;
    while (hi < n && canon[hi].hash == canon[lo].hash) ++hi;
    if (hi - lo < 2) return {};
    parlay::sequence<Id> out;
    std::size_t i = lo;
    while (i < hi) {
      std::size_t j = i + 1;
      while (j < hi && equal_fn(canon[i], canon[j])) ++j;
      if (j - i >= 2) {
        const Id ref = canon[i].root;
        bool all_same = true;
        for (std::size_t k = i + 1; k < j; ++k) {
          if (canon[k].root != ref) { all_same = false; break; }
        }
        if (!all_same) {
          const Id survivor = dnc_union(canon, i, j, uf);
          for (std::size_t k = i; k < j; ++k) {
            if (canon[k].root != survivor) out.push_back(canon[k].root);
          }
        }
      }
      i = j;
    }
    return out;
  });
  auto out = parlay::flatten(per_run);
  if (timings) timings->per_group_ms = ms_since(t2);
  return out;
}

template <typename EqualFn>
parlay::sequence<Id> semisort_with_equality(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    SemisortTimings* timings, EqualFn equal_fn) {
  if (canon.empty()) {
    if (timings) *timings = {0.0, 0.0, 0.0};
    return {};
  }
  if (canon.size() < semisort_integer_sort_cutoff()) {
    return semisort_group_by_key(std::move(canon), uf, timings, equal_fn);
  }
  return semisort_integer_sort(std::move(canon), uf, timings, equal_fn);
}

}  // namespace pe::detail
