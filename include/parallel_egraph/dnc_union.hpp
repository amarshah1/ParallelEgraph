#pragma once
// Internal shared helper for both `merge_and_collect_semisort` impls
// (semisort_secondary.hpp = default, semisort_sound.hpp = sound). Both
// go through `dnc_union` to apply the per-group unions; the bucket type
// is templated because the impls carry different per-element payloads —
// `as_root()` overloads pick out the class id regardless.
//
// Not part of the stable public API. Header-only because every helper
// here is either `inline` or a template.

#include <cstdint>
#include <cstdlib>

#include <parlay/parallel.h>

#include "parallel_egraph/detail.hpp"
#include "parallel_egraph/unionfind.hpp"

namespace pe::detail {

// PE_DNC_CUTOFF env var lets us sweep without recompiling.
inline std::size_t dnc_cutoff() {
  static const std::size_t v = [] {
    const char* s = std::getenv("PE_DNC_CUTOFF");
    return s ? static_cast<std::size_t>(std::atoll(s)) : std::size_t{16};
  }();
  return v;
}

// Bucket-element accessors so dnc_union can dispatch over CanonEntry and
// plain Id without a runtime branch.
inline Id as_root(Id x) { return x; }
inline Id as_root(const CanonEntry& e) { return e.root; }

// Returns the representative of the merged class (post-call
// find_root of any bucket member). Threaded through union_'s return
// rather than recomputed via find_root.
template <typename Bucket>
Id dnc_union(Bucket& bucket, std::size_t lo, std::size_t hi,
             ConcurrentUnionFind& uf) {
  if (hi - lo <= 1) return as_root(bucket[lo]);
  const std::size_t DNC_SEQ_CUTOFF = dnc_cutoff();
  if (hi - lo <= DNC_SEQ_CUTOFF) {
    Id surv = as_root(bucket[lo]);
    for (std::size_t i = lo + 1; i < hi; ++i) {
      surv = uf.union_(surv, as_root(bucket[i]));
    }
    return surv;
  }
  std::size_t mid = lo + (hi - lo) / 2;
  Id left_surv, right_surv;
  parlay::par_do(
      [&] { left_surv  = dnc_union(bucket, lo, mid, uf); },
      [&] { right_surv = dnc_union(bucket, mid, hi, uf); });
  return uf.union_(left_surv, right_surv);
}

// Variant taking a caller-supplied union method. Used by the MIN_ID
// async closure to substitute `union_min_id` for the default `union_`
// without duplicating dnc_union's dnc structure. `union_fn` must
// return Id (the surviving root after its merge).
template <typename Bucket, typename UnionFn>
Id dnc_union_with(Bucket& bucket, std::size_t lo, std::size_t hi,
                  ConcurrentUnionFind& uf, UnionFn union_fn) {
  if (hi - lo <= 1) return as_root(bucket[lo]);
  const std::size_t DNC_SEQ_CUTOFF = dnc_cutoff();
  if (hi - lo <= DNC_SEQ_CUTOFF) {
    Id surv = as_root(bucket[lo]);
    for (std::size_t i = lo + 1; i < hi; ++i) {
      surv = union_fn(uf, surv, as_root(bucket[i]));
    }
    return surv;
  }
  std::size_t mid = lo + (hi - lo) / 2;
  Id left_surv, right_surv;
  parlay::par_do(
      [&] { left_surv  = dnc_union_with(bucket, lo, mid, uf, union_fn); },
      [&] { right_surv = dnc_union_with(bucket, mid, hi, uf, union_fn); });
  return union_fn(uf, left_surv, right_surv);
}

}  // namespace pe::detail
