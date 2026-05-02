#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/detail.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>

namespace pe {

// `sig_hash`, `sig_hashes`, and `sigs_equal` are templated on the UF type
// and live in egraph.hpp — one definition serves both the parallel
// (ConcurrentUnionFind) and sequential (SequentialUnionFind) paths. The
// previous `_seq` variants and the dead `sig_cmp` (used by the removed
// sample-sort variant) are gone.
//
// `EGraph<UF>::EGraph` is defined inline in egraph.hpp; only the two
// close methods are out-of-line here, each as an explicit specialization
// on the UF type that algorithm requires.

// ---- parallel consolidation helper ---------------------------------------
//
// For each c in `cs` where c != root_of_c, migrate parents[c] into
// parents[root_of_c]. Outer parallel: distinct root r per group, so
// writes to parents[r] are isolated across groups. Inner parallel:
// precompute offsets via parlay::scan, resize parents[r] once, then
// scatter each c's entries into its pre-computed slot.

namespace detail {

void parallel_consolidate(
    parlay::sequence<parlay::sequence<Id>>& parents,
    const parlay::sequence<Id>& cs,
    const parlay::sequence<Id>& roots) {
  auto non_root_pairs = parlay::map_maybe(
      parlay::iota(cs.size()),
      [&](std::size_t i) -> std::optional<std::pair<Id, Id>> {
        if (cs[i] == roots[i]) return std::nullopt;
        return std::make_pair(roots[i], cs[i]);
      });
  if (non_root_pairs.empty()) return;

  auto groups = parlay::group_by_key(std::move(non_root_pairs));

  parlay::parallel_for(0, groups.size(), [&](std::size_t g) {
    Id rep = groups[g].first;
    const auto& members = groups[g].second;

    // Replace parents[r] with the concatenation of parents[r] (its own
    // pre-existing parents) and parents[c] for every non-root c grouped
    // under r. Each source is moved out; parlay::flatten allocates the
    // destination buffer once and does the parallel scatter internally.
    parlay::sequence<Id> sources(members);
    sources.push_back(rep);
    auto inputs = parlay::map(sources, [&](Id c) {
      return std::move(parents[c]);
    });
    parents[rep] = parlay::flatten(std::move(inputs));
  });
}

}  // namespace detail

// `merge_and_collect_semisort` lives in semisort_secondary.hpp (default)
// or semisort_sound.hpp (PE_SEMISORT_SOUND=ON). The shared dnc helpers
// — dnc_cutoff() and dnc_union() — are in dnc_union.hpp, included by
// both impls.

// ---- parallel_close ------------------------------------------------------

template <>
void EGraph<ConcurrentUnionFind>::parallel_close(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf_.union_(initial_unions[i].first, initial_unions[i].second);
  });

  auto work = parlay::flatten(parlay::map(initial_unions, [](auto p) {
    return parlay::sequence<Id>{p.first, p.second};
  }));

  std::size_t round = 0;
  while (!work.empty()) {
    work = parlay::remove_duplicates(std::move(work));

    auto t_consolidate = clk::now();
    auto roots = parlay::map(work, [&](Id c) { return uf_.find_root(c); });
    detail::parallel_consolidate(parents_, work, roots);
    double consolidate_ms = trace ? ms_since(t_consolidate) : 0.0;

    auto t_frontier = clk::now();
    auto unique_roots = parlay::remove_duplicates(roots);
    auto frontier = parlay::flatten(parlay::map(unique_roots, [&](Id r) {
      return parlay::sequence<std::uint32_t>(std::begin(parents_[r]),
                                             std::end(parents_[r]));
    }));
    double frontier_ms = trace ? ms_since(t_frontier) : 0.0;

    if (frontier.empty()) {
      if (trace) {
        std::fprintf(stderr,
                     "[pe] round=%3zu work=%9zu frontier=        0 next=        0 "
                     "consolidate=%7.3fms frontier=%7.3fms semisort=%7.3fms (break)\n",
                     round, work.size(), consolidate_ms, frontier_ms, 0.0);
      }
      break;
    }

    auto t_semisort = clk::now();
    auto canon = parlay::map(frontier, [&](std::uint32_t idx) {
      // `idx` is a frontier element — under sparse `nodes_` storage it
      // is the class id of a parent node and the index into `nodes_`.
      const auto& node = nodes_[idx];
#ifdef PE_SEMISORT_SOUND
      return detail::CanonEntry{sig_hash(node, uf_), uf_.find_root(idx), idx};
#else
      auto [hash, secondary_hash] = sig_hashes(node, uf_);
      return detail::CanonEntry{hash, uf_.find_root(idx), secondary_hash};
#endif
    });

    detail::SemisortTimings semi_t{};
    auto next_work = detail::merge_and_collect_semisort(
        std::move(canon), uf_, nodes_, trace ? &semi_t : nullptr);
    next_work = parlay::remove_duplicates(std::move(next_work));
    double semisort_ms = trace ? ms_since(t_semisort) : 0.0;

    if (trace) {
      std::fprintf(stderr,
                   "[pe] round=%3zu work=%9zu frontier=%9zu next=%9zu "
                   "consolidate=%7.3fms frontier=%7.3fms semisort=%7.3fms "
                   "(keyed=%6.3fms group_by=%7.3fms per_group=%6.3fms)\n",
                   round, work.size(), frontier.size(), next_work.size(),
                   consolidate_ms, frontier_ms, semisort_ms,
                   semi_t.keyed_ms, semi_t.group_by_ms, semi_t.per_group_ms);
    }

    work = std::move(next_work);
    ++round;
  }
}

// ---- sequential_close_nelson --------------------------------------------

template <>
void EGraph<SequentialUnionFind>::sequential_close_nelson(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  const std::size_t n_classes = uf_.len();

  std::vector<Id> worklist;
  worklist.reserve(initial_unions.size() * 2);
  for (auto [a, b] : initial_unions) {
    Id ra = uf_.find_root(a);
    Id rb = uf_.find_root(b);
    if (ra != rb) {
      uf_.union_(ra, rb);
      worklist.push_back(ra);
      worklist.push_back(rb);
    }
  }

  // Flat hashmap (robin-hood) replaces std::unordered_map's chained
  // buckets. ~3x faster on this insert-heavy workload. Reserve up-front
  // to skip rehash spikes — sig_table holds at most one entry per visited
  // parent class; n_classes is a safe over-estimate of the steady state.
  ankerl::unordered_dense::map<std::uint64_t, std::vector<std::uint32_t>> sig_table;
  sig_table.reserve(n_classes / 4);

  while (!worklist.empty()) {
    Id c = worklist.back();
    worklist.pop_back();

    parlay::sequence<std::uint32_t> frontier = std::move(parents_[c]);
    parents_[c].clear();
    if (frontier.empty()) continue;

    for (std::uint32_t pidx : frontier) {
      // class_id == pidx since nodes_ is class-id-indexed.
      const auto& node = nodes_[pidx];
      std::uint64_t h = sig_hash(node, uf_);
      Id my_class = uf_.find_root(static_cast<Id>(pidx));

      auto& bucket = sig_table[h];
      std::int64_t match_o = -1;
      for (std::uint32_t o : bucket) {
        if (sigs_equal(pidx, o, uf_, nodes_)) {
          match_o = static_cast<std::int64_t>(o);
          break;
        }
      }
      if (match_o >= 0) {
        Id other_class = uf_.find_root(static_cast<Id>(match_o));
        if (my_class != other_class) {
          uf_.union_(my_class, other_class);
          worklist.push_back(my_class);
          worklist.push_back(other_class);
        }
      } else {
        bucket.push_back(pidx);
      }
    }
  }
}

}  // namespace pe
