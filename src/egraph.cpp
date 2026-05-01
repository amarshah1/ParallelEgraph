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

// ---- EGraph ctor / bulk init ---------------------------------------------

EGraph::EGraph(std::size_t capacity)
    : uf_(capacity), parent_index_(capacity) {}

// Bulk parallel construction. nodes_ is sparsely indexed by class id (the
// array index *is* the class id), so leaves occupy nodes_ slots but
// contribute nothing to parent_index_. No prefix-scan or packing — just
// the inverse from (parent → children) into (class → parents).
std::unique_ptr<EGraph> EGraph::bulk_init(parlay::sequence<ENode> nodes) {
  const std::size_t n = nodes.size();
  auto eg = std::make_unique<EGraph>(n);
  eg->uf_.bulk_init(n);

  auto child_pairs = parlay::flatten(parlay::tabulate(
      n, [&](std::size_t i) -> parlay::sequence<std::pair<Id, Id>> {
        const auto& cs = nodes[i].children;
        parlay::sequence<std::pair<Id, Id>> result;
        result.reserve(cs.size());
        Id parent = static_cast<Id>(i);
        for (Id c : cs) result.emplace_back(c, parent);
        return result;
      }));

  eg->parent_index_ = parlay::group_by_index(
      std::move(child_pairs), static_cast<Id>(n));
  eg->nodes_ = std::move(nodes);
  return eg;
}

// ---- parallel consolidation helper ---------------------------------------
//
// For each c in `cs` where c != root_of_c, migrate parent_index[c] into
// parent_index[root_of_c]. Outer parallel: distinct root r per group, so
// writes to parent_index[r] are isolated across groups. Inner parallel:
// precompute offsets via parlay::scan, resize parent_index[r] once, then
// scatter each c's entries into its pre-computed slot.

namespace detail {

void parallel_consolidate(
    parlay::sequence<parlay::sequence<Id>>& parent_index,
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
    Id r = groups[g].first;
    const auto& cs_for_r = groups[g].second;

    // Replace parent_index[r] with the concatenation of its existing
    // entries plus parent_index[c] for every non-root c grouped under r.
    // Each source is moved out (so cs_for_r's parent_index slots end up
    // empty); parlay::flatten allocates the destination buffer once and
    // does the parallel scatter internally.
    auto inputs = parlay::tabulate(
        cs_for_r.size() + 1,
        [&](std::size_t i) -> parlay::sequence<Id> {
          if (i == 0) return std::move(parent_index[r]);
          Id c = cs_for_r[i - 1];
          auto out = std::move(parent_index[c]);
          parent_index[c].clear();  // defensive: post-move state is unspecified
          return out;
        });

    parent_index[r] = parlay::flatten(std::move(inputs));
  });
}

}  // namespace detail

// `merge_and_collect_semisort` lives in src/semisort_ordered.cpp (default)
// or src/semisort_hash.cpp (kernel; PE_GROUPBY_HASH=ON). The shared dnc
// helpers — dnc_cutoff(), union_style_adjacent(), dnc_union() — are in
// include/parallel_egraph/dnc_union.hpp, included by both impls.

// ---- parallel_close ------------------------------------------------------

void EGraph::parallel_close(parlay::sequence<std::pair<Id, Id>> initial_unions) {
  auto& uf = uf_;
  auto& nodes = nodes_;
  auto& parent_index = parent_index_;
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf.union_(initial_unions[i].first, initial_unions[i].second);
  });

  auto work = parlay::flatten(parlay::map(initial_unions, [](auto p) {
    return parlay::sequence<Id>{p.first, p.second};
  }));

  std::size_t round = 0;
  while (!work.empty()) {
    work = parlay::remove_duplicates(std::move(work));

    auto t_consolidate = clk::now();
    auto roots = parlay::map(work, [&](Id c) { return uf.find_root(c); });
    detail::parallel_consolidate(parent_index, work, roots);
    double consolidate_ms = trace ? ms_since(t_consolidate) : 0.0;

    auto t_frontier = clk::now();
    auto unique_roots = parlay::remove_duplicates(roots);
    auto frontier = parlay::flatten(parlay::map(unique_roots, [&](Id r) {
      return parlay::sequence<std::uint32_t>(std::begin(parent_index[r]),
                                             std::end(parent_index[r]));
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
      const auto& node = nodes[idx];
#ifdef PE_GROUPBY_HASH
      return detail::CanonEntry{sig_hash(node, uf), uf.find_root(idx), idx};
#else
      auto [hash, secondary_hash] = sig_hashes(node, uf);
      return detail::CanonEntry{hash, uf.find_root(idx), secondary_hash};
#endif
    });

    detail::SemisortTimings semi_t{};
    auto next_work = detail::merge_and_collect_semisort(
        std::move(canon), uf, nodes, trace ? &semi_t : nullptr);
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

void EGraph::sequential_close_nelson(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  const std::size_t n_classes = uf_.len();

  SequentialUnionFind uf(n_classes);
  // Copy existing uf_ partition into the local sequential UF.
  for (std::size_t c = 0; c < n_classes; ++c) {
    Id id = static_cast<Id>(c);
    Id r = uf_.find_root(id);
    if (r != id) uf.union_(id, r);
  }

  std::vector<Id> worklist;
  worklist.reserve(initial_unions.size() * 2);
  for (auto [a, b] : initial_unions) {
    Id ra = uf.find_root(a);
    Id rb = uf.find_root(b);
    if (ra != rb) {
      uf.union_(ra, rb);
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

    parlay::sequence<std::uint32_t> frontier = std::move(parent_index_[c]);
    parent_index_[c].clear();
    if (frontier.empty()) continue;

    for (std::uint32_t pidx : frontier) {
      // class_id == pidx since nodes_ is class-id-indexed.
      const auto& node = nodes_[pidx];
      std::uint64_t h = sig_hash(node, uf);
      Id my_class = uf.find_root(static_cast<Id>(pidx));

      auto& bucket = sig_table[h];
      std::int64_t match_o = -1;
      for (std::uint32_t o : bucket) {
        if (sigs_equal(pidx, o, uf, nodes_)) {
          match_o = static_cast<std::int64_t>(o);
          break;
        }
      }
      if (match_o >= 0) {
        Id other_class = uf.find_root(static_cast<Id>(match_o));
        if (my_class != other_class) {
          uf.union_(my_class, other_class);
          worklist.push_back(my_class);
          worklist.push_back(other_class);
        }
      } else {
        bucket.push_back(pidx);
      }
    }
  }

  // Replay the final partition into self.uf_ so that equiv() reflects the
  // closure.
  for (std::size_t c = 0; c < n_classes; ++c) {
    Id id = static_cast<Id>(c);
    Id r = uf.find_root(id);
    if (r != id) uf_.union_(id, r);
  }
}

}  // namespace pe
