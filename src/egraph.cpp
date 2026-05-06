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

// `sig_hash`, `sig_hashes`, and `sigs_equal` are templated on the UF
// type and live in egraph.hpp — one definition serves both the parallel
// (ConcurrentUnionFind) and sequential (SequentialUnionFind) paths.
//
// `EGraph<UF>::EGraph` is defined inline in egraph.hpp; only the two
// close methods are out-of-line here, each as an explicit specialization
// on the UF type that algorithm requires.

// ---- parallel consolidation helper ---------------------------------------
//
// For each c in `cs` where c != root_of_c, migrate parents[c] into
// parents[root_of_c]. Outer parallel: group_by_key partitions the (root,
// non_root) pairs so each root r is written by exactly one task; the
// per-group body then concatenates parents[r] with each parents[c_i] via
// a single parlay::flatten that allocates the destination buffer once
// and scatters in parallel internally.

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

// ---- parallel_close_topo ------------------------------------------------
//
// Depth-stratified BSP. Round d processes every class at depth d in
// parallel. Because children sit strictly below their parents in depth,
// every signature read in round d sees a UF state that no other round-d
// thread is mutating; intra-round matches are resolved by the semisort
// (same primitive used by parallel_close). Total rounds = max depth.
//
// Unlike `sequential_close_topo`, this is correct on arbitrary DAGs: at
// every level the UF snapshot used to compute sigs is fixed for the
// whole batch, so no level-d node misses a congruence with another
// level-d node — they end up in the same semisort bucket.

template <>
void EGraph<ConcurrentUnionFind>::parallel_close_topo(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };

  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf_.union_(initial_unions[i].first, initial_unions[i].second);
  });

  // Depth 0 = leaves; no signatures to canonicalize. Walk d = 1..max.
  for (std::size_t d = 1; d < depth_buckets_.size(); ++d) {
    const auto& bucket = depth_buckets_[d];
    if (bucket.empty()) continue;

    auto t_round = clk::now();
    auto canon = parlay::map(bucket, [&](Id idx) {
      const auto& node = nodes_[idx];
#ifdef PE_SEMISORT_SOUND
      return detail::CanonEntry{sig_hash(node, uf_), uf_.find_root(idx), idx};
#else
      auto [hash, secondary] = sig_hashes(node, uf_);
      return detail::CanonEntry{hash, uf_.find_root(idx), secondary};
#endif
    });

    detail::apply_congruence_semisort(std::move(canon), uf_, nodes_);

    if (trace) {
      std::fprintf(stderr,
                   "[pe-topo] depth=%3zu bucket=%9zu round=%7.3fms\n",
                   d, bucket.size(), ms_since(t_round));
    }
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

// ---- sequential_close_topo ----------------------------------------------
//
// One-shot upward propagation: walk every class in reverse topological
// order (children before parents). For each node, build its canonical
// signature (op, find_root(c) for c in children) and try_emplace into a
// hashmap keyed by that signature. Repeated signature ⇒ union the current
// class with the prior holder. Because we visit children before parents,
// every child's root is already its final-for-this-pass value when its
// parent's signature is computed; no worklist, no fixed-point loop, one
// linear pass through nodes_.
//
// Caveat: this is a *single pass*. If two nodes A and B at lower indices
// were inserted with non-matching signatures and later become congruent
// because of a union deduced *after* both were visited, this pass will
// not retroactively merge them. The forward DAG walk catches all
// cascading congruences that flow upward from the initial unions, which
// is sufficient for the synthetic workloads in closure_compare_bench;
// callers needing the full Nelson fixed-point should use
// sequential_close_nelson.

namespace {

// Canonical signature key: op (pointer into the node's op string —
// nodes_ is stable for the closure's lifetime) plus the find_root'd
// children. Two signatures are equal iff op strings match and
// canonicalized child sequences match element-wise.
struct Signature {
  const std::string* op;
  std::vector<Id> children;

  bool operator==(const Signature& o) const noexcept {
    if (*op != *o.op) return false;
    if (children.size() != o.children.size()) return false;
    return std::equal(children.begin(), children.end(), o.children.begin());
  }
};

struct SignatureHash {
  // FxHasher avalanches well; tell ankerl to skip its own mixing.
  using is_avalanching = void;

  std::size_t operator()(const Signature& s) const noexcept {
    FxHasher h;
    h.write_str(*s.op);
    for (Id c : s.children) h.write_u32(c);
    return static_cast<std::size_t>(h.finish());
  }
};

}  // namespace

template <>
void EGraph<SequentialUnionFind>::sequential_close_topo(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  ankerl::unordered_dense::map<Signature, std::uint32_t, SignatureHash>
      bucket;
  bucket.reserve(nodes_.size() * 2);

  const std::size_t n = nodes_.size();
  for (std::uint32_t pidx = 0; pidx < n; ++pidx) {
    const auto& node = nodes_[pidx];

    Signature sig;
    sig.op = &node.op;
    sig.children.reserve(node.children.size());
    for (Id c : node.children) {
      sig.children.push_back(uf_.find_root(c));
    }

    auto [it, inserted] = bucket.try_emplace(std::move(sig), pidx);
    if (!inserted) {
      Id ra = uf_.find_root(pidx);
      Id rb = uf_.find_root(it->second);
      if (ra != rb) uf_.union_(ra, rb);
    }
  }
}

}  // namespace pe
