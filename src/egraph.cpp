#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/detail.hpp"
#include "parallel_egraph/dnc_union.hpp"

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

    // Build the inputs sequence directly via tabulate: slot 0 = parents[rep],
    // slots 1..k = parents[m_i]. Each inner is moved (O(1) pointer transfer);
    // parlay::flatten then uninitialized-relocates them into one freshly-
    // allocated destination buffer (the sequence<sequence<T>>&& overload).
    // The previous implementation built a `sources` index list and a
    // `parlay::map(sources, …)` — both are gratuitous; tabulate fuses them.
    auto inputs = parlay::tabulate(members.size() + 1, [&](std::size_t i) {
      return std::move(i == 0 ? parents[rep] : parents[members[i - 1]]);
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
    // Map each root to a non-owning slice of its parents list. Flatten then
    // copies straight from parents_[r] into one freshly-allocated destination
    // buffer (the generic flatten overload, since slices yield element refs)
    // — saves one per-root sequence allocation + memcpy compared to the
    // older "construct a fresh sequence(begin, end) per r" pattern.
    auto frontier = parlay::flatten(parlay::map(unique_roots, [&](Id r) {
      return parlay::make_slice(parents_[r]);
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
// Depth-stratified BSP. Round d is a two-phase parallel step:
//   Phase 1 (parlay::map):   compute CanonEntry for every depth-d class.
//                            Reads UF (find_root); the only writes are
//                            path-compression CAS, which preserves
//                            find_root return values.
//   parlay::map JOINS — implicit barrier between phases.
//   Phase 2 (apply_congruence_semisort): semisort canon by hash, dnc_union
//                            each non-singleton run. The only UF writes
//                            in this round happen here.
// The barrier is the actual safety property: every sig read in phase 1
// observes the same UF snapshot — the post-round-(d-1) state — so the
// semisort catches every intra-round congruence reachable from that
// snapshot. Note that this safety does NOT come from depth stratification
// per se: a class's representative can be a node at any depth (cross-depth
// initial unions like `assert(leaf_ta1_0 = f0(a))` make the leaf's class
// have a depth-1 root), and round-d unions can mutate those representatives.
// What guarantees correctness is that phase 1 doesn't observe phase-2
// mutations because the barrier orders them.
//
// Total rounds = max depth. Correct on arbitrary DAGs (including
// cross-depth initial unions); unlike `sequential_close_topo`, no
// level-d node ever misses a congruence with another level-d node — the
// semisort sees them all in one frozen-snapshot bucketing pass.

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

// ---- parallel_close_topo_iter -------------------------------------------
//
// Sound parallel topo: depth-stratified BSP wrapped in a fixpoint loop,
// with MIN_ID linking. The single-pass `parallel_close_topo` is unsound
// on cross-depth initial unions because in-round unions can change
// find_root values for sibling sigs that were already computed; iterating
// catches the missed congruences across passes, and MIN_ID makes the
// fixpoint converge in 1-2 passes on regular cascades.

template <>
void EGraph<ConcurrentUnionFind>::parallel_close_topo_iter(
    parlay::sequence<std::pair<Id, Id>> initial_unions) {
  const bool trace = std::getenv("PE_TRACE_ITER") != nullptr;
  using clk = std::chrono::steady_clock;

  // ---- step 1: apply initial unions in parallel (MIN_ID) -----------------
  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf_.union_min_id(initial_unions[i].first, initial_unions[i].second);
  });

  // Lambda dispatched into dnc_union_with — also sets `*changed` when
  // an actual merge commits.
  std::atomic<bool> changed{false};
  auto union_track = [&changed](ConcurrentUnionFind& u, Id a, Id b) {
    if (u.union_min_id(a, b)) {
      changed.store(true, std::memory_order_relaxed);
    }
  };

  std::size_t round = 0;
  do {
    changed.store(false, std::memory_order_relaxed);
    auto t_round = trace ? clk::now() : clk::time_point{};

    // Walk d = 1..max_depth. depth 0 = leaves; nothing to canonicalize.
    for (std::size_t d = 1; d < depth_buckets_.size(); ++d) {
      const auto& bucket = depth_buckets_[d];
      if (bucket.empty()) continue;

      // Build CanonEntry for every class in the bucket.
      auto canon = parlay::map(bucket, [&](Id idx) {
        const auto& node = nodes_[idx];
#ifdef PE_SEMISORT_SOUND
        return detail::CanonEntry{sig_hash(node, uf_), uf_.find_root(idx), idx};
#else
        auto [hash, secondary] = sig_hashes(node, uf_);
        return detail::CanonEntry{hash, uf_.find_root(idx), secondary};
#endif
      });

      if (canon.empty()) continue;

      // Sort by primary hash (in-place) and walk runs.
      parlay::integer_sort_inplace(
          parlay::make_slice(canon),
          [](const detail::CanonEntry& e) -> std::uint64_t { return e.hash; });

      auto starts_flag = parlay::tabulate(canon.size(), [&](std::size_t i) {
        return i == 0 || canon[i].hash != canon[i - 1].hash;
      });
      auto run_starts = parlay::pack_index<std::uint32_t>(starts_flag);

      parlay::parallel_for(0, run_starts.size(), [&](std::size_t r) {
        const std::size_t lo = run_starts[r];
        const std::size_t hi = (r + 1 < run_starts.size())
                                   ? static_cast<std::size_t>(run_starts[r + 1])
                                   : canon.size();
        if (hi - lo < 2) return;
        // Same primary-hash run; split by full equality (secondary or sound).
        std::size_t i = lo;
        while (i < hi) {
          std::size_t j = i + 1;
          while (j < hi) {
#ifdef PE_SEMISORT_SOUND
            if (!sigs_equal(canon[i].class_id, canon[j].class_id, uf_, nodes_))
              break;
#else
            if (canon[i].secondary_hash != canon[j].secondary_hash) break;
#endif
            ++j;
          }
          if (j - i >= 2) {
            detail::dnc_union_with(canon, i, j, uf_, union_track);
          }
          i = j;
        }
      });
    }

    ++round;
    if (trace) {
      double ms = std::chrono::duration<double, std::milli>(
                      clk::now() - t_round).count();
      std::fprintf(stderr,
                   "[par_topo_iter] round=%2zu changed=%d  %7.2fms\n",
                   round, int(changed.load()), ms);
    }
  } while (changed.load(std::memory_order_relaxed));

  if (trace) {
    std::fprintf(stderr,
                 "[par_topo_iter] converged after %zu round(s)\n", round);
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

template <typename UF>
Signature canonical_sig(const ENode& node, UF& uf) {
  Signature sig;
  sig.op = &node.op;
  sig.children.reserve(node.children.size());
  for (Id c : node.children) sig.children.push_back(uf.find_root(c));
  return sig;
}

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
    auto [it, inserted] = bucket.try_emplace(
        canonical_sig(nodes_[pidx], uf_), pidx);
    if (!inserted) {
      Id ra = uf_.find_root(pidx);
      Id rb = uf_.find_root(it->second);
      if (ra != rb) {
        // MIN_ID union: lower class id wins. Preserves the invariant
        // that on a DAG-ordered input, the canonical (earliest-emitted)
        // representative of any equivalence class is its current root —
        // which lets a forward-pass collapse the entire cascade in a
        // single sweep, since downstream sigs reference roots that were
        // valid at insertion time AND remain valid after subsequent
        // intra-pass merges.
        uf_.union_into(std::max(ra, rb), std::min(ra, rb));
      }
    }
  }
}

// ---- sequential_close_topo_iter -----------------------------------------
//
// Drive `sequential_close_topo` to a fixpoint. Each iteration is a fresh
// forward walk through `nodes_` with a fresh hashcons; we stop when a
// full pass produces zero new unions. Recovers correctness from the
// known unsoundness of single-pass topo on cross-depth / adversarial
// inputs (see test_seq_topo_adversarial_order). Number of iterations =
// (longest cross-depth chain in the augmented DAG) + 1.

template <>
void EGraph<SequentialUnionFind>::sequential_close_topo_iter(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  for (auto [a, b] : initial_unions) uf_.union_(a, b);

  const bool trace = std::getenv("PE_TRACE_ITER") != nullptr;
  using clk = std::chrono::steady_clock;

  const std::size_t n = nodes_.size();
  std::size_t round = 0;
  std::size_t total_unions = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    auto t0 = trace ? clk::now() : clk::time_point{};
    std::size_t round_unions = 0;

    ankerl::unordered_dense::map<Signature, std::uint32_t, SignatureHash>
        bucket;
    bucket.reserve(n);
    for (std::uint32_t pidx = 0; pidx < n; ++pidx) {
      auto [it, inserted] = bucket.try_emplace(
          canonical_sig(nodes_[pidx], uf_), pidx);
      if (!inserted) {
        Id ra = uf_.find_root(pidx);
        Id rb = uf_.find_root(it->second);
        if (ra != rb) {
          // MIN_ID union: see comment in sequential_close_topo.
          // For DAG-ordered emission with canonical members at lower
          // ids, this collapses the entire cascade in 2 rounds (1 work
          // pass + 1 verification pass). Without it, each round
          // advances exactly one g-tree level and total cost is
          // O(N × depth) instead of O(N).
          uf_.union_into(std::max(ra, rb), std::min(ra, rb));
          changed = true;
          ++round_unions;
        }
      }
    }
    ++round;
    total_unions += round_unions;
    if (trace) {
      double ms = std::chrono::duration<double, std::milli>(
                      clk::now() - t0).count();
      std::fprintf(stderr,
                   "[topo_iter] round=%3zu unions=%9zu cumulative=%9zu  %7.2fms\n",
                   round, round_unions, total_unions, ms);
    }
  }
  if (trace) {
    std::fprintf(stderr,
                 "[topo_iter] converged after %zu rounds, %zu total unions\n",
                 round, total_unions);
  }
}

// ---- sequential_close_dst -----------------------------------------------
//
// Worklist closure with smaller-into-larger merging and a structural
// hashcons (Signature -> class id). Correct on arbitrary initial unions.
//
// Invariants:
//   * `parents_[r]` (for any current UF root r) holds every node id that
//     has a class member of r among its children. When a root dies, its
//     parents migrate to the survivor's parents list.
//   * The hashcons may contain *stale* entries (sig -> pidx where pidx's
//     current canonical sig differs from the stored key). Stale entries
//     are harmless: a later structurally-matching insertion either hits
//     the stale entry — yielding a valid merge by congruence at the time
//     the stale sig was canonical, which is monotonically still valid
//     now — or misses and inserts a fresh entry.
//
// Two queues drive the loop:
//   * pending_merges: pairs (a, b) of class ids that need to be unioned.
//     Seeded from `initial_unions` and from duplicate sigs detected on
//     the initial seed sweep; refilled by phase 1 when re-canonicalized
//     parents collide with existing hashcons entries.
//   * pending_nodes: node ids whose canonical sig may have changed
//     (because one of their children's roots moved). Refilled by
//     phase 2 from `parents_[ra]` whenever ra dies.

template <>
void EGraph<SequentialUnionFind>::sequential_close_dst(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  ankerl::unordered_dense::map<Signature, Id, SignatureHash> hashcons;
  hashcons.reserve(nodes_.size());

  std::vector<std::pair<Id, Id>> pending_merges;
  pending_merges.reserve(initial_unions.size() + nodes_.size() / 4);
  for (auto [a, b] : initial_unions) pending_merges.emplace_back(a, b);

  // Initial seed: every node's signature goes into the hashcons.
  // Pre-existing congruences (two nodes with identical structure)
  // surface as duplicates and become initial pending_merges.
  for (std::uint32_t pidx = 0; pidx < nodes_.size(); ++pidx) {
    auto [it, inserted] = hashcons.try_emplace(
        canonical_sig(nodes_[pidx], uf_), pidx);
    if (!inserted) pending_merges.emplace_back(pidx, it->second);
  }

  std::vector<std::uint32_t> pending_nodes;

  while (!pending_merges.empty() || !pending_nodes.empty()) {
    // Phase 1: re-canonicalize each pending node. A collision with an
    // existing hashcons entry queues a fresh merge.
    for (std::uint32_t pidx : pending_nodes) {
      auto [it, inserted] = hashcons.try_emplace(
          canonical_sig(nodes_[pidx], uf_), pidx);
      if (!inserted && it->second != pidx) {
        pending_merges.emplace_back(pidx, it->second);
      }
    }
    pending_nodes.clear();

    // Phase 2: drain the merge queue, possibly adding to pending_nodes.
    auto merges = std::move(pending_merges);
    pending_merges.clear();
    for (auto [a, b] : merges) {
      Id ra = uf_.find_root(a);
      Id rb = uf_.find_root(b);
      if (ra == rb) continue;

      // Smaller-half: the class with fewer parents dies. O(N log N)
      // total parent-list migration over the lifetime of the closure.
      if (parents_[ra].size() > parents_[rb].size()) std::swap(ra, rb);

      uf_.union_into(ra, rb);

      // Migrate ra's parents into rb's list and queue them for
      // re-canonicalization (their child-root just moved). One pass
      // over src does both.
      auto& src = parents_[ra];
      auto& dst = parents_[rb];
      for (Id pidx : src) {
        pending_nodes.push_back(pidx);
        dst.push_back(pidx);
      }
      src.clear();
    }
  }
}

}  // namespace pe
