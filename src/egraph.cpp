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

// ---- ENode hashing --------------------------------------------------------

std::size_t ENodeHash::operator()(const ENode& n) const noexcept {
  FxHasher h;
  h.write_str(n.op);
  for (Id c : n.children) h.write_u32(c);
  return static_cast<std::size_t>(h.finish());
}

// ---- Signature helpers ----------------------------------------------------

std::uint64_t sig_hash(const ENode& node, ConcurrentUnionFind& uf) {
  FxHasher h;
  h.write_str(node.op);
  for (Id c : node.children) h.write_u32(uf.find_root(c));
  return h.finish();
}

// Constant for h2's seed: golden-ratio-derived; same constant rustc uses
// for its `RandomState` initialisation seed. Picked because it has good
// bit distribution and is structurally unrelated to FxHasher's mixing
// constant (0x517C…), keeping h1 and h2 effectively independent.
namespace { constexpr std::uint64_t kSigHash2Seed = 0x9E37'79B9'7F4A'7C15ULL; }

std::pair<std::uint64_t, std::uint32_t> sig_hashes(
    const ENode& node, ConcurrentUnionFind& uf) {
  FxHasher h1;
  FxHasher h2(kSigHash2Seed);
  h1.write_str(node.op);
  h2.write_str(node.op);
  for (Id c : node.children) {
    Id r = uf.find_root(c);
    h1.write_u32(r);
    h2.write_u32(r);
  }
  return {h1.finish(), static_cast<std::uint32_t>(h2.finish())};
}

std::uint64_t sig_hash_seq(const ENode& node, SequentialUnionFind& uf) {
  FxHasher h;
  h.write_str(node.op);
  for (Id c : node.children) h.write_u32(uf.find_root(c));
  return h.finish();
}

bool sigs_equal(std::uint32_t ia, std::uint32_t ib, ConcurrentUnionFind& uf,
                const parlay::sequence<std::pair<ENode, Id>>& nodes) {
  const auto& na = nodes[ia].first;
  const auto& nb = nodes[ib].first;
  if (na.op != nb.op) return false;
  if (na.children.size() != nb.children.size()) return false;
  for (std::size_t i = 0; i < na.children.size(); ++i) {
    if (uf.find_root(na.children[i]) != uf.find_root(nb.children[i])) {
      return false;
    }
  }
  return true;
}

bool sigs_equal_seq(std::uint32_t ia, std::uint32_t ib, SequentialUnionFind& uf,
                    const parlay::sequence<std::pair<ENode, Id>>& nodes) {
  const auto& na = nodes[ia].first;
  const auto& nb = nodes[ib].first;
  if (na.op != nb.op) return false;
  if (na.children.size() != nb.children.size()) return false;
  for (std::size_t i = 0; i < na.children.size(); ++i) {
    if (uf.find_root(na.children[i]) != uf.find_root(nb.children[i])) {
      return false;
    }
  }
  return true;
}

int sig_cmp(std::uint32_t ia, std::uint32_t ib, ConcurrentUnionFind& uf,
            const parlay::sequence<std::pair<ENode, Id>>& nodes) {
  const auto& na = nodes[ia].first;
  const auto& nb = nodes[ib].first;
  int c = na.op.compare(nb.op);
  if (c != 0) return c < 0 ? -1 : 1;
  if (na.children.size() != nb.children.size()) {
    return na.children.size() < nb.children.size() ? -1 : 1;
  }
  for (std::size_t i = 0; i < na.children.size(); ++i) {
    Id ra = uf.find_root(na.children[i]);
    Id rb = uf.find_root(nb.children[i]);
    if (ra != rb) return ra < rb ? -1 : 1;
  }
  return 0;
}

// ---- EGraph ctor / build helpers -----------------------------------------

EGraph::EGraph(std::size_t capacity)
    : uf_(capacity), parent_index_(capacity) {}

ENode EGraph::canonicalize(const ENode& node) {
  ENode out;
  out.op = node.op;
  out.children.reserve(node.children.size());
  for (Id c : node.children) out.children.push_back(find(c));
  return out;
}

// Bulk parallel construction. See egraph.hpp for semantics.
std::unique_ptr<EGraph> EGraph::bulk_init(parlay::sequence<ENode> nodes) {
  const std::size_t n = nodes.size();
  auto eg = std::make_unique<EGraph>(n);

  // 1. UF: bump size_ to n in one shot (slots already at make_rank(0)).
  eg->uf_.bulk_init(n);

  // 2. node_idx prefix-scan: only non-leaves go into nodes_, so each
  //    non-leaf at input position i gets node_idx = (# non-leaves before i).
  auto is_nonleaf = parlay::tabulate(n, [&](std::size_t i) {
    return nodes[i].children.empty() ? std::size_t{0} : std::size_t{1};
  });
  auto scan_pair = parlay::scan(is_nonleaf);
  auto& offsets = scan_pair.first;
  std::size_t total_nonleaves = scan_pair.second;

  // 3. Build (child_class, node_idx) pairs in parallel before we move out
  //    of `nodes`. We tabulate per-node-index sequences and flatten.
  auto child_pairs = parlay::flatten(parlay::tabulate(
      n, [&](std::size_t i) -> parlay::sequence<std::pair<Id, std::uint32_t>> {
        const auto& cs = nodes[i].children;
        if (cs.empty()) return {};
        std::uint32_t j = static_cast<std::uint32_t>(offsets[i]);
        return parlay::tabulate(cs.size(), [&, j](std::size_t k) {
          return std::pair<Id, std::uint32_t>{cs[k], j};
        });
      }));

  // 4. group_by_index by child class → parent_index_[c] = node_idx list.
  eg->parent_index_ = parlay::group_by_index(
      std::move(child_pairs), static_cast<Id>(n));

  // 5. Pack non-leaves into eg->nodes_ in parallel (consumes `nodes`).
  //    Each non-leaf has a unique offsets[i], so the parallel writes never
  //    collide.
  eg->nodes_ = parlay::sequence<std::pair<ENode, Id>>(total_nonleaves);
  parlay::parallel_for(0, n, [&](std::size_t i) {
    if (!nodes[i].children.empty()) {
      eg->nodes_[offsets[i]] = std::pair<ENode, Id>{
          std::move(nodes[i]), static_cast<Id>(i)};
    }
  });

  // hashcons_ left empty: bulk_init skips dedup by contract.
  return eg;
}

Id EGraph::add(ENode node) {
  ENode canon = canonicalize(node);

  auto it = hashcons_.find(canon);
  if (it != hashcons_.end()) return find(it->second);

  Id id = make_id();

  if (!canon.children.empty()) {
    std::uint32_t node_idx = static_cast<std::uint32_t>(nodes_.size());
    nodes_.emplace_back(canon, id);
    for (Id child : canon.children) {
      Id child_root = find(child);
      parent_index_[child_root].push_back(node_idx);
    }
  }

  hashcons_.emplace(std::move(canon), id);
  return id;
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

    auto sizes = parlay::tabulate(cs_for_r.size(), [&](std::size_t i) {
      return parent_index[cs_for_r[i]].size();
    });
    auto scan_result = parlay::scan(sizes);
    auto& offsets = scan_result.first;
    std::size_t total_new = scan_result.second;

    auto& dst = parent_index[r];
    std::size_t old_size = dst.size();
    dst.resize(old_size + total_new);

    constexpr std::size_t COPY_SEQ_CUTOFF = 1024;
    parlay::parallel_for(0, cs_for_r.size(), [&](std::size_t i) {
      Id c = cs_for_r[i];
      auto& src = parent_index[c];
      std::size_t ofs = old_size + offsets[i];
      if (src.size() < COPY_SEQ_CUTOFF) {
        for (std::size_t j = 0; j < src.size(); ++j) dst[ofs + j] = src[j];
      } else {
        parlay::parallel_for(0, src.size(),
                             [&](std::size_t j) { dst[ofs + j] = src[j]; });
      }
      src.clear();
    });
  });
}

}  // namespace detail

namespace {

// PE_DNC_CUTOFF env var lets us sweep without recompiling.
inline std::size_t dnc_cutoff() {
  static const std::size_t v = [] {
    const char* s = std::getenv("PE_DNC_CUTOFF");
    return s ? static_cast<std::size_t>(std::atoll(s)) : std::size_t{16};
  }();
  return v;
}

// PE_UNION_STYLE=adjacent uses a flat parlay::parallel_for over adjacent
// pairs instead of the D&C tree. Higher CAS contention (each pair shares
// an endpoint with its neighbour) but trivially simple. Empirically wins
// on `large` (391ms vs 434ms) because group sizes are small enough that
// contention is bounded.
inline bool union_style_adjacent() {
  static const bool v = [] {
    const char* s = std::getenv("PE_UNION_STYLE");
    return s && std::string(s) == "adjacent";
  }();
  return v;
}


// dnc_union accepts buckets of CanonEntry, raw Id, or RootH2 pairs; only
// the root is consumed for the union. RootH2 carries h2 alongside root so
// the per-group reducer in merge_and_collect_semisort can verify the group
// is collision-free with a single 32-bit compare per element instead of
// invoking sigs_equal (which does N UF lookups per call).
struct RootH2 {
  Id root;
  std::uint32_t h2;
};
static_assert(sizeof(RootH2) == 8);

inline Id as_root(Id x) { return x; }
inline Id as_root(const detail::CanonEntry& e) { return e.root; }
inline Id as_root(const RootH2& v) { return v.root; }

template <typename Bucket>
void dnc_union(Bucket& bucket, std::size_t lo, std::size_t hi,
               ConcurrentUnionFind& uf) {
  if (hi - lo <= 1) return;
  if (union_style_adjacent()) {
    parlay::parallel_for(lo, hi - 1, [&](std::size_t i) {
      uf.union_(as_root(bucket[i]), as_root(bucket[i + 1]));
    });
    return;
  }
  const std::size_t DNC_SEQ_CUTOFF = dnc_cutoff();
  if (hi - lo <= DNC_SEQ_CUTOFF) {
    for (std::size_t i = lo + 1; i < hi; ++i) {
      uf.union_(as_root(bucket[lo]), as_root(bucket[i]));
    }
    return;
  }
  std::size_t mid = lo + (hi - lo) / 2;
  parlay::par_do([&] { dnc_union(bucket, lo, mid, uf); },
                 [&] { dnc_union(bucket, mid, hi, uf); });
  uf.union_(as_root(bucket[lo]), as_root(bucket[mid]));
}

}  // namespace

namespace detail {

// Given a canon sequence, merge classes that share both hash and signature,
// applying unions inline as the per-bucket scan proceeds, and return the
// set of touched class ids (pre-round roots of any union issued).
//
// Uses parlay::group_by_key with custom hash (= sig_hash) and equal
// (= sigs_equal). Hash distributes; the equal predicate filters 64-bit
// collisions, so groups are exact and no in-bucket sort is needed.
parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    const parlay::sequence<std::pair<ENode, Id>>& nodes,
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

  // Two implementations selected at build time. Default (ordered) uses
  // parlay::group_by_key_ordered on the 64-bit primary hash with a 32-bit
  // secondary hash carried in the value for collision-safe per-group
  // equality (combined 96-bit entropy → ~10⁻¹⁴ collision probability per
  // 64M-element batch). Falls back to a slow path for adversarial inputs.
  //
  // PE_GROUPBY_HASH selects the legacy hash + open-addressing variant
  // with sigs_equal as the equality predicate — deterministically correct
  // under any hash function but ~1.2× slower because each equal_fn call
  // does N UF lookups.
  parlay::sequence<parlay::sequence<Id>> per_group;

#ifndef PE_GROUPBY_HASH
  auto t0 = timings ? clk::now() : clk::time_point{};
  auto keyed = parlay::delayed_tabulate(n, [&](std::size_t i) {
    return std::pair<std::uint64_t, RootH2>{
        canon[i].h, RootH2{canon[i].root, canon[i].h2}};
  });
  if (timings) timings->keyed_ms = ms_since(t0);

  auto t1 = timings ? clk::now() : clk::time_point{};
  auto groups = parlay::group_by_key_ordered(keyed);
  if (timings) timings->group_by_ms = ms_since(t1);

  (void)nodes;  // sigs_equal not invoked from this path
  auto t2 = timings ? clk::now() : clk::time_point{};
  per_group = parlay::map(groups, [&](auto& kv) -> parlay::sequence<Id> {
    auto& values = kv.second;
    const std::size_t k = values.size();
    if (k < 2) return {};

    // Fast path: every element shares h2 with the first → same signature.
    bool same = true;
    const std::uint32_t h2_ref = values[0].h2;
    for (std::size_t i = 1; i < k; ++i) {
      if (values[i].h2 != h2_ref) { same = false; break; }
    }
    if (same) {
      // Skip groups whose members all already share the same root — the
      // unions would be no-ops and the touched ids would re-enter
      // next_work, looping the BSP forever. Sequential O(k) scan; same
      // shape as the h2 check above and short-circuits on the first
      // mismatch (i.e. on every group that's actually doing useful work).
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

    // Slow path: bucket by h2 within this h1-equal group. With 96-bit
    // combined entropy this should never fire on real inputs.
    parlay::sequence<bool> consumed(k, false);
    parlay::sequence<Id> touched;
    for (std::size_t i = 0; i < k; ++i) {
      if (consumed[i]) continue;
      consumed[i] = true;
      parlay::sequence<RootH2> bucket;
      bucket.push_back(values[i]);
      for (std::size_t j = i + 1; j < k; ++j) {
        if (!consumed[j] && values[j].h2 == values[i].h2) {
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
  if (timings) timings->per_group_ms = ms_since(t2);
#else
  // Hash variant: original group_by_key + sigs_equal. Value half of the
  // keyed pair is just Id — only the root is consumed downstream — so we
  // don't double the count-sort bandwidth carrying CanonEntry as the
  // value (a strict-win optimization that's independent of the algorithm).
  auto t0 = timings ? clk::now() : clk::time_point{};
  auto keyed = parlay::delayed_tabulate(n, [&](std::size_t i) {
    return std::pair<CanonEntry, Id>{canon[i], canon[i].root};
  });
  if (timings) timings->keyed_ms = ms_since(t0);

  auto hash_fn = [](const CanonEntry& e) -> std::size_t { return e.h; };
  auto equal_fn = [&](const CanonEntry& a, const CanonEntry& b) -> bool {
    return a.h == b.h && sigs_equal(a.idx, b.idx, uf, nodes);
  };
  auto t1 = timings ? clk::now() : clk::time_point{};
  auto groups = parlay::group_by_key(keyed, hash_fn, equal_fn);
  if (timings) timings->group_by_ms = ms_since(t1);

  auto t2 = timings ? clk::now() : clk::time_point{};
  per_group = parlay::map(groups, [&](auto& kv) -> parlay::sequence<Id> {
    auto& values = kv.second;
    if (values.size() < 2) return {};
    dnc_union(values, 0, values.size(), uf);
    return std::move(values);
  });
  if (timings) timings->per_group_ms = ms_since(t2);
#endif

  auto out = parlay::flatten(per_group);
  if (timings) timings->per_group_ms = ms_since(t2);
  return out;
}

}  // namespace detail

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
      const auto& [node, class_id] = nodes[idx];
#ifdef PE_GROUPBY_HASH
      return detail::CanonEntry{sig_hash(node, uf), uf.find_root(class_id), idx};
#else
      auto [h1, h2] = sig_hashes(node, uf);
      return detail::CanonEntry{h1, uf.find_root(class_id), h2};
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
      const auto& [node, my_class_raw] = nodes_[pidx];
      std::uint64_t h = sig_hash_seq(node, uf);
      Id my_class = uf.find_root(my_class_raw);

      auto& bucket = sig_table[h];
      std::int64_t match_o = -1;
      for (std::uint32_t o : bucket) {
        if (sigs_equal_seq(pidx, o, uf, nodes_)) {
          match_o = static_cast<std::int64_t>(o);
          break;
        }
      }
      if (match_o >= 0) {
        Id other_class = uf.find_root(
            nodes_[static_cast<std::size_t>(match_o)].second);
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
