#include "parallel_egraph/egraph.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <parlay/delayed.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>

// NOTE on remaining user-authored sequential work inside rounds:
//   1. Per-root-group concat in `parallel_consolidate`: we parallelize across
//      groups (each root gets its own thread), but within a group we append
//      every c's parent_index entries to parent_index[root] sequentially.
//      Group sizes are bounded by how many class ids share the same current
//      root among entries in `work`, which is usually a small constant.
//      Span is therefore O(max_group_size), typically O(1)–O(few).
// Everything else on the parallel hot path (canon/sort/scan/union/
// next_work) uses parlay primitives only.

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

// ---- EGraph ctor ----------------------------------------------------------

EGraph::EGraph(std::size_t capacity, bool parallel)
    : uf_(capacity),
      parent_index_(capacity),
      changed_(capacity),
      parallel_(parallel) {
  for (auto& b : changed_) b.store(false, std::memory_order_relaxed);
}

// ---- clone ----------------------------------------------------------------
// Produces an independent EGraph with identical state. Used by the SAT
// integration to snapshot a search level. The atomic-bool / atomic-u32
// fields cannot be copy-assigned, so we mirror their values element-wise.
std::unique_ptr<EGraph> EGraph::clone() const {
  auto out = std::make_unique<EGraph>(uf_.capacity(), parallel_);
  out->uf_.copy_state_from(uf_);
  out->nodes_ = nodes_;
  out->parent_index_ = parent_index_;
  // changed_ already has the right size from the ctor; copy values.
  for (std::size_t i = 0; i < changed_.size(); ++i) {
    out->changed_[i].store(changed_[i].load(std::memory_order_acquire),
                           std::memory_order_release);
  }
  out->classes_ = classes_;
  out->parents_ = parents_;
  out->worklist_ = worklist_;
  out->hashcons_ = hashcons_;
  // parallel_ already set by ctor.
  return out;
}

// ---- canonicalize ---------------------------------------------------------

ENode EGraph::canonicalize(const ENode& node) {
  ENode out;
  out.op = node.op;
  out.children.reserve(node.children.size());
  for (Id c : node.children) out.children.push_back(find(c));
  return out;
}

// ---- add ------------------------------------------------------------------

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

  if (!parallel_) {
    for (Id child : canon.children) {
      parents_[child].emplace_back(canon, id);
    }
    classes_[id].push_back(canon);
  }

  hashcons_.emplace(std::move(canon), id);
  return id;
}

// ---- merge (sequential) ---------------------------------------------------

Id EGraph::merge(Id a, Id b) {
  a = find(a);
  b = find(b);
  if (a == b) return a;

  uf_.union_(a, b);
  Id root = find(a);
  Id merged = (root == a) ? b : a;

  auto it_c = classes_.find(merged);
  if (it_c != classes_.end()) {
    auto& dst = classes_[root];
    for (auto& n : it_c->second) dst.push_back(std::move(n));
    classes_.erase(it_c);
  }

  auto it_p = parents_.find(merged);
  if (it_p != parents_.end()) {
    auto& dst = parents_[root];
    for (auto& p : it_p->second) dst.push_back(std::move(p));
    parents_.erase(it_p);
  }

  worklist_.push_back(root);
  return root;
}

// ---- repair ---------------------------------------------------------------

void EGraph::repair(Id id) {
  id = find(id);

  auto it = parents_.find(id);
  if (it == parents_.end()) return;

  std::vector<std::pair<ENode, Id>> old_parents = std::move(it->second);
  parents_.erase(it);

  for (auto& entry : old_parents) {
    ENode p_node = std::move(entry.first);
    Id p_class = entry.second;

    hashcons_.erase(p_node);
    ENode p_canon = canonicalize(p_node);
    Id p_id = find(p_class);

    auto existing_it = hashcons_.find(p_canon);
    if (existing_it != hashcons_.end()) {
      Id existing = find(existing_it->second);
      p_id = merge(p_id, existing);
    }

    p_id = find(p_id);
    hashcons_[p_canon] = p_id;
    Id root = find(id);
    parents_[root].emplace_back(std::move(p_canon), p_id);
  }
}

// ---- rebuild (sequential) ------------------------------------------------

void EGraph::rebuild() {
  if (parallel_) {
    parallel_rebuild();
    return;
  }
  while (!worklist_.empty()) {
    std::vector<Id> todo;
    todo.swap(worklist_);
    for (Id id : todo) {
      Id id_root = find(id);
      repair(id_root);
    }
  }
}

// ---- parallel_merge_all --------------------------------------------------

void EGraph::grow_changed_to(std::size_t) {
  // No-op: changed_ is sized to UF capacity at construction.
}

void EGraph::parallel_merge(Id a, Id b) {
  uf_.union_(a, b);
  changed_[a].store(true, std::memory_order_release);
  changed_[b].store(true, std::memory_order_release);
}

void EGraph::parallel_merge_all(
    const parlay::sequence<std::pair<Id, Id>>& pairs) {
  if (pairs.empty()) return;
  parlay::parallel_for(0, pairs.size(), [&](std::size_t i) {
    parallel_merge(pairs[i].first, pairs[i].second);
  });
}

// ---- parallel consolidation helper ---------------------------------------
//
// For each c in `cs` where c != root_of_c, migrate parent_index[c] into
// parent_index[root_of_c]. Fully parallel across groups (one group per
// distinct root); within a group the appends are sequential because they
// all target the same destination slot (see NOTE at top of file).

namespace {

void parallel_consolidate(
    parlay::sequence<parlay::sequence<Id>>& parent_index,
    const parlay::sequence<Id>& cs,
    const parlay::sequence<Id>& roots) {
  auto non_root_pairs = parlay::map_maybe(
      parlay::iota(cs.size()),
      [&](std::size_t i) -> std::optional<std::pair<Id, Id>> {
        if (cs[i] == roots[i]) return std::nullopt;
        // Key on root so group_by_key groups all c's that share a root.
        return std::make_pair(roots[i], cs[i]);
      });
  if (non_root_pairs.empty()) return;

  auto groups = parlay::group_by_key(std::move(non_root_pairs));

  // Outer parallel: distinct root r per group, so writes to parent_index[r]
  // are isolated across groups. Inner parallel: precompute offsets via
  // parlay::scan, resize parent_index[r] once, then scatter each c's
  // entries into its pre-computed slot. No user-authored sequential loop.
  // parent_index is sized to uf_ capacity at construction; class ids
  // never exceed that, so the bounds checks present in earlier drafts are
  // dead and have been removed.
  parlay::parallel_for(0, groups.size(), [&](std::size_t g) {
    Id r = groups[g].first;
    const auto& cs_for_r = groups[g].second;

    auto sizes = parlay::tabulate(cs_for_r.size(), [&](std::size_t i) {
      return parent_index[cs_for_r[i]].size();
    });
    auto scan_result = parlay::scan(sizes);        // (offsets, total)
    auto& offsets = scan_result.first;
    std::size_t total_new = scan_result.second;

    auto& dst = parent_index[r];
    std::size_t old_size = dst.size();
    dst.resize(old_size + total_new);

    // Sequential cutoff for the inner copy: tiny src.size() doesn't
    // benefit from forking, and many groups have parent_index entries of
    // size 1–4. Threshold matches a typical parlay base-case.
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

struct CanonEntry {
  std::uint64_t h;
  std::uint32_t idx;
  Id root;
};

// Divide-and-conquer union over a contiguous same-sig run [lo, hi) of a
// sorted bucket: recursively merges the left half and the right half in
// parallel (parlay::par_do), then unions the two sub-representatives.
// Span O(log n); contention is minimal because sibling recursive unions
// target disjoint class ids.
//
// Below DNC_SEQ_CUTOFF, fall through to a sequential pairwise sweep —
// par_do fork/join overhead dominates for tiny groups, which are common.
template <typename Bucket>
void dnc_union(Bucket& bucket, std::size_t lo, std::size_t hi,
               ConcurrentUnionFind& uf) {
  constexpr std::size_t DNC_SEQ_CUTOFF = 16;
  if (hi - lo <= 1) return;
  if (hi - lo <= DNC_SEQ_CUTOFF) {
    for (std::size_t i = lo + 1; i < hi; ++i) {
      uf.union_(bucket[lo].root, bucket[i].root);
    }
    return;
  }
  std::size_t mid = lo + (hi - lo) / 2;
  parlay::par_do([&] { dnc_union(bucket, lo, mid, uf); },
                 [&] { dnc_union(bucket, mid, hi, uf); });
  uf.union_(bucket[lo].root, bucket[mid].root);
}

// Given a canon sequence, merge classes that share both hash and signature,
// and return the set of touched class ids (pre-round roots of any union
// issued).
//
// Uses parlay::group_by_key with a CUSTOM HASH and a CUSTOM EQUAL, which is
// the semisort primitive parlay exposes for "hash-to-distribute,
// equality-check-to-group" workloads. Hash = sig_hash (already computed in
// canon.h). Equal = sigs_equal (op + find_root(children) component-wise).
//
// Groups are exact: every group contains only entries that compare
// equal under sigs_equal. Hash collisions don't cause incorrect merges —
// the equal predicate filters them. This removes the need for any
// in-bucket sort.
//
// Within each group (same-sig run), D&C union collapses all class roots
// into one via parlay::par_do — O(log group_size) span, minimal contention.
parlay::sequence<Id> merge_and_collect_semisort(
    parlay::sequence<CanonEntry> canon, ConcurrentUnionFind& uf,
    const parlay::sequence<std::pair<ENode, Id>>& nodes) {
  const std::size_t n = canon.size();
  if (n == 0) return {};

  // group_by_key consumes a range of pair<K, V>. We carry the whole
  // CanonEntry as both key and value so the hash/equal functors can see
  // both the cached hash (.h) and the node index (.idx, for sigs_equal).
  auto keyed = parlay::tabulate(n, [&](std::size_t i) {
    return std::pair<CanonEntry, CanonEntry>{canon[i], canon[i]};
  });

  auto hash_fn = [](const CanonEntry& e) -> std::size_t { return e.h; };
  auto equal_fn = [&](const CanonEntry& a, const CanonEntry& b) -> bool {
    return a.h == b.h && sigs_equal(a.idx, b.idx, uf, nodes);
  };
  auto groups =
      parlay::group_by_key(std::move(keyed), hash_fn, equal_fn);

  // Each group's values[] is a sequence<CanonEntry> with all sigs equal.
  // D&C union its roots and emit every root as "touched" for next round.
  auto per_group = parlay::map(groups, [&](auto& kv) -> parlay::sequence<Id> {
    auto& values = kv.second;  // sequence<CanonEntry>
    if (values.size() < 2) return {};
    dnc_union(values, 0, values.size(), uf);
    return parlay::tabulate(values.size(),
                            [&](std::size_t i) { return values[i].root; });
  });

  return parlay::flatten(per_group);
}

}  // namespace

// ---- parallel_rebuild dispatcher -----------------------------------------

void EGraph::parallel_rebuild() {
  const char* v = std::getenv("PE_REBUILD");
  if (v) {
    if (std::strcmp(v, "sort") == 0) {
      parallel_rebuild_sort();
      return;
    }
    if (std::strcmp(v, "close") == 0) {
      auto initial = parlay::map_maybe(
          parlay::iota<std::size_t>(changed_.size()),
          [&](std::size_t i) -> std::optional<std::pair<Id, Id>> {
            if (changed_[i].exchange(false, std::memory_order_acq_rel)) {
              Id id = static_cast<Id>(i);
              return std::make_pair(id, id);
            }
            return std::nullopt;
          });
      parallel_close(std::move(initial));
      return;
    }
  }
  parallel_rebuild_semisort();
}

// ---- parallel_rebuild_semisort -------------------------------------------
//
// Mirrors src/egraph.rs:775-890 with the consolidation fix. Uses parlay's
// `group_by_index` semisort primitive (bucket-groups via radix/counting
// sort) then sorts each bucket with parlay::sort_inplace.

void EGraph::parallel_rebuild_semisort() {
  auto& uf = uf_;
  auto& nodes = nodes_;
  auto& parent_index = parent_index_;
  auto& changed = changed_;

  for (;;) {
    // 1. Collect changed classes in parallel.
    auto changed_classes = parlay::map_maybe(
        parlay::iota<std::size_t>(changed.size()),
        [&](std::size_t i) -> std::optional<Id> {
          if (changed[i].exchange(false, std::memory_order_acq_rel)) {
            return static_cast<Id>(i);
          }
          return std::nullopt;
        });
    if (changed_classes.empty()) break;

    // 2. Roots in parallel.
    auto roots = parlay::map(changed_classes,
                             [&](Id c) { return uf.find_root(c); });

    // 3. Consolidation (parallel across distinct roots).
    parallel_consolidate(parent_index, changed_classes, roots);

    // 4. Dedup roots so the frontier doesn't double-iterate the same slot.
    // Hash-based dedup (no sort). Avoids inflating the frontier when the
    // work list contains multiple class ids that share a current root.
    auto unique_roots = parlay::remove_duplicates(roots);

    // 5. Frontier: flat_map root -> parent_index[root].
    auto frontier = parlay::flatten(parlay::map(unique_roots, [&](Id r) {
      return parlay::sequence<std::uint32_t>(std::begin(parent_index[r]),
                                             std::end(parent_index[r]));
    }));
    if (frontier.empty()) break;

    // 6. Canonicalize: (hash, idx, class_root) per frontier entry.
    auto canon = parlay::map(frontier, [&](std::uint32_t idx) {
      const auto& [node, class_id] = nodes[idx];
      return CanonEntry{sig_hash(node, uf), idx, uf.find_root(class_id)};
    });

    // 7. Merge via semisort (group_by_index + per-bucket D&C union inline);
    //    return touched class ids (pre-round roots of any applied union).
    auto touched = merge_and_collect_semisort(std::move(canon), uf, nodes);
    if (touched.empty()) break;

    // 8. Mark changed (pre-round roots; next round's consolidation handles
    //    the other members of the merged classes).
    parlay::parallel_for(0, touched.size(), [&](std::size_t i) {
      changed[touched[i]].store(true, std::memory_order_release);
    });
  }
}

// ---- parallel_rebuild_sort -----------------------------------------------
//
// Single global parlay::sort_inplace with (hash, sig_cmp) comparator
// (sample sort). Same consolidation fix as _semisort.

void EGraph::parallel_rebuild_sort() {
  auto& uf = uf_;
  auto& nodes = nodes_;
  auto& parent_index = parent_index_;
  auto& changed = changed_;

  for (;;) {
    auto changed_classes = parlay::map_maybe(
        parlay::iota<std::size_t>(changed.size()),
        [&](std::size_t i) -> std::optional<Id> {
          if (changed[i].exchange(false, std::memory_order_acq_rel)) {
            return static_cast<Id>(i);
          }
          return std::nullopt;
        });
    if (changed_classes.empty()) break;

    auto roots = parlay::map(changed_classes,
                             [&](Id c) { return uf.find_root(c); });
    parallel_consolidate(parent_index, changed_classes, roots);

    // Hash-based dedup (no sort). Avoids inflating the frontier when the
    // work list contains multiple class ids that share a current root.
    auto unique_roots = parlay::remove_duplicates(roots);

    auto frontier = parlay::flatten(parlay::map(unique_roots, [&](Id r) {
      return parlay::sequence<std::uint32_t>(std::begin(parent_index[r]),
                                             std::end(parent_index[r]));
    }));
    if (frontier.empty()) break;

    auto canon = parlay::map(frontier, [&](std::uint32_t idx) {
      const auto& [node, class_id] = nodes[idx];
      return CanonEntry{sig_hash(node, uf), idx, uf.find_root(class_id)};
    });

    parlay::sort_inplace(canon,
                         [&](const CanonEntry& a, const CanonEntry& b) {
                           if (a.h != b.h) return a.h < b.h;
                           return sig_cmp(a.idx, b.idx, uf, nodes) < 0;
                         });

    // Apply unions inline, collect touched class ids.
    auto touched = parlay::flatten(parlay::map_maybe(
        parlay::iota<std::size_t>(canon.size() == 0 ? 0 : canon.size() - 1),
        [&](std::size_t i) -> std::optional<parlay::sequence<Id>> {
          if (canon[i].h != canon[i + 1].h) return std::nullopt;
          if (!sigs_equal(canon[i].idx, canon[i + 1].idx, uf, nodes))
            return std::nullopt;
          Id a = canon[i].root;
          Id b = canon[i + 1].root;
          if (a == b) return std::nullopt;
          uf.union_(a, b);
          return parlay::sequence<Id>{a, b};
        }));
    if (touched.empty()) break;

    parlay::parallel_for(0, touched.size(), [&](std::size_t i) {
      changed[touched[i]].store(true, std::memory_order_release);
    });
  }
}

// ---- parallel_close ------------------------------------------------------
//
// BSP closure on explicit initial unions. Fully parallel within each round
// (the only user-authored sequential loop is the per-group concat inside
// parallel_consolidate — see NOTE at top). Mirrors src/egraph.rs:619-759
// with the consolidation fix.

void EGraph::parallel_close(parlay::sequence<std::pair<Id, Id>> initial_unions) {
  auto& uf = uf_;
  auto& nodes = nodes_;
  auto& parent_index = parent_index_;
  const bool trace = std::getenv("PE_TRACE") != nullptr;

  // 1. Apply initial unions in parallel.
  parlay::parallel_for(0, initial_unions.size(), [&](std::size_t i) {
    uf.union_(initial_unions[i].first, initial_unions[i].second);
  });

  // 2. Seed work list in parallel: flatten [(a,b) -> [a, b]].
  auto work = parlay::flatten(parlay::map(initial_unions, [](auto p) {
    return parlay::sequence<Id>{p.first, p.second};
  }));

  std::size_t round = 0;
  while (!work.empty()) {
    // Dedup work — initial_unions may repeat a class id (e.g. the dispatcher
    // that converts changed flags to (id, id) pairs doubles every id), and
    // duplicates would break the pre-scanned offsets inside
    // parallel_consolidate by double-counting a slot's size. Hash-based
    // remove_duplicates, no sort required.
    work = parlay::remove_duplicates(std::move(work));

    // 3. Roots.
    auto roots = parlay::map(work, [&](Id c) { return uf.find_root(c); });

    // 4. Consolidation.
    parallel_consolidate(parent_index, work, roots);

    // 5. Dedup roots.
    // Hash-based dedup (no sort). Avoids inflating the frontier when the
    // work list contains multiple class ids that share a current root.
    auto unique_roots = parlay::remove_duplicates(roots);

    // 6. Frontier via flat_map.
    auto frontier = parlay::flatten(parlay::map(unique_roots, [&](Id r) {
      return parlay::sequence<std::uint32_t>(std::begin(parent_index[r]),
                                             std::end(parent_index[r]));
    }));
    if (frontier.empty()) {
      if (trace) {
        std::fprintf(stderr, "[pe] round=%3zu work=%9zu frontier=0 (break)\n",
                     round, work.size());
      }
      break;
    }

    // 7. Canonicalize (hash, idx, class_root) so we can feed the shared
    //    semisort helper. The `root` field carries the pre-round class
    //    root, which becomes the (a, b) in the merge pair.
    auto canon = parlay::map(frontier, [&](std::uint32_t idx) {
      const auto& [node, class_id] = nodes[idx];
      return CanonEntry{sig_hash(node, uf), idx, uf.find_root(class_id)};
    });

    // 8. Merge via semisort — same primitive as _semisort. group_by_key
    //    with custom hash (sig_hash) and custom equal (sigs_equal) groups
    //    same-sig entries exactly; D&C-union within each group. Applies
    //    unions inline and returns touched class ids.
    auto next_work = merge_and_collect_semisort(std::move(canon), uf, nodes);

    // 9. Hash-based dedup for the next round's work list.
    next_work = parlay::remove_duplicates(std::move(next_work));

    if (trace) {
      std::fprintf(stderr,
                   "[pe] round=%3zu work=%9zu frontier=%9zu next=%9zu\n",
                   round, work.size(), frontier.size(), next_work.size());
    }

    work = std::move(next_work);
    ++round;
  }
}

// ---- sequential_close_nelson --------------------------------------------

void EGraph::sequential_close_nelson(
    const parlay::sequence<std::pair<Id, Id>>& initial_unions) {
  const std::size_t n_classes = uf_.len();
  // parent_index_ is sized to UF capacity at construction; n_classes is
  // bounded by that, so the resize on legacy entry-points is dead.

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

  std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> sig_table;

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

  for (std::size_t c = 0; c < n_classes; ++c) {
    Id id = static_cast<Id>(c);
    Id r = uf.find_root(id);
    if (r != id) uf_.union_(id, r);
  }
}

// ---- diagnostics ---------------------------------------------------------

std::size_t EGraph::num_enodes() const {
  std::size_t total = 0;
  for (const auto& kv : classes_) total += kv.second.size();
  return total;
}

void EGraph::print() const {
  std::printf("EGraph: %zu classes, %zu e-nodes\n", num_classes(), num_enodes());
}

}  // namespace pe
