#pragma once
// E-graph trimmed to exactly what the closure benchmark needs: a single
// ctor that takes a flat sequence of ENodes, then parallel_parents /
// sequential_close_nelson on a list of equalities. No incremental add() —
// all consumers go through the bulk ctor, so nodes_ is sparsely indexed
// by class id (the array index *is* the class id).
//
// Templated on the union-find type. `EGraph<ConcurrentUnionFind>` carries
// `parallel_parents` and is used from the parallel BSP path;
// `EGraph<SequentialUnionFind>` carries `sequential_close_nelson` and is
// used from the sequential Nelson baseline. Each instance constructs only
// the UF it needs — no bridging between them.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/group_by.h>

#include "parallel_egraph/fxhash.hpp"
#include "parallel_egraph/unionfind.hpp"

namespace pe {

struct ENode {
  std::string op;
  std::vector<Id> children;

  bool operator==(const ENode& other) const {
    return op == other.op && children == other.children;
  }
};

// ---- Signature helpers (used during BSP rounds) ---------------------------
//
// Templated on the union-find type so the parallel path
// (`ConcurrentUnionFind`) and sequential Nelson baseline
// (`SequentialUnionFind`) share one implementation. Both UF types expose
// `find_root(Id) -> Id`; that's the only requirement on `UF`.

template <typename UF>
std::uint64_t sig_hash(const ENode& node, UF& uf) {
  FxHasher h;
  h.write_str(node.op);
  for (Id c : node.children) h.write_u32(uf.find_root(c));
  return h.finish();
}

// Returns a 64-bit primary hash and a 32-bit secondary hash computed in a
// single pass over the node's children (one set of UF lookups). The
// secondary uses a different FxHasher seed; combined 96-bit entropy makes
// hash collisions across distinct signatures effectively impossible
// (≈10⁻¹⁴ across a 64M-element batch on the largest synthetic workload).
template <typename UF>
std::pair<std::uint64_t, std::uint32_t> sig_hashes(
    const ENode& node, UF& uf) {
  // Golden-ratio-derived seed; structurally unrelated to FxHasher's mixing
  // constant, keeping h1 and h2 effectively independent.
  static constexpr std::uint64_t kSigHash2Seed = 0x9E37'79B9'7F4A'7C15ULL;
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

template <typename UF>
bool sigs_equal(std::uint32_t ia, std::uint32_t ib, UF& uf,
                const parlay::sequence<ENode>& nodes) {
  const auto& na = nodes[ia];
  const auto& nb = nodes[ib];
  if (na.op != nb.op) return false;
  if (na.children.size() != nb.children.size()) return false;
  for (std::size_t i = 0; i < na.children.size(); ++i) {
    if (uf.find_root(na.children[i]) != uf.find_root(nb.children[i])) {
      return false;
    }
  }
  return true;
}

// ---- EGraph ---------------------------------------------------------------

// Tag type selecting the auxiliary state populated by the EGraph ctor.
// The default (no tag) is the BSP layout — `parents_` populated, used by
// `parallel_parents`. The filter layout populates `last_marked_` instead
// (per-class round-stamp tracking when each class was most recently
// rerooted) and is used by `parallel_filter`. Each closure
// path uses only its own auxiliary state, so the bench can A/B compare
// without paying for the wrong-flavor setup.
struct filter_t { explicit filter_t() = default; };
inline constexpr filter_t filter{};

// Tag selecting the depth-stratified parallel closure layout. The ctor
// computes each class's depth (longest path from any leaf) and groups
// class ids by depth into `depth_buckets_`. Used by
// `parallel_topo`. Skips parents_ and last_marked_; calling
// either of the other parallel closure variants on an instance built
// this way is undefined.
struct topo_t { explicit topo_t() = default; };
inline constexpr topo_t topo{};

// Tag selecting the bare-UF closure layout used by
// `parallel_naive_rounds`. The ctor only initializes the union-find
// and stores `nodes_`; it skips `parents_`, `last_marked_`, and
// `depth_buckets_` so the naive variant doesn't pay for auxiliary state
// it never reads. Calling any other closure variant on an instance built
// this way is undefined.
struct naive_t { explicit naive_t() = default; };
inline constexpr naive_t naive{};

// Tag selecting the hybrid closure layout used by
// `parallel_filter_hybrid`. Builds BOTH `parents_` (used by the
// par_parents-style parents-walk path) AND `last_marked_` (used by the
// par_filter-style filter scan path). The algorithm starts in filter
// mode and switches to parents-walk mode once the dirty-set ratio
// drops below a threshold — see PE_HYBRID_SWITCH_RATIO in
// `parallel_filter_hybrid`.
struct hybrid_t { explicit hybrid_t() = default; };
inline constexpr hybrid_t hybrid{};

// Round-stamp slot used by the async closure path. Padded to 64 bytes
// (one cache line) so that concurrent CAS-max updates on adjacent
// class ids don't false-share. Without padding, eight slots fit in
// one cache line: at high core counts the MESI traffic from
// updating slot[3] invalidates slot[0..7] on every other core,
// turning an O(unions) atomic workload into an O(unions × cores)
// cache-coherence storm. With padding, distinct slots are on
// distinct cache lines, so threads can update unrelated classes in
// parallel without trashing each other's caches. Cost is 8× memory
// (64B per slot vs 8B), which is fine on the supercomputer at
// quintic-n=18 scale (~240MB).
struct alignas(64) PaddedMark {
  std::atomic<std::uint64_t> v;
  // Trailing padding fills the rest of the cache line; sized to keep
  // the struct exactly 64 bytes regardless of std::atomic<uint64_t>'s
  // alignment requirements on this target.
  char _pad[64 - sizeof(std::atomic<std::uint64_t>)];
};
static_assert(sizeof(PaddedMark) == 64,
              "PaddedMark must be exactly one cache line");
static_assert(alignof(PaddedMark) == 64,
              "PaddedMark must be cache-line aligned");

template <typename UF>
class EGraph {
 public:
  // BSP-flavor ctor: populate `parents_` (the inverted child→parent
  // index used by the BSP frontier walk). Leaves `last_marked_`
  // empty; calling `parallel_filter` on an instance
  // built this way is undefined.
  //
  // Bulk-construct from a sequence of ENodes given in DAG order: the i-th
  // node receives class id i, and every child id in node i must be < i.
  // The caller is responsible for ensuring no two nodes are structurally
  // identical (no hashcons dedup). nodes_ ends up class-id-indexed
  // (sparse), so leaves occupy slots in nodes_ but contribute nothing to
  // parents_. EGraph<ConcurrentUnionFind> holds atomics (non-movable),
  // so callers heap-allocate via `std::make_unique<EGraph<UF>>(...)`.
  explicit EGraph(parlay::sequence<ENode> nodes) : uf_(nodes.size()) {
    const std::size_t n = nodes.size();
    uf_.bulk_init(n);

    auto child_pairs = parlay::flatten(parlay::tabulate(
        n, [&](std::size_t i) -> parlay::sequence<std::pair<Id, Id>> {
          const auto& cs = nodes[i].children;
          parlay::sequence<std::pair<Id, Id>> result;
          result.reserve(cs.size());
          Id parent = static_cast<Id>(i);
          for (Id c : cs) result.emplace_back(c, parent);
          return result;
        }));

    parents_ = parlay::group_by_index(
        std::move(child_pairs), static_cast<Id>(n));
    nodes_ = std::move(nodes);
  }

  // Async-flavor ctor: skip `parents_` setup entirely; allocate
  // `last_marked_` with one slot per class, default-initialized to 0.
  // The closure uses these as round-stamps: when class c gets unioned
  // into a new root r, last_marked_[r] is set to the current global
  // round R via a monotone CAS-max. A term `t` is dirty in round R iff
  // some child c of t has last_marked_[find_root(c)] ∈ {R-1, R}.
  // Calling `parallel_parents` on an instance built this way is
  // undefined: `parents_` is empty.
  EGraph(parlay::sequence<ENode> nodes, filter_t) : uf_(nodes.size()) {
    const std::size_t n = nodes.size();
    uf_.bulk_init(n);
    nodes_ = std::move(nodes);
    // parlay::sequence<PaddedMark> supports in-place construction via
    // uninitialized + parallel placement; we then zero-init each
    // slot's atomic. PaddedMark's default ctor would leave .v
    // unspecified, so the explicit store is required. PaddedMark is
    // 64B so distinct slots live on distinct cache lines — eliminates
    // false sharing on `last_marked_` at high core counts.
    last_marked_ = parlay::sequence<PaddedMark>::uninitialized(n);
    parlay::parallel_for(0, n, [&](std::size_t i) {
      new (&last_marked_[i]) PaddedMark{std::atomic<std::uint64_t>(0), {}};
    });
  }

  // Naive-flavor ctor: bulk-init the UF and stash nodes_; skip every
  // auxiliary index (`parents_`, `last_marked_`, `depth_buckets_`).
  // Used only by `parallel_naive_rounds`, which semisorts every
  // non-leaf term every round and so needs nothing beyond the UF and
  // the node list.
  EGraph(parlay::sequence<ENode> nodes, naive_t) : uf_(nodes.size()) {
    const std::size_t n = nodes.size();
    uf_.bulk_init(n);
    nodes_ = std::move(nodes);
  }

  // Hybrid-flavor ctor: build BOTH `parents_` (inverted child→parent
  // index, as in the BSP ctor) AND `last_marked_` (per-class round
  // stamps, as in the async ctor). Used by
  // `parallel_filter_hybrid`, which starts in async-style filter
  // mode and switches to BSP-style parents-walk mode when the
  // dirty-set ratio falls below threshold.
  EGraph(parlay::sequence<ENode> nodes, hybrid_t) : uf_(nodes.size()) {
    const std::size_t n = nodes.size();
    uf_.bulk_init(n);

    auto child_pairs = parlay::flatten(parlay::tabulate(
        n, [&](std::size_t i) -> parlay::sequence<std::pair<Id, Id>> {
          const auto& cs = nodes[i].children;
          parlay::sequence<std::pair<Id, Id>> result;
          result.reserve(cs.size());
          Id parent = static_cast<Id>(i);
          for (Id c : cs) result.emplace_back(c, parent);
          return result;
        }));
    parents_ = parlay::group_by_index(
        std::move(child_pairs), static_cast<Id>(n));

    nodes_ = std::move(nodes);

    last_marked_ = parlay::sequence<PaddedMark>::uninitialized(n);
    parlay::parallel_for(0, n, [&](std::size_t i) {
      new (&last_marked_[i]) PaddedMark{std::atomic<std::uint64_t>(0), {}};
    });
  }

  // Topo-flavor ctor: compute each class's depth (longest path from any
  // leaf — leaves are 0, parents are 1 + max(child depth)) via a single
  // sequential bottom-up sweep (cheap because nodes_ is already in DAG
  // order, so children are at indices < i). Then group class ids by depth
  // into `depth_buckets_`, used as the per-round frontier in
  // `parallel_topo`. Skips parents_/last_marked_ entirely.
  EGraph(parlay::sequence<ENode> nodes, topo_t) : uf_(nodes.size()) {
    const std::size_t n = nodes.size();
    uf_.bulk_init(n);
    nodes_ = std::move(nodes);

    std::vector<std::uint32_t> depth(n, 0);
    std::uint32_t max_depth = 0;
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t d = 0;
      for (Id c : nodes_[i].children) {
        const std::uint32_t cd = depth[c] + 1;
        if (cd > d) d = cd;
      }
      depth[i] = d;
      if (d > max_depth) max_depth = d;
    }

    auto pairs = parlay::tabulate(n, [&](std::size_t i) {
      return std::pair<Id, Id>{static_cast<Id>(depth[i]),
                               static_cast<Id>(i)};
    });
    depth_buckets_ = parlay::group_by_index(
        std::move(pairs), static_cast<Id>(max_depth + 1));
  }

  // Lock-free for ConcurrentUnionFind; safe to call concurrently. Path
  // compression CAS is a write, so this is non-const.
  Id find(Id id) { return uf_.find_root(id); }

  // BSP closure on explicit initial unions; fully parallel within rounds.
  // See DESIGN.md §1 (parallel_consolidate) and §2 (merge_and_collect_semisort)
  // for the algorithm. Defined out-of-line in src/egraph.cpp only as an
  // explicit specialization on `ConcurrentUnionFind` — calling it on the
  // sequential flavor is a link error.
  void parallel_parents(parlay::sequence<std::pair<Id, Id>> initial_unions);

  // BSP closure that replaces the integer-sort+run-walk semisort with
  // `parlay::group_by_key` over per-arity `SigK<K>` keys (K = 1..4)
  // plus a vector-backed `Signature` fallback for arity > 4. Same
  // `parents_`-driven round structure as `parallel_parents`; differs
  // only in the per-round canonicalize+merge step. Structurally
  // sound by construction — group equality uses the SigK<K>'s
  // compile-time-unrolled `operator==`, no probabilistic hash
  // fallback. Defined out-of-line as an explicit specialization on
  // `ConcurrentUnionFind`.
  void parallel_parents_groupby_sigk(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Async-style closure (still rounds-based for now). Drops the
  // `parents_` machinery in favor of round-stamp tracking on each
  // class: `last_marked_[r]` is the round number when class r last
  // gained a new member (or was demoted into another root). The
  // algorithm:
  //   (1) apply pending unions in parallel; mark each affected root
  //       with the current round number R.
  //   (2) Increment R.
  //   (3) Filter to dirty: terms with at least one child whose
  //       find_root has last_marked ∈ {R-1, R}. The "or R" clause is
  //       belt-and-suspenders for the eventual async case where
  //       another thread might mark with R-of-the-just-bumped-R during
  //       our scan; in the rounds-based version it's a no-op since R
  //       only changes between rounds.
  //   (4) Semisort dirty by current canonical sig; union per bucket;
  //       mark the surviving root of each unioned bucket with R.
  //   (5) Loop until dirty is empty.
  // Defined out-of-line as an explicit specialization on
  // `ConcurrentUnionFind`.
  void parallel_filter(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Same algorithm as `parallel_filter` but using
  // `parlay::group_by_key` (hash table + per-bucket `sequence<
  // CanonEntry>` materialization) instead of in-place integer-sort.
  // Kept as an A/B baseline; the integer-sort variant is the
  // production path. Defined out-of-line as an explicit specialization
  // on `ConcurrentUnionFind`.
  void parallel_filter_groupby(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Naive rounds-based closure: same skeleton as
  // `parallel_filter` but without dirty tracking or the
  // `last_marked_` filter. Every round semisorts *all* non-leaf terms
  // by their current canonical signature, dnc_unions every multi-root
  // bucket, and loops until a full pass produces no new union (detected
  // by checking whether any bucket's snapshot roots disagreed). Costs
  // O(non_leaves) work per round vs the async variant's O(dirty), so
  // it serves as the "no filter" baseline for measuring what
  // last_marked_-driven filtering buys. Defined out-of-line as an
  // explicit specialization on `ConcurrentUnionFind`.
  void parallel_naive_rounds(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Hybrid filter / parents-walk closure. Starts in `par_filter`-style
  // filter mode: each round scans the non_leaf set checking
  // `last_marked_[find_root(child)] == R-1` to find dirty terms. Once
  // the dirty-set ratio falls below threshold (default 1% of
  // non_leaves, configurable via PE_HYBRID_SWITCH_RATIO), switches to
  // `par_parents`-style parents-walk mode: one bulk
  // `parallel_consolidate` of every loser collected so far, then each
  // subsequent round walks `parents_[find_root(losers_prev_round)]`
  // to form the dirty set. Closes the deep-cascade gap on workloads
  // where the filter scan's O(non_leaves) per-round overhead
  // dominates after the wide-merge phase ends. Requires the
  // hybrid-flavor ctor (`pe::hybrid`).
  void parallel_filter_hybrid(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // MIN_ID variant of `parallel_filter`. Same algorithm —
  // dirty filter via `last_marked_`, semisort by canonical sig, dnc_union
  // per multi-member bucket — but every union (initial unions and
  // dnc_union both) uses `union_min_id` instead of by-rank. On
  // DAG-ordered inputs where the lowest-id member of each equivalence
  // class is its structurally-canonical representative, this preserves
  // the canonical-id-stable invariant across BSP rounds: a single sweep
  // collapses the entire cascade for regular workloads, instead of one
  // round per cascade level. Defined out-of-line as an explicit
  // specialization on `ConcurrentUnionFind`.
  void parallel_filter_min_id(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Depth-stratified parallel closure. Round d processes every class at
  // depth d as a two-phase BSP step: (1) parallel canon-build —
  // `parlay::map` over the depth-d bucket reads `find_root` to compute
  // each node's CanonEntry, no UF unions occur yet; (2) parallel
  // semisort + dnc_union writes the new unions. The map join between (1)
  // and (2) is a barrier, so all sig reads in (1) observe the same UF
  // snapshot — the post-round-(d-1) state — and the semisort catches
  // every intra-round congruence reachable from that snapshot. (Note:
  // depth-stratification orders the *work* — children before parents,
  // bucket d after bucket d-1 — but it does not by itself isolate UF
  // state across threads. The actual safety property comes from the
  // phase-1 / phase-2 separation: phase 1 writes only path-compression
  // CAS, which is idempotent w.r.t. find_root return values.) Total
  // rounds = max depth — strictly fewer than parallel_parents's
  // frontier-driven BSP cadence on workloads with bounded depth.
  // Defined out-of-line as an explicit specialization on
  // `ConcurrentUnionFind`.
  void parallel_topo(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Sound iterated variant of `parallel_topo`. Same depth-stratified
  // structure (parallel canon-build per depth, semisort + dnc_union per
  // bucket), but every union uses MIN_ID linking AND the entire
  // depth-walk is wrapped in a fixpoint loop that repeats until a full
  // pass produces no new unions. Recovers correctness on cross-depth
  // initial unions; on regular cascades (Family C) MIN_ID makes a single
  // depth-walk usually suffice, so the verification pass is the only
  // overhead. Defined out-of-line as an explicit specialization on
  // `ConcurrentUnionFind`.
  void parallel_topo_iter(
      parlay::sequence<std::pair<Id, Id>> initial_unions);

  // Sequential Nelson-style closure baseline. Defined out-of-line only as
  // an explicit specialization on `SequentialUnionFind`.
  void sequential_close_nelson(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Simple sequential CC baseline: seed a worklist with every non-leaf
  // node, pop one at a time, canonicalize against a sig→class hashcons.
  // Repeated sig ⇒ union the two classes; on union, splice the loser's
  // class predecessors into the winner's list and re-queue them (their
  // signatures contain a now-stale child root). Self-contained class-
  // predecessor map built fresh inside the function (post-initial-
  // unions) — does not mutate `parents_`. Distinguishing feature vs
  // `sequential_close_dst`: seeded with every non-leaf rather than
  // driven by pre-existing congruences in the hashcons. Defined out-
  // of-line as an explicit specialization on `SequentialUnionFind`.
  void sequential_close_simple(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Hash-only variant of `sequential_close_simple`. Identical control
  // flow, but the hashcons is keyed by the same 96-bit canonical
  // signature (`sig_hashes`) used in the parallel `CanonEntry`, and
  // equality is decided by hash compare rather than structural
  // recheck. Matches the parallel algorithms' probabilistic
  // correctness model exactly (collision probability ≤ N²/2⁹⁷ per
  // closure), enabling apples-to-apples speedup comparisons.
  // Defined out-of-line as an explicit specialization on
  // `SequentialUnionFind`.
  void sequential_close_simple_hash(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Arity-specialized variant of `sequential_close_simple`. Same
  // deterministic structural correctness, but splits the signature
  // hashcons by arity: arity ≤ 2 uses an inline-only `Sig2` (no
  // heap allocation per term), arity > 2 falls back to the
  // vector-backed `Signature`. On workloads where most terms have
  // arity ≤ 2 (boolean gates, NOTs, equalities), this eliminates
  // the per-term `malloc` that dominates the baseline's cost while
  // keeping the same correctness model. Defined out-of-line as an
  // explicit specialization on `SequentialUnionFind`.
  void sequential_close_simple_arity(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Variant of `sequential_close_simple` that pulls signature
  // children out of a single chunked bump arena rather than a per-
  // term `std::vector<Id>`. Same deterministic structural correctness;
  // eliminates the per-term `malloc/free` pair without specializing
  // on arity. Defined out-of-line as an explicit specialization on
  // `SequentialUnionFind`.
  void sequential_close_simple_bump(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Combined variant: arity ≤ 2 uses the inline `Sig2` hashtable (no
  // allocation), arity > 2 uses the bump-allocated `SigBump`. Aims
  // for the best of both — zero allocation on the arity-≤2 fast path
  // and amortized-constant bump on the slow path. Defined out-of-line
  // as an explicit specialization on `SequentialUnionFind`.
  void sequential_close_simple_arity_bump(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Most general variant: one per-arity inline-storage hashtable
  // (`SigK<K>`) for arities 1..K, plus a bump-allocated fallback for
  // arity > K. Each per-arity hashtable's compare and hash loops are
  // fully unrolled at compile time. Currently K = 4 (compile-time
  // constant in the implementation; one new switch arm + table per
  // additional arity). Same deterministic structural correctness as
  // `sequential_close_simple`. Defined out-of-line as an explicit
  // specialization on `SequentialUnionFind`.
  void sequential_close_simple_inline(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Single-pass sequential closure: walks `nodes_` in reverse topological
  // order (children before parents — guaranteed by the EGraph ctor's DAG-
  // order invariant, so plain forward index iteration suffices) and
  // canonicalizes each node's signature once. Each repeated signature
  // unions the current node's class with the prior one. Defined
  // out-of-line as an explicit specialization on `SequentialUnionFind`.
  //
  // Order-dependent on cross-depth initial unions: see
  // tests/closure_test.cpp::test_seq_topo_adversarial_order for a
  // concrete failure case. Use `sequential_close_dst` for arbitrary
  // initial-union shapes.
  void sequential_close_topo(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Fixpoint over `sequential_close_topo`: repeats the single-pass
  // topological closure until a full pass yields no new unions. Recovers
  // correctness on cross-depth and adversarial inputs at the cost of
  // (cross-depth-chain-length + 1) passes. Each pass is the same forward
  // walk through `nodes_` with a fresh hashcons; once the UF is stable
  // across one full pass, the closure has converged. Defined out-of-line
  // as an explicit specialization on `SequentialUnionFind`.
  void sequential_close_topo_iter(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  // Worklist-driven sequential closure with smaller-into-larger merging.
  // Maintains a structural hashcons (Signature -> class id) seeded with
  // every node's initial signature; initial unions and seed-detected
  // duplicates are queued as pending merges. Each merge re-pends the
  // dying class's parents so their stale signatures get refreshed
  // against the new UF state. Continues until the merge queue and the
  // re-canonicalization queue are both empty. Correct on arbitrary
  // initial unions (cross-depth, cycles in the augmented DAG, etc.).
  // Requires `parents_` — populated by the default (BSP-flavor) ctor.
  // Defined out-of-line as an explicit specialization on
  // `SequentialUnionFind`.
  void sequential_close_dst(
      const parlay::sequence<std::pair<Id, Id>>& initial_unions);

  bool equiv(Id a, Id b) { return uf_.find_root(a) == uf_.find_root(b); }

  // Internal but exposed for tests / benches that need raw access.
  UF& uf() { return uf_; }
  const parlay::sequence<ENode>& nodes() const { return nodes_; }
  parlay::sequence<parlay::sequence<Id>>& parents() { return parents_; }
  parlay::sequence<PaddedMark>& last_marked() {
    return last_marked_;
  }

 private:
  UF uf_;
  parlay::sequence<ENode> nodes_;
  // BSP-only state. Populated by the default ctor; left empty by the
  // async ctor.
  parlay::sequence<parlay::sequence<Id>> parents_;
  // Async-only state. Populated by the `filter_t`-tagged ctor; left
  // empty by the default ctor. last_marked_[r].v = highest round
  // number for which class r was unioned-into-as-the-new-root.
  // Updated via parlay::write_max (monotone CAS-max), so a tail
  // worker holding a stale R can never overwrite a fresher mark.
  // Each entry is cache-line padded (PaddedMark = 64B) to prevent
  // false sharing between adjacent class ids' marks. Indexed by
  // class id.
  parlay::sequence<PaddedMark> last_marked_;
  // Topo-only state. Populated by the `topo_t`-tagged ctor; left empty
  // by the default and async ctors. depth_buckets_[d] is the (parallel)
  // sequence of class ids whose nodes lie at depth d in the DAG; depth
  // 0 = leaves, no congruence work. Indexed by depth; size = max_depth+1.
  parlay::sequence<parlay::sequence<Id>> depth_buckets_;
};

using ConcurrentEGraph = EGraph<ConcurrentUnionFind>;
using SequentialEGraph = EGraph<SequentialUnionFind>;

}  // namespace pe
