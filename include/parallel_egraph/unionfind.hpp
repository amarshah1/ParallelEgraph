#pragma once
// Lock-free concurrent union-find (Jayanti & Tarjan Listing 3) + plain
// sequential UF.
//
// No dynamic resize. Construct with a known capacity; make_set() bumps a
// single-threaded counter within that capacity. The caller (solver /
// benchmark) knows the upper bound on class count before e-graph
// construction — for the SMT solver, a one-pass AST walk counts subterms.

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

namespace pe {

using Id = std::uint32_t;

// High bit set = node is a root holding its rank.
// High bit clear = value is a parent pointer.
inline constexpr std::uint32_t RANK_FLAG = 0x8000'0000u;

inline bool is_rank(std::uint32_t v) { return (v & RANK_FLAG) != 0; }
inline std::uint32_t rank_value(std::uint32_t v) { return v & ~RANK_FLAG; }
inline std::uint32_t make_rank(std::uint32_t r) { return r | RANK_FLAG; }

class ConcurrentUnionFind {
 public:
  ConcurrentUnionFind() = default;
  explicit ConcurrentUnionFind(std::size_t capacity);

  // Current live size (number of make_set calls so far).
  std::size_t len() const { return size_; }
  std::size_t capacity() const { return data_.size(); }

  // Single-threaded; called only during the add phase.
  // Bumps size_ up to capacity_. Asserts on overflow.
  Id make_set();

  // One-shot equivalent of calling `make_set()` n times. Used by the
  // EGraph<UF> ctor to populate the UF without a per-element loop. Slots
  // are already at make_rank(0) from the ctor.
  void bulk_init(std::size_t n);

  // Thread-safe but non-const: CAS writes (path compression / rank bump)
  // are genuine mutations, so these are declared non-const. Safe to call
  // concurrently from multiple threads via a shared reference/pointer —
  // thread-safety in C++ is orthogonal to const-correctness.
  std::pair<Id, std::uint32_t> find(Id u);
  Id find_root(Id u) { return find(u).first; }
  // Returns the surviving root after the merge — i.e., the post-call
  // find_root(u) (== find_root(v)). Lets callers (e.g., dnc_union)
  // hand the survivor outward without a separate find_root call.
  Id union_(Id u, Id v);

  // MIN_ID variant: lower class id always wins (becomes root). Drops
  // by-rank balancing — path compression keeps chains short in practice.
  // Used by the `parallel_filter_min_id` variant; preserves
  // the canonical-id-stable invariant that lets DAG-ordered cascades
  // collapse in one BSP round on regular workloads (Family C).
  //
  // Returns (survivor, did_merge): survivor is the post-call root
  // (always min(u_root, v_root)); did_merge is true iff the call
  // actually merged two distinct classes (false if u and v were
  // already in the same class). Callers that need fixpoint detection
  // (e.g., `parallel_topo_iter`) capture did_merge.
  std::pair<Id, bool> union_min_id(Id u, Id v);

  bool same_set(Id u, Id v);

 private:
  // std::atomic<T> is non-movable. std::vector's size-constructor default-
  // constructs atomics in place — that works fine as long as we never grow
  // the vector (we don't: size_ bumps up to capacity). Access pattern is
  // identical to a raw array: one pointer-load + offset.
  std::vector<std::atomic<std::uint32_t>> data_;
  std::size_t size_ = 0;  // single-threaded counter, ≤ data_.size()
};

class SequentialUnionFind {
 public:
  SequentialUnionFind() = default;
  explicit SequentialUnionFind(std::size_t size)
      : data_(size, make_rank(0)) {}

  std::size_t len() const { return data_.size(); }

  // Symmetric with ConcurrentUnionFind::bulk_init so the EGraph<UF> ctor
  // can call it generically. The ctor here already populated data_; this
  // is a no-op.
  void bulk_init(std::size_t /*n*/) {}

  Id find_root(Id u);
  // Returns the surviving root (= find_root(u) == find_root(v) post-call).
  Id union_(Id u, Id v);

  // Merge `dying` into `survivor` unconditionally — `dying` becomes a
  // child of `survivor`, no by-rank tie-breaking. Caller must guarantee
  // `dying` and `survivor` are distinct roots. Used by
  // sequential_close_dst's smaller-into-larger merge, which decides which
  // root dies based on the size of `parents_[r]` rather than UF rank.
  void union_into(Id dying, Id survivor);

 private:
  std::uint32_t rank_of(Id root) const { return rank_value(data_[root]); }
  std::vector<std::uint32_t> data_;
};

}  // namespace pe
