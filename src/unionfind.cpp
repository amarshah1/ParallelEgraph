#include "parallel_egraph/unionfind.hpp"

#include <cassert>

// Memory orderings:
//   Acquire on load, Release on CAS success / store, Relaxed on CAS failure.

namespace pe {

// ---- ConcurrentUnionFind --------------------------------------------------

ConcurrentUnionFind::ConcurrentUnionFind(std::size_t capacity)
    : data_(capacity), size_(0) {
  for (std::size_t i = 0; i < capacity; ++i) {
    data_[i].store(make_rank(0), std::memory_order_relaxed);
  }
}

Id ConcurrentUnionFind::make_set() {
  assert(size_ < data_.size() && "UF capacity exceeded — no dynamic resize");
  Id id = static_cast<Id>(size_);
  data_[id].store(make_rank(0), std::memory_order_relaxed);
  ++size_;
  return id;
}

void ConcurrentUnionFind::bulk_init(std::size_t n) {
  assert(n <= data_.size() && "bulk_init: n exceeds capacity");
  size_ = n;
  // Slots already at make_rank(0) from the ctor — nothing else to do.
}

std::pair<Id, std::uint32_t> ConcurrentUnionFind::find(Id u) {
  std::uint32_t p = data_[u].load(std::memory_order_acquire);
  if (is_rank(p)) {
    return {u, rank_value(p)};
  }
  auto [root, rank] = find(p);
  if (p != root) {
    std::uint32_t expected = p;
    // Ignore the result — whether or not the CAS wins, the invariant holds.
    data_[u].compare_exchange_strong(expected, root,
                                     std::memory_order_release,
                                     std::memory_order_relaxed);
  }
  return {root, rank};
}

void ConcurrentUnionFind::union_(Id u, Id v) {
  for (;;) {
    auto [u_root, ru] = find(u);
    auto [v_root, rv] = find(v);
    if (u_root == v_root) return;

    if (ru < rv) {
      std::uint32_t expected = make_rank(ru);
      if (data_[u_root].compare_exchange_strong(expected, v_root,
                                                std::memory_order_release,
                                                std::memory_order_relaxed)) {
        return;
      }
    } else if (ru > rv) {
      std::uint32_t expected = make_rank(rv);
      if (data_[v_root].compare_exchange_strong(expected, u_root,
                                                std::memory_order_release,
                                                std::memory_order_relaxed)) {
        return;
      }
    } else {
      // Equal ranks: smaller id merges into larger id.
      Id lo, hi;
      if (u_root < v_root) {
        lo = u_root;
        hi = v_root;
      } else {
        lo = v_root;
        hi = u_root;
      }
      std::uint32_t r = ru;  // == rv
      std::uint32_t expected = make_rank(r);
      if (data_[lo].compare_exchange_strong(expected, hi,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
        // Best-effort rank bump on the winner.
        std::uint32_t expected_winner = make_rank(r);
        data_[hi].compare_exchange_strong(expected_winner, make_rank(r + 1),
                                          std::memory_order_release,
                                          std::memory_order_relaxed);
        return;
      }
    }
    // CAS failed; retry from the top.
  }
}

void ConcurrentUnionFind::union_min_id(Id u, Id v) {
  // MIN_ID merge: the lower-id root always survives. Repeatedly resolve
  // both sides to roots, then attempt to swing the higher-id root's slot
  // to point at the lower-id one. CAS races: if some other thread merges
  // either root in the meantime, we retry from the top — find() returns
  // the up-to-date root regardless. We only swing the *loser*'s slot,
  // and we only succeed if the slot is still its own rank-encoded root
  // value; if it was already merged into something else, retry. The
  // lower-id root's slot is never written by this routine.
  for (;;) {
    auto [u_root, ru] = find(u);
    auto [v_root, rv] = find(v);
    if (u_root == v_root) return;

    Id loser, winner;
    std::uint32_t loser_rank;
    if (u_root < v_root) {
      winner = u_root;
      loser  = v_root;
      loser_rank = rv;
    } else {
      winner = v_root;
      loser  = u_root;
      loser_rank = ru;
    }

    std::uint32_t expected = make_rank(loser_rank);
    if (data_[loser].compare_exchange_strong(expected, winner,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
      return;
    }
    // CAS failed: someone else either bumped loser's rank or merged
    // loser into a different root. Retry — find() will follow the new
    // parent pointer and we'll redo the comparison.
  }
}

bool ConcurrentUnionFind::same_set(Id u, Id v) {
  for (;;) {
    Id u_root = find(u).first;
    Id v_root = find(v).first;
    if (u_root == v_root) return true;
    // Verify u_root is still a root (linearizability check).
    std::uint32_t p = data_[u_root].load(std::memory_order_acquire);
    if (is_rank(p)) return false;
    // u_root was concurrently merged; retry.
  }
}

// ---- SequentialUnionFind --------------------------------------------------

Id SequentialUnionFind::find_root(Id u) {
  std::uint32_t p = data_[u];
  if (is_rank(p)) return u;
  Id root = find_root(p);
  if (p != root) data_[u] = root;
  return root;
}

void SequentialUnionFind::union_(Id u, Id v) {
  Id ru = find_root(u);
  Id rv = find_root(v);
  if (ru == rv) return;
  std::uint32_t ranku = rank_of(ru);
  std::uint32_t rankv = rank_of(rv);
  if (ranku < rankv) {
    data_[ru] = rv;
  } else if (ranku > rankv) {
    data_[rv] = ru;
  } else {
    Id lo, hi;
    if (ru < rv) { lo = ru; hi = rv; } else { lo = rv; hi = ru; }
    data_[lo] = hi;
    data_[hi] = make_rank(ranku + 1);
  }
}

void SequentialUnionFind::union_into(Id dying, Id survivor) {
  // Caller's responsibility to ensure both are roots and distinct.
  // We just splice: dying's slot points at survivor; survivor's rank
  // bookkeeping is left alone (it's irrelevant for correctness — we
  // never compete with by-rank union_ on the same call sequence).
  data_[dying] = survivor;
}

}  // namespace pe
