#pragma once
// ConcurrentAppendVec<T>: a chunked, lock-free, append-only vector.
//
// Use case: many threads concurrently appending events (merge log
// entries, adjacency edges) without contention beyond a single
// fetch_add per append. No element-level locks, no global lock, no
// reallocation that would invalidate prior pointers.
//
// Layout: an array of NUM_CHUNKS power-of-two-sized chunks. Chunk k
// has 2^(k + INITIAL_LOG) slots. Capacity sums to 2^(NUM_CHUNKS +
// INITIAL_LOG) − 2^INITIAL_LOG slots, which with the defaults below
// is ~ 2^32 slots — far more than we'd ever use.
//
// Append protocol:
//   idx = size_.fetch_add(1, relaxed)
//   compute (chunk, slot) from idx
//   if chunks_[chunk] is null: try to allocate it (CAS); other threads
//     racing on the same chunk lose the CAS and free their copy.
//   chunks_[chunk][slot] = value
//
// The fetch_add gives every appender a unique idx, and the chunk
// indexed by idx is determined deterministically. Different idx
// values touch different slots, so writes to slots never race. Only
// the per-chunk allocation step contends, and it's bounded by NUM_CHUNKS
// failed CASes across the entire lifetime of the vector.
//
// Reads via at(idx) are safe once size() > idx, provided the producer
// has finished writing the slot (acquire/release on chunks_ pointer
// gives a happens-before; the slot write itself is unsynchronized but
// a reader who sees size() > idx must have synchronized through
// fetch_add and therefore can observe the slot write — see the comment
// in append() for details).
//
// In our use case readers and writers are not concurrent: the parallel
// rebuild round writes into the log, and the consumer (cb_check_found_model
// or explain) reads after the round is complete. This is the simplest
// regime; the data structure also supports concurrent read+write.

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <utility>

namespace pe {

template <typename T,
          std::size_t INITIAL_LOG = 5,    // first chunk has 32 slots
          std::size_t NUM_CHUNKS  = 28>   // total ≈ 2^32 slots
class ConcurrentAppendVec {
 public:
  ConcurrentAppendVec() {
    for (auto& c : chunks_) c.store(nullptr, std::memory_order_relaxed);
  }

  ~ConcurrentAppendVec() { reset(); }

  ConcurrentAppendVec(const ConcurrentAppendVec& other) { copy_from(other); }

  ConcurrentAppendVec& operator=(const ConcurrentAppendVec& other) {
    if (this == &other) return *this;
    reset();
    copy_from(other);
    return *this;
  }

  // No move (atomic pointers can't be moved trivially without races; we
  // keep this simple and rely on copy + RVO).
  ConcurrentAppendVec(ConcurrentAppendVec&&) = delete;
  ConcurrentAppendVec& operator=(ConcurrentAppendVec&&) = delete;

  // Append a value; return the index it was placed at.
  std::size_t append(T value) {
    std::size_t idx = size_.fetch_add(1, std::memory_order_relaxed);
    auto [k, slot] = decompose(idx);
    T* chunk = chunks_[k].load(std::memory_order_acquire);
    if (chunk == nullptr) {
      // Allocate this chunk. Multiple threads may race here for the
      // same chunk; the loser frees its allocation.
      T* fresh = new T[chunk_size(k)];
      T* expected = nullptr;
      if (chunks_[k].compare_exchange_strong(
              expected, fresh,
              std::memory_order_release,
              std::memory_order_acquire)) {
        chunk = fresh;
      } else {
        delete[] fresh;
        chunk = expected;
      }
    }
    chunk[slot] = std::move(value);
    return idx;
  }

  // Number of appends successfully claimed. May briefly include slots
  // whose write is still in flight; callers wanting a stable size
  // should fence after their writers are done.
  std::size_t size() const {
    return size_.load(std::memory_order_acquire);
  }

  // Random access. Precondition: idx < size() AND the writer that
  // claimed idx has finished writing.
  const T& at(std::size_t idx) const {
    auto [k, slot] = decompose(idx);
    const T* chunk = chunks_[k].load(std::memory_order_acquire);
    assert(chunk != nullptr);
    return chunk[slot];
  }

  T& at(std::size_t idx) {
    auto [k, slot] = decompose(idx);
    T* chunk = chunks_[k].load(std::memory_order_acquire);
    assert(chunk != nullptr);
    return chunk[slot];
  }

  // Clear all elements and free chunks. Not thread-safe.
  void clear() { reset(); }

 private:
  static constexpr std::size_t chunk_size(std::size_t k) {
    return std::size_t(1) << (k + INITIAL_LOG);
  }

  // Map a global index to (chunk, slot_in_chunk). Chunk k holds slots
  // [first_idx_in_chunk(k), first_idx_in_chunk(k+1)). The geometry
  // gives first_idx_in_chunk(k) = (2^k − 1) * 2^INITIAL_LOG.
  static std::pair<std::size_t, std::size_t> decompose(std::size_t idx) {
    // Add 2^INITIAL_LOG so 1 + (idx >> INITIAL_LOG) lands in [1, ...);
    // log2 of that gives the chunk index.
    std::size_t shifted = (idx >> INITIAL_LOG) + 1;
    // floor(log2(shifted)).
    std::size_t k = 0;
    std::size_t s = shifted;
    while (s > 1) { s >>= 1; ++k; }
    std::size_t first_in_chunk =
        ((std::size_t(1) << k) - 1) << INITIAL_LOG;
    return {k, idx - first_in_chunk};
  }

  void reset() {
    for (std::size_t k = 0; k < NUM_CHUNKS; ++k) {
      T* c = chunks_[k].load(std::memory_order_relaxed);
      if (c) {
        delete[] c;
        chunks_[k].store(nullptr, std::memory_order_relaxed);
      }
    }
    size_.store(0, std::memory_order_relaxed);
  }

  void copy_from(const ConcurrentAppendVec& other) {
    std::size_t n = other.size();
    size_.store(n, std::memory_order_relaxed);
    for (std::size_t k = 0; k < NUM_CHUNKS; ++k) {
      T* src = other.chunks_[k].load(std::memory_order_acquire);
      if (src == nullptr) {
        chunks_[k].store(nullptr, std::memory_order_relaxed);
        continue;
      }
      // We allocate the full chunk regardless of how full it is in
      // `other`. The unwritten slots are default-constructed, which is
      // fine because at() is only ever called for idx < size().
      T* dst = new T[chunk_size(k)];
      // Copy only the valid slots (those before `n`).
      std::size_t first_in_chunk =
          ((std::size_t(1) << k) - 1) << INITIAL_LOG;
      std::size_t end = chunk_size(k);
      if (first_in_chunk + end > n) {
        end = (n > first_in_chunk) ? (n - first_in_chunk) : 0;
      }
      for (std::size_t i = 0; i < end; ++i) dst[i] = src[i];
      chunks_[k].store(dst, std::memory_order_release);
    }
  }

  std::atomic<std::size_t> size_{0};
  std::array<std::atomic<T*>, NUM_CHUNKS> chunks_{};
};

}  // namespace pe
