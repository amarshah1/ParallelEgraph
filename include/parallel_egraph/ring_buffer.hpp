#pragma once
// Fixed-capacity FIFO ring buffer for single-threaded use. Backed by a
// std::vector<T> sized at construction time; push/pop are pointer-bump
// + wrap. The caller is responsible for never exceeding the declared
// capacity (no resize, no overflow check on push beyond a debug
// assert).
//
// Used by `sequential_close_simple` as the worklist: its `in_queue`
// flag guarantees each node is enqueued at most once at any time, so
// a capacity equal to the node count is sufficient and the buffer
// can't overflow by construction.

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

namespace pe {

template <typename T>
class RingBuffer {
 public:
  explicit RingBuffer(std::size_t capacity)
      : buf_(capacity), cap_(capacity) {
    assert(capacity > 0 && "RingBuffer needs positive capacity");
  }

  std::size_t capacity() const noexcept { return cap_; }
  std::size_t size()     const noexcept { return count_; }
  bool        empty()    const noexcept { return count_ == 0; }

  // Append at the tail. Undefined behavior if `size() == capacity()`
  // (assert in debug). Callers should size capacity to a known upper
  // bound on simultaneously-live entries.
  void push(T x) {
    assert(count_ < cap_ && "RingBuffer overflow");
    buf_[tail_] = std::move(x);
    if (++tail_ == cap_) tail_ = 0;
    ++count_;
  }

  // Take from the head. Undefined behavior if `empty()` (assert in
  // debug). Use `empty()` / `size() > 0` to guard at the call site.
  T pop() {
    assert(count_ > 0 && "RingBuffer underflow");
    T x = std::move(buf_[head_]);
    if (++head_ == cap_) head_ = 0;
    --count_;
    return x;
  }

 private:
  std::vector<T>   buf_;
  std::size_t      cap_;
  std::size_t      head_  = 0;  // pop position
  std::size_t      tail_  = 0;  // push position
  std::size_t      count_ = 0;
};

}  // namespace pe
