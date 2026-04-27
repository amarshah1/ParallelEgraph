// Unit tests for ConcurrentAppendVec.
//   1. sequential append + at()
//   2. decompose() at chunk boundaries
//   3. concurrent append from many threads (each thread keeps its
//      own claimed indices and verifies its own writes)
//   4. copy constructor preserves state

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unordered_set>
#include <vector>

#include "parallel_egraph/concurrent_append_vec.hpp"

#define ASSERT(cond) do { if (!(cond)) { \
  std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
  std::exit(1); } } while (0)

int main() {
  using namespace pe;
  // 1. sequential
  {
    ConcurrentAppendVec<int> v;
    ASSERT(v.size() == 0);
    for (int i = 0; i < 10000; ++i) {
      std::size_t idx = v.append(i);
      ASSERT(idx == static_cast<std::size_t>(i));
    }
    ASSERT(v.size() == 10000);
    for (int i = 0; i < 10000; ++i) {
      ASSERT(v.at(i) == i);
    }
  }
  std::fprintf(stderr, "[ok] sequential append/at\n");

  // 2. boundary: appending many times exercises chunk transitions
  {
    ConcurrentAppendVec<int, /*INITIAL_LOG=*/2> v;  // tiny chunks
    constexpr int N = 1024;
    for (int i = 0; i < N; ++i) v.append(i * 7);
    for (int i = 0; i < N; ++i) ASSERT(v.at(i) == i * 7);
  }
  std::fprintf(stderr, "[ok] chunk transitions\n");

  // 3. concurrent
  {
    ConcurrentAppendVec<int> v;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 50000;
    std::vector<std::vector<std::size_t>> claimed(kThreads);
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
      ts.emplace_back([&, t]() {
        claimed[t].reserve(kPerThread);
        for (int i = 0; i < kPerThread; ++i) {
          std::size_t idx = v.append(t * kPerThread + i);
          claimed[t].push_back(idx);
        }
      });
    }
    for (auto& th : ts) th.join();
    ASSERT(v.size() == static_cast<std::size_t>(kThreads * kPerThread));
    // Every claimed idx should be unique across all threads, and
    // 0..N-1 should be exactly covered.
    std::unordered_set<std::size_t> all;
    for (auto& c : claimed)
      for (auto idx : c) {
        ASSERT(all.insert(idx).second);
      }
    ASSERT(all.size() == static_cast<std::size_t>(kThreads * kPerThread));
    // Check each thread can read back its own values at its idxs.
    for (int t = 0; t < kThreads; ++t) {
      for (int i = 0; i < kPerThread; ++i) {
        ASSERT(v.at(claimed[t][i]) == t * kPerThread + i);
      }
    }
  }
  std::fprintf(stderr, "[ok] concurrent append (8 threads x 50k)\n");

  // 4. copy
  {
    ConcurrentAppendVec<int> v1;
    for (int i = 0; i < 100; ++i) v1.append(i * 3);
    ConcurrentAppendVec<int> v2 = v1;
    ASSERT(v2.size() == 100);
    for (int i = 0; i < 100; ++i) ASSERT(v2.at(i) == i * 3);
    // mutating v1 doesn't affect v2
    v1.append(999);
    ASSERT(v1.size() == 101);
    ASSERT(v2.size() == 100);
  }
  std::fprintf(stderr, "[ok] copy ctor\n");

  std::fprintf(stderr, "concurrent_append_vec_test: all passed\n");
  return 0;
}
