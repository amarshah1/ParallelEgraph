// 1:1 port of the #[test] items in src/unionfind.rs.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <parlay/parallel.h>

#include "parallel_egraph/unionfind.hpp"

using namespace pe;

namespace {

int failures = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__,     \
                   __LINE__, #cond);                                    \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

void basic_union_find() {
  ConcurrentUnionFind uf(3);
  Id a = uf.make_set();
  Id b = uf.make_set();
  Id c = uf.make_set();

  CHECK(!uf.same_set(a, b));
  CHECK(!uf.same_set(b, c));

  uf.union_(a, b);
  CHECK(uf.same_set(a, b));
  CHECK(!uf.same_set(a, c));

  uf.union_(b, c);
  CHECK(uf.same_set(a, c));
}

void find_returns_consistent_root() {
  constexpr std::size_t N = 10;
  ConcurrentUnionFind uf(N);
  std::vector<Id> ids;
  for (std::size_t i = 0; i < N; ++i) ids.push_back(uf.make_set());

  for (std::size_t i = 0; i + 1 < N; ++i) {
    uf.union_(ids[i], ids[i + 1]);
  }

  Id root = uf.find_root(ids[0]);
  for (Id id : ids) {
    CHECK(uf.find_root(id) == root);
  }
}

void concurrent_unions() {
  constexpr std::size_t N = 1000;
  ConcurrentUnionFind uf(N);
  for (std::size_t i = 0; i < N; ++i) uf.make_set();

  // Union even neighbours: (0,2), (2,4), ...
  std::vector<std::pair<Id, Id>> pairs;
  for (std::size_t i = 0; i + 2 < N; i += 2) {
    pairs.emplace_back(static_cast<Id>(i), static_cast<Id>(i + 2));
  }

  parlay::parallel_for(0, pairs.size(),
                       [&](std::size_t k) {
                         uf.union_(pairs[k].first, pairs[k].second);
                       });

  // All evens share a root.
  for (std::size_t i = 0; i < N; i += 2) {
    CHECK(uf.same_set(0, static_cast<Id>(i)));
  }
  // Odds are separate from evens.
  CHECK(!uf.same_set(0, 1));
}

void sequential_union_find() {
  constexpr std::size_t N = 8;
  SequentialUnionFind uf(N);
  // Chain: 0-1, 2-3, ..., 6-7
  for (std::size_t i = 0; i + 1 < N; i += 2) {
    uf.union_(static_cast<Id>(i), static_cast<Id>(i + 1));
  }
  CHECK(uf.find_root(0) == uf.find_root(1));
  CHECK(uf.find_root(2) == uf.find_root(3));
  CHECK(uf.find_root(0) != uf.find_root(2));
  uf.union_(0, 2);
  CHECK(uf.find_root(0) == uf.find_root(3));
}

}  // namespace

int main() {
  basic_union_find();
  find_returns_consistent_root();
  concurrent_unions();
  sequential_union_find();

  if (failures == 0) {
    std::puts("unionfind_test: PASS");
    return 0;
  }
  std::fprintf(stderr, "unionfind_test: %d FAILURES\n", failures);
  return 1;
}
