// Closure correctness test: builds small e-graphs by hand, runs both
// parallel_close and sequential_close_nelson on the same input, and
// verifies they agree on equivalence relations. Replaces the SMT-LIB
// regression test (which was deleted along with the CLI).

#include <cassert>
#include <cstdio>
#include <utility>

#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"

using namespace pe;

namespace {

#define EXPECT(cond)                                                      \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
      return false;                                                       \
    }                                                                     \
  } while (0)

// Build helper: each ENode's class id is its index in `nodes`.
parlay::sequence<ENode> make_nodes(std::initializer_list<ENode> ns) {
  parlay::sequence<ENode> out;
  out.reserve(ns.size());
  for (const auto& n : ns) out.push_back(n);
  return out;
}

// ---------------------------------------------------------------------------
// Simple congruence: f(a) and f(b) become equivalent when a == b.
//
//   class 0: a (leaf)
//   class 1: b (leaf)
//   class 2: f(a)        ← children {0}
//   class 3: f(b)        ← children {1}
// merge a = b => f(a) ≡ f(b)
// ---------------------------------------------------------------------------
bool test_basic_congruence() {
  auto nodes = make_nodes({
      ENode{"a", {}},          // 0
      ENode{"b", {}},          // 1
      ENode{"f", {0}},         // 2
      ENode{"f", {1}},         // 3
  });
  auto eg = EGraph::bulk_init(std::move(nodes));
  parlay::sequence<std::pair<Id, Id>> eqs;
  eqs.push_back({0, 1});  // a = b
  eg->parallel_close(std::move(eqs));

  EXPECT(eg->equiv(0, 1));   // a ≡ b
  EXPECT(eg->equiv(2, 3));   // f(a) ≡ f(b)  (the cascade)
  return true;
}

// ---------------------------------------------------------------------------
// Two-level cascade: g(f(a)) ≡ g(f(b)) requires propagation up two levels.
// ---------------------------------------------------------------------------
bool test_two_level_cascade() {
  auto nodes = make_nodes({
      ENode{"a", {}},          // 0
      ENode{"b", {}},          // 1
      ENode{"f", {0}},         // 2 = f(a)
      ENode{"f", {1}},         // 3 = f(b)
      ENode{"g", {2}},         // 4 = g(f(a))
      ENode{"g", {3}},         // 5 = g(f(b))
  });
  auto eg = EGraph::bulk_init(std::move(nodes));
  parlay::sequence<std::pair<Id, Id>> eqs;
  eqs.push_back({0, 1});
  eg->parallel_close(std::move(eqs));

  EXPECT(eg->equiv(0, 1));
  EXPECT(eg->equiv(2, 3));
  EXPECT(eg->equiv(4, 5));
  return true;
}

// ---------------------------------------------------------------------------
// Disjoint terms stay disjoint when no relevant union is asserted.
// ---------------------------------------------------------------------------
bool test_no_spurious_merges() {
  auto nodes = make_nodes({
      ENode{"a", {}},          // 0
      ENode{"b", {}},          // 1
      ENode{"c", {}},          // 2
      ENode{"f", {0}},         // 3 = f(a)
      ENode{"f", {1}},         // 4 = f(b)
      ENode{"g", {2}},         // 5 = g(c)
  });
  auto eg = EGraph::bulk_init(std::move(nodes));
  parlay::sequence<std::pair<Id, Id>> eqs;
  eqs.push_back({0, 1});  // only a = b
  eg->parallel_close(std::move(eqs));

  EXPECT(eg->equiv(0, 1));
  EXPECT(eg->equiv(3, 4));    // f(a) ≡ f(b)
  EXPECT(!eg->equiv(0, 2));   // a ≢ c
  EXPECT(!eg->equiv(3, 5));   // f(a) ≢ g(c)
  return true;
}

// ---------------------------------------------------------------------------
// Mixed arities: f1(a), f2(a, b) participate in distinct classes.
// Cross-arity initial union puts them in the same class but they don't
// trigger congruence with each other (different op + arity).
// ---------------------------------------------------------------------------
bool test_mixed_arity() {
  auto nodes = make_nodes({
      ENode{"a", {}},             // 0
      ENode{"b", {}},             // 1
      ENode{"c", {}},             // 2
      ENode{"f1", {0}},           // 3 = f1(a)       arity 1
      ENode{"f2", {0, 1}},        // 4 = f2(a, b)    arity 2
      ENode{"f1", {2}},           // 5 = f1(c)       arity 1
  });
  auto eg = EGraph::bulk_init(std::move(nodes));
  parlay::sequence<std::pair<Id, Id>> eqs;
  // Cross-arity: f1(a) = f2(a, b). Direct merge, no congruence cascade
  // because they have different signatures.
  eqs.push_back({3, 4});
  // Also: a = c → f1(a) ≡ f1(c) via congruence.
  eqs.push_back({0, 2});
  eg->parallel_close(std::move(eqs));

  EXPECT(eg->equiv(3, 4));    // initial union held
  EXPECT(eg->equiv(0, 2));    // initial union held
  EXPECT(eg->equiv(3, 5));    // f1(a) ≡ f1(c) via congruence
  EXPECT(eg->equiv(4, 5));    // transitively (f1(a) ≡ f2(a,b), f1(a) ≡ f1(c))
  return true;
}

// ---------------------------------------------------------------------------
// par_close and sequential_close_nelson agree on a randomly-shaped input.
// ---------------------------------------------------------------------------
bool test_par_vs_seq_agree() {
  // 6 leaves, plus binary + ternary fn nodes to exercise variable arity
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a", {}},        // 0
        ENode{"b", {}},        // 1
        ENode{"c", {}},        // 2
        ENode{"d", {}},        // 3
        ENode{"e", {}},        // 4
        ENode{"g", {}},        // 5
        ENode{"f", {0, 1}},    // 6 = f(a, b)
        ENode{"f", {2, 3}},    // 7 = f(c, d)
        ENode{"h", {0, 1, 4}}, // 8 = h(a, b, e)
        ENode{"h", {2, 3, 5}}, // 9 = h(c, d, g)
        ENode{"k", {6}},       // 10 = k(f(a,b))
        ENode{"k", {7}},       // 11 = k(f(c,d))
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 2});  // a = c
    eqs.push_back({1, 3});  // b = d
    eqs.push_back({4, 5});  // e = g
    return eqs;
  };

  auto eg_par = EGraph::bulk_init(build_nodes());
  eg_par->parallel_close(unions());

  auto eg_seq = EGraph::bulk_init(build_nodes());
  eg_seq->sequential_close_nelson(unions());

  // Both should derive: f(a,b) ≡ f(c,d), h(a,b,e) ≡ h(c,d,g),
  // k(f(a,b)) ≡ k(f(c,d)).
  for (Id i = 0; i < 12; ++i) {
    for (Id j = i + 1; j < 12; ++j) {
      bool p = eg_par->equiv(i, j);
      bool s = eg_seq->equiv(i, j);
      if (p != s) {
        std::fprintf(stderr,
                     "FAIL par/seq disagree on (%u, %u): par=%d seq=%d\n",
                     i, j, int(p), int(s));
        return false;
      }
    }
  }

  // And the expected equivalences hold:
  EXPECT(eg_par->equiv(6, 7));
  EXPECT(eg_par->equiv(8, 9));
  EXPECT(eg_par->equiv(10, 11));
  EXPECT(!eg_par->equiv(6, 8));   // f and h not equivalent
  return true;
}

}  // namespace

int main() {
  struct Case { const char* name; bool (*fn)(); };
  Case cases[] = {
      {"basic_congruence",   test_basic_congruence},
      {"two_level_cascade",  test_two_level_cascade},
      {"no_spurious_merges", test_no_spurious_merges},
      {"mixed_arity",        test_mixed_arity},
      {"par_vs_seq_agree",   test_par_vs_seq_agree},
  };

  int passed = 0;
  int total = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
  for (const auto& c : cases) {
    if (c.fn()) {
      std::printf("[ OK ] %s\n", c.name);
      ++passed;
    } else {
      std::printf("[FAIL] %s\n", c.name);
    }
  }
  std::printf("\n%d/%d closure tests passed\n", passed, total);
  return passed == total ? 0 : 1;
}
