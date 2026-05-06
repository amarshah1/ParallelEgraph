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
  return parlay::sequence<ENode>(ns.begin(), ns.end());
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
  auto eg = std::make_unique<ConcurrentEGraph>(std::move(nodes));
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
  auto eg = std::make_unique<ConcurrentEGraph>(std::move(nodes));
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
  auto eg = std::make_unique<ConcurrentEGraph>(std::move(nodes));
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
  auto eg = std::make_unique<ConcurrentEGraph>(std::move(nodes));
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
// ---------------------------------------------------------------------------
// sequential_close_topo agrees with Nelson on the synthetic benchmark
// shape (initial unions only between leaves, function nodes referencing
// only the level immediately below). Same hand-built input as
// par_vs_seq_agree, plus the topo path.
// ---------------------------------------------------------------------------
bool test_topo_agrees_with_nelson() {
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

  auto eg_nel = std::make_unique<SequentialEGraph>(build_nodes());
  eg_nel->sequential_close_nelson(unions());

  auto eg_top = std::make_unique<SequentialEGraph>(build_nodes());
  eg_top->sequential_close_topo(unions());

  for (Id i = 0; i < 12; ++i) {
    for (Id j = i + 1; j < 12; ++j) {
      bool n = eg_nel->equiv(i, j);
      bool t = eg_top->equiv(i, j);
      if (n != t) {
        std::fprintf(stderr,
                     "FAIL nelson/topo disagree on (%u, %u): nel=%d topo=%d\n",
                     i, j, int(n), int(t));
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// parallel_close_topo (depth-stratified BSP) agrees with Nelson on the
// same hand-built input as par_vs_seq_agree.
// ---------------------------------------------------------------------------
bool test_par_topo_agrees_with_nelson() {
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a", {}},        // 0
        ENode{"b", {}},        // 1
        ENode{"c", {}},        // 2
        ENode{"d", {}},        // 3
        ENode{"e", {}},        // 4
        ENode{"g", {}},        // 5
        ENode{"f", {0, 1}},    // 6
        ENode{"f", {2, 3}},    // 7
        ENode{"h", {0, 1, 4}}, // 8
        ENode{"h", {2, 3, 5}}, // 9
        ENode{"k", {6}},       // 10
        ENode{"k", {7}},       // 11
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 2});
    eqs.push_back({1, 3});
    eqs.push_back({4, 5});
    return eqs;
  };

  auto eg_nel = std::make_unique<SequentialEGraph>(build_nodes());
  eg_nel->sequential_close_nelson(unions());

  auto eg_pt = std::make_unique<ConcurrentEGraph>(build_nodes(), pe::topo);
  eg_pt->parallel_close_topo(unions());

  for (Id i = 0; i < 12; ++i) {
    for (Id j = i + 1; j < 12; ++j) {
      bool n = eg_nel->equiv(i, j);
      bool p = eg_pt->equiv(i, j);
      if (n != p) {
        std::fprintf(stderr,
                     "FAIL nelson/par_topo disagree on (%u, %u): nel=%d pt=%d\n",
                     i, j, int(n), int(p));
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Cross-depth initial unions can create intra-round congruence
// dependencies that depth-stratified BSP misses.
//
//   class 0 = a, 1 = b           (leaves)
//   class 2 = ta1, 3 = ta2, 4 = t (leaves used as proxies)
//   class 5 = f(a)               (depth 1, child = 0)
//   class 6 = f(b)               (depth 1, child = 1)
//   class 7 = g(ta1, t)          (depth 1, children = 2, 4)
//   class 8 = g(ta2, t)          (depth 1, children = 3, 4)
//
//   init: a = b, ta1 = f(a), ta2 = f(b)  (cross-depth unions)
//
// Nelson reasons sequentially: f(a) ≡ f(b) via a=b ⇒ classes 5 and 6
// merge ⇒ classes 2 and 3 transitively merge through their init-unions
// with 5 and 6 ⇒ g(ta1, t) ≡ g(ta2, t).
//
// par_topo computes all four depth-1 sigs at the start of round 1,
// before phase 2's union(5, 6) commits. So sig(7) reads find_root(2) =
// pre-union root of {2, 5}, and sig(8) reads find_root(3) = pre-union
// root of {3, 6}; those roots differ. The semisort buckets 7 and 8
// separately, no union. Round 1 ends; no more depths. Congruence missed.
// ---------------------------------------------------------------------------
bool test_par_topo_cross_depth_init() {
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a",  {}},          // 0
        ENode{"b",  {}},          // 1
        ENode{"ta1", {}},         // 2
        ENode{"ta2", {}},         // 3
        ENode{"t",  {}},          // 4
        ENode{"f", {0}},          // 5 = f(a)
        ENode{"f", {1}},          // 6 = f(b)
        ENode{"g", {2, 4}},       // 7 = g(ta1, t)
        ENode{"g", {3, 4}},       // 8 = g(ta2, t)
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 1});  // a = b
    eqs.push_back({2, 5});  // ta1 = f(a)   — cross-depth
    eqs.push_back({3, 6});  // ta2 = f(b)   — cross-depth
    return eqs;
  };

  // parallel_close uses the BSP frontier (`parents_`) and re-processes
  // affected classes after each round of unions, so it reaches the
  // mathematical closure on this input. We use it as the oracle.
  auto eg_par = std::make_unique<ConcurrentEGraph>(build_nodes());
  eg_par->parallel_close(unions());
  if (!eg_par->equiv(7, 8)) {
    std::fprintf(stderr,
                 "FAIL: parallel_close didn't derive g(ta1,t) ≡ g(ta2,t) "
                 "(test setup issue — closure should give 7 ≡ 8)\n");
    return false;
  }

  auto eg_pt = std::make_unique<ConcurrentEGraph>(build_nodes(), pe::topo);
  eg_pt->parallel_close_topo(unions());

  // Compare every equivalence class against the parallel_close oracle.
  for (Id i = 0; i < 9; ++i) {
    for (Id j = i + 1; j < 9; ++j) {
      bool oracle = eg_par->equiv(i, j);
      bool pt = eg_pt->equiv(i, j);
      if (oracle != pt) {
        std::fprintf(stderr,
                     "FAIL par_topo vs parallel_close on (%u, %u): "
                     "oracle=%d pt=%d\n",
                     i, j, int(oracle), int(pt));
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// `sequential_close_topo` is correct only when the input ordering happens
// to put any "dependency-creating" classes before classes that read those
// dependencies. The DAG-order invariant (child < parent) does not pin
// down ordering between independent same-depth classes, so adversarial
// orderings can still miss congruences from cross-depth inits.
//
// Same scenario as test_par_topo_cross_depth_init, with the g's placed
// BEFORE the f's: g(ta2,t)=5, g(ta1,t)=6, f(a)=7, f(b)=8. By the time
// the f-union fires (at class 8), classes 5 and 6 have already been
// inserted with stale sigs. The transitive merge of ta1's and ta2's
// classes is missed.
// ---------------------------------------------------------------------------
bool test_seq_topo_adversarial_order() {
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a",  {}},          // 0
        ENode{"b",  {}},          // 1
        ENode{"ta1", {}},         // 2
        ENode{"ta2", {}},         // 3
        ENode{"t",  {}},          // 4
        ENode{"g", {3, 4}},       // 5 = g(ta2, t)  — appears BEFORE f's
        ENode{"g", {2, 4}},       // 6 = g(ta1, t)
        ENode{"f", {0}},          // 7 = f(a)
        ENode{"f", {1}},          // 8 = f(b)
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 1});  // a = b
    eqs.push_back({2, 7});  // ta1 = f(a)   — cross-depth (note: f indices)
    eqs.push_back({3, 8});  // ta2 = f(b)   — cross-depth
    return eqs;
  };

  auto eg_par = std::make_unique<ConcurrentEGraph>(build_nodes());
  eg_par->parallel_close(unions());
  if (!eg_par->equiv(5, 6)) {
    std::fprintf(stderr,
                 "FAIL: parallel_close didn't derive g(ta1,t) ≡ g(ta2,t) "
                 "(test setup issue)\n");
    return false;
  }

  auto eg_st = std::make_unique<SequentialEGraph>(build_nodes());
  eg_st->sequential_close_topo(unions());

  for (Id i = 0; i < 9; ++i) {
    for (Id j = i + 1; j < 9; ++j) {
      bool oracle = eg_par->equiv(i, j);
      bool st = eg_st->equiv(i, j);
      if (oracle != st) {
        std::fprintf(stderr,
                     "FAIL seq_topo vs parallel_close on (%u, %u): "
                     "oracle=%d seq_topo=%d\n",
                     i, j, int(oracle), int(st));
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// `sequential_close_dst` is correct on the same cross-depth init scenario
// that breaks both topo paths. Same DAG as test_par_topo_cross_depth_init.
// ---------------------------------------------------------------------------
bool test_seq_dst_cross_depth() {
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a",  {}}, ENode{"b",  {}},
        ENode{"ta1", {}}, ENode{"ta2", {}}, ENode{"t",  {}},
        ENode{"f", {0}}, ENode{"f", {1}},
        ENode{"g", {2, 4}}, ENode{"g", {3, 4}},
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 1});
    eqs.push_back({2, 5});
    eqs.push_back({3, 6});
    return eqs;
  };

  auto eg_par = std::make_unique<ConcurrentEGraph>(build_nodes());
  eg_par->parallel_close(unions());

  auto eg_dst = std::make_unique<SequentialEGraph>(build_nodes());
  eg_dst->sequential_close_dst(unions());

  for (Id i = 0; i < 9; ++i) {
    for (Id j = i + 1; j < 9; ++j) {
      if (eg_par->equiv(i, j) != eg_dst->equiv(i, j)) {
        std::fprintf(stderr,
                     "FAIL seq_dst vs parallel_close on (%u, %u): "
                     "oracle=%d dst=%d\n",
                     i, j, int(eg_par->equiv(i, j)), int(eg_dst->equiv(i, j)));
        return false;
      }
    }
  }
  return true;
}

// Same input shape but with the g's reordered before the f's — the
// adversarial ordering that defeats sequential_close_topo. The dst path
// is order-independent and should still match parallel_close.
bool test_seq_dst_adversarial_order() {
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a",  {}}, ENode{"b",  {}},
        ENode{"ta1", {}}, ENode{"ta2", {}}, ENode{"t",  {}},
        ENode{"g", {3, 4}},  // 5 = g(ta2, t)
        ENode{"g", {2, 4}},  // 6 = g(ta1, t)
        ENode{"f", {0}},     // 7 = f(a)
        ENode{"f", {1}},     // 8 = f(b)
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 1});
    eqs.push_back({2, 7});
    eqs.push_back({3, 8});
    return eqs;
  };

  auto eg_par = std::make_unique<ConcurrentEGraph>(build_nodes());
  eg_par->parallel_close(unions());

  auto eg_dst = std::make_unique<SequentialEGraph>(build_nodes());
  eg_dst->sequential_close_dst(unions());

  for (Id i = 0; i < 9; ++i) {
    for (Id j = i + 1; j < 9; ++j) {
      if (eg_par->equiv(i, j) != eg_dst->equiv(i, j)) {
        std::fprintf(stderr,
                     "FAIL seq_dst (adversarial) vs parallel_close on "
                     "(%u, %u): oracle=%d dst=%d\n",
                     i, j, int(eg_par->equiv(i, j)), int(eg_dst->equiv(i, j)));
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// `sequential_close_topo_iter` should recover correctness on the same
// adversarial-order scenario that defeats single-pass `sequential_close_topo`.
// ---------------------------------------------------------------------------
bool test_seq_topo_iter_adversarial_order() {
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a",  {}}, ENode{"b",  {}},
        ENode{"ta1", {}}, ENode{"ta2", {}}, ENode{"t",  {}},
        ENode{"g", {3, 4}},  // 5 = g(ta2, t)
        ENode{"g", {2, 4}},  // 6 = g(ta1, t)
        ENode{"f", {0}},     // 7 = f(a)
        ENode{"f", {1}},     // 8 = f(b)
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 1});
    eqs.push_back({2, 7});
    eqs.push_back({3, 8});
    return eqs;
  };

  auto eg_par = std::make_unique<ConcurrentEGraph>(build_nodes());
  eg_par->parallel_close(unions());

  auto eg_iter = std::make_unique<SequentialEGraph>(build_nodes());
  eg_iter->sequential_close_topo_iter(unions());

  for (Id i = 0; i < 9; ++i) {
    for (Id j = i + 1; j < 9; ++j) {
      if (eg_par->equiv(i, j) != eg_iter->equiv(i, j)) {
        std::fprintf(stderr,
                     "FAIL seq_topo_iter vs parallel_close on (%u, %u): "
                     "oracle=%d iter=%d\n",
                     i, j, int(eg_par->equiv(i, j)),
                     int(eg_iter->equiv(i, j)));
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// `parallel_close_async_rounds` should match `parallel_close` on the
// cross-depth init scenario — confirms the async-rounds variant doesn't
// inherit the same depth-stratification bug as `parallel_close_topo`.
// ---------------------------------------------------------------------------
bool test_par_async_cross_depth() {
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a",  {}}, ENode{"b",  {}},
        ENode{"ta1", {}}, ENode{"ta2", {}}, ENode{"t",  {}},
        ENode{"f", {0}}, ENode{"f", {1}},
        ENode{"g", {2, 4}}, ENode{"g", {3, 4}},
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 1});
    eqs.push_back({2, 5});
    eqs.push_back({3, 6});
    return eqs;
  };

  auto eg_par = std::make_unique<ConcurrentEGraph>(build_nodes());
  eg_par->parallel_close(unions());

  auto eg_async = std::make_unique<ConcurrentEGraph>(build_nodes(), pe::async);
  eg_async->parallel_close_async_rounds(unions());

  for (Id i = 0; i < 9; ++i) {
    for (Id j = i + 1; j < 9; ++j) {
      if (eg_par->equiv(i, j) != eg_async->equiv(i, j)) {
        std::fprintf(stderr,
                     "FAIL par_async vs parallel_close on (%u, %u): "
                     "oracle=%d async=%d\n",
                     i, j, int(eg_par->equiv(i, j)),
                     int(eg_async->equiv(i, j)));
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// `parallel_close_async_rounds_min_id` should match `parallel_close` on
// the cross-depth init scenario — confirms the MIN_ID variant doesn't
// regress correctness when classes get merged across depth boundaries.
// ---------------------------------------------------------------------------
bool test_par_async_min_cross_depth() {
  auto build_nodes = []() {
    return make_nodes({
        ENode{"a",  {}}, ENode{"b",  {}},
        ENode{"ta1", {}}, ENode{"ta2", {}}, ENode{"t",  {}},
        ENode{"f", {0}}, ENode{"f", {1}},
        ENode{"g", {2, 4}}, ENode{"g", {3, 4}},
    });
  };
  auto unions = []() {
    parlay::sequence<std::pair<Id, Id>> eqs;
    eqs.push_back({0, 1});
    eqs.push_back({2, 5});
    eqs.push_back({3, 6});
    return eqs;
  };

  auto eg_par = std::make_unique<ConcurrentEGraph>(build_nodes());
  eg_par->parallel_close(unions());

  auto eg_min = std::make_unique<ConcurrentEGraph>(build_nodes(), pe::async);
  eg_min->parallel_close_async_rounds_min_id(unions());

  for (Id i = 0; i < 9; ++i) {
    for (Id j = i + 1; j < 9; ++j) {
      if (eg_par->equiv(i, j) != eg_min->equiv(i, j)) {
        std::fprintf(stderr,
                     "FAIL par_async_min vs parallel_close on (%u, %u): "
                     "oracle=%d min=%d\n",
                     i, j, int(eg_par->equiv(i, j)),
                     int(eg_min->equiv(i, j)));
        return false;
      }
    }
  }
  return true;
}

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

  auto eg_par = std::make_unique<ConcurrentEGraph>(build_nodes());
  eg_par->parallel_close(unions());

  auto eg_seq = std::make_unique<SequentialEGraph>(build_nodes());
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

// ---------------------------------------------------------------------------
// Regression: parent-class merged with descendant. Asserting f(a,b) = a
// drops the f node into the same class as its own children. After
// parallel_consolidate, parents_[R] = [class_of_f, class_of_f] (one ref
// from each child slot), every CanonEntry in the resulting hash group
// shares root R, and dnc_union is a no-op. Without the all_same_root
// skip in semisort_common.hpp, next_work = [R] each round forever.
// ---------------------------------------------------------------------------
bool test_self_merge_no_loop() {
  auto nodes = make_nodes({
      ENode{"a", {}},          // 0
      ENode{"b", {}},          // 1
      ENode{"f", {0, 1}},      // 2 = f(a, b) — parent of both children
  });
  auto eg = std::make_unique<ConcurrentEGraph>(std::move(nodes));
  parlay::sequence<std::pair<Id, Id>> eqs;
  eqs.push_back({0, 1});  // a = b
  eqs.push_back({2, 0});  // f(a, b) = a  — collapses parent into children
  eg->parallel_close(std::move(eqs));  // must not loop

  EXPECT(eg->equiv(0, 1));
  EXPECT(eg->equiv(0, 2));
  EXPECT(eg->equiv(1, 2));
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
      {"topo_agrees",        test_topo_agrees_with_nelson},
      {"par_topo_agrees",    test_par_topo_agrees_with_nelson},
      {"par_topo_cross_depth", test_par_topo_cross_depth_init},
      {"seq_topo_adversarial", test_seq_topo_adversarial_order},
      {"seq_dst_cross_depth", test_seq_dst_cross_depth},
      {"seq_dst_adversarial", test_seq_dst_adversarial_order},
      {"seq_topo_iter_adversarial", test_seq_topo_iter_adversarial_order},
      {"par_async_cross_depth", test_par_async_cross_depth},
      {"par_async_min_cross_depth", test_par_async_min_cross_depth},
      {"self_merge_no_loop", test_self_merge_no_loop},
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
