// Reproduction test for a suspected bug in sequential_close_nelson:
// `sig_table` accumulates entries indefinitely, but a node's sig can
// shift when a later union changes one of its children's roots. The
// stale entry sits at the old hash forever, so a sibling with the new
// sig fails to find it.
//
// Construction (all hashcons-deduped):
//   0: a, 1: b, 2: c
//   4: f(0)        parents_[0] += {4}
//   5: f(2)        parents_[2] += {5}
//   6: g(4)        parents_[4] += {6}
//   7: g(1)        parents_[1] += {7}
//   8: g(5)        parents_[5] += {8}
//
// initial_unions: (0, 2), (4, 1)
//
// Expected closure:
//   0≡2 (initial)
//   4≡1 (initial, direct)
//   4≡5 (congruence: f(0) and f(2), since 0≡2)
//   6≡7 (congruence: g(4) and g(1), since 4≡1)
//   6≡8 (congruence: g(4) and g(5), since 4≡5)  <-- this is what the bug misses
//
// All of {6, 7, 8} should be in one class, and that class should equal
// {1, 4, 5}.
//
// Compare against parallel_close which is independently correct.

#include <cstdio>
#include "parallel_egraph/egraph.hpp"

using namespace pe;

static parlay::sequence<ENode> nodes() {
  parlay::sequence<ENode> ns;
  ns.push_back(ENode{"a", {}});       // 0
  ns.push_back(ENode{"b", {}});       // 1
  ns.push_back(ENode{"c", {}});       // 2
  ns.push_back(ENode{"_pad_", {}});   // 3 — pad, unused (just to make 4 line up)
  ns.push_back(ENode{"f", {0}});      // 4 = f(a)
  ns.push_back(ENode{"f", {2}});      // 5 = f(c)
  ns.push_back(ENode{"g", {4}});      // 6 = g(f(a))
  ns.push_back(ENode{"g", {1}});      // 7 = g(b)
  ns.push_back(ENode{"g", {5}});      // 8 = g(f(c))
  return ns;
}

static parlay::sequence<std::pair<Id, Id>> unions() {
  parlay::sequence<std::pair<Id, Id>> eqs;
  eqs.push_back({0, 2});
  eqs.push_back({4, 1});
  return eqs;
}

int main() {
  // BSP path is independently correct; use it as the oracle.
  auto eg_par = std::make_unique<ConcurrentEGraph>(nodes());
  eg_par->parallel_close(unions());

  // Sequential path under test.
  auto eg_seq = std::make_unique<SequentialEGraph>(nodes());
  eg_seq->sequential_close_nelson(unions());

  std::puts("=== par_close (oracle) classes ===");
  for (Id i = 0; i < 9; ++i) {
    if (i == 3) continue;
    std::printf("  %u → root %u\n", i, eg_par->uf().find_root(i));
  }
  std::puts("=== sequential_close_nelson classes ===");
  for (Id i = 0; i < 9; ++i) {
    if (i == 3) continue;
    std::printf("  %u → root %u\n", i, eg_seq->uf().find_root(i));
  }

  // Specifically check the alleged-missed congruence.
  bool par_678_one_class =
      eg_par->equiv(6, 7) && eg_par->equiv(7, 8);
  bool seq_678_one_class =
      eg_seq->equiv(6, 7) && eg_seq->equiv(7, 8);
  std::printf("par: 6≡7≡8? %s\n", par_678_one_class ? "YES" : "NO");
  std::printf("seq: 6≡7≡8? %s\n", seq_678_one_class ? "YES" : "NO");

  if (par_678_one_class && !seq_678_one_class) {
    std::puts("\nBUG CONFIRMED: par_close says 6≡7≡8, but "
              "sequential_close_nelson does not.");
    return 1;
  }
  if (par_678_one_class && seq_678_one_class) {
    std::puts("\nNo bug observed. Both say 6≡7≡8.");
    return 0;
  }
  std::puts("\nUnexpected: par_close itself says 6 NOT≡ 7≡8. Test setup wrong.");
  return 2;
}
