// In-process port of gen_bench.py's synthetic families. Builds the same
// DAGs directly into an EGraph via its bulk ctor (skipping the SMT-LIB
// parse/build path) and times sequential_close_nelson vs parallel_parents on
// the resulting equality batch. Only closure time is measured — graph
// construction is amortized out of the hot loop the same way
// closure_compare.cpp does it.
//
// Supported families: chain, grid, cube, quartic, quintic. (exp is skipped
// per user request.)
//
// Env knobs:
//   PE_SYNTH_FAMILIES=chain,grid,...   restrict families (default: all five)
//   PE_SYNTH_NS=5,10,20                workload sizes per family (default: 5,10,20)
//   PE_BENCH_SKIP_NELSON=1             skip sequential baseline
//   PE_BENCH_FORMAT=csv                emit one row per (family, n, algo, trial)
//   PE_BENCH_HEADER=1                  emit CSV header (gated for appended runs)
//   PE_DNC_CUTOFF                      tagged into CSV for plot grouping
//   PE_SYNTH_DUMP=1                    print every generated node + initial
//                                      union to stderr, then exit (no timing).
//                                      Useful for sanity-checking what's in
//                                      the e-graph before closure.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"

using namespace pe;

namespace {

// Compile-time defaults; overridable at runtime via PE_BENCH_TRIALS /
// PE_BENCH_WARMUP. Defaults are 1 warmup + 5 trials per the harness's
// statistical-noise-vs-runtime tradeoff; override to 0+1 for big sweeps
// where you'd rather do many more configurations than re-time each one.
constexpr int DEFAULT_TRIALS = 5;
constexpr int DEFAULT_WARMUP = 1;

enum class Family { Chain, Grid, Cube, Quartic, Quintic, MixedDepth };

const char* family_name(Family f) {
  switch (f) {
    case Family::Chain:      return "chain";
    case Family::Grid:       return "grid";
    case Family::Cube:       return "cube";
    case Family::Quartic:    return "quartic";
    case Family::Quintic:    return "quintic";
    case Family::MixedDepth: return "mixed_depth";
  }
  return "?";
}

bool parse_family(const std::string& s, Family& out) {
  if (s == "chain")       { out = Family::Chain;      return true; }
  if (s == "grid")        { out = Family::Grid;       return true; }
  if (s == "cube")        { out = Family::Cube;       return true; }
  if (s == "quartic")     { out = Family::Quartic;    return true; }
  if (s == "quintic")     { out = Family::Quintic;    return true; }
  if (s == "mixed_depth") { out = Family::MixedDepth; return true; }
  return false;
}

template <typename UF>
struct BuiltGraph {
  std::unique_ptr<EGraph<UF>> eg;
  parlay::sequence<std::pair<Id, Id>> eqs;
  std::size_t n_classes;
  std::size_t n_eqs;
};

// ---- Workload generation --------------------------------------------------
// Each generator lays out nodes in DAG order (children before parents).
// Class id == position in the sequence, which we exploit when constructing
// the equality batch. Generators return raw (nodes, eqs) so we can either
// hand them to the EGraph ctor (for benchmarking) or print them (for dumping).

struct Workload {
  parlay::sequence<ENode> nodes;
  parlay::sequence<std::pair<Id, Id>> eqs;
};

// chain: a0,b0 leaves; for i in 1..n, a_i = f(a_{i-1}), b_i = f(b_{i-1}).
// Layout: [a0, b0, a1, b1, ..., a_n, b_n]. Initial union: (a0, b0).
Workload gen_chain(std::size_t n) {
  const std::size_t total = 2 * (n + 1);
  auto nodes = parlay::tabulate(total, [&](std::size_t i) -> ENode {
    const std::size_t lvl  = i / 2;
    const bool is_b        = (i % 2) == 1;
    if (lvl == 0) {
      return ENode{is_b ? std::string("b0") : std::string("a0"), {}};
    }
    Id parent = static_cast<Id>(2 * (lvl - 1) + (is_b ? 1 : 0));
    return ENode{std::string("f"), {parent}};
  });
  parlay::sequence<std::pair<Id, Id>> eqs;
  eqs.push_back({Id{0}, Id{1}});
  return {std::move(nodes), std::move(eqs)};
}

// Polynomial-arity family: 2n leaves + 2 * n^arity f-applications +
// a balanced binary `g`-tree wrapper on each side, mirroring
// gen_bench.py's nest_balanced("g", ...) construction.
//
// Layout (children-before-parents, required by the EGraph ctor):
//   [a_0 .. a_{n-1}, b_0 .. b_{n-1},                        # 2n leaves
//    f(a-tuple_0) .. f(a-tuple_{m-1}),                       # m  a-side f-apps
//    f(b-tuple_0) .. f(b-tuple_{m-1}),                       # m  b-side f-apps
//    g-internal-nodes for a-side (m-1 of them, bottom-up),   # m-1 g-nodes
//    g-internal-nodes for b-side (m-1 of them, bottom-up)]   # m-1 g-nodes
// where m = n^arity.
//
// Initial unions: a_i = b_i for i in [0, n). With the g-tree in place,
// closure has to propagate congruences from the leaves all the way up
// to the two g-tree roots — log_2(m) extra rounds beyond the f-layer
// fan-out. This matches what running the SMT-LIB synthetic_benchmarks/
// files through smt_bench would do, just without the parse/build cost.

// Build a balanced d-ary g-tree over the f-app id range [lo, hi). Each
// internal g-node consumes up to `d` children (the last bucket may be
// short when len isn't a multiple of d). For len=1 the "tree" is the
// single f-app id itself (no g-node added). Returns the class id of the
// root g-node, or the sole f-app id when the range has size 1.
//
// Round behavior on this shape: when level-i has m_i nodes touched, the
// next level has m_i/d nodes (roughly), so frontier shrinks by d each
// round and total g-tree depth is ⌈log_d(m)⌉.
//
// Appends new g-nodes to `out` and assigns them ids starting at
// `next_id` (which is the current end of the workload sequence).
Id emit_g_tree(parlay::sequence<ENode>& out, Id& next_id,
               std::size_t lo, std::size_t hi, std::size_t d) {
  const std::size_t len = hi - lo;
  if (len == 1) {
    return static_cast<Id>(lo);
  }
  if (len <= d) {
    // Bottom of the tree: one g-node consuming `len` direct f-app ids.
    std::vector<Id> children(len);
    for (std::size_t i = 0; i < len; ++i) {
      children[i] = static_cast<Id>(lo + i);
    }
    Id node_id = next_id++;
    out.push_back(ENode{std::string("g"), std::move(children)});
    return node_id;
  }
  // Recursive case: split [lo, hi) into d roughly-equal chunks. Emit a
  // subtree per chunk, then a single g-node above them with the d
  // subtree-root ids as children.
  const std::size_t per_chunk = (len + d - 1) / d;
  std::vector<Id> children;
  children.reserve(d);
  for (std::size_t c_lo = lo; c_lo < hi; c_lo += per_chunk) {
    const std::size_t c_hi = std::min(c_lo + per_chunk, hi);
    children.push_back(emit_g_tree(out, next_id, c_lo, c_hi, d));
  }
  Id node_id = next_id++;
  out.push_back(ENode{std::string("g"), std::move(children)});
  return node_id;
}

// Total number of internal g-nodes in a balanced d-ary tree built by
// emit_g_tree over `m` leaves. Used to reserve storage up front. d≥2.
// Closed form: nodes_per_level summed = m/d + m/d² + ... + 1, but
// emit_g_tree's chunking sometimes leaves an extra parent (e.g. d=2
// over 5 leaves), so we just simulate the level reduction.
std::size_t count_g_internal(std::size_t m, std::size_t d) {
  if (m < 2) return 0;
  std::size_t total = 0;
  std::size_t cur = m;
  while (cur > 1) {
    const std::size_t next = (cur + d - 1) / d;
    total += next;
    cur = next;
  }
  return total;
}

// `g_arity`: fan-in of each internal g-node. d=2 is the original binary
// tree; larger d gives shallower trees (depth ⌈log_d(per_side)⌉) and
// reduces frontier by a factor of d each round.
Workload gen_poly(std::size_t n, std::size_t arity, std::size_t g_arity) {
  std::size_t per_side = 1;
  for (std::size_t k = 0; k < arity; ++k) per_side *= n;
  const std::size_t leaves    = 2 * n;
  const std::size_t f_total   = leaves + 2 * per_side;
  // Per-side g-tree size depends on d (the g-arity). We compute it
  // exactly by simulating emit_g_tree's level reduction; this matches
  // what gets pushed into `nodes` below.
  const std::size_t g_per_side = count_g_internal(per_side, g_arity);
  const std::size_t total = f_total + 2 * g_per_side;

  // f-layer: leaves + f-apps, all positions deterministic, fully parallel.
  auto nodes = parlay::tabulate(f_total, [&](std::size_t i) -> ENode {
    if (i < n) {
      return ENode{std::string("a") + std::to_string(i), {}};
    }
    if (i < 2 * n) {
      return ENode{std::string("b") + std::to_string(i - n), {}};
    }
    const std::size_t off = i - leaves;
    const bool is_b       = off >= per_side;
    const std::size_t idx = is_b ? off - per_side : off;
    std::vector<Id> children(arity);
    std::size_t rem = idx;
    for (std::size_t d = arity; d-- > 0;) {
      const std::size_t digit = rem % n;
      rem /= n;
      children[d] = static_cast<Id>((is_b ? n : 0) + digit);
    }
    return ENode{std::string("f"), std::move(children)};
  });

  // g-tree appended sequentially. Each side's recursion is self-contained
  // and order-of-emission is bottom-up (children before parents), so
  // The EGraph ctor's child<parent invariant holds. The work is O(m) total —
  // small relative to the f-layer for the workload sizes we run.
  nodes.reserve(total);
  Id next_id = static_cast<Id>(f_total);
  if (g_per_side > 0) {
    const std::size_t a_lo = leaves;
    const std::size_t a_hi = leaves + per_side;
    const std::size_t b_lo = a_hi;
    const std::size_t b_hi = a_hi + per_side;
    emit_g_tree(nodes, next_id, a_lo, a_hi, g_arity);
    emit_g_tree(nodes, next_id, b_lo, b_hi, g_arity);
  }

  auto eqs = parlay::tabulate(n, [&](std::size_t i) {
    return std::pair<Id, Id>{static_cast<Id>(i), static_cast<Id>(n + i)};
  });

  return {std::move(nodes), std::move(eqs)};
}

// mixed_depth: cube-shape DAG (arity-3 f-apps + balanced d-ary g-tree)
// with extra cross-depth initial unions on top of the standard
// a_i = b_i leaf-leaf pairs. Designed to stress closure algorithms
// whose convergence relies on a topological order of inputs:
// par_topo_iter's outer fixpoint loop has to redo all depths on a
// pass when a cross-depth merge happened, while async/BSP don't care.
//
// Layout is identical to gen_poly(n, 3, g_arity); on top of the
// returned eqs (n leaf-leaf pairs), we append `n` cross-depth pairs.
// Each cross-depth pair = (random leaf, random g-tree class). Both
// endpoints are valid class ids in nodes_; the leaf is at depth 0
// and the g-tree class is at some depth >= 2, so they're guaranteed
// non-same-depth.
//
// Determinism: the seed is fixed (seeded from n + a magic constant)
// so successive runs with the same n produce the same workload.
Workload gen_mixed_depth(std::size_t n, std::size_t g_arity) {
  // Build the cube base.
  Workload w = gen_poly(n, 3, g_arity);

  // Class-id ranges in the gen_poly layout:
  //   [0, 2n)         : leaves (depth 0)
  //   [2n, 2n + 2n^3) : f-apps (depth 1)
  //   [2n + 2n^3, total) : g-tree internal nodes (depth >= 2)
  const std::size_t per_side = n * n * n;          // n^3 f-apps per side
  const std::size_t leaves_end = 2 * n;
  const std::size_t f_total = leaves_end + 2 * per_side;
  const std::size_t total = w.nodes.size();
  const std::size_t g_count = total - f_total;     // g-tree classes

  if (g_count == 0) {
    // Degenerate (n is too small for g-tree to exist) — return base
    // workload unchanged. mixed_depth at n=1 has nothing extra to do.
    return w;
  }

  // Append n cross-depth unions: one leaf + one g-tree class each.
  // 64-bit splitmix RNG seeded from n; deterministic per workload.
  std::uint64_t seed = 0xC0FFEE'BADC0DEull ^ (static_cast<std::uint64_t>(n) << 16);
  auto next_rand = [&]() {
    seed += 0x9E37'79B9'7F4A'7C15ull;
    std::uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return z;
  };

  for (std::size_t i = 0; i < n; ++i) {
    Id leaf = static_cast<Id>(next_rand() % leaves_end);
    Id g_cls = static_cast<Id>(f_total + (next_rand() % g_count));
    w.eqs.push_back({leaf, g_cls});
  }
  return w;
}

// `g_arity` (a.k.a. d, the decomposition rate): fan-in of each internal
// g-node in the polynomial families' wrapper. Ignored for chain (no
// g-tree). d=2 is the historical default (balanced binary tree).
Workload gen_workload(Family f, std::size_t n, std::size_t g_arity) {
  switch (f) {
    case Family::Chain:      return gen_chain(n);
    case Family::Grid:       return gen_poly(n, 2, g_arity);
    case Family::Cube:       return gen_poly(n, 3, g_arity);
    case Family::Quartic:    return gen_poly(n, 4, g_arity);
    case Family::Quintic:    return gen_poly(n, 5, g_arity);
    case Family::MixedDepth: return gen_mixed_depth(n, g_arity);
  }
  std::abort();
}

template <typename UF>
BuiltGraph<UF> build(Family f, std::size_t n, std::size_t g_arity) {
  auto w = gen_workload(f, n, g_arity);
  const std::size_t total = w.nodes.size();
  const std::size_t n_eqs = w.eqs.size();
  auto eg = std::make_unique<EGraph<UF>>(std::move(w.nodes));
  return {std::move(eg), std::move(w.eqs), total, n_eqs};
}

// Async-flavor build: workload generation identical, EGraph via the
// async-tagged ctor (populates last_marked_, skips parents_).
template <typename UF>
BuiltGraph<UF> build_async(Family f, std::size_t n, std::size_t g_arity) {
  auto w = gen_workload(f, n, g_arity);
  const std::size_t total = w.nodes.size();
  const std::size_t n_eqs = w.eqs.size();
  auto eg = std::make_unique<EGraph<UF>>(std::move(w.nodes), pe::filter);
  return {std::move(eg), std::move(w.eqs), total, n_eqs};
}

// Topo-flavor build: workload generation identical, EGraph via the
// topo-tagged ctor (populates depth_buckets_, skips parents_/last_marked_).
template <typename UF>
BuiltGraph<UF> build_topo(Family f, std::size_t n, std::size_t g_arity) {
  auto w = gen_workload(f, n, g_arity);
  const std::size_t total = w.nodes.size();
  const std::size_t n_eqs = w.eqs.size();
  auto eg = std::make_unique<EGraph<UF>>(std::move(w.nodes), pe::topo);
  return {std::move(eg), std::move(w.eqs), total, n_eqs};
}

// Naive-flavor build: workload generation identical, EGraph via the
// naive-tagged ctor (UF only — no auxiliary state). Used by
// bench_parallel_naive.
template <typename UF>
BuiltGraph<UF> build_naive(Family f, std::size_t n, std::size_t g_arity) {
  auto w = gen_workload(f, n, g_arity);
  const std::size_t total = w.nodes.size();
  const std::size_t n_eqs = w.eqs.size();
  auto eg = std::make_unique<EGraph<UF>>(std::move(w.nodes), pe::naive);
  return {std::move(eg), std::move(w.eqs), total, n_eqs};
}

// Prints every node + initial union to stderr in a human-readable form.
// One line per node:  "id: op(child_id, child_id, ...)" or "id: op  // leaf"
// One line per union: "  union: a_id = b_id"
void dump_workload(Family f, std::size_t n, std::size_t g_arity) {
  auto w = gen_workload(f, n, g_arity);
  std::fprintf(stderr,
               "==== %s n=%zu d=%zu  classes=%zu  unions=%zu ====\n",
               family_name(f), n, g_arity,
               w.nodes.size(), w.eqs.size());
  for (std::size_t i = 0; i < w.nodes.size(); ++i) {
    const auto& nd = w.nodes[i];
    if (nd.children.empty()) {
      std::fprintf(stderr, "  %6zu: %s  // leaf\n", i, nd.op.c_str());
    } else {
      std::fprintf(stderr, "  %6zu: %s(", i, nd.op.c_str());
      for (std::size_t k = 0; k < nd.children.size(); ++k) {
        std::fprintf(stderr, "%s%u",
                     k ? ", " : "",
                     static_cast<unsigned>(nd.children[k]));
      }
      std::fprintf(stderr, ")\n");
    }
  }
  for (const auto& eq : w.eqs) {
    std::fprintf(stderr, "  union: %u = %u\n",
                 static_cast<unsigned>(eq.first),
                 static_cast<unsigned>(eq.second));
  }
}

// ---- Timing ---------------------------------------------------------------

double median(std::vector<double> xs) {
  std::sort(xs.begin(), xs.end());
  return xs[xs.size() / 2];
}

using clk = std::chrono::steady_clock;
double elapsed_ms(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

std::vector<double> bench_nelson(Family f, std::size_t n,
                                 std::size_t g_arity,
                                 int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    g.eg->sequential_close_nelson(g.eqs);
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->sequential_close_nelson(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_topo(Family f, std::size_t n,
                                      std::size_t g_arity,
                                      int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    g.eg->sequential_close_topo(g.eqs);
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->sequential_close_topo(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_topo_iter(Family f, std::size_t n,
                                             std::size_t g_arity,
                                             int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    g.eg->sequential_close_topo_iter(g.eqs);
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->sequential_close_topo_iter(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_dst(Family f, std::size_t n,
                                      std::size_t g_arity,
                                      int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    g.eg->sequential_close_dst(g.eqs);
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->sequential_close_dst(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_simple(Family f, std::size_t n,
                                         std::size_t g_arity,
                                         int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    g.eg->sequential_close_simple(g.eqs);
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->sequential_close_simple(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_simple_hash(Family f, std::size_t n,
                                              std::size_t g_arity,
                                              int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    g.eg->sequential_close_simple_hash(g.eqs);
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->sequential_close_simple_hash(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_simple_inline(Family f, std::size_t n,
                                                std::size_t g_arity,
                                                int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    g.eg->sequential_close_simple_inline(g.eqs);
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build<SequentialUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->sequential_close_simple_inline(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_parents(Family f, std::size_t n,
                                          std::size_t g_arity,
                                          int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build<ConcurrentUnionFind>(f, n, g_arity);
    g.eg->parallel_parents(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build<ConcurrentUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->parallel_parents(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_filter_groupby(
    Family f, std::size_t n, std::size_t g_arity, int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build_async<ConcurrentUnionFind>(f, n, g_arity);
    g.eg->parallel_filter_groupby(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build_async<ConcurrentUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->parallel_filter_groupby(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_filter(Family f, std::size_t n,
                                                 std::size_t g_arity,
                                                 int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build_async<ConcurrentUnionFind>(f, n, g_arity);
    g.eg->parallel_filter(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build_async<ConcurrentUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->parallel_filter(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_filter_min(Family f, std::size_t n,
                                                     std::size_t g_arity,
                                                     int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build_async<ConcurrentUnionFind>(f, n, g_arity);
    g.eg->parallel_filter_min_id(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build_async<ConcurrentUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->parallel_filter_min_id(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_naive(Family f, std::size_t n,
                                                 std::size_t g_arity,
                                                 int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build_naive<ConcurrentUnionFind>(f, n, g_arity);
    g.eg->parallel_naive_rounds(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build_naive<ConcurrentUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->parallel_naive_rounds(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_topo_iter(Family f, std::size_t n,
                                                     std::size_t g_arity,
                                                     int warmup, int trials) {
  for (int i = 0; i < warmup; ++i) {
    auto g = build_topo<ConcurrentUnionFind>(f, n, g_arity);
    g.eg->parallel_topo_iter(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    auto g = build_topo<ConcurrentUnionFind>(f, n, g_arity);
    auto t0 = clk::now();
    g.eg->parallel_topo_iter(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

// ---- Env parsing ----------------------------------------------------------

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= s.size()) {
    std::size_t end = s.find(',', start);
    std::string tok = (end == std::string::npos) ? s.substr(start)
                                                  : s.substr(start, end - start);
    if (!tok.empty()) out.push_back(std::move(tok));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return out;
}

}  // namespace

int main() {
  const std::size_t par_threads = parlay::num_workers();

  const char* fams_env = std::getenv("PE_SYNTH_FAMILIES");
  const char* ns_env   = std::getenv("PE_SYNTH_NS");
  // PE_BENCH_PAR_ONLY=1 skips every sequential algorithm (nelson_seq,
  // nelson_topo, nelson_topo_iter, nelson_dst). Useful for T>1 sweeps
  // where re-running thread-independent baselines is wasted work.
  // Implies PE_BENCH_SKIP_NELSON=1.
  const bool par_only   = std::getenv("PE_BENCH_PAR_ONLY") != nullptr;
  const bool skip_nelson = par_only ||
                           std::getenv("PE_BENCH_SKIP_NELSON") != nullptr;

  // PE_BENCH_ALGOS=algo1,algo2,... restricts which algorithms run+emit
  // (saves wall time for narrow sweeps). Names must match the CSV
  // `algorithm` column tags exactly. Empty/unset = run all (subject to
  // par_only / skip_nelson). Combining with par_only is fine: the union
  // of both restrictions applies (PE_BENCH_ALGOS narrows further).
  std::set<std::string> algo_filter;
  if (const char* algos_env = std::getenv("PE_BENCH_ALGOS")) {
    for (auto& tok : split_csv(algos_env)) algo_filter.insert(std::move(tok));
  }
  auto algo_enabled = [&](const char* name) {
    return algo_filter.empty() || algo_filter.count(name) > 0;
  };
  const char* fmt = std::getenv("PE_BENCH_FORMAT");
  const bool csv = fmt && std::strcmp(fmt, "csv") == 0;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  const char* dnc_cutoff_env  = std::getenv("PE_DNC_CUTOFF");
  const std::string dnc_cutoff  = dnc_cutoff_env  ? dnc_cutoff_env  : "16";
  // PE_SYNTH_D = decomposition rate / g-tree fan-in (>= 2). Default 2 is
  // the historical balanced binary tree. Larger d → shallower tree,
  // frontier shrinks by a factor of d each round.
  std::size_t g_arity = 2;
  if (const char* d_env = std::getenv("PE_SYNTH_D")) {
    char* endp = nullptr;
    unsigned long long v = std::strtoull(d_env, &endp, 10);
    if (!endp || *endp != '\0' || v < 2) {
      std::fprintf(stderr, "PE_SYNTH_D must be an integer >= 2 (got '%s')\n",
                   d_env);
      return 2;
    }
    g_arity = static_cast<std::size_t>(v);
  }

  // Trial counts override compile-time defaults (1 warmup + 5 trials).
  // Set PE_BENCH_TRIALS=1 PE_BENCH_WARMUP=0 for fast sweeps where
  // per-config repetition is too expensive.
  int trials = DEFAULT_TRIALS;
  int warmup = DEFAULT_WARMUP;
  if (const char* t_env = std::getenv("PE_BENCH_TRIALS")) {
    trials = std::atoi(t_env);
    if (trials < 1) {
      std::fprintf(stderr, "PE_BENCH_TRIALS must be >= 1 (got '%s')\n", t_env);
      return 2;
    }
  }
  if (const char* w_env = std::getenv("PE_BENCH_WARMUP")) {
    warmup = std::atoi(w_env);
    if (warmup < 0) {
      std::fprintf(stderr, "PE_BENCH_WARMUP must be >= 0 (got '%s')\n", w_env);
      return 2;
    }
  }

  std::vector<Family> families;
  if (fams_env) {
    for (const auto& tok : split_csv(fams_env)) {
      Family f;
      if (!parse_family(tok, f)) {
        std::fprintf(stderr, "unknown family '%s' in PE_SYNTH_FAMILIES\n",
                     tok.c_str());
        return 2;
      }
      families.push_back(f);
    }
  } else {
    families = {Family::Chain, Family::Grid, Family::Cube,
                Family::Quartic, Family::Quintic};
  }

  std::vector<std::size_t> ns;
  if (ns_env) {
    for (const auto& tok : split_csv(ns_env)) {
      char* endp = nullptr;
      unsigned long long v = std::strtoull(tok.c_str(), &endp, 10);
      if (!endp || *endp != '\0' || v == 0) {
        std::fprintf(stderr, "bad n '%s' in PE_SYNTH_NS\n", tok.c_str());
        return 2;
      }
      ns.push_back(static_cast<std::size_t>(v));
    }
  } else {
    ns = {5, 10, 20};
  }

  // PE_SYNTH_DUMP=1: print every node + initial union for each (family,n)
  // to stderr, then exit without timing anything. The dump cost on big
  // workloads is dominated by I/O, not generation, so cap n implicitly
  // by trusting the caller to pick small ones.
  if (std::getenv("PE_SYNTH_DUMP")) {
    for (Family f : families) {
      for (std::size_t n : ns) {
        dump_workload(f, n, g_arity);
      }
    }
    return 0;
  }

  if (!csv) {
    std::printf("synthetic_bench  trials=%d  warmup=%d  par_threads=%zu\n",
                trials, warmup, par_threads);
    std::printf("(* = unsound on cross-depth inits; par_spd = par_parents vs topo_iter)\n");
    std::printf("%-8s %5s %10s %9s | %11s %11s %11s %11s | %11s %11s %11s %11s %7s\n",
                "family", "n", "classes", "merges",
                "nelson_seq", "nelson_topo*", "topo_iter", "nelson_dst",
                "par_parents", "par_topo_it", "par_filter", "par_filter_m", "par_spd");
  } else if (csv_header) {
    std::printf("family,n,d,classes,merges,algorithm,trial,"
                "parlay_threads,dnc_cutoff,wallclock_ms\n");
  }

  auto emit_csv = [&](Family f, std::size_t n, std::size_t classes,
                      std::size_t merges, const char* algorithm,
                      const std::vector<double>& times) {
    for (std::size_t i = 0; i < times.size(); ++i) {
      std::printf("%s,%zu,%zu,%zu,%zu,%s,%zu,%zu,%s,%.4f\n",
                  family_name(f), n, g_arity, classes, merges,
                  algorithm, i, par_threads, dnc_cutoff.c_str(), times[i]);
    }
  };

  for (Family f : families) {
    for (std::size_t n : ns) {
      // Probe sizes once so we can print classes/merges even when nelson is
      // skipped. The build is cheap relative to the timed work; UF flavor
      // doesn't matter — we only read n_classes / n_eqs.
      auto probe = build<ConcurrentUnionFind>(f, n, g_arity);
      const std::size_t classes = probe.n_classes;
      const std::size_t merges  = probe.n_eqs;

      std::fprintf(stderr,
                   "[synthetic_bench] %s n=%zu d=%zu classes=%zu merges=%zu ...\n",
                   family_name(f), n, g_arity, classes, merges);

      std::vector<double> nel;
      double mn = 0.0;
      if (!skip_nelson && algo_enabled("nelson_seq")) {
        nel = bench_nelson(f, n, g_arity, warmup, trials);
        mn = median(nel);
      }
      std::vector<double> top, iter, dst;
      double mt = 0.0, mi = 0.0, md = 0.0;
      if (!par_only) {
        if (algo_enabled("nelson_topo")) {
          top  = bench_nelson_topo(f, n, g_arity, warmup, trials);
          mt   = median(top);
        }
        if (algo_enabled("nelson_topo_iter")) {
          iter = bench_nelson_topo_iter(f, n, g_arity, warmup, trials);
          mi   = median(iter);
        }
        if (algo_enabled("nelson_dst")) {
          dst  = bench_nelson_dst(f, n, g_arity, warmup, trials);
          md   = median(dst);
        }
      }
      std::vector<double> nsim;
      if (!par_only && algo_enabled("nelson_simple")) {
        nsim = bench_nelson_simple(f, n, g_arity, warmup, trials);
      }
      std::vector<double> nsimh;
      if (!par_only && algo_enabled("nelson_simple_hash")) {
        nsimh = bench_nelson_simple_hash(f, n, g_arity, warmup, trials);
      }
      std::vector<double> nsimi;
      if (!par_only && algo_enabled("nelson_simple_inline")) {
        nsimi = bench_nelson_simple_inline(f, n, g_arity, warmup, trials);
      }
      std::vector<double> par, pti, pa, pam, pnv, pagbk;
      double mp = 0.0, mpti = 0.0, mpa = 0.0, mpam = 0.0;
      if (algo_enabled("par_parents")) {
        par = bench_parallel_parents(f, n, g_arity, warmup, trials);
        mp  = median(par);
      }
      if (algo_enabled("par_topo_iter")) {
        pti  = bench_parallel_topo_iter(f, n, g_arity, warmup, trials);
        mpti = median(pti);
      }
      if (algo_enabled("par_filter")) {
        pa  = bench_parallel_filter(f, n, g_arity, warmup, trials);
        mpa = median(pa);
      }
      if (algo_enabled("par_filter_min_id")) {
        pam  = bench_parallel_filter_min(f, n, g_arity, warmup, trials);
        mpam = median(pam);
      }
      if (algo_enabled("par_naive")) {
        pnv = bench_parallel_naive(f, n, g_arity, warmup, trials);
      }
      if (algo_enabled("par_filter_gbk")) {
        pagbk = bench_parallel_filter_groupby(f, n, g_arity,
                                                    warmup, trials);
      }

      if (csv) {
        if (!skip_nelson && algo_enabled("nelson_seq"))
          emit_csv(f, n, classes, merges, "nelson_seq", nel);
        if (!par_only) {
          if (algo_enabled("nelson_topo"))
            emit_csv(f, n, classes, merges, "nelson_topo", top);
          if (algo_enabled("nelson_topo_iter"))
            emit_csv(f, n, classes, merges, "nelson_topo_iter", iter);
          if (algo_enabled("nelson_dst"))
            emit_csv(f, n, classes, merges, "nelson_dst", dst);
          if (algo_enabled("nelson_simple"))
            emit_csv(f, n, classes, merges, "nelson_simple", nsim);
          if (algo_enabled("nelson_simple_hash"))
            emit_csv(f, n, classes, merges, "nelson_simple_hash", nsimh);
          if (algo_enabled("nelson_simple_inline"))
            emit_csv(f, n, classes, merges, "nelson_simple_inline", nsimi);
        }
        if (algo_enabled("par_parents"))
          emit_csv(f, n, classes, merges, "par_parents", par);
        if (algo_enabled("par_topo_iter"))
          emit_csv(f, n, classes, merges, "par_topo_iter", pti);
        if (algo_enabled("par_filter"))
          emit_csv(f, n, classes, merges, "par_filter", pa);
        if (algo_enabled("par_filter_min_id"))
          emit_csv(f, n, classes, merges, "par_filter_min_id", pam);
        if (algo_enabled("par_naive"))
          emit_csv(f, n, classes, merges, "par_naive", pnv);
        if (algo_enabled("par_filter_gbk"))
          emit_csv(f, n, classes, merges, "par_filter_gbk", pagbk);
      } else if (skip_nelson) {
        std::printf("%-8s %5zu %10zu %9zu |   skipped   %9.2fms %9.2fms %9.2fms | %9.2fms %9.2fms %9.2fms %9.2fms %6.2fx\n",
                    family_name(f), n, classes, merges, mt, mi, md, mp, mpti, mpa, mpam, mi / mp);
      } else {
        std::printf("%-8s %5zu %10zu %9zu | %9.2fms %9.2fms %9.2fms %9.2fms | %9.2fms %9.2fms %9.2fms %9.2fms %6.2fx\n",
                    family_name(f), n, classes, merges,
                    mn, mt, mi, md, mp, mpti, mpa, mpam, mi / mp);
      }
      std::fflush(stdout);
    }
  }
  return 0;
}
