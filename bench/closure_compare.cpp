// Port of benches/rebuild_compare.rs.
//
// Same 6 workloads, same trials/warmup, same output format. The synthetic
// DAGs are *not* bit-identical to the Rust bench (we use std::mt19937
// instead of Rust's SmallRng), but the shape parameters and within-language
// determinism are preserved.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"

using namespace pe;

namespace {

constexpr int TRIALS = 11;
constexpr int WARMUP = 3;

struct Workload {
  const char* name;
  std::size_t n_leaves;
  std::size_t n_fns;
  std::size_t n_nodes;
  std::size_t n_merges;
  std::size_t depth;
};

const std::vector<Workload> WORKLOADS = {
  {"small",   1'000,   4,  10'000,    2'000,   1},
  {"medium",  10'000,  8,  200'000,   20'000,  1},
  {"large",   50'000,  16, 1'000'000, 100'000, 1},
  {"deep-s",  1'000,   4,  10'000,    2'000,   3},
  {"deep-m",  10'000,  8,  200'000,   20'000,  3},
  {"deep-l",  50'000,  16, 1'000'000, 100'000, 3},
};

struct BuiltGraph {
  // EGraph is heap-allocated because it contains atomics (non-movable).
  std::unique_ptr<EGraph> eg;
  parlay::sequence<std::pair<Id, Id>> eqs;
};

// Workload generation is fully parallel:
//   * Every ENode (leaves + per-level function nodes) is built in a single
//     parlay::tabulate.
//   * The whole e-graph (uf_, nodes_, parent_index_) is constructed in one
//     parallel pass via EGraph::bulk_init.
//   * Equality pairs come out of a parlay::tabulate.
// parlay::random_generator gives per-index forked sub-generators, so each
// node draws independent random numbers without sequential RNG state.
BuiltGraph build(const Workload& w) {
  const std::size_t depth = std::max<std::size_t>(w.depth, 1);
  const std::size_t per_level = w.n_nodes / depth;

  // Compute level boundaries: level_starts[k] is the first class id of
  // level k's function nodes. Leaves (x then y) occupy [0, 2*n_leaves);
  // level k function nodes occupy [level_starts[k], level_starts[k+1]).
  std::vector<std::size_t> level_starts;
  level_starts.reserve(depth + 1);
  level_starts.push_back(2 * w.n_leaves);
  for (std::size_t lvl = 0; lvl < depth; ++lvl) {
    const std::size_t count = (lvl + 1 == depth)
                                ? (w.n_nodes - per_level * (depth - 1))
                                : per_level;
    level_starts.push_back(level_starts[lvl] + count);
  }
  const std::size_t total = level_starts.back();   // 2*n_leaves + n_nodes

  parlay::random_generator gen(0xC0FFEEULL ^ static_cast<std::size_t>(w.n_nodes));

  // ---- All nodes in one parallel tabulate ----
  auto all_nodes = parlay::tabulate(total, [&](std::size_t i) -> ENode {
    if (i < w.n_leaves) {
      return ENode{std::string("x") + std::to_string(i), {}};
    }
    if (i < 2 * w.n_leaves) {
      return ENode{std::string("y") + std::to_string(i - w.n_leaves), {}};
    }
    // Function node: locate its level via binary search on level_starts.
    std::size_t lvl = static_cast<std::size_t>(
        std::upper_bound(level_starts.begin(), level_starts.end(), i) -
        level_starts.begin()) - 1;
    const std::size_t prev_start = (lvl == 0) ? 0 : level_starts[lvl - 1];
    const std::size_t prev_size = level_starts[lvl] - prev_start;

    auto r = gen[i];
    ENode n;
    n.op = std::string("f") + std::to_string(lvl) + "_" +
           std::to_string(r() % w.n_fns);
    n.children = {static_cast<Id>(prev_start + r() % prev_size),
                  static_cast<Id>(prev_start + r() % prev_size)};
    return n;
  });

  auto eg = EGraph::bulk_init(std::move(all_nodes));

  // ---- Equalities: fully parallel. ----
  const std::size_t n_x_merges = w.n_merges / 2;
  auto eq_seq = parlay::tabulate(w.n_merges, [&](std::size_t i) {
    auto r = gen[total + i];   // distinct subkey from any node-gen seed
    if (i < n_x_merges) {
      // x-side merges: ids 0..n_leaves-1
      return std::pair<Id, Id>{static_cast<Id>(r() % w.n_leaves),
                               static_cast<Id>(r() % w.n_leaves)};
    }
    // y-side merges: ids n_leaves..2*n_leaves-1
    return std::pair<Id, Id>{
        static_cast<Id>(w.n_leaves + r() % w.n_leaves),
        static_cast<Id>(w.n_leaves + r() % w.n_leaves)};
  });

  return {std::move(eg), std::move(eq_seq)};
}

double median(std::vector<double> xs) {
  std::sort(xs.begin(), xs.end());
  return xs[xs.size() / 2];
}

using clk = std::chrono::steady_clock;
double elapsed_ms(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

std::vector<double> bench_nelson(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build(w);
    g.eg->sequential_close_nelson(g.eqs);
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build(w);
    auto t0 = clk::now();
    g.eg->sequential_close_nelson(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build(w);
    g.eg->parallel_close(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build(w);
    auto t0 = clk::now();
    g.eg->parallel_close(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

}  // namespace

int main() {
  std::size_t par_threads = parlay::num_workers();

  std::printf("close_compare  trials=%d  warmup=%d  par_threads=%zu\n",
              TRIALS, WARMUP, par_threads);
  std::printf("%-8s %8s %10s %9s | %11s | %11s %11s\n",
              "name", "leaves", "nodes", "merges",
              "nelson_seq", "par_close", "par_spd");

  // PE_BENCH_ONLY=large (etc.) restricts to a single workload — handy for
  // quick parameter sweeps without paying for the other five.
  const char* only = std::getenv("PE_BENCH_ONLY");
  // PE_BENCH_SKIP_NELSON=1 skips the sequential baseline (sweep-only).
  const bool skip_nelson = std::getenv("PE_BENCH_SKIP_NELSON") != nullptr;

  for (const auto& w : WORKLOADS) {
    if (only && std::string(only) != w.name) continue;
    std::fprintf(stderr, "[bench] running %s ...\n", w.name);
    double mn = 0.0;
    if (!skip_nelson) {
      auto nel = bench_nelson(w);
      mn = median(nel);
    }
    auto par = bench_parallel_close(w);
    double mp = median(par);
    if (skip_nelson) {
      std::printf("%-8s %8zu %10zu %9zu |   skipped  | %9.2fms\n",
                  w.name, w.n_leaves, w.n_nodes, w.n_merges, mp);
    } else {
      std::printf("%-8s %8zu %10zu %9zu | %9.2fms | %9.2fms %9.2fx\n",
                  w.name, w.n_leaves, w.n_nodes, w.n_merges, mn, mp, mn / mp);
    }
    std::fflush(stdout);
  }
  return 0;
}
