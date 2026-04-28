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
#include <random>
#include <string>
#include <vector>

#include <parlay/parallel.h>
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

BuiltGraph build(const Workload& w) {
  std::mt19937_64 rng(0xC0FFEE ^ static_cast<std::uint64_t>(w.n_nodes));
  std::size_t capacity = 2 * w.n_leaves + w.n_nodes;
  auto eg = std::make_unique<EGraph>(capacity, /*parallel=*/true);

  std::vector<Id> x_ids;
  x_ids.reserve(w.n_leaves);
  for (std::size_t i = 0; i < w.n_leaves; ++i) {
    ENode n;
    n.op = "x" + std::to_string(i);
    x_ids.push_back(eg->add(std::move(n)));
  }
  std::vector<Id> y_ids;
  y_ids.reserve(w.n_leaves);
  for (std::size_t i = 0; i < w.n_leaves; ++i) {
    ENode n;
    n.op = "y" + std::to_string(i);
    y_ids.push_back(eg->add(std::move(n)));
  }

  std::size_t depth = std::max<std::size_t>(w.depth, 1);
  std::size_t per_level = w.n_nodes / depth;

  std::vector<Id> prev_level;
  prev_level.reserve(2 * w.n_leaves);
  prev_level.insert(prev_level.end(), x_ids.begin(), x_ids.end());
  prev_level.insert(prev_level.end(), y_ids.begin(), y_ids.end());

  for (std::size_t lvl = 0; lvl < depth; ++lvl) {
    std::size_t count = (lvl + 1 == depth)
                          ? (w.n_nodes - per_level * (depth - 1))
                          : per_level;
    std::vector<Id> this_level;
    this_level.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      std::uniform_int_distribution<std::size_t> d_fn(0, w.n_fns - 1);
      std::uniform_int_distribution<std::size_t> d_prev(0, prev_level.size() - 1);
      std::size_t k = d_fn(rng);
      Id a = prev_level[d_prev(rng)];
      Id b = prev_level[d_prev(rng)];
      ENode n;
      n.op = "f" + std::to_string(lvl) + "_" + std::to_string(k);
      n.children = {a, b};
      this_level.push_back(eg->add(std::move(n)));
    }
    prev_level = std::move(this_level);
  }

  parlay::sequence<std::pair<Id, Id>> equalities;
  equalities.reserve(w.n_merges);
  std::uniform_int_distribution<std::size_t> d_leaf(0, w.n_leaves - 1);
  for (std::size_t i = 0; i < w.n_merges / 2; ++i) {
    equalities.emplace_back(x_ids[d_leaf(rng)], x_ids[d_leaf(rng)]);
  }
  for (std::size_t i = 0; i < w.n_merges - w.n_merges / 2; ++i) {
    equalities.emplace_back(y_ids[d_leaf(rng)], y_ids[d_leaf(rng)]);
  }

  return {std::move(eg), std::move(equalities)};
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
