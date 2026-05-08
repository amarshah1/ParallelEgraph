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
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"

using namespace pe;

namespace {

// Default trial counts; overridable at runtime via PE_BENCH_TRIALS /
// PE_BENCH_WARMUP env vars. Lazy lookup the first time they're used so
// the cost is paid once and every subsequent call sees a fast int load.
inline int trials() {
  static const int v = [] {
    const char* s = std::getenv("PE_BENCH_TRIALS");
    return s ? std::max(1, std::atoi(s)) : 5;
  }();
  return v;
}
inline int warmup() {
  static const int v = [] {
    const char* s = std::getenv("PE_BENCH_WARMUP");
    return s ? std::max(0, std::atoi(s)) : 1;
  }();
  return v;
}
// Kept as identifiers for the existing call sites; macro-substitute via
// using-declarations so we don't have to touch every loop bound.
#define TRIALS (trials())
#define WARMUP (warmup())

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

template <typename UF>
struct BuiltGraph {
  // EGraph is heap-allocated because ConcurrentUnionFind contains atomics
  // (non-movable). The same shape works for both flavors.
  std::unique_ptr<EGraph<UF>> eg;
  parlay::sequence<std::pair<Id, Id>> eqs;
};

// Workload generation is fully parallel:
//   * Every ENode (leaves + per-level function nodes) is built in a single
//     parlay::tabulate.
//   * The whole e-graph (uf_, nodes_, parents_) is constructed in one
//     parallel pass via the EGraph<UF> ctor.
//   * Equality pairs come out of a parlay::tabulate.
// parlay::random_generator gives per-index forked sub-generators, so each
// node draws independent random numbers without sequential RNG state.
template <typename UF>
BuiltGraph<UF> build(const Workload& w) {
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

  auto eg = std::make_unique<EGraph<UF>>(std::move(all_nodes));

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

// Topo-flavor variant of build(): identical node + eq generation, but
// constructs the EGraph through the topo-tagged ctor (populates
// depth_buckets_; skips parents_/last_marked_).
template <typename UF>
BuiltGraph<UF> build_topo(const Workload& w) {
  const std::size_t depth = std::max<std::size_t>(w.depth, 1);
  const std::size_t per_level = w.n_nodes / depth;
  std::vector<std::size_t> level_starts;
  level_starts.reserve(depth + 1);
  level_starts.push_back(2 * w.n_leaves);
  for (std::size_t lvl = 0; lvl < depth; ++lvl) {
    const std::size_t count = (lvl + 1 == depth)
                                ? (w.n_nodes - per_level * (depth - 1))
                                : per_level;
    level_starts.push_back(level_starts[lvl] + count);
  }
  const std::size_t total = level_starts.back();
  parlay::random_generator gen(0xC0FFEEULL ^ static_cast<std::size_t>(w.n_nodes));

  auto all_nodes = parlay::tabulate(total, [&](std::size_t i) -> ENode {
    if (i < w.n_leaves) {
      return ENode{std::string("x") + std::to_string(i), {}};
    }
    if (i < 2 * w.n_leaves) {
      return ENode{std::string("y") + std::to_string(i - w.n_leaves), {}};
    }
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

  auto eg = std::make_unique<EGraph<UF>>(std::move(all_nodes), pe::topo);

  const std::size_t n_x_merges = w.n_merges / 2;
  auto eq_seq = parlay::tabulate(w.n_merges, [&](std::size_t i) {
    auto r = gen[total + i];
    if (i < n_x_merges) {
      return std::pair<Id, Id>{static_cast<Id>(r() % w.n_leaves),
                               static_cast<Id>(r() % w.n_leaves)};
    }
    return std::pair<Id, Id>{
        static_cast<Id>(w.n_leaves + r() % w.n_leaves),
        static_cast<Id>(w.n_leaves + r() % w.n_leaves)};
  });

  return {std::move(eg), std::move(eq_seq)};
}

// Async-flavor build: same workload generation, EGraph constructed via
// the async-tagged ctor (populates last_marked_, skips parents_/
// depth_buckets_). Used by bench_parallel_close_async.
template <typename UF>
BuiltGraph<UF> build_async(const Workload& w) {
  const std::size_t depth = std::max<std::size_t>(w.depth, 1);
  const std::size_t per_level = w.n_nodes / depth;
  std::vector<std::size_t> level_starts;
  level_starts.reserve(depth + 1);
  level_starts.push_back(2 * w.n_leaves);
  for (std::size_t lvl = 0; lvl < depth; ++lvl) {
    const std::size_t count = (lvl + 1 == depth)
                                ? (w.n_nodes - per_level * (depth - 1))
                                : per_level;
    level_starts.push_back(level_starts[lvl] + count);
  }
  const std::size_t total = level_starts.back();
  parlay::random_generator gen(0xC0FFEEULL ^ static_cast<std::size_t>(w.n_nodes));

  auto all_nodes = parlay::tabulate(total, [&](std::size_t i) -> ENode {
    if (i < w.n_leaves) {
      return ENode{std::string("x") + std::to_string(i), {}};
    }
    if (i < 2 * w.n_leaves) {
      return ENode{std::string("y") + std::to_string(i - w.n_leaves), {}};
    }
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

  auto eg = std::make_unique<EGraph<UF>>(std::move(all_nodes), pe::async);

  const std::size_t n_x_merges = w.n_merges / 2;
  auto eq_seq = parlay::tabulate(w.n_merges, [&](std::size_t i) {
    auto r = gen[total + i];
    if (i < n_x_merges) {
      return std::pair<Id, Id>{static_cast<Id>(r() % w.n_leaves),
                               static_cast<Id>(r() % w.n_leaves)};
    }
    return std::pair<Id, Id>{
        static_cast<Id>(w.n_leaves + r() % w.n_leaves),
        static_cast<Id>(w.n_leaves + r() % w.n_leaves)};
  });

  return {std::move(eg), std::move(eq_seq)};
}

// Naive-flavor build: workload generation identical, EGraph via the
// naive-tagged ctor (UF only — no parents_/last_marked_/depth_buckets_).
// Used by bench_parallel_close_naive.
template <typename UF>
BuiltGraph<UF> build_naive(const Workload& w) {
  const std::size_t depth = std::max<std::size_t>(w.depth, 1);
  const std::size_t per_level = w.n_nodes / depth;
  std::vector<std::size_t> level_starts;
  level_starts.reserve(depth + 1);
  level_starts.push_back(2 * w.n_leaves);
  for (std::size_t lvl = 0; lvl < depth; ++lvl) {
    const std::size_t count = (lvl + 1 == depth)
                                ? (w.n_nodes - per_level * (depth - 1))
                                : per_level;
    level_starts.push_back(level_starts[lvl] + count);
  }
  const std::size_t total = level_starts.back();
  parlay::random_generator gen(0xC0FFEEULL ^ static_cast<std::size_t>(w.n_nodes));

  auto all_nodes = parlay::tabulate(total, [&](std::size_t i) -> ENode {
    if (i < w.n_leaves) {
      return ENode{std::string("x") + std::to_string(i), {}};
    }
    if (i < 2 * w.n_leaves) {
      return ENode{std::string("y") + std::to_string(i - w.n_leaves), {}};
    }
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

  auto eg = std::make_unique<EGraph<UF>>(std::move(all_nodes), pe::naive);

  const std::size_t n_x_merges = w.n_merges / 2;
  auto eq_seq = parlay::tabulate(w.n_merges, [&](std::size_t i) {
    auto r = gen[total + i];
    if (i < n_x_merges) {
      return std::pair<Id, Id>{static_cast<Id>(r() % w.n_leaves),
                               static_cast<Id>(r() % w.n_leaves)};
    }
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
    auto g = build<SequentialUnionFind>(w);
    g.eg->sequential_close_nelson(g.eqs);
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build<SequentialUnionFind>(w);
    auto t0 = clk::now();
    g.eg->sequential_close_nelson(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_topo(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build<SequentialUnionFind>(w);
    g.eg->sequential_close_topo(g.eqs);
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build<SequentialUnionFind>(w);
    auto t0 = clk::now();
    g.eg->sequential_close_topo(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_dst(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build<SequentialUnionFind>(w);
    g.eg->sequential_close_dst(g.eqs);
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build<SequentialUnionFind>(w);
    auto t0 = clk::now();
    g.eg->sequential_close_dst(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_nelson_topo_iter(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build<SequentialUnionFind>(w);
    g.eg->sequential_close_topo_iter(g.eqs);
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build<SequentialUnionFind>(w);
    auto t0 = clk::now();
    g.eg->sequential_close_topo_iter(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build<ConcurrentUnionFind>(w);
    g.eg->parallel_close(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_close(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close_async_continuous(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build_async<ConcurrentUnionFind>(w);
    g.eg->parallel_close_async_continuous(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build_async<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_close_async_continuous(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close_async_groupby(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build_async<ConcurrentUnionFind>(w);
    g.eg->parallel_close_async_rounds_groupby(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build_async<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_close_async_rounds_groupby(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close_async(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build_async<ConcurrentUnionFind>(w);
    g.eg->parallel_close_async_rounds(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build_async<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_close_async_rounds(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close_async_min(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build_async<ConcurrentUnionFind>(w);
    g.eg->parallel_close_async_rounds_min_id(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build_async<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_close_async_rounds_min_id(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close_naive(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build_naive<ConcurrentUnionFind>(w);
    g.eg->parallel_close_naive_rounds(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build_naive<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_close_naive_rounds(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close_topo_iter(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build_topo<ConcurrentUnionFind>(w);
    g.eg->parallel_close_topo_iter(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build_topo<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_close_topo_iter(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

}  // namespace

// Parse "leaves,fns,nodes,merges,depth" → Workload. Returns false on
// malformed input.
bool parse_custom_workload(const char* spec, Workload& out) {
  out.name = "custom";
  std::size_t* fields[] = {&out.n_leaves, &out.n_fns, &out.n_nodes,
                           &out.n_merges, &out.depth};
  std::string s(spec);
  std::size_t start = 0;
  for (int i = 0; i < 5; ++i) {
    std::size_t end = s.find(',', start);
    std::string tok = (end == std::string::npos) ? s.substr(start)
                                                  : s.substr(start, end - start);
    if (tok.empty()) return false;
    char* endp = nullptr;
    unsigned long long v = std::strtoull(tok.c_str(), &endp, 10);
    if (!endp || *endp != '\0') return false;
    *fields[i] = static_cast<std::size_t>(v);
    if (end == std::string::npos) {
      if (i != 4) return false;
      break;
    }
    start = end + 1;
  }
  return out.depth >= 1 && out.n_leaves > 0 && out.n_nodes > 0;
}

int main() {
  std::size_t par_threads = parlay::num_workers();

  // PE_BENCH_ONLY=large (etc.) restricts to a single workload — handy for
  // quick parameter sweeps without paying for the other five.
  const char* only = std::getenv("PE_BENCH_ONLY");
  // PE_BENCH_SKIP_NELSON=1 skips the sequential nelson_seq baseline.
  // PE_BENCH_PAR_ONLY=1 additionally skips nelson_topo, nelson_topo_iter,
  // nelson_dst — i.e. every sequential algorithm. Useful for T>1 sweeps
  // where the thread-independent baselines have already been measured
  // at T=1. Implies PE_BENCH_SKIP_NELSON=1.
  const bool par_only    = std::getenv("PE_BENCH_PAR_ONLY") != nullptr;
  const bool skip_nelson = par_only ||
                           std::getenv("PE_BENCH_SKIP_NELSON") != nullptr;

  // PE_BENCH_ALGOS=algo1,algo2,... restricts which algorithms run+emit.
  // Names must match CSV `algorithm` tags: nelson_seq, nelson_topo,
  // nelson_topo_iter, nelson_dst, par_close, par_topo_iter, par_async,
  // par_async_min_id, par_naive. Empty/unset = run all (subject to
  // par_only and skip_nelson). Combining narrows further.
  std::set<std::string> algo_filter;
  if (const char* algos_env = std::getenv("PE_BENCH_ALGOS")) {
    std::string s(algos_env);
    std::size_t start = 0;
    while (start <= s.size()) {
      std::size_t end = s.find(',', start);
      std::string tok = (end == std::string::npos)
                          ? s.substr(start)
                          : s.substr(start, end - start);
      if (!tok.empty()) algo_filter.insert(std::move(tok));
      if (end == std::string::npos) break;
      start = end + 1;
    }
  }
  auto algo_enabled = [&](const char* name) {
    return algo_filter.empty() || algo_filter.count(name) > 0;
  };
  // PE_BENCH_FORMAT=csv emits one row per (workload, algorithm, trial)
  // instead of the aligned summary table. Header gated by PE_BENCH_HEADER=1
  // so multiple driver invocations can append.
  const char* fmt = std::getenv("PE_BENCH_FORMAT");
  const bool csv = fmt && std::strcmp(fmt, "csv") == 0;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  // PE_BENCH_CUSTOM=leaves,fns,nodes,merges,depth replaces the 6 baked-in
  // workloads with a single caller-specified one.
  const char* custom_spec = std::getenv("PE_BENCH_CUSTOM");
  // PE_DNC_CUTOFF isn't read here, but we tag CSV rows with its value
  // so downstream plots can group correctly.
  const char* dnc_cutoff_env = std::getenv("PE_DNC_CUTOFF");
  const std::string dnc_cutoff = dnc_cutoff_env ? dnc_cutoff_env : "16";
  std::vector<Workload> workloads;
  Workload custom_w{};
  if (custom_spec) {
    if (!parse_custom_workload(custom_spec, custom_w)) {
      std::fprintf(stderr,
                   "PE_BENCH_CUSTOM must be 'leaves,fns,nodes,merges,depth' "
                   "(got '%s')\n",
                   custom_spec);
      return 2;
    }
    workloads.push_back(custom_w);
  } else {
    workloads = WORKLOADS;
  }

  if (!csv) {
    std::printf("close_compare  trials=%d  warmup=%d  par_threads=%zu\n",
                TRIALS, WARMUP, par_threads);
    std::printf("(* = unsound on cross-depth inits; topo_iter is the "
                "iterated-to-fixpoint sound version. par_spd is "
                "par_close vs topo_iter.)\n");
    std::printf("%-8s %8s %10s %9s | %11s %11s %11s %11s | %11s %11s %11s %11s %7s\n",
                "name", "leaves", "nodes", "merges",
                "nelson_seq", "nelson_topo*", "topo_iter", "nelson_dst",
                "par_close", "par_topo_it", "par_async", "par_async_m", "par_spd");
  } else if (csv_header) {
    std::printf("workload,leaves,fns,nodes,merges,depth,algorithm,trial,"
                "parlay_threads,dnc_cutoff,wallclock_ms\n");
  }

  auto emit_csv = [&](const Workload& w, const char* algorithm,
                      const std::vector<double>& times) {
    for (std::size_t i = 0; i < times.size(); ++i) {
      std::printf("%s,%zu,%zu,%zu,%zu,%zu,%s,%zu,%zu,%s,%.4f\n",
                  w.name, w.n_leaves, w.n_fns, w.n_nodes, w.n_merges, w.depth,
                  algorithm, i, par_threads, dnc_cutoff.c_str(), times[i]);
    }
  };

  for (const auto& w : workloads) {
    if (only && std::string(only) != w.name) continue;
    std::fprintf(stderr, "[bench] running %s ...\n", w.name);
    std::vector<double> nel;
    double mn = 0.0;
    if (!skip_nelson && algo_enabled("nelson_seq")) {
      nel = bench_nelson(w);
      mn = median(nel);
    }
    std::vector<double> top, iter, dst;
    double mt = 0.0, mi = 0.0, md = 0.0;
    if (!par_only) {
      if (algo_enabled("nelson_topo")) {
        top  = bench_nelson_topo(w);
        mt   = median(top);
      }
      if (algo_enabled("nelson_topo_iter")) {
        iter = bench_nelson_topo_iter(w);
        mi   = median(iter);
      }
      if (algo_enabled("nelson_dst")) {
        dst  = bench_nelson_dst(w);
        md   = median(dst);
      }
    }
    std::vector<double> par, pti, pa, pam, pnv;
    double mp = 0.0, mpti = 0.0, mpa = 0.0, mpam = 0.0;
    if (algo_enabled("par_close")) {
      par = bench_parallel_close(w);
      mp  = median(par);
    }
    if (algo_enabled("par_topo_iter")) {
      pti  = bench_parallel_close_topo_iter(w);
      mpti = median(pti);
    }
    if (algo_enabled("par_async")) {
      pa  = bench_parallel_close_async(w);
      mpa = median(pa);
    }
    if (algo_enabled("par_async_min_id")) {
      pam  = bench_parallel_close_async_min(w);
      mpam = median(pam);
    }
    if (algo_enabled("par_naive")) {
      pnv = bench_parallel_close_naive(w);
    }
    std::vector<double> pac;
    if (algo_enabled("par_async_cont")) {
      pac = bench_parallel_close_async_continuous(w);
    }
    std::vector<double> pagbk;
    if (algo_enabled("par_async_gbk")) {
      pagbk = bench_parallel_close_async_groupby(w);
    }

    if (csv) {
      if (!skip_nelson && algo_enabled("nelson_seq"))
        emit_csv(w, "nelson_seq", nel);
      if (!par_only) {
        if (algo_enabled("nelson_topo"))      emit_csv(w, "nelson_topo", top);
        if (algo_enabled("nelson_topo_iter")) emit_csv(w, "nelson_topo_iter", iter);
        if (algo_enabled("nelson_dst"))       emit_csv(w, "nelson_dst", dst);
      }
      if (algo_enabled("par_close"))        emit_csv(w, "par_close", par);
      if (algo_enabled("par_topo_iter"))    emit_csv(w, "par_topo_iter", pti);
      if (algo_enabled("par_async"))        emit_csv(w, "par_async", pa);
      if (algo_enabled("par_async_min_id")) emit_csv(w, "par_async_min_id", pam);
      if (algo_enabled("par_naive"))        emit_csv(w, "par_naive", pnv);
      if (algo_enabled("par_async_cont"))   emit_csv(w, "par_async_cont", pac);
      if (algo_enabled("par_async_gbk"))    emit_csv(w, "par_async_gbk", pagbk);
    } else if (skip_nelson) {
      std::printf("%-8s %8zu %10zu %9zu |   skipped   %9.2fms %9.2fms %9.2fms | %9.2fms %9.2fms %9.2fms %9.2fms %6.2fx\n",
                  w.name, w.n_leaves, w.n_nodes, w.n_merges, mt, mi, md, mp, mpti, mpa, mpam, mi / mp);
    } else {
      std::printf("%-8s %8zu %10zu %9zu | %9.2fms %9.2fms %9.2fms %9.2fms | %9.2fms %9.2fms %9.2fms %9.2fms %6.2fx\n",
                  w.name, w.n_leaves, w.n_nodes, w.n_merges,
                  mn, mt, mi, md, mp, mpti, mpa, mpam, mi / mp);
    }
    std::fflush(stdout);
  }
  return 0;
}
