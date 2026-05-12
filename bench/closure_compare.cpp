// closure_compare_bench (ae branch).
//
// Times three algorithms against six synthetic workloads:
//   nelson_simple_inline — sequential, arity-bucketed signature tables
//   par_parents          — BSP closure on parents_, parlay-semisort frontier
//   par_filter           — filter-mode closure on last_marked_ round stamps
//
// Trimmed for the FMCAD 2025 artifact: removed every algorithm not in
// the canonical `--algos nelson_simple_inline,par_parents,par_filter`
// command line used to produce the paper's results.

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
#include <parlay/random.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"

using namespace pe;

namespace {

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
  std::unique_ptr<EGraph<UF>> eg;
  parlay::sequence<std::pair<Id, Id>> eqs;
};

// Construct nodes + initial-union pairs for a workload. Parameterized
// on a ctor-tag lambda (or empty for the default ctor) so callers can
// pick the auxiliary-state flavor needed by their algorithm.
template <typename UF, typename Build>
BuiltGraph<UF> build_with(const Workload& w, Build build_eg) {
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

  parlay::random_generator gen(0xC0FFEEULL ^
                               static_cast<std::size_t>(w.n_nodes));

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

  auto eg = build_eg(std::move(all_nodes));

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

template <typename UF>
BuiltGraph<UF> build(const Workload& w) {
  return build_with<UF>(w, [](parlay::sequence<ENode> nodes) {
    return std::make_unique<EGraph<UF>>(std::move(nodes));
  });
}

template <typename UF>
BuiltGraph<UF> build_filter(const Workload& w) {
  return build_with<UF>(w, [](parlay::sequence<ENode> nodes) {
    return std::make_unique<EGraph<UF>>(std::move(nodes), pe::filter);
  });
}

double median(std::vector<double> xs) {
  std::sort(xs.begin(), xs.end());
  return xs[xs.size() / 2];
}

using clk = std::chrono::steady_clock;
double elapsed_ms(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

std::vector<double> bench_nelson_simple_inline(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build<SequentialUnionFind>(w);
    g.eg->sequential_close_simple_inline(g.eqs);
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build<SequentialUnionFind>(w);
    auto t0 = clk::now();
    g.eg->sequential_close_simple_inline(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_parents(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build<ConcurrentUnionFind>(w);
    g.eg->parallel_parents(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_parents(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_filter(const Workload& w) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build_filter<ConcurrentUnionFind>(w);
    g.eg->parallel_filter(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build_filter<ConcurrentUnionFind>(w);
    auto t0 = clk::now();
    g.eg->parallel_filter(std::move(g.eqs));
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

}  // namespace

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

  const char* only = std::getenv("PE_BENCH_ONLY");
  const bool par_only    = std::getenv("PE_BENCH_PAR_ONLY") != nullptr;
  const bool skip_nelson = par_only ||
                           std::getenv("PE_BENCH_SKIP_NELSON") != nullptr;

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

  const char* fmt = std::getenv("PE_BENCH_FORMAT");
  const bool csv = fmt && std::strcmp(fmt, "csv") == 0;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  const char* custom_spec = std::getenv("PE_BENCH_CUSTOM");
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
    std::printf("%-8s %8s %10s %9s | %14s | %12s %12s\n",
                "name", "leaves", "nodes", "merges",
                "nel_simp_inline", "par_parents", "par_filter");
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

    std::vector<double> nsimi, par, pa;
    double mnsimi = 0.0, mp = 0.0, mpa = 0.0;

    if (!par_only && !skip_nelson && algo_enabled("nelson_simple_inline")) {
      nsimi = bench_nelson_simple_inline(w);
      mnsimi = median(nsimi);
    }
    if (algo_enabled("par_parents")) {
      par = bench_parallel_parents(w);
      mp  = median(par);
    }
    if (algo_enabled("par_filter")) {
      pa  = bench_parallel_filter(w);
      mpa = median(pa);
    }

    if (csv) {
      if (!par_only && !skip_nelson && algo_enabled("nelson_simple_inline"))
        emit_csv(w, "nelson_simple_inline", nsimi);
      if (algo_enabled("par_parents")) emit_csv(w, "par_parents", par);
      if (algo_enabled("par_filter"))  emit_csv(w, "par_filter",  pa);
    } else {
      std::printf("%-8s %8zu %10zu %9zu | %12.2fms | %10.2fms %10.2fms\n",
                  w.name, w.n_leaves, w.n_nodes, w.n_merges,
                  mnsimi, mp, mpa);
    }
    std::fflush(stdout);
  }
  return 0;
}
