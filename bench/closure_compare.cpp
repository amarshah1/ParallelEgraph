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
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"

using namespace pe;

namespace {

constexpr int TRIALS = 5;
constexpr int WARMUP = 1;

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
  // PE_BENCH_SKIP_NELSON=1 skips the sequential baseline (sweep-only).
  const bool skip_nelson = std::getenv("PE_BENCH_SKIP_NELSON") != nullptr;
  // PE_BENCH_FORMAT=csv emits one row per (workload, algorithm, trial)
  // instead of the aligned summary table. Header gated by PE_BENCH_HEADER=1
  // so multiple driver invocations can append.
  const char* fmt = std::getenv("PE_BENCH_FORMAT");
  const bool csv = fmt && std::strcmp(fmt, "csv") == 0;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  // PE_BENCH_CUSTOM=leaves,fns,nodes,merges,depth replaces the 6 baked-in
  // workloads with a single caller-specified one.
  const char* custom_spec = std::getenv("PE_BENCH_CUSTOM");
  // PE_UNION_STYLE / PE_DNC_CUTOFF aren't read here, but we tag CSV rows
  // with their values so downstream plots can group correctly.
  const char* union_style_env = std::getenv("PE_UNION_STYLE");
  const char* dnc_cutoff_env = std::getenv("PE_DNC_CUTOFF");
  const std::string union_style = union_style_env ? union_style_env : "dnc";
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
    std::printf("%-8s %8s %10s %9s | %11s | %11s %11s\n",
                "name", "leaves", "nodes", "merges",
                "nelson_seq", "par_close", "par_spd");
  } else if (csv_header) {
    std::printf("workload,leaves,fns,nodes,merges,depth,algorithm,trial,"
                "parlay_threads,union_style,dnc_cutoff,wallclock_ms\n");
  }

  auto emit_csv = [&](const Workload& w, const char* algorithm,
                      const std::vector<double>& times) {
    for (std::size_t i = 0; i < times.size(); ++i) {
      std::printf("%s,%zu,%zu,%zu,%zu,%zu,%s,%zu,%zu,%s,%s,%.4f\n",
                  w.name, w.n_leaves, w.n_fns, w.n_nodes, w.n_merges, w.depth,
                  algorithm, i, par_threads, union_style.c_str(),
                  dnc_cutoff.c_str(), times[i]);
    }
  };

  for (const auto& w : workloads) {
    if (only && std::string(only) != w.name) continue;
    std::fprintf(stderr, "[bench] running %s ...\n", w.name);
    std::vector<double> nel;
    double mn = 0.0;
    if (!skip_nelson) {
      nel = bench_nelson(w);
      mn = median(nel);
    }
    auto par = bench_parallel_close(w);
    double mp = median(par);

    if (csv) {
      if (!skip_nelson) emit_csv(w, "nelson_seq", nel);
      emit_csv(w, "par_close", par);
    } else if (skip_nelson) {
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
