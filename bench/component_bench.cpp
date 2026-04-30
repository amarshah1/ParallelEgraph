// Component microbench: times parallel_consolidate and
// merge_and_collect_semisort in isolation, on round-0 mid-state of a real
// closure. Each trial rebuilds the graph (cheap, parallel) and runs only
// up to the timed phase, so the timing reflects the phase alone.
//
// Output: CSV with one row per (phase, trial); identical schema across runs.
//   phase,trial,parlay_threads,workload,wallclock_ms

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>

#include "parallel_egraph/detail.hpp"
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
  std::unique_ptr<EGraph> eg;
  parlay::sequence<std::pair<Id, Id>> eqs;
};

// Mirror of bench/closure_compare.cpp's build(). Kept as a copy rather
// than factored out to avoid a shared bench/common.hpp for two callers.
BuiltGraph build(const Workload& w) {
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

  auto eg = EGraph::bulk_init(std::move(all_nodes));

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

using clk = std::chrono::steady_clock;
double elapsed_ms(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct PhaseTimes {
  double consolidate_ms;
  double semisort_ms;
};

// One trial: build graph, apply initial unions, then time each phase
// in turn over round-0 inputs.
PhaseTimes run_trial(const Workload& w) {
  auto g = build(w);
  auto& uf = g.eg->uf();
  auto& parent_index = g.eg->parent_index();
  const auto& nodes = g.eg->nodes();

  parlay::parallel_for(0, g.eqs.size(), [&](std::size_t i) {
    uf.union_(g.eqs[i].first, g.eqs[i].second);
  });

  auto work = parlay::flatten(parlay::map(g.eqs, [](auto p) {
    return parlay::sequence<Id>{p.first, p.second};
  }));
  work = parlay::remove_duplicates(std::move(work));
  auto roots = parlay::map(work, [&](Id c) { return uf.find_root(c); });

  auto t0 = clk::now();
  detail::parallel_consolidate(parent_index, work, roots);
  double consolidate_ms = elapsed_ms(t0);

  auto unique_roots = parlay::remove_duplicates(roots);
  auto frontier = parlay::flatten(parlay::map(unique_roots, [&](Id r) {
    return parlay::sequence<std::uint32_t>(std::begin(parent_index[r]),
                                           std::end(parent_index[r]));
  }));

  if (frontier.empty()) return {consolidate_ms, 0.0};

  auto canon = parlay::map(frontier, [&](std::uint32_t idx) {
    const auto& [node, class_id] = nodes[idx];
#ifdef PE_GROUPBY_HASH
    return detail::CanonEntry{sig_hash(node, uf), uf.find_root(class_id), idx};
#else
    auto [h1, h2] = sig_hashes(node, uf);
    return detail::CanonEntry{h1, uf.find_root(class_id), h2};
#endif
  });

  auto t1 = clk::now();
  auto next_work =
      detail::merge_and_collect_semisort(std::move(canon), uf, nodes);
  (void)next_work;
  double semisort_ms = elapsed_ms(t1);

  return {consolidate_ms, semisort_ms};
}

// Parse "leaves,fns,nodes,merges,depth" → Workload. Mirrors
// parse_custom_workload in closure_compare.cpp.
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

}  // namespace

int main() {
  std::size_t par_threads = parlay::num_workers();

  const char* only = std::getenv("PE_COMPONENT_WORKLOAD");
  const char* fmt = std::getenv("PE_BENCH_FORMAT");
  const bool csv = fmt && std::strcmp(fmt, "csv") == 0;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  // PE_COMPONENT_CUSTOM=leaves,fns,nodes,merges,depth replaces the baked-in
  // WORKLOADS table with one caller-specified workload.
  const char* custom_spec = std::getenv("PE_COMPONENT_CUSTOM");

  std::vector<Workload> workloads;
  Workload custom_w{};
  if (custom_spec) {
    if (!parse_custom_workload(custom_spec, custom_w)) {
      std::fprintf(stderr,
                   "PE_COMPONENT_CUSTOM must be 'leaves,fns,nodes,merges,depth' "
                   "(got '%s')\n",
                   custom_spec);
      return 2;
    }
    workloads.push_back(custom_w);
  } else {
    workloads = WORKLOADS;
  }

  if (csv && csv_header) {
    std::printf("phase,trial,parlay_threads,workload,wallclock_ms\n");
  }
  if (!csv) {
    std::printf("component_bench  trials=%d  warmup=%d  par_threads=%zu\n",
                TRIALS, WARMUP, par_threads);
    std::printf("%-8s %15s %15s\n", "name", "consolidate(med)", "semisort(med)");
  }

  for (const auto& w : workloads) {
    if (only && std::string(only) != w.name) continue;
    std::fprintf(stderr, "[component] running %s ...\n", w.name);

    for (int i = 0; i < WARMUP; ++i) (void)run_trial(w);

    std::vector<double> consolidate_ms, semisort_ms;
    consolidate_ms.reserve(TRIALS);
    semisort_ms.reserve(TRIALS);
    for (int i = 0; i < TRIALS; ++i) {
      auto pt = run_trial(w);
      consolidate_ms.push_back(pt.consolidate_ms);
      semisort_ms.push_back(pt.semisort_ms);
      if (csv) {
        std::printf("consolidate,%d,%zu,%s,%.4f\n",
                    i, par_threads, w.name, pt.consolidate_ms);
        std::printf("semisort,%d,%zu,%s,%.4f\n",
                    i, par_threads, w.name, pt.semisort_ms);
      }
    }
    if (!csv) {
      auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      std::printf("%-8s %12.3fms %12.3fms\n", w.name,
                  med(consolidate_ms), med(semisort_ms));
    }
    std::fflush(stdout);
  }
  return 0;
}
