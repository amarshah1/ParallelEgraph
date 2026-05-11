// Irregular workload bench: variable arities + cross-arity initial unions.
//
// The closure_compare bench builds a uniform binary-tree DAG; this bench
// stresses code paths the uniform workload doesn't:
//
//   * Variable arity: each level has 25% unary, 50% binary, 25% ternary
//     function nodes. Each arity has its own per-level operator pool, so
//     `f1_<lvl>_<r>` (arity 1), `f2_<lvl>_<r>` (arity 2), and `f3_<lvl>_<r>`
//     (arity 3) are textually disjoint operators. Children are drawn iid
//     uniform from the previous level's full class-id range.
//
//   * Cross-arity initial unions: 60% of merges are leaf-leaf (same as
//     closure_compare), 40% are unary-vs-binary at level 0. A cross-arity
//     union puts structurally-incompatible nodes in the same class — their
//     own signatures never collide (different op + arity), but parents of
//     either side now share a canonical child class, which can trigger
//     congruence cascades upward through the DAG.
//
// Output schema matches closure_compare so the row drops directly into
// width_grid analysis. Workload column is "irregular" (or "irregular-custom"
// when PE_BENCH_CUSTOM is set).

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

constexpr int TRIALS = 11;
constexpr int WARMUP = 3;

struct Workload {
  const char* name;
  std::size_t n_leaves;
  std::size_t n_fns;        // operator-pool size *per arity per level*
  std::size_t n_nodes;
  std::size_t n_merges;
  std::size_t depth;
};

// Default sized to match xl-d3 in closure_compare for direct comparability.
constexpr Workload DEFAULT_WORKLOAD = {
  "irregular", 100'000, 16, 2'000'000, 200'000, 3
};

// Per-level layout: each level subdivided into unary / binary / ternary bands.
struct LevelLayout {
  std::size_t start;
  std::size_t unary_start, n_unary;
  std::size_t binary_start, n_binary;
  std::size_t ternary_start, n_ternary;
  std::size_t end;
};

template <typename UF>
struct BuiltGraph {
  std::unique_ptr<EGraph<UF>> eg;
  parlay::sequence<std::pair<Id, Id>> eqs;
};

std::vector<LevelLayout> compute_layouts(const Workload& w) {
  const std::size_t depth = std::max<std::size_t>(w.depth, 1);
  const std::size_t per_level = w.n_nodes / depth;
  std::vector<LevelLayout> layouts(depth);
  std::size_t cursor = 2 * w.n_leaves;
  for (std::size_t k = 0; k < depth; ++k) {
    const std::size_t count = (k + 1 == depth)
                                ? (w.n_nodes - per_level * (depth - 1))
                                : per_level;
    LevelLayout L;
    L.start = cursor;
    L.unary_start = cursor;
    L.n_unary = count / 4;                              // 25%
    L.binary_start = L.unary_start + L.n_unary;
    L.n_binary = count / 2;                             // 50%
    L.ternary_start = L.binary_start + L.n_binary;
    L.n_ternary = count - L.n_unary - L.n_binary;       // remaining ~25%
    L.end = L.ternary_start + L.n_ternary;
    layouts[k] = L;
    cursor = L.end;
  }
  return layouts;
}

// Find which level contains class id `i` via linear scan over layouts.
// Levels are typically <10 so this beats binary search.
std::size_t locate_level(const std::vector<LevelLayout>& layouts, std::size_t i) {
  for (std::size_t k = 0; k < layouts.size(); ++k) {
    if (i >= layouts[k].start && i < layouts[k].end) return k;
  }
  return layouts.size() - 1;  // shouldn't happen
}

template <typename UF>
BuiltGraph<UF> build(const Workload& w) {
  const auto layouts = compute_layouts(w);
  const std::size_t total = layouts.back().end;

  // Distinct seed namespace from closure_compare so the irregular workload
  // doesn't share random structure with any uniform workload of equal size.
  parlay::random_generator gen(0xBADC'0FFE'EBAD'BABEULL ^
                                static_cast<std::size_t>(w.n_nodes));

  auto all_nodes = parlay::tabulate(total, [&](std::size_t i) -> ENode {
    if (i < w.n_leaves) {
      return ENode{std::string("x") + std::to_string(i), {}};
    }
    if (i < 2 * w.n_leaves) {
      return ENode{std::string("y") + std::to_string(i - w.n_leaves), {}};
    }
    const std::size_t lvl = locate_level(layouts, i);
    const auto& L = layouts[lvl];

    int arity;
    if (i < L.binary_start)        arity = 1;
    else if (i < L.ternary_start)  arity = 2;
    else                           arity = 3;

    auto r = gen[i];
    ENode n;
    n.op = std::string("f") + std::to_string(arity) + "_" +
           std::to_string(lvl) + "_" +
           std::to_string(r() % w.n_fns);

    const std::size_t prev_start = (lvl == 0) ? 0 : layouts[lvl - 1].start;
    const std::size_t prev_end   = (lvl == 0) ? 2 * w.n_leaves
                                              : layouts[lvl - 1].end;
    const std::size_t prev_size  = prev_end - prev_start;

    n.children.reserve(static_cast<std::size_t>(arity));
    for (int a = 0; a < arity; ++a) {
      n.children.push_back(static_cast<Id>(prev_start + r() % prev_size));
    }
    return n;
  });

  auto eg = std::make_unique<EGraph<UF>>(std::move(all_nodes));

  // Initial unions: 60% same-arity leaf-leaf (half x-x, half y-y),
  // 40% cross-arity unary-vs-binary at level 0.
  const std::size_t n_cross    = w.n_merges * 4 / 10;
  const std::size_t n_leaf     = w.n_merges - n_cross;
  const std::size_t n_x_merges = n_leaf / 2;
  const auto& L0 = layouts[0];

  auto eq_seq = parlay::tabulate(w.n_merges, [&](std::size_t i) {
    auto r = gen[total + i];
    if (i < n_x_merges) {
      return std::pair<Id, Id>{
        static_cast<Id>(r() % w.n_leaves),
        static_cast<Id>(r() % w.n_leaves)
      };
    }
    if (i < n_leaf) {
      return std::pair<Id, Id>{
        static_cast<Id>(w.n_leaves + r() % w.n_leaves),
        static_cast<Id>(w.n_leaves + r() % w.n_leaves)
      };
    }
    // Cross-arity: pick a level-0 unary class and a level-0 binary class.
    return std::pair<Id, Id>{
      static_cast<Id>(L0.unary_start  + r() % L0.n_unary),
      static_cast<Id>(L0.binary_start + r() % L0.n_binary)
    };
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

bool parse_custom_workload(const char* spec, Workload& out) {
  out.name = "irregular-custom";
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
  const std::size_t par_threads = parlay::num_workers();

  const bool skip_nelson = std::getenv("PE_BENCH_SKIP_NELSON") != nullptr;
  const char* fmt = std::getenv("PE_BENCH_FORMAT");
  const bool csv = fmt && std::strcmp(fmt, "csv") == 0;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  const char* custom_spec = std::getenv("PE_BENCH_CUSTOM");
  const char* dnc_cutoff_env = std::getenv("PE_DNC_CUTOFF");
  const std::string dnc_cutoff = dnc_cutoff_env ? dnc_cutoff_env : "16";

  Workload w = DEFAULT_WORKLOAD;
  if (custom_spec) {
    if (!parse_custom_workload(custom_spec, w)) {
      std::fprintf(stderr,
                   "PE_BENCH_CUSTOM must be 'leaves,fns,nodes,merges,depth' "
                   "(got '%s')\n", custom_spec);
      return 2;
    }
  }

  if (csv && csv_header) {
    std::printf("workload,leaves,fns,nodes,merges,depth,algorithm,trial,"
                "parlay_threads,dnc_cutoff,wallclock_ms\n");
  } else if (!csv) {
    std::printf("irregular_bench  trials=%d  warmup=%d  par_threads=%zu\n",
                TRIALS, WARMUP, par_threads);
    std::printf("workload: %s  leaves=%zu  fns=%zu (per arity per level)  "
                "nodes=%zu  merges=%zu  depth=%zu\n",
                w.name, w.n_leaves, w.n_fns, w.n_nodes, w.n_merges, w.depth);
    std::printf("arity mix per level: 25%% unary, 50%% binary, 25%% ternary\n");
    std::printf("union mix: 60%% leaf-leaf same-arity, 40%% level-0 "
                "unary<->binary cross-arity\n\n");
  }

  auto emit_csv = [&](const char* algorithm, const std::vector<double>& times) {
    for (std::size_t i = 0; i < times.size(); ++i) {
      std::printf("%s,%zu,%zu,%zu,%zu,%zu,%s,%zu,%zu,%s,%.4f\n",
                  w.name, w.n_leaves, w.n_fns, w.n_nodes, w.n_merges, w.depth,
                  algorithm, i, par_threads, dnc_cutoff.c_str(), times[i]);
    }
  };

  std::fprintf(stderr, "[irregular] running %s ...\n", w.name);
  std::vector<double> nel;
  double mn = 0.0;
  if (!skip_nelson) {
    nel = bench_nelson(w);
    mn = median(nel);
  }
  auto par = bench_parallel_parents(w);
  const double mp = median(par);

  if (csv) {
    if (!skip_nelson) emit_csv("nelson_seq", nel);
    emit_csv("par_parents", par);
  } else {
    if (!skip_nelson) std::printf("nelson_seq median: %9.2fms\n", mn);
    std::printf("par_parents median:  %9.2fms\n", mp);
    if (!skip_nelson) std::printf("speedup:           %9.2fx\n", mn / mp);
  }

  return 0;
}
