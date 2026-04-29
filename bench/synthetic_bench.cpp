// In-process port of gen_bench.py's synthetic families. Builds the same
// DAGs directly into an EGraph via bulk_init (skipping the SMT-LIB
// parse/build path) and times sequential_close_nelson vs parallel_close on
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
//   PE_UNION_STYLE / PE_DNC_CUTOFF     tagged into CSV for plot grouping

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"

using namespace pe;

namespace {

constexpr int TRIALS = 1;
constexpr int WARMUP = 0;

enum class Family { Chain, Grid, Cube, Quartic, Quintic };

const char* family_name(Family f) {
  switch (f) {
    case Family::Chain:   return "chain";
    case Family::Grid:    return "grid";
    case Family::Cube:    return "cube";
    case Family::Quartic: return "quartic";
    case Family::Quintic: return "quintic";
  }
  return "?";
}

bool parse_family(const std::string& s, Family& out) {
  if (s == "chain")   { out = Family::Chain;   return true; }
  if (s == "grid")    { out = Family::Grid;    return true; }
  if (s == "cube")    { out = Family::Cube;    return true; }
  if (s == "quartic") { out = Family::Quartic; return true; }
  if (s == "quintic") { out = Family::Quintic; return true; }
  return false;
}

struct BuiltGraph {
  std::unique_ptr<EGraph> eg;
  parlay::sequence<std::pair<Id, Id>> eqs;
  std::size_t n_classes;
  std::size_t n_eqs;
};

// ---- Builders -------------------------------------------------------------
// Each builder lays out nodes in DAG order (children before parents) and
// hands the sequence to bulk_init. Class id == position in the input
// sequence, which we exploit when constructing the equality batch.

// chain: a0,b0 leaves; for i in 1..n, a_i = f(a_{i-1}), b_i = f(b_{i-1}).
// Layout: [a0, b0, a1, b1, ..., a_n, b_n]. Initial union: (a0, b0).
BuiltGraph build_chain(std::size_t n) {
  const std::size_t total = 2 * (n + 1);
  auto nodes = parlay::tabulate(total, [&](std::size_t i) -> ENode {
    const std::size_t lvl  = i / 2;     // 0 = leaves, k = layer-k constants
    const bool is_b        = (i % 2) == 1;
    if (lvl == 0) {
      return ENode{is_b ? std::string("b0") : std::string("a0"), {}};
    }
    // Parent is the same-side node one layer down.
    Id parent = static_cast<Id>(2 * (lvl - 1) + (is_b ? 1 : 0));
    return ENode{std::string("f"), {parent}};
  });

  parlay::sequence<std::pair<Id, Id>> eqs;
  eqs.push_back({Id{0}, Id{1}});
  const std::size_t n_eqs = eqs.size();

  auto eg = EGraph::bulk_init(std::move(nodes));
  return {std::move(eg), std::move(eqs), total, n_eqs};
}

// Helpers to build the polynomial-arity families. They share a common
// shape: 2n leaves, then 2 * n^k function nodes, with merges (a_i, b_i).
//
// Layout:
//   [a_0 .. a_{n-1}, b_0 .. b_{n-1},
//    f(a-tuples in lex order), f(b-tuples in lex order)]
// Class id of a_i = i, of b_i = n + i. Function nodes occupy
// [2n, 2n + n^k) for the a-side and [2n + n^k, 2n + 2*n^k) for the b-side.

BuiltGraph build_poly(std::size_t n, std::size_t arity) {
  // n^arity. We rely on the caller to pick n small enough that this fits.
  std::size_t per_side = 1;
  for (std::size_t k = 0; k < arity; ++k) per_side *= n;
  const std::size_t leaves = 2 * n;
  const std::size_t total  = leaves + 2 * per_side;

  auto nodes = parlay::tabulate(total, [&](std::size_t i) -> ENode {
    if (i < n) {
      return ENode{std::string("a") + std::to_string(i), {}};
    }
    if (i < 2 * n) {
      return ENode{std::string("b") + std::to_string(i - n), {}};
    }
    const std::size_t off = i - leaves;
    const bool is_b       = off >= per_side;
    const std::size_t idx = is_b ? off - per_side : off;
    // Decode lex index -> (arity)-tuple of digits in base n.
    std::vector<Id> children(arity);
    std::size_t rem = idx;
    for (std::size_t d = arity; d-- > 0;) {
      const std::size_t digit = rem % n;
      rem /= n;
      children[d] = static_cast<Id>((is_b ? n : 0) + digit);
    }
    return ENode{std::string("f"), std::move(children)};
  });

  // Initial unions: a_i = b_i for i in [0, n).
  auto eqs_seq = parlay::tabulate(n, [&](std::size_t i) {
    return std::pair<Id, Id>{static_cast<Id>(i), static_cast<Id>(n + i)};
  });

  auto eg = EGraph::bulk_init(std::move(nodes));
  return {std::move(eg), std::move(eqs_seq), total, n};
}

BuiltGraph build_grid   (std::size_t n) { return build_poly(n, 2); }
BuiltGraph build_cube   (std::size_t n) { return build_poly(n, 3); }
BuiltGraph build_quartic(std::size_t n) { return build_poly(n, 4); }
BuiltGraph build_quintic(std::size_t n) { return build_poly(n, 5); }

BuiltGraph build(Family f, std::size_t n) {
  switch (f) {
    case Family::Chain:   return build_chain(n);
    case Family::Grid:    return build_grid(n);
    case Family::Cube:    return build_cube(n);
    case Family::Quartic: return build_quartic(n);
    case Family::Quintic: return build_quintic(n);
  }
  std::abort();
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

std::vector<double> bench_nelson(Family f, std::size_t n) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build(f, n);
    g.eg->sequential_close_nelson(g.eqs);
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build(f, n);
    auto t0 = clk::now();
    g.eg->sequential_close_nelson(g.eqs);
    times.push_back(elapsed_ms(t0));
  }
  return times;
}

std::vector<double> bench_parallel_close(Family f, std::size_t n) {
  for (int i = 0; i < WARMUP; ++i) {
    auto g = build(f, n);
    g.eg->parallel_close(std::move(g.eqs));
  }
  std::vector<double> times;
  times.reserve(TRIALS);
  for (int i = 0; i < TRIALS; ++i) {
    auto g = build(f, n);
    auto t0 = clk::now();
    g.eg->parallel_close(std::move(g.eqs));
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
  const bool skip_nelson = std::getenv("PE_BENCH_SKIP_NELSON") != nullptr;
  const char* fmt = std::getenv("PE_BENCH_FORMAT");
  const bool csv = fmt && std::strcmp(fmt, "csv") == 0;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  const char* union_style_env = std::getenv("PE_UNION_STYLE");
  const char* dnc_cutoff_env  = std::getenv("PE_DNC_CUTOFF");
  const std::string union_style = union_style_env ? union_style_env : "dnc";
  const std::string dnc_cutoff  = dnc_cutoff_env  ? dnc_cutoff_env  : "16";

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

  if (!csv) {
    std::printf("synthetic_bench  trials=%d  warmup=%d  par_threads=%zu\n",
                TRIALS, WARMUP, par_threads);
    std::printf("%-8s %5s %10s %9s | %11s | %11s %11s\n",
                "family", "n", "classes", "merges",
                "nelson_seq", "par_close", "par_spd");
  } else if (csv_header) {
    std::printf("family,n,classes,merges,algorithm,trial,"
                "parlay_threads,union_style,dnc_cutoff,wallclock_ms\n");
  }

  auto emit_csv = [&](Family f, std::size_t n, std::size_t classes,
                      std::size_t merges, const char* algorithm,
                      const std::vector<double>& times) {
    for (std::size_t i = 0; i < times.size(); ++i) {
      std::printf("%s,%zu,%zu,%zu,%s,%zu,%zu,%s,%s,%.4f\n",
                  family_name(f), n, classes, merges,
                  algorithm, i, par_threads, union_style.c_str(),
                  dnc_cutoff.c_str(), times[i]);
    }
  };

  for (Family f : families) {
    for (std::size_t n : ns) {
      // Probe sizes once so we can print classes/merges even when nelson is
      // skipped. The build is cheap relative to the timed work.
      auto probe = build(f, n);
      const std::size_t classes = probe.n_classes;
      const std::size_t merges  = probe.n_eqs;

      std::fprintf(stderr, "[synthetic_bench] %s n=%zu classes=%zu merges=%zu ...\n",
                   family_name(f), n, classes, merges);

      std::vector<double> nel;
      double mn = 0.0;
      if (!skip_nelson) {
        nel = bench_nelson(f, n);
        mn = median(nel);
      }
      auto par = bench_parallel_close(f, n);
      double mp = median(par);

      if (csv) {
        if (!skip_nelson) emit_csv(f, n, classes, merges, "nelson_seq", nel);
        emit_csv(f, n, classes, merges, "par_close", par);
      } else if (skip_nelson) {
        std::printf("%-8s %5zu %10zu %9zu |   skipped  | %9.2fms\n",
                    family_name(f), n, classes, merges, mp);
      } else {
        std::printf("%-8s %5zu %10zu %9zu | %9.2fms | %9.2fms %9.2fx\n",
                    family_name(f), n, classes, merges, mn, mp, mn / mp);
      }
      std::fflush(stdout);
    }
  }
  return 0;
}
