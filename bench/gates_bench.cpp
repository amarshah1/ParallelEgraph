// Bench harness for the .gates file format used by the
// miter-cc-benchmarks suite. Loads each file, builds an EGraph, runs
// the three algorithms exercised on the FMCAD pipeline
// (nelson_simple_inline, par_parents, par_filter), and emits per-trial
// CSV.
//
// CSV schema:
//   file, suite, n_gates, n_literals, n_not_terms, total_classes,
//   algorithm, trial, parlay_threads, read_s, parse_s, build_s,
//   close_ms
//
// Env knobs:
//   PE_BENCH_TRIALS=N        measured trials per (file, algo) (default 5)
//   PE_BENCH_WARMUP=N        warmup invocations (default 1)
//   PE_BENCH_HEADER=1        emit CSV header (default off)
//   PE_BENCH_ALGOS=...       comma-separated EXACT names from
//                            {nelson_simple_inline, par_parents,
//                             par_filter}. Default = run all three.
//   PE_BENCH_PAR_ONLY=1      skip every sequential algo. Applies after
//                            PE_BENCH_ALGOS.
//   PE_TRACE=1               propagate to closure routines for per-round
//                            detail.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/gates_io.hpp"

namespace fs = std::filesystem;

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

std::string parent_suite(const fs::path& p) {
  return p.parent_path().filename().string();
}

template <typename UF, typename Tag, typename CloseFn>
double run_one(parlay::sequence<pe::ENode> nodes,
               parlay::sequence<std::pair<pe::Id, pe::Id>> eqs,
               Tag tag, CloseFn close) {
  auto eg = std::make_unique<pe::EGraph<UF>>(std::move(nodes), tag);
  auto t0 = clk::now();
  close(*eg, std::move(eqs));
  return ms_since(t0);
}

template <typename UF, typename CloseFn>
double run_one_seq(parlay::sequence<pe::ENode> nodes,
                   parlay::sequence<std::pair<pe::Id, pe::Id>> eqs,
                   CloseFn close) {
  auto eg = std::make_unique<pe::EGraph<UF>>(std::move(nodes));
  auto t0 = clk::now();
  close(*eg, eqs);
  return ms_since(t0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <file.gates> [more...]\n", argv[0]);
    return 2;
  }

  const int trials = std::getenv("PE_BENCH_TRIALS")
      ? std::atoi(std::getenv("PE_BENCH_TRIALS")) : 5;
  const int warmup = std::getenv("PE_BENCH_WARMUP")
      ? std::atoi(std::getenv("PE_BENCH_WARMUP")) : 1;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  const std::size_t par_threads = parlay::num_workers();

  std::set<std::string> algo_filter;
  if (const char* a = std::getenv("PE_BENCH_ALGOS")) {
    std::string s(a);
    std::size_t start = 0;
    while (start <= s.size()) {
      std::size_t end = s.find(',', start);
      if (end == std::string::npos) end = s.size();
      std::string tok = s.substr(start, end - start);
      while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t'))
        tok.erase(tok.begin());
      while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t'))
        tok.pop_back();
      if (!tok.empty()) algo_filter.insert(std::move(tok));
      if (end == s.size()) break;
      start = end + 1;
    }
  }
  auto algo_enabled = [&](const char* name) {
    return algo_filter.empty() || algo_filter.count(name) > 0;
  };
  const bool par_only = std::getenv("PE_BENCH_PAR_ONLY") != nullptr;
  const bool run_nelson_simple_inline = !par_only && algo_enabled("nelson_simple_inline");
  const bool run_par_parents          = algo_enabled("par_parents");
  const bool run_par_filter           = algo_enabled("par_filter");
  if (!run_nelson_simple_inline && !run_par_parents && !run_par_filter) {
    std::fprintf(stderr,
        "PE_BENCH_ALGOS matched nothing; valid algos are "
        "nelson_simple_inline, par_parents, par_filter\n");
    return 2;
  }

  if (csv_header) {
    std::printf("file,suite,n_gates,n_literals,n_not_terms,total_classes,"
                "algorithm,trial,parlay_threads,read_s,parse_s,build_s,"
                "close_ms\n");
  }

  for (int ai = 1; ai < argc; ++ai) {
    fs::path path(argv[ai]);
    std::string name = path.filename().string();
    std::string suite = parent_suite(path);
    std::fprintf(stderr, "[gates_bench] %s ...\n", path.string().c_str());

    pe::LoadedGates loaded;
    try {
      loaded = pe::load_gates_file(path.string());
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  load failed: %s\n", e.what());
      continue;
    }
    const std::size_t total_classes = loaded.nodes.size();
    std::fprintf(stderr,
        "  literals=%zu  NOT=%zu  gates=%zu  total_classes=%zu  "
        "read=%.3fs parse=%.3fs build=%.3fs\n",
        loaded.n_literals, loaded.n_not_terms, loaded.n_gates,
        total_classes, loaded.read_s, loaded.parse_s, loaded.build_s);

    auto emit_row = [&](const char* algo, int trial, double close_ms) {
      std::printf("%s,%s,%zu,%zu,%zu,%zu,%s,%d,%zu,%.6f,%.6f,%.6f,%.4f\n",
                  name.c_str(), suite.c_str(),
                  loaded.n_gates, loaded.n_literals, loaded.n_not_terms,
                  total_classes,
                  algo, trial, par_threads,
                  loaded.read_s, loaded.parse_s, loaded.build_s,
                  close_ms);
      std::fflush(stdout);
    };

    auto deep_copy_nodes = [&]() {
      return parlay::sequence<pe::ENode>(loaded.nodes);
    };
    auto deep_copy_eqs = [&]() {
      return parlay::sequence<std::pair<pe::Id, pe::Id>>(loaded.eqs);
    };

    if (run_nelson_simple_inline) {
      for (int i = 0; i < warmup; ++i) {
        run_one_seq<pe::SequentialUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              eg.sequential_close_simple_inline(eqs);
            });
      }
      for (int i = 0; i < trials; ++i) {
        double ms = run_one_seq<pe::SequentialUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              eg.sequential_close_simple_inline(eqs);
            });
        emit_row("nelson_simple_inline", i, ms);
      }
    }

    if (run_par_parents) {
      for (int i = 0; i < warmup; ++i) {
        run_one_seq<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              auto e = parlay::sequence<std::pair<pe::Id, pe::Id>>(eqs);
              eg.parallel_parents(std::move(e));
            });
      }
      for (int i = 0; i < trials; ++i) {
        double ms = run_one_seq<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              auto e = parlay::sequence<std::pair<pe::Id, pe::Id>>(eqs);
              eg.parallel_parents(std::move(e));
            });
        emit_row("par_parents", i, ms);
      }
    }

    if (run_par_filter) {
      for (int i = 0; i < warmup; ++i) {
        run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::filter,
            [](auto& eg, auto eqs) { eg.parallel_filter(std::move(eqs)); });
      }
      for (int i = 0; i < trials; ++i) {
        double ms = run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::filter,
            [](auto& eg, auto eqs) { eg.parallel_filter(std::move(eqs)); });
        emit_row("par_filter", i, ms);
      }
    }
  }
  return 0;
}
