// Bench harness for the .gates file format used by the
// miter-cc-benchmarks suite. Loads each file, builds an EGraph, runs
// nelson_topo_iter (sequential), par_topo_iter, and
// parallel_close_async_rounds with PE_BENCH_WARMUP warmup +
// PE_BENCH_TRIALS measured trials per algorithm, and emits per-trial
// CSV.
//
// CSV schema:
//   file, suite, n_gates, n_literals, n_not_terms, total_classes,
//   algorithm, trial, parlay_threads, read_s, parse_s, build_s,
//   close_ms
//
// Phase timings (read/parse/build) are reported once per file (they
// don't depend on the algorithm or thread count); the algorithm-
// specific row reports them too so each row is self-describing.
//
// Env knobs:
//   PE_BENCH_TRIALS=N        measured trials per (file, algo) (default 5)
//   PE_BENCH_WARMUP=N        warmup invocations (default 1)
//   PE_BENCH_HEADER=1        emit CSV header (default off; let the
//                            python driver gate it)
//   PE_BENCH_ALGOS=...       comma-separated EXACT names from
//                            {nelson_simple, nelson_topo_iter,
//                             par_close, par_topo_iter, par_async,
//                             par_async_cont, par_async_hybrid}.
//                            Default = run all seven.
//   PE_BENCH_PAR_ONLY=1      skip every sequential algo (nelson_*).
//                            Set on T>1 invocations so we don't
//                            re-measure the thread-independent
//                            sequential baseline at every thread
//                            count. Applies after PE_BENCH_ALGOS.
//   PE_TRACE=1               propagate to the closure routines for
//                            per-round detail
//
// Usage:
//   ./build/gates_bench <file.gates> [<file.gates> ...]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

// "iwls22/h5m-n01-ands.gates" → "iwls22"
std::string parent_suite(const fs::path& p) {
  return p.parent_path().filename().string();
}

// Build an EGraph from the loaded gates with the given ctor tag and
// run `close` on it. Returns wallclock ms of the close call only.
template <typename UF, typename Tag, typename CloseFn>
double run_one(parlay::sequence<pe::ENode> nodes,
               parlay::sequence<std::pair<pe::Id, pe::Id>> eqs,
               Tag tag, CloseFn close) {
  auto eg = std::make_unique<pe::EGraph<UF>>(std::move(nodes), tag);
  auto t0 = clk::now();
  close(*eg, std::move(eqs));
  return ms_since(t0);
}

// Tagless overload for the sequential path (default ctor; nelson-style
// sequential closures don't use the topo/async/parents auxiliary
// state).
template <typename UF, typename CloseFn>
double run_one_seq(parlay::sequence<pe::ENode> nodes,
                   parlay::sequence<std::pair<pe::Id, pe::Id>> eqs,
                   CloseFn close) {
  auto eg = std::make_unique<pe::EGraph<UF>>(std::move(nodes));
  auto t0 = clk::now();
  close(*eg, eqs);  // sequential closures take eqs by const ref
  return ms_since(t0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <file.gates> [more...]\n", argv[0]);
    return 2;
  }

  const int trials = std::getenv("PE_BENCH_TRIALS")
      ? std::atoi(std::getenv("PE_BENCH_TRIALS"))
      : 5;
  const int warmup = std::getenv("PE_BENCH_WARMUP")
      ? std::atoi(std::getenv("PE_BENCH_WARMUP"))
      : 1;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  const std::size_t par_threads = parlay::num_workers();

  // Parse PE_BENCH_ALGOS as a comma-separated list of EXACT algorithm
  // names. Default (env unset) = run every algorithm. If the env is set,
  // only the listed ones run. Exact-match (not substring) so e.g.
  // "par_async" does NOT match "par_async_cont".
  bool run_nelson_simple    = true;
  bool run_nelson_topo_iter = true;
  bool run_par_close        = true;
  bool run_topo             = true;
  bool run_async            = true;
  bool run_async_cont       = true;
  bool run_async_hybrid     = true;
  if (const char* a = std::getenv("PE_BENCH_ALGOS")) {
    std::string s(a);
    // Tokenize on commas.
    std::vector<std::string> toks;
    std::size_t start = 0;
    while (start <= s.size()) {
      std::size_t end = s.find(',', start);
      if (end == std::string::npos) end = s.size();
      std::string tok = s.substr(start, end - start);
      // Trim leading/trailing whitespace.
      while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t'))
        tok.erase(tok.begin());
      while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t'))
        tok.pop_back();
      if (!tok.empty()) toks.push_back(std::move(tok));
      if (end == s.size()) break;
      start = end + 1;
    }
    auto has = [&](const std::string& want) {
      for (const auto& t : toks) if (t == want) return true;
      return false;
    };
    run_nelson_simple    = has("nelson_simple");
    run_nelson_topo_iter = has("nelson_topo_iter");
    run_par_close        = has("par_close");
    run_topo             = has("par_topo_iter");
    run_async            = has("par_async");
    run_async_cont       = has("par_async_cont");
    run_async_hybrid     = has("par_async_hybrid");
    if (!run_nelson_simple && !run_nelson_topo_iter && !run_par_close &&
        !run_topo && !run_async && !run_async_cont && !run_async_hybrid) {
      std::fprintf(stderr,
          "PE_BENCH_ALGOS=%s matched nothing; valid algos are "
          "nelson_simple, nelson_topo_iter, par_close, par_topo_iter, "
          "par_async, par_async_cont, par_async_hybrid\n", a);
      return 2;
    }
  }
  // PE_BENCH_PAR_ONLY=1: skip every sequential algorithm. Mirrors the
  // synthetic_bench / closure_compare convention. The intent is "we
  // already measured the sequential at T=1, don't re-run it at T>1
  // where it produces the same number 4 times more slowly." Applies
  // AFTER the PE_BENCH_ALGOS filter, so PE_BENCH_ALGOS=nelson_simple
  // + PE_BENCH_PAR_ONLY=1 = no algorithms run.
  if (std::getenv("PE_BENCH_PAR_ONLY")) {
    run_nelson_simple    = false;
    run_nelson_topo_iter = false;
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
      // Each trial needs a fresh EGraph (closure is destructive). Copy
      // the nodes; eqs we move once per trial below from a fresh
      // copy too.
      return parlay::sequence<pe::ENode>(loaded.nodes);
    };
    auto deep_copy_eqs = [&]() {
      return parlay::sequence<std::pair<pe::Id, pe::Id>>(loaded.eqs);
    };

    if (run_nelson_simple) {
      for (int i = 0; i < warmup; ++i) {
        run_one_seq<pe::SequentialUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              eg.sequential_close_simple(eqs);
            });
      }
      for (int i = 0; i < trials; ++i) {
        double ms = run_one_seq<pe::SequentialUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              eg.sequential_close_simple(eqs);
            });
        emit_row("nelson_simple", i, ms);
      }
    }

    if (run_nelson_topo_iter) {
      // Warmup
      for (int i = 0; i < warmup; ++i) {
        run_one_seq<pe::SequentialUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              eg.sequential_close_topo_iter(eqs);
            });
      }
      // Trials
      for (int i = 0; i < trials; ++i) {
        double ms = run_one_seq<pe::SequentialUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              eg.sequential_close_topo_iter(eqs);
            });
        emit_row("nelson_topo_iter", i, ms);
      }
    }

    if (run_par_close) {
      // BSP path uses the default ctor (parents_-driven, no async/topo
      // auxiliary state). Mirrors run_one_seq's tagless build.
      for (int i = 0; i < warmup; ++i) {
        run_one_seq<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              // parallel_close takes eqs by value (moves); copy here.
              auto e = parlay::sequence<std::pair<pe::Id, pe::Id>>(eqs);
              eg.parallel_close(std::move(e));
            });
      }
      for (int i = 0; i < trials; ++i) {
        double ms = run_one_seq<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(),
            [](auto& eg, const auto& eqs) {
              auto e = parlay::sequence<std::pair<pe::Id, pe::Id>>(eqs);
              eg.parallel_close(std::move(e));
            });
        emit_row("par_close", i, ms);
      }
    }

    if (run_topo) {
      // Warmup
      for (int i = 0; i < warmup; ++i) {
        run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::topo,
            [](auto& eg, auto eqs) { eg.parallel_close_topo_iter(std::move(eqs)); });
      }
      // Trials
      for (int i = 0; i < trials; ++i) {
        double ms = run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::topo,
            [](auto& eg, auto eqs) { eg.parallel_close_topo_iter(std::move(eqs)); });
        emit_row("par_topo_iter", i, ms);
      }
    }

    if (run_async) {
      for (int i = 0; i < warmup; ++i) {
        run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::async,
            [](auto& eg, auto eqs) { eg.parallel_close_async_rounds(std::move(eqs)); });
      }
      for (int i = 0; i < trials; ++i) {
        double ms = run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::async,
            [](auto& eg, auto eqs) { eg.parallel_close_async_rounds(std::move(eqs)); });
        emit_row("par_async", i, ms);
      }
    }

    if (run_async_cont) {
      // Truly-async (continuous): same async_t-tagged ctor as par_async.
      for (int i = 0; i < warmup; ++i) {
        run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::async,
            [](auto& eg, auto eqs) { eg.parallel_close_async_continuous(std::move(eqs)); });
      }
      for (int i = 0; i < trials; ++i) {
        double ms = run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::async,
            [](auto& eg, auto eqs) { eg.parallel_close_async_continuous(std::move(eqs)); });
        emit_row("par_async_cont", i, ms);
      }
    }

    if (run_async_hybrid) {
      // Hybrid filter / parents-walk: uses the hybrid_t-tagged ctor,
      // which builds both `parents_` and `last_marked_`.
      for (int i = 0; i < warmup; ++i) {
        run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::hybrid,
            [](auto& eg, auto eqs) { eg.parallel_close_async_hybrid(std::move(eqs)); });
      }
      for (int i = 0; i < trials; ++i) {
        double ms = run_one<pe::ConcurrentUnionFind>(
            deep_copy_nodes(), deep_copy_eqs(), pe::hybrid,
            [](auto& eg, auto eqs) { eg.parallel_close_async_hybrid(std::move(eqs)); });
        emit_row("par_async_hybrid", i, ms);
      }
    }
  }
  return 0;
}
