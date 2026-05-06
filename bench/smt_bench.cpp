// SMT-LIB bench: runs parallel_close (and optionally sequential_close_nelson)
// on each .smt2 file under one or more paths. Filename convention follows
// gen_bench.py: <family>_n<N>_(sat|unsat).smt2.
//
// Output (CSV):
//   file,family,n,classes,equalities,algorithm,trial,parlay_threads,wallclock_ms
//
// Run as:
//   ./build/smt_bench <dir-or-file> [<dir-or-file> ...]
// Env:
//   PE_BENCH_HEADER=1        emit CSV header (once across appended runs)
//   PE_BENCH_SKIP_NELSON=1   skip the sequential baseline
//   PE_SMT_TRIALS=N          override default trials (11)
//   PE_SMT_WARMUP=N          override default warmup (3)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/smt_to_egraph.hpp"
#include "parallel_egraph/smtlib.hpp"

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path.string());
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

// `SmtToEGraphBuilder` lives in include/parallel_egraph/smt_to_egraph.hpp
// and is shared with the egraph-cc CLI (src/main.cpp).

// Build EGraph from a parsed script's asserted equalities (skipping
// disequalities — we don't time the sat/unsat verdict here). Construction
// happens once: walk all assertions through the builder, then construct
// the resulting flat ENode sequence.
template <typename UF>
struct Built {
  std::unique_ptr<pe::EGraph<UF>> eg;
  parlay::sequence<std::pair<pe::Id, pe::Id>> eqs;
  std::size_t classes;
};

// `Tag` selects the EGraph ctor flavor. The default-constructed value
// of pe::Script::Tag isn't a thing — Tag is meant to be one of:
//   * std::monostate-like sentinel (default ctor: parents_)
//   * pe::async_t (last_marked_)
//   * pe::topo_t  (depth_buckets_)
template <typename UF, typename Tag = std::monostate>
Built<UF> build_from_script(const pe::Script& script, Tag tag = {}) {
  pe::SmtToEGraphBuilder builder;
  parlay::sequence<std::pair<pe::Id, pe::Id>> eqs;
  for (const auto& cmd : script.commands) {
    if (cmd.kind != pe::Command::Kind::Assert) continue;
    const pe::Term& t = cmd.term;
    if (t.kind == pe::Term::Kind::Eq) {
      pe::Id a = builder.add_term(t.args[0]);
      pe::Id b = builder.add_term(t.args[1]);
      eqs.emplace_back(a, b);
    }
    // disequalities ignored for benchmarking
  }
  std::size_t classes = builder.size();
  std::unique_ptr<pe::EGraph<UF>> eg;
  if constexpr (std::is_same_v<Tag, std::monostate>) {
    eg = std::make_unique<pe::EGraph<UF>>(std::move(builder).take_nodes());
  } else {
    eg = std::make_unique<pe::EGraph<UF>>(std::move(builder).take_nodes(), tag);
  }
  return {std::move(eg), std::move(eqs), classes};
}

using clk = std::chrono::steady_clock;
double elapsed_ms(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Parse <family>_n<N>_(sat|unsat).smt2 → (family, n). Returns {"", 0} on
// non-matching filenames (e.g. hand-written regression cases).
std::pair<std::string, std::size_t> parse_filename(const std::string& stem) {
  static const std::regex re(R"(^([a-z]+)_n(\d+)_(?:sat|unsat)$)");
  std::smatch m;
  if (!std::regex_match(stem, m, re)) return {"", 0};
  return {m[1].str(), static_cast<std::size_t>(std::stoull(m[2].str()))};
}

template <typename UF, typename Tag, typename Close>
void run_one(const pe::Script& script, const std::string& stem,
             const std::string& family, std::size_t n,
             int trials, int warmup, std::size_t par_threads,
             const char* algorithm, Tag tag, Close close) {
  for (int i = 0; i < warmup; ++i) {
    auto b = build_from_script<UF>(script, tag);
    close(*b.eg, b.eqs);
  }
  for (int i = 0; i < trials; ++i) {
    auto b = build_from_script<UF>(script, tag);
    const std::size_t neq = b.eqs.size();
    auto t0 = clk::now();
    close(*b.eg, b.eqs);
    double ms = elapsed_ms(t0);
    std::printf("%s,%s,%zu,%zu,%zu,%s,%d,%zu,%.4f\n",
                stem.c_str(), family.c_str(), n,
                b.classes, neq, algorithm, i, par_threads, ms);
    std::fflush(stdout);
  }
}

void collect_paths(const fs::path& root, std::vector<fs::path>& out) {
  if (fs::is_regular_file(root) && root.extension() == ".smt2") {
    out.push_back(root);
    return;
  }
  if (!fs::is_directory(root)) {
    std::fprintf(stderr, "skip: %s (not a file or directory)\n",
                 root.string().c_str());
    return;
  }
  for (const auto& entry : fs::directory_iterator(root)) {
    if (entry.is_regular_file() && entry.path().extension() == ".smt2") {
      out.push_back(entry.path());
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <dir-or-file> [more...]\n", argv[0]);
    return 2;
  }

  const bool skip_nelson = std::getenv("PE_BENCH_SKIP_NELSON") != nullptr;
  const bool csv_header = std::getenv("PE_BENCH_HEADER") != nullptr;
  const int trials =
      std::getenv("PE_SMT_TRIALS") ? std::atoi(std::getenv("PE_SMT_TRIALS")) : 11;
  const int warmup =
      std::getenv("PE_SMT_WARMUP") ? std::atoi(std::getenv("PE_SMT_WARMUP")) : 3;
  const std::size_t par_threads = parlay::num_workers();

  if (csv_header) {
    std::printf("file,family,n,classes,equalities,algorithm,trial,"
                "parlay_threads,wallclock_ms\n");
  }

  std::vector<fs::path> paths;
  for (int i = 1; i < argc; ++i) collect_paths(argv[i], paths);
  std::sort(paths.begin(), paths.end());

  for (const auto& p : paths) {
    std::string stem = p.stem().string();
    auto fam_n = parse_filename(stem);
    const std::string& family = fam_n.first;
    const std::size_t n = fam_n.second;
    std::fprintf(stderr, "[smt_bench] %s ...\n", p.string().c_str());

    pe::Script script;
    try {
      script = pe::parse_smtlib(read_file(p));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "  parse failed: %s\n", e.what());
      continue;
    }

    if (!skip_nelson) {
      run_one<pe::SequentialUnionFind>(script, stem, family, n, trials,
          warmup, par_threads, "nelson_seq", std::monostate{},
          [](auto& eg, auto& eqs) { eg.sequential_close_nelson(eqs); });
    }
    run_one<pe::SequentialUnionFind>(script, stem, family, n, trials,
        warmup, par_threads, "nelson_topo", std::monostate{},
        [](auto& eg, auto& eqs) { eg.sequential_close_topo(eqs); });
    run_one<pe::SequentialUnionFind>(script, stem, family, n, trials,
        warmup, par_threads, "nelson_topo_iter", std::monostate{},
        [](auto& eg, auto& eqs) { eg.sequential_close_topo_iter(eqs); });
    run_one<pe::SequentialUnionFind>(script, stem, family, n, trials,
        warmup, par_threads, "nelson_dst", std::monostate{},
        [](auto& eg, auto& eqs) { eg.sequential_close_dst(eqs); });
    run_one<pe::ConcurrentUnionFind>(script, stem, family, n, trials,
        warmup, par_threads, "par_close", std::monostate{},
        [](auto& eg, auto& eqs) { eg.parallel_close(std::move(eqs)); });
    run_one<pe::ConcurrentUnionFind>(script, stem, family, n, trials,
        warmup, par_threads, "par_async", pe::async,
        [](auto& eg, auto& eqs) {
          eg.parallel_close_async_rounds(std::move(eqs));
        });
    run_one<pe::ConcurrentUnionFind>(script, stem, family, n, trials,
        warmup, par_threads, "par_async_min_id", pe::async,
        [](auto& eg, auto& eqs) {
          eg.parallel_close_async_rounds_min_id(std::move(eqs));
        });
  }
  return 0;
}
