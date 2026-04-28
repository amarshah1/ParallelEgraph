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
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
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

std::size_t count_subterms(const pe::Term& t) {
  std::size_t n = 1;
  for (const auto& a : t.args) n += count_subterms(a);
  return n;
}

// Matches main.cpp's add_term: iterative add of a Term tree to an EGraph.
pe::Id add_term(pe::EGraph& eg, const pe::Term& term) {
  enum class WK { Process, Build };
  struct W { WK kind; const pe::Term* term; std::string op; std::size_t nargs; };
  std::vector<W> stack;
  std::vector<pe::Id> results;
  stack.push_back({WK::Process, &term, {}, 0});
  while (!stack.empty()) {
    W w = std::move(stack.back());
    stack.pop_back();
    if (w.kind == WK::Process) {
      const pe::Term& t = *w.term;
      switch (t.kind) {
        case pe::Term::Kind::Const: {
          pe::ENode leaf;
          leaf.op = t.op;
          results.push_back(eg.add(std::move(leaf)));
          break;
        }
        case pe::Term::Kind::App:
          stack.push_back({WK::Build, nullptr, t.op, t.args.size()});
          for (auto it = t.args.rbegin(); it != t.args.rend(); ++it) {
            stack.push_back({WK::Process, &*it, {}, 0});
          }
          break;
        case pe::Term::Kind::Eq:
        case pe::Term::Kind::Not:
          throw std::runtime_error("= and not are not first-class terms");
      }
    } else {
      std::size_t start = results.size() - w.nargs;
      std::vector<pe::Id> children(results.begin() + start, results.end());
      results.erase(results.begin() + start, results.end());
      pe::ENode node;
      node.op = std::move(w.op);
      node.children = std::move(children);
      results.push_back(eg.add(std::move(node)));
    }
  }
  return results.back();
}

// Build EGraph from a parsed script's asserted equalities (skipping
// disequalities — we don't time the sat/unsat verdict here).
struct Built {
  std::unique_ptr<pe::EGraph> eg;
  parlay::sequence<std::pair<pe::Id, pe::Id>> eqs;
  std::size_t classes;
};

Built build_from_script(const pe::Script& script) {
  std::size_t capacity = 0;
  for (const auto& cmd : script.commands) {
    if (cmd.kind == pe::Command::Kind::Assert) {
      capacity += count_subterms(cmd.term);
    }
  }
  auto eg = std::make_unique<pe::EGraph>(capacity);
  parlay::sequence<std::pair<pe::Id, pe::Id>> eqs;
  for (const auto& cmd : script.commands) {
    if (cmd.kind != pe::Command::Kind::Assert) continue;
    const pe::Term& t = cmd.term;
    if (t.kind == pe::Term::Kind::Eq) {
      pe::Id a = add_term(*eg, t.args[0]);
      pe::Id b = add_term(*eg, t.args[1]);
      eqs.emplace_back(a, b);
    }
    // disequalities ignored for benchmarking
  }
  std::size_t classes = eg->nodes().size();
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

    auto run_one = [&](const char* algorithm) {
      for (int i = 0; i < warmup; ++i) {
        auto b = build_from_script(script);
        if (std::strcmp(algorithm, "par_close") == 0) {
          b.eg->parallel_close(std::move(b.eqs));
        } else {
          b.eg->sequential_close_nelson(b.eqs);
        }
      }
      for (int i = 0; i < trials; ++i) {
        auto b = build_from_script(script);
        const std::size_t neq = b.eqs.size();
        auto t0 = clk::now();
        if (std::strcmp(algorithm, "par_close") == 0) {
          b.eg->parallel_close(std::move(b.eqs));
        } else {
          b.eg->sequential_close_nelson(b.eqs);
        }
        double ms = elapsed_ms(t0);
        std::printf("%s,%s,%zu,%zu,%zu,%s,%d,%zu,%.4f\n",
                    stem.c_str(), family.c_str(), n,
                    b.classes, neq,
                    algorithm, i, par_threads, ms);
        std::fflush(stdout);
      }
    };

    if (!skip_nelson) run_one("nelson_seq");
    run_one("par_close");
  }
  return 0;
}
