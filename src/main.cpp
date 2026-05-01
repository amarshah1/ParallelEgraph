// egraph-cc — congruence closure on a QF_UF SMT-LIB 2 input.
//
// Parses the script, builds the e-graph (parser-side hashcons + bulk_init)
// from the asserted equalities, runs parallel_close on the equality list,
// and reports sat / unsat: unsat iff any asserted disequality (a != b)
// collapses (find_root(a) == find_root(b)) after closure.
//
// Pass --timing for a final stderr line:
//   timing: read=...ms parse=...ms build=...ms close=...ms check=...ms dtor=...ms
// (consumed by run_all_benchmarks.py's egg suite). PE_TRACE=1 also still
// works and emits the per-round breakdowns.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/smt_to_egraph.hpp"
#include "parallel_egraph/smtlib.hpp"

namespace {

int usage(const char* prog) {
  std::fprintf(stderr, "usage: %s [--timing] <file.smt2>\n", prog);
  return 2;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

using clk = std::chrono::steady_clock;
double elapsed_ms(clk::time_point t0, clk::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  bool emit_timing = false;
  const char* path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--timing") == 0) {
      emit_timing = true;
    } else if (path == nullptr) {
      path = argv[i];
    } else {
      return usage(argv[0]);
    }
  }
  if (path == nullptr) return usage(argv[0]);

  auto t_start = clk::now();

  std::string input;
  try {
    input = read_file(path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  auto t_read = clk::now();

  pe::Script script;
  try {
    script = pe::parse_smtlib(input);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  auto t_parse = clk::now();

  // Build phase: walk every assertion's terms through the parser-side
  // hashcons, collect (a, b) for equalities and disequalities.
  pe::SmtToEGraphBuilder builder;
  parlay::sequence<std::pair<pe::Id, pe::Id>> equalities;
  std::vector<std::pair<pe::Id, pe::Id>> disequalities;

  try {
    for (const auto& cmd : script.commands) {
      if (cmd.kind != pe::Command::Kind::Assert) continue;
      const pe::Term& t = cmd.term;
      if (t.kind == pe::Term::Kind::Eq) {
        pe::Id a = builder.add_term(t.args[0]);
        pe::Id b = builder.add_term(t.args[1]);
        equalities.emplace_back(a, b);
      } else if (t.kind == pe::Term::Kind::Not && t.args.size() == 1 &&
                 t.args[0].kind == pe::Term::Kind::Eq) {
        pe::Id a = builder.add_term(t.args[0].args[0]);
        pe::Id b = builder.add_term(t.args[0].args[1]);
        disequalities.emplace_back(a, b);
      } else {
        std::fprintf(stderr,
            "unsupported assertion shape (only (= a b) and (not (= a b)))\n");
        return 1;
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  auto eg = pe::EGraph::bulk_init(std::move(builder).take_nodes());
  auto t_build = clk::now();

  eg->parallel_close(std::move(equalities));
  auto t_close = clk::now();

  bool unsat = false;
  for (auto [a, b] : disequalities) {
    if (eg->equiv(a, b)) { unsat = true; break; }
  }
  auto t_check = clk::now();

  std::puts(unsat ? "unsat" : "sat");
  std::fflush(stdout);

  // The EGraph dtor runs at function exit; capture its cost too.
  eg.reset();
  auto t_dtor = clk::now();

  if (emit_timing) {
    std::fprintf(stderr,
                 "timing: read=%.2fms parse=%.2fms build=%.2fms "
                 "close=%.2fms check=%.2fms dtor=%.2fms\n",
                 elapsed_ms(t_start, t_read),
                 elapsed_ms(t_read,  t_parse),
                 elapsed_ms(t_parse, t_build),
                 elapsed_ms(t_build, t_close),
                 elapsed_ms(t_close, t_check),
                 elapsed_ms(t_check, t_dtor));
  }

  return 0;
}
