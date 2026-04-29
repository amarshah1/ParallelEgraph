// egraph-cc — congruence closure on a QF_UF SMT-LIB 2 input.
//
// Parse the script, build an EGraph from the asserted equalities and
// disequalities, run parallel_close on the equality list, and report
// sat / unsat: unsat iff any asserted disequality (a != b) collapses
// (find_root(a) == find_root(b)) after closure.

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
#include "parallel_egraph/smtlib.hpp"

namespace {

int usage(const char* prog) {
  std::fprintf(stderr,
               "usage: %s [--timing] <file.smt2>\n"
               "  --timing  print per-phase timings to stderr "
               "(read/parse/build/close/check/dtor)\n",
               prog);
  return 2;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

// Subterm count of a term tree — used as an upper bound on EGraph capacity.
std::size_t count_subterms(const pe::Term& t) {
  std::size_t n = 1;
  for (const auto& a : t.args) n += count_subterms(a);
  return n;
}

// Iteratively add a Term to the e-graph, returning its e-class id.
// Iterative form (not recursive) to handle deeply nested inputs.
pe::Id add_term(pe::EGraph& eg, const pe::Term& term) {
  enum class WK { Process, Build };
  struct W {
    WK kind;
    const pe::Term* term;     // for Process
    std::string op;           // for Build
    std::size_t nargs;        // for Build
  };
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
        case pe::Term::Kind::App: {
          stack.push_back({WK::Build, nullptr, t.op, t.args.size()});
          // Reverse so first child is processed first.
          for (auto it = t.args.rbegin(); it != t.args.rend(); ++it) {
            stack.push_back({WK::Process, &*it, {}, 0});
          }
          break;
        }
        case pe::Term::Kind::Eq:
        case pe::Term::Kind::Not:
          throw std::runtime_error(
              "= and not are not first-class terms; only allowed at "
              "the top of an assertion");
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

}  // namespace

// Phase-timing harness. When --timing is on, we record clock points
// between phases and print them to stderr just before returning. We
// also estimate destructor time via a static guard that captures the
// "main returned" timestamp and emits the dtor delta from its own
// destructor (which fires after main's locals have unwound).
struct TimingState {
  bool enabled = false;
  using clk = std::chrono::steady_clock;
  clk::time_point t_start;
  clk::time_point t_after_read;
  clk::time_point t_after_parse;
  clk::time_point t_after_build;
  clk::time_point t_after_close;
  clk::time_point t_after_check;
  clk::time_point t_after_main;  // set just before main returns
  bool main_returned = false;

  static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  }
};
static TimingState g_timing;

// Module-scoped guard whose destructor fires after main's locals have
// unwound (file-scoped statics are destroyed in reverse construction
// order; this one's last). Uses g_timing.t_after_main as its baseline.
struct DtorGuard {
  ~DtorGuard() {
    if (!g_timing.enabled || !g_timing.main_returned) return;
    auto t_now = TimingState::clk::now();
    double dtor_s = TimingState::secs(g_timing.t_after_main, t_now);
    std::fprintf(stderr, " dtor=%.6f\n", dtor_s);
  }
};
static DtorGuard g_dtor_guard;

int main(int argc, char** argv) {
  g_timing.t_start = TimingState::clk::now();

  // Parse our own flags. Single positional <file.smt2> + optional --timing.
  const char* file = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--timing") == 0) {
      g_timing.enabled = true;
    } else if (file == nullptr) {
      file = argv[i];
    } else {
      return usage(argv[0]);
    }
  }
  if (file == nullptr) return usage(argv[0]);

  std::string input;
  try {
    input = read_file(file);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  g_timing.t_after_read = TimingState::clk::now();

  pe::Script script;
  try {
    script = pe::parse_smtlib(input);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  g_timing.t_after_parse = TimingState::clk::now();

  // Pre-walk to bound the e-graph capacity.
  std::size_t capacity = 0;
  for (const auto& cmd : script.commands) {
    if (cmd.kind == pe::Command::Kind::Assert) {
      capacity += count_subterms(cmd.term);
    }
  }

  pe::EGraph eg(capacity);
  parlay::sequence<std::pair<pe::Id, pe::Id>> equalities;
  std::vector<std::pair<pe::Id, pe::Id>> disequalities;

  try {
    for (const auto& cmd : script.commands) {
      if (cmd.kind != pe::Command::Kind::Assert) continue;
      const pe::Term& t = cmd.term;
      if (t.kind == pe::Term::Kind::Eq) {
        pe::Id a = add_term(eg, t.args[0]);
        pe::Id b = add_term(eg, t.args[1]);
        equalities.emplace_back(a, b);
      } else if (t.kind == pe::Term::Kind::Not && t.args.size() == 1 &&
                 t.args[0].kind == pe::Term::Kind::Eq) {
        pe::Id a = add_term(eg, t.args[0].args[0]);
        pe::Id b = add_term(eg, t.args[0].args[1]);
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
  g_timing.t_after_build = TimingState::clk::now();

  eg.parallel_close(std::move(equalities));
  g_timing.t_after_close = TimingState::clk::now();

  bool unsat = false;
  for (auto [a, b] : disequalities) {
    if (eg.equiv(a, b)) {
      unsat = true;
      break;
    }
  }
  g_timing.t_after_check = TimingState::clk::now();

  std::puts(unsat ? "unsat" : "sat");

  if (g_timing.enabled) {
    // Print a single timing: line that the bench harness can regex.
    // The dtor delta is appended by DtorGuard after all local
    // destructors run (DtorGuard emits the trailing newline).
    auto& ts = g_timing;
    std::fprintf(stderr,
                 "timing: read=%.6f parse=%.6f build=%.6f "
                 "close=%.6f check=%.6f",
                 TimingState::secs(ts.t_start, ts.t_after_read),
                 TimingState::secs(ts.t_after_read, ts.t_after_parse),
                 TimingState::secs(ts.t_after_parse, ts.t_after_build),
                 TimingState::secs(ts.t_after_build, ts.t_after_close),
                 TimingState::secs(ts.t_after_close, ts.t_after_check));
    g_timing.t_after_main = TimingState::clk::now();
    g_timing.main_returned = true;
    // No newline yet — DtorGuard appends " dtor=..." then \n.
  }
  return 0;
}
