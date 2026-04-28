// egraph-cc — congruence closure on a QF_UF SMT-LIB 2 input.
//
// Parse the script, build an EGraph from the asserted equalities and
// disequalities, run parallel_close on the equality list, and report
// sat / unsat: unsat iff any asserted disequality (a != b) collapses
// (find_root(a) == find_root(b)) after closure.

#include <cstdio>
#include <cstdlib>
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
  std::fprintf(stderr, "usage: %s <file.smt2>\n", prog);
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

int main(int argc, char** argv) {
  if (argc != 2) return usage(argv[0]);
  std::string input;
  try {
    input = read_file(argv[1]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  pe::Script script;
  try {
    script = pe::parse_smtlib(input);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

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

  eg.parallel_close(std::move(equalities));

  for (auto [a, b] : disequalities) {
    if (eg.equiv(a, b)) {
      std::puts("unsat");
      return 0;
    }
  }
  std::puts("sat");
  return 0;
}
