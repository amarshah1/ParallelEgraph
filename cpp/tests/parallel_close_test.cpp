// Regression for the `parallel_close` BSP closure. Exercises it directly
// (not via solve_with_mode) on exp_n11_unsat.smt2 — the bench that first
// exposed the parent_index consolidation soundness bug.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/smtlib.hpp"

namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "open(%s): fail\n", path.c_str()); std::exit(1); }
  std::stringstream buf; buf << f.rdbuf(); return buf.str();
}

std::size_t count_subterms(const pe::Term& t) {
  std::size_t n = 1;
  for (const auto& a : t.args) n += count_subterms(a);
  return n;
}

pe::Id add_term(pe::EGraph& eg, const pe::Term& term) {
  enum class WK { Proc, Build };
  struct W { WK kind; const pe::Term* term; std::string op; std::size_t nargs; };
  std::vector<W> stack; std::vector<pe::Id> res;
  stack.push_back({WK::Proc, &term, {}, 0});
  while (!stack.empty()) {
    auto w = std::move(stack.back()); stack.pop_back();
    if (w.kind == WK::Proc) {
      const auto& t = *w.term;
      if (t.kind == pe::Term::Kind::Const) {
        pe::ENode leaf; leaf.op = t.op;
        res.push_back(eg.add(std::move(leaf)));
      } else if (t.kind == pe::Term::Kind::App) {
        stack.push_back({WK::Build, nullptr, t.op, t.args.size()});
        for (auto it = t.args.rbegin(); it != t.args.rend(); ++it)
          stack.push_back({WK::Proc, &*it, {}, 0});
      }
    } else {
      std::size_t s = res.size() - w.nargs;
      std::vector<pe::Id> ch(res.begin() + s, res.end());
      res.erase(res.begin() + s, res.end());
      pe::ENode n; n.op = std::move(w.op); n.children = std::move(ch);
      res.push_back(eg.add(std::move(n)));
    }
  }
  return res.back();
}

}  // namespace

int main() {
#ifndef SYNTHETIC_DIR
#error "SYNTHETIC_DIR must be defined"
#endif
  std::string path = std::string(SYNTHETIC_DIR) + "/exp_n11_unsat.smt2";
  std::string input = read_file(path);

  pe::Script script = pe::parse_smtlib(input);

  // Capacity = subterm count across all assertions.
  std::size_t capacity = 0;
  for (const auto& c : script.commands) {
    if (c.kind == pe::Command::Kind::Assert) capacity += count_subterms(c.term);
  }
  pe::EGraph eg(capacity, /*parallel=*/true);

  parlay::sequence<std::pair<pe::Id, pe::Id>> eqs;
  std::vector<std::pair<pe::Id, pe::Id>> diseqs;
  for (const auto& c : script.commands) {
    if (c.kind != pe::Command::Kind::Assert) continue;
    const auto& t = c.term;
    if (t.kind == pe::Term::Kind::Eq) {
      pe::Id a = add_term(eg, t.args[0]);
      pe::Id b = add_term(eg, t.args[1]);
      eqs.emplace_back(a, b);
    } else if (t.kind == pe::Term::Kind::Not && t.args.size() == 1 &&
               t.args[0].kind == pe::Term::Kind::Eq) {
      pe::Id a = add_term(eg, t.args[0].args[0]);
      pe::Id b = add_term(eg, t.args[0].args[1]);
      diseqs.emplace_back(a, b);
    }
  }

  eg.parallel_close(std::move(eqs));

  bool unsat = false;
  for (auto [a, b] : diseqs) {
    if (eg.equiv(a, b)) { unsat = true; break; }
  }

  if (unsat) {
    std::puts("parallel_close_test: PASS (exp_n11 -> unsat via parallel_close)");
    return 0;
  }
  std::fprintf(stderr,
               "parallel_close_test: FAIL — expected unsat on exp_n11_unsat.smt2\n");
  return 1;
}
