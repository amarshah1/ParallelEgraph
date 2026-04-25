#include "parallel_egraph/solve.hpp"

#include <chrono>
#include <stdexcept>

#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/smtlib.hpp"

namespace pe {

void begin_solve_region() {}
void end_solve_region() {}

const char* to_string(SolveResult r) {
  return r == SolveResult::Sat ? "sat" : "unsat";
}

namespace {

// Count the number of (sub)term nodes in a Term tree. Used to compute an
// upper bound on UF capacity for the EGraph — the fixed-cap UF means we
// need to know size before construction.
std::size_t count_subterms(const Term& t) {
  std::size_t n = 1;
  for (const auto& a : t.args) n += count_subterms(a);
  return n;
}

// Iterative add of a term to an EGraph; mirrors src/process.rs add_term.
Id add_term(EGraph& eg, const Term& term) {
  enum class WorkKind { Process, Build };
  struct Work {
    WorkKind kind;
    const Term* term;     // for Process
    std::string op;       // for Build
    std::size_t nargs;    // for Build
  };
  std::vector<Work> stack;
  std::vector<Id> results;
  stack.push_back({WorkKind::Process, &term, {}, 0});

  while (!stack.empty()) {
    Work w = std::move(stack.back());
    stack.pop_back();

    if (w.kind == WorkKind::Process) {
      const Term& t = *w.term;
      switch (t.kind) {
        case Term::Kind::Const: {
          ENode leaf;
          leaf.op = t.op;
          results.push_back(eg.add(std::move(leaf)));
          break;
        }
        case Term::Kind::App: {
          stack.push_back({WorkKind::Build, nullptr, t.op, t.args.size()});
          // Process children in reverse so first child is processed first.
          for (auto it = t.args.rbegin(); it != t.args.rend(); ++it) {
            stack.push_back({WorkKind::Process, &*it, {}, 0});
          }
          break;
        }
        case Term::Kind::Eq:
        case Term::Kind::Not:
          throw std::runtime_error(
              "add_term: = and not are not first-class terms; only applies "
              "inside assertions");
      }
    } else {
      // Build
      std::size_t start = results.size() - w.nargs;
      std::vector<Id> children(results.begin() + start, results.end());
      results.erase(results.begin() + start, results.end());
      ENode node;
      node.op = std::move(w.op);
      node.children = std::move(children);
      results.push_back(eg.add(std::move(node)));
    }
  }

  return results.back();
}

// An assertion is either equality (to merge) or disequality (to check).
struct Assertion {
  enum class Kind { Eq, Diseq };
  Kind kind;
  Id a, b;
};

Assertion process_assertion(EGraph& eg, const Term& term) {
  if (term.kind == Term::Kind::Eq) {
    Id a = add_term(eg, term.args[0]);
    Id b = add_term(eg, term.args[1]);
    return {Assertion::Kind::Eq, a, b};
  }
  if (term.kind == Term::Kind::Not && term.args.size() == 1 &&
      term.args[0].kind == Term::Kind::Eq) {
    Id a = add_term(eg, term.args[0].args[0]);
    Id b = add_term(eg, term.args[0].args[1]);
    return {Assertion::Kind::Diseq, a, b};
  }
  throw std::runtime_error(
      "unsupported assertion: only (= ...) and (not (= ...)) are handled");
}

double secs_since(std::chrono::steady_clock::time_point t) {
  using namespace std::chrono;
  return duration<double>(steady_clock::now() - t).count();
}

}  // namespace

std::pair<SolveResult, SolveTimings> solve_timed(const std::string& input,
                                                 bool parallel) {
  using clk = std::chrono::steady_clock;
  SolveTimings timings;
  auto total_start = clk::now();

  // --- Parse ---
  auto parse_start = clk::now();
  Script script = parse_smtlib(input);
  std::vector<const Term*> assertions;
  for (const auto& c : script.commands) {
    if (c.kind == Command::Kind::Assert) assertions.push_back(&c.term);
  }
  timings.parse_s = secs_since(parse_start);

  // --- Capacity pre-walk ---
  // Every subterm occurrence adds at most one fresh e-class (hashcons dedups),
  // so total subterm count is a safe upper bound.
  std::size_t capacity = 0;
  for (const Term* t : assertions) capacity += count_subterms(*t);

  // --- Build ---
  begin_solve_region();
  EGraph eg(capacity, parallel);
  parlay::sequence<std::pair<Id, Id>> equalities;
  std::vector<std::pair<Id, Id>> disequalities;

  auto build_start = clk::now();
  double merge_accum = 0.0;
  if (parallel) {
    for (const Term* t : assertions) {
      Assertion a = process_assertion(eg, *t);
      if (a.kind == Assertion::Kind::Eq) equalities.emplace_back(a.a, a.b);
      else                                 disequalities.emplace_back(a.a, a.b);
    }
    timings.build_s = secs_since(build_start);

    auto merge_start = clk::now();
    eg.parallel_merge_all(equalities);
    timings.merge_s = secs_since(merge_start);

    auto rebuild_start = clk::now();
    eg.parallel_rebuild();
    timings.rebuild_s = secs_since(rebuild_start);
  } else {
    // Sequential: time build (add_term) and merge separately.
    double build_accum = 0.0;
    for (const Term* t : assertions) {
      auto t0 = clk::now();
      Assertion a = process_assertion(eg, *t);
      build_accum += secs_since(t0);
      if (a.kind == Assertion::Kind::Eq) {
        auto t1 = clk::now();
        eg.merge(a.a, a.b);
        merge_accum += secs_since(t1);
      } else {
        disequalities.emplace_back(a.a, a.b);
      }
    }
    timings.build_s = build_accum;
    timings.merge_s = merge_accum;

    auto rebuild_start = clk::now();
    eg.rebuild();
    timings.rebuild_s = secs_since(rebuild_start);
  }

  auto check_start = clk::now();
  SolveResult result = SolveResult::Sat;
  for (auto [a, b] : disequalities) {
    if (eg.equiv(a, b)) {
      result = SolveResult::Unsat;
      break;
    }
  }
  timings.check_s = secs_since(check_start);

  end_solve_region();
  timings.total_s = secs_since(total_start);
  timings.solve_s = timings.build_s + timings.merge_s + timings.rebuild_s +
                    timings.check_s;
  return {result, timings};
}

SolveResult solve_with_mode(const std::string& input, bool parallel) {
  return solve_timed(input, parallel).first;
}

}  // namespace pe
