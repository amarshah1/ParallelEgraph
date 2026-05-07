#pragma once
// QF_UF subset of SMT-LIB 2: declare-sort/fun/const, assert, check-sat,
// set-logic, with terms = constant identifiers, function applications,
// (= a b), (not T), (distinct t1 ... tn). Other Boolean connectives
// (and/or/=>/xor/ite/let/true/false) are intentionally not modeled —
// the SAT integration that needed them was removed; restore from git
// history if you need it back.

#include <cstdint>
#include <string>
#include <vector>

namespace pe {

struct Term {
  enum class Kind : std::uint8_t { Const, App, Eq, Not, Distinct };

  Kind kind = Kind::Const;
  std::string op;                 // Const / App: identifier
  std::vector<Term> args;         // children, in source order. For
                                  // Distinct, all n operands; the
                                  // assertion (distinct t1 ... tn)
                                  // is expanded by the consumer
                                  // into n*(n-1)/2 pairwise
                                  // disequalities.
};

struct Command {
  enum class Kind : std::uint8_t {
    SetLogic, DeclareSort, DeclareFun, DeclareConst,
    Assert, CheckSat, Other,
  };
  Kind kind;
  std::string name;               // for declare-const/fun/sort/set-logic
  Term term;                      // for assert
};

struct Script {
  std::vector<Command> commands;
};

// Throws std::runtime_error with line info on parse failure.
Script parse_smtlib(const std::string& input);

}  // namespace pe
