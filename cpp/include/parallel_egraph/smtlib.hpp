#pragma once
// QF_UF subset of SMT-LIB 2. Mirrors src/process.rs plus the yaspar-ir
// bits we actually use (constants, function applications, =, not, assert).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pe {

struct Term {
  enum class Kind : std::uint8_t { Const, App, Eq, Not };

  Kind kind;
  std::string op;                 // for Const and App; empty for Eq/Not
  std::vector<Term> args;         // children
};

struct Command {
  enum class Kind : std::uint8_t { SetLogic, DeclareSort, DeclareFun,
                                   DeclareConst, Assert, CheckSat, Other };
  Kind kind;
  std::string name;               // for declare-const/fun/sort
  Term term;                      // for assert
};

struct Script {
  std::vector<Command> commands;
};

// Throws std::runtime_error with line info on parse failure.
Script parse_smtlib(const std::string& input);

}  // namespace pe
