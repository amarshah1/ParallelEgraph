#pragma once
// QF_UF SMT-LIB 2 subset, *with full Boolean fragment*.
//
// The pre-SAT version of this header only modeled Eq/Not as assertion
// shapes. Now we model the entire Boolean skeleton (and / or / not / =>
// / xor / ite-over-Bool / distinct / Bool consts), so the SAT-driven
// solver can clausify it. UF terms (Const, App) live as before.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pe {

struct Term {
  enum class Kind : std::uint8_t {
    // UF / generic shapes
    Const,        // leaf identifier (UF constant or Bool variable)
    App,          // function application f(x1, ..., xk)
    // Boolean / equality
    Eq,           // (= a b)        — UF equality OR Bool iff (handled identically)
    Not,          // (not x)
    And,          // (and x1 ... xk)    k >= 0
    Or,           // (or x1 ... xk)     k >= 0
    Implies,      // (=> a b ... z)     right-associative chain
    Xor,          // (xor a b ... z)    left-associative chain
    Ite,          // (ite c a b)        Bool here (ite over UF terms is a UF App)
    Distinct,     // (distinct x1 ... xk)
    True,         // 'true' literal
    False,        // 'false' literal
  };

  Kind kind = Kind::Const;
  std::string op;                 // Const/App: identifier
  std::vector<Term> args;         // children, in source order
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

// Returns true iff every assertion in `s` is one of:
//   (= a b)           where a, b are UF terms (no Booleans inside)
//   (not (= a b))     same
// Used by solve.cpp to take the legacy fast path that skips CaDiCaL
// for trivially-conjunctive QF_UF instances. Anything Boolean (and/or/
// implies/ite over assertions) goes through the SAT-driven pipeline.
bool is_pure_conjunctive(const Script& s);

}  // namespace pe
