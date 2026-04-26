#include "parallel_egraph/smtlib.hpp"

#include <cctype>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pe {
namespace {

// ---- Lexer ----------------------------------------------------------------

struct Token {
  enum class Kind { LParen, RParen, Atom, End };
  Kind kind;
  std::string text;   // for Atom
  std::size_t line;
};

class Lexer {
 public:
  explicit Lexer(const std::string& s) : s_(s) {}

  Token next() {
    skip_ws();
    if (pos_ >= s_.size()) return {Token::Kind::End, {}, line_};
    char c = s_[pos_];
    if (c == '(') { ++pos_; return {Token::Kind::LParen, {}, line_}; }
    if (c == ')') { ++pos_; return {Token::Kind::RParen, {}, line_}; }
    // Atom: run of non-whitespace non-paren chars (or quoted symbol |...|).
    std::size_t start = pos_;
    if (c == '|') {
      ++pos_;
      while (pos_ < s_.size() && s_[pos_] != '|') {
        if (s_[pos_] == '\n') ++line_;
        ++pos_;
      }
      if (pos_ >= s_.size()) {
        throw std::runtime_error("unterminated |...| symbol at line " +
                                 std::to_string(line_));
      }
      ++pos_;  // consume closing |
      return {Token::Kind::Atom, s_.substr(start, pos_ - start), line_};
    }
    while (pos_ < s_.size()) {
      char ch = s_[pos_];
      if (std::isspace(static_cast<unsigned char>(ch)) || ch == '(' ||
          ch == ')' || ch == ';') {
        break;
      }
      ++pos_;
    }
    return {Token::Kind::Atom, s_.substr(start, pos_ - start), line_};
  }

 private:
  void skip_ws() {
    while (pos_ < s_.size()) {
      char c = s_[pos_];
      if (c == '\n') { ++line_; ++pos_; continue; }
      if (std::isspace(static_cast<unsigned char>(c))) { ++pos_; continue; }
      if (c == ';') {
        while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
        continue;
      }
      break;
    }
  }

  const std::string& s_;
  std::size_t pos_ = 0;
  std::size_t line_ = 1;
};

// ---- Parser ---------------------------------------------------------------

class Parser {
 public:
  explicit Parser(const std::string& s) : lexer_(s) { advance(); }

  Script parse_script() {
    Script script;
    while (cur_.kind != Token::Kind::End) {
      script.commands.push_back(parse_command());
    }
    return script;
  }

 private:
  // Lexical scope stack for `let`. Each frame is a single let's bindings;
  // a Const atom is resolved by walking the stack top-down.
  std::vector<std::vector<std::pair<std::string, Term>>> let_scopes_;

  // Look up a name in the current scope chain. Returns a copy of the bound
  // term if found, else nullopt.
  std::optional<Term> resolve_let(const std::string& name) {
    for (auto it = let_scopes_.rbegin(); it != let_scopes_.rend(); ++it) {
      for (const auto& kv : *it) {
        if (kv.first == name) return kv.second;
      }
    }
    return std::nullopt;
  }

 public:

 private:
  void advance() { cur_ = lexer_.next(); }

  [[noreturn]] void error(const std::string& msg) {
    throw std::runtime_error("parse error at line " +
                             std::to_string(cur_.line) + ": " + msg);
  }

  void expect(Token::Kind k, const char* what) {
    if (cur_.kind != k) error(std::string("expected ") + what);
    advance();
  }

  std::string expect_atom(const char* what) {
    if (cur_.kind != Token::Kind::Atom)
      error(std::string("expected ") + what);
    std::string t = cur_.text;
    advance();
    return t;
  }

  Command parse_command() {
    expect(Token::Kind::LParen, "'(' at command start");
    std::string head = expect_atom("command keyword");
    Command cmd;
    if (head == "set-logic") {
      cmd.kind = Command::Kind::SetLogic;
      cmd.name = expect_atom("logic name");
    } else if (head == "declare-sort") {
      cmd.kind = Command::Kind::DeclareSort;
      cmd.name = expect_atom("sort name");
      // Skip arity atom.
      if (cur_.kind == Token::Kind::Atom) advance();
    } else if (head == "declare-const") {
      cmd.kind = Command::Kind::DeclareConst;
      cmd.name = expect_atom("const name");
      // Skip sort atom(s) / s-expression.
      skip_term_like();
    } else if (head == "declare-fun") {
      cmd.kind = Command::Kind::DeclareFun;
      cmd.name = expect_atom("function name");
      // Skip arg list "(SORT SORT ...)".
      expect(Token::Kind::LParen, "'(' for arg sort list");
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated arg list");
        skip_term_like();
      }
      expect(Token::Kind::RParen, "')' closing arg list");
      // Skip return sort.
      skip_term_like();
    } else if (head == "assert") {
      cmd.kind = Command::Kind::Assert;
      cmd.term = parse_term();
    } else if (head == "check-sat") {
      cmd.kind = Command::Kind::CheckSat;
    } else {
      cmd.kind = Command::Kind::Other;
      cmd.name = head;
      // Skip any remaining atoms / nested sexps until matching ')'.
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated command");
        skip_term_like();
      }
    }
    expect(Token::Kind::RParen, "')' closing command");
    return cmd;
  }

  // Skip an atom or a balanced s-expression. Used for sort expressions and
  // other non-term bits we don't model.
  void skip_term_like() {
    if (cur_.kind == Token::Kind::Atom) { advance(); return; }
    if (cur_.kind == Token::Kind::LParen) {
      int depth = 1;
      advance();
      while (depth > 0) {
        if (cur_.kind == Token::Kind::End) error("unterminated s-expression");
        if (cur_.kind == Token::Kind::LParen) ++depth;
        if (cur_.kind == Token::Kind::RParen) --depth;
        advance();
      }
      return;
    }
    error("expected atom or '('");
  }

  Term parse_term() {
    if (cur_.kind == Token::Kind::Atom) {
      Term t;
      // Recognize the Boolean literals; everything else is a Const,
      // unless it's bound by an enclosing `let` — in which case we
      // substitute the bound term in place.
      if (cur_.text == "true")       t.kind = Term::Kind::True;
      else if (cur_.text == "false") t.kind = Term::Kind::False;
      else if (auto bound = resolve_let(cur_.text); bound.has_value()) {
        t = std::move(*bound);
      } else {
        t.kind = Term::Kind::Const;
        t.op = cur_.text;
      }
      advance();
      return t;
    }
    expect(Token::Kind::LParen, "'(' at term start");
    std::string head = expect_atom("term head");
    Term t;
    if (head == "let") {
      // (let ((x1 e1) (x2 e2) ...) body)
      // SMT-LIB `let` is parallel: each ei is parsed in the *outer*
      // scope, then all bindings are pushed as one frame for the body.
      expect(Token::Kind::LParen, "'(' for let binding list");
      std::vector<std::pair<std::string, Term>> frame;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated let bindings");
        expect(Token::Kind::LParen, "'(' for binding");
        std::string name = expect_atom("binding name");
        Term value = parse_term();   // outer scope (frame not yet pushed)
        expect(Token::Kind::RParen, "')' closing binding");
        frame.emplace_back(std::move(name), std::move(value));
      }
      expect(Token::Kind::RParen, "')' closing binding list");
      let_scopes_.push_back(std::move(frame));
      Term body = parse_term();
      let_scopes_.pop_back();
      // Closing ')' of (let ...) is consumed below by the shared
      // expect at the bottom of parse_term.
      t = std::move(body);
    } else if (head == "=") {
      t.kind = Term::Kind::Eq;
      // SMT-LIB allows n-ary =, but for QF_UF we only need binary. If we
      // ever see n>2 we left-fold into pairwise equalities under and.
      std::vector<Term> args;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated (= ...)");
        args.push_back(parse_term());
      }
      if (args.size() < 2) error("(=) needs at least 2 args");
      if (args.size() == 2) {
        t.args = std::move(args);
      } else {
        // Desugar (= x1 x2 x3 ...) → (and (= x1 x2) (= x2 x3) ...).
        t.kind = Term::Kind::And;
        for (std::size_t i = 0; i + 1 < args.size(); ++i) {
          Term eq;
          eq.kind = Term::Kind::Eq;
          eq.args = {args[i], args[i + 1]};
          t.args.push_back(std::move(eq));
        }
      }
    } else if (head == "not") {
      t.kind = Term::Kind::Not;
      t.args.push_back(parse_term());
    } else if (head == "and") {
      t.kind = Term::Kind::And;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated (and ...)");
        t.args.push_back(parse_term());
      }
    } else if (head == "or") {
      t.kind = Term::Kind::Or;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated (or ...)");
        t.args.push_back(parse_term());
      }
    } else if (head == "=>" || head == "implies") {
      t.kind = Term::Kind::Implies;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated (=> ...)");
        t.args.push_back(parse_term());
      }
      if (t.args.size() < 2) error("(=>) needs at least 2 args");
    } else if (head == "xor") {
      t.kind = Term::Kind::Xor;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated (xor ...)");
        t.args.push_back(parse_term());
      }
      if (t.args.size() < 2) error("(xor) needs at least 2 args");
    } else if (head == "ite") {
      t.kind = Term::Kind::Ite;
      t.args.push_back(parse_term());  // cond
      t.args.push_back(parse_term());  // then
      t.args.push_back(parse_term());  // else
    } else if (head == "distinct") {
      t.kind = Term::Kind::Distinct;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated (distinct ...)");
        t.args.push_back(parse_term());
      }
      if (t.args.size() < 2) error("(distinct) needs at least 2 args");
    } else {
      // Function application f(args...). UF term (or, if appearing in a
      // Boolean context, a Bool-typed UF predicate — we don't distinguish
      // sorts at parse time; the encoder figures it out from context.)
      t.kind = Term::Kind::App;
      t.op = head;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated term");
        t.args.push_back(parse_term());
      }
    }
    expect(Token::Kind::RParen, "')' closing term");
    return t;
  }

  Lexer lexer_;
  Token cur_;
};

}  // namespace

Script parse_smtlib(const std::string& input) {
  Parser p(input);
  return p.parse_script();
}

namespace {

// Returns true iff `t` is a pure UF term: only Const and App, no Boolean
// kinds, no Eq, no Ite, etc. The legacy fast path requires this — its
// add_term() asserts on anything else.
bool is_pure_uf(const Term& t) {
  if (t.kind == Term::Kind::Const) return true;
  if (t.kind == Term::Kind::App) {
    for (const auto& a : t.args) if (!is_pure_uf(a)) return false;
    return true;
  }
  return false;
}

// True if the term tree is "(= a b)" or "(not (= a b))" with no inner
// Boolean structure beyond that. Anything else (and / or / nested =, ite
// inside an arg, etc.) returns false.
bool is_pure_assertion(const Term& t) {
  if (t.kind == Term::Kind::Eq) {
    if (t.args.size() != 2) return false;
    return is_pure_uf(t.args[0]) && is_pure_uf(t.args[1]);
  }
  if (t.kind == Term::Kind::Not && t.args.size() == 1 &&
      t.args[0].kind == Term::Kind::Eq &&
      t.args[0].args.size() == 2 &&
      is_pure_uf(t.args[0].args[0]) &&
      is_pure_uf(t.args[0].args[1])) {
    return true;
  }
  return false;
}

}  // namespace

bool is_pure_conjunctive(const Script& s) {
  for (const auto& c : s.commands) {
    if (c.kind != Command::Kind::Assert) continue;
    if (!is_pure_assertion(c.term)) return false;
  }
  return true;
}

}  // namespace pe
