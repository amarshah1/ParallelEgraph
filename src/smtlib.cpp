#include "parallel_egraph/smtlib.hpp"

#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace pe {
namespace {

// ---- Lexer ----------------------------------------------------------------

struct Token {
  enum class Kind { LParen, RParen, Atom, End };
  Kind kind;
  std::string text;       // for Atom
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
      if (std::isspace(static_cast<unsigned char>(ch)) ||
          ch == '(' || ch == ')' || ch == ';') {
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

  // Skip an atom or a balanced s-expression (used for sort exprs etc.).
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
      if (cur_.kind == Token::Kind::Atom) advance();   // arity
    } else if (head == "declare-const") {
      cmd.kind = Command::Kind::DeclareConst;
      cmd.name = expect_atom("const name");
      skip_term_like();                                // sort
    } else if (head == "declare-fun") {
      cmd.kind = Command::Kind::DeclareFun;
      cmd.name = expect_atom("function name");
      expect(Token::Kind::LParen, "'(' for arg sort list");
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated arg list");
        skip_term_like();
      }
      expect(Token::Kind::RParen, "')' closing arg list");
      skip_term_like();                                // return sort
    } else if (head == "assert") {
      cmd.kind = Command::Kind::Assert;
      cmd.term = parse_term();
    } else if (head == "check-sat") {
      cmd.kind = Command::Kind::CheckSat;
    } else {
      cmd.kind = Command::Kind::Other;
      cmd.name = head;
      while (cur_.kind != Token::Kind::RParen) {
        if (cur_.kind == Token::Kind::End) error("unterminated command");
        skip_term_like();
      }
    }
    expect(Token::Kind::RParen, "')' closing command");
    return cmd;
  }

  Term parse_term() {
    if (cur_.kind == Token::Kind::Atom) {
      Term t;
      t.kind = Term::Kind::Const;
      t.op = cur_.text;
      advance();
      return t;
    }
    expect(Token::Kind::LParen, "'(' at term start");
    std::string head = expect_atom("term head");
    Term t;
    if (head == "=") {
      t.kind = Term::Kind::Eq;
      t.args.push_back(parse_term());
      t.args.push_back(parse_term());
    } else if (head == "not") {
      t.kind = Term::Kind::Not;
      t.args.push_back(parse_term());
    } else {
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

}  // namespace pe
