// Drives every examples/regression/*.smt2 through the parser → EGraph →
// parallel_close pipeline and verifies the result against the
// `_sat` / `_unsat` suffix in the filename.

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
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

enum class Expected { Sat, Unsat };

Expected expected_from_name(const std::string& name) {
  if (name.find("_unsat") != std::string::npos) return Expected::Unsat;
  if (name.find("_sat") != std::string::npos) return Expected::Sat;
  std::fprintf(stderr,
               "cannot determine expected result from filename %s\n",
               name.c_str());
  std::exit(1);
}

std::vector<std::string> list_smt2(const char* dir) {
  std::vector<std::string> out;
  DIR* d = opendir(dir);
  if (!d) {
    std::fprintf(stderr, "opendir(%s): %s\n", dir, std::strerror(errno));
    std::exit(1);
  }
  while (struct dirent* ent = readdir(d)) {
    std::string name = ent->d_name;
    if (name.size() > 5 && name.substr(name.size() - 5) == ".smt2") {
      out.push_back(std::string(dir) + "/" + name);
    }
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

std::size_t count_subterms(const pe::Term& t) {
  std::size_t n = 1;
  for (const auto& a : t.args) n += count_subterms(a);
  return n;
}

// Iterative add (mirrors src/main.cpp).
pe::Id add_term(pe::EGraph& eg, const pe::Term& term) {
  enum class WK { Process, Build };
  struct W { WK kind; const pe::Term* term; std::string op; std::size_t nargs; };
  std::vector<W> stack;
  std::vector<pe::Id> results;
  stack.push_back({WK::Process, &term, {}, 0});
  while (!stack.empty()) {
    W w = std::move(stack.back());
    stack.pop_back();
    if (w.kind == WK::Process) {
      const pe::Term& t = *w.term;
      if (t.kind == pe::Term::Kind::Const) {
        pe::ENode leaf; leaf.op = t.op;
        results.push_back(eg.add(std::move(leaf)));
      } else if (t.kind == pe::Term::Kind::App) {
        stack.push_back({WK::Build, nullptr, t.op, t.args.size()});
        for (auto it = t.args.rbegin(); it != t.args.rend(); ++it) {
          stack.push_back({WK::Process, &*it, {}, 0});
        }
      }
    } else {
      std::size_t start = results.size() - w.nargs;
      std::vector<pe::Id> children(results.begin() + start, results.end());
      results.erase(results.begin() + start, results.end());
      pe::ENode node; node.op = std::move(w.op); node.children = std::move(children);
      results.push_back(eg.add(std::move(node)));
    }
  }
  return results.back();
}

Expected solve(const pe::Script& script) {
  std::size_t capacity = 0;
  for (const auto& cmd : script.commands) {
    if (cmd.kind == pe::Command::Kind::Assert) {
      capacity += count_subterms(cmd.term);
    }
  }

  pe::EGraph eg(capacity);
  parlay::sequence<std::pair<pe::Id, pe::Id>> equalities;
  std::vector<std::pair<pe::Id, pe::Id>> disequalities;

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
      throw std::runtime_error("unsupported assertion shape");
    }
  }

  eg.parallel_close(std::move(equalities));
  for (auto [a, b] : disequalities) {
    if (eg.equiv(a, b)) return Expected::Unsat;
  }
  return Expected::Sat;
}

const char* to_string(Expected e) { return e == Expected::Sat ? "sat" : "unsat"; }

}  // namespace

int main() {
#ifndef REGRESSION_DIR
#error "REGRESSION_DIR must be defined"
#endif
  int ok = 0, total = 0, failures = 0;
  for (const auto& path : list_smt2(REGRESSION_DIR)) {
    std::string base = path.substr(path.find_last_of('/') + 1);
    Expected expected = expected_from_name(base);
    ++total;
    try {
      pe::Script script = pe::parse_smtlib(read_file(path));
      Expected got = solve(script);
      if (got == expected) {
        ++ok;
      } else {
        std::fprintf(stderr, "FAIL %s: expected %s got %s\n",
                     base.c_str(), to_string(expected), to_string(got));
        ++failures;
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "FAIL %s: %s\n", base.c_str(), e.what());
      ++failures;
    }
  }
  std::printf("regression_test: %d/%d passed\n", ok, total);
  return failures == 0 ? 0 : 1;
}
