// Runs every .smt2 in ../tests/ in both sequential and parallel modes,
// verifying the answer matches the {_sat, _unsat} filename convention.
// Mirrors tests/regression.rs.

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "parallel_egraph/solve.hpp"

namespace {

pe::SolveResult expected_from_name(const std::string& name) {
  if (name.find("_unsat") != std::string::npos) return pe::SolveResult::Unsat;
  if (name.find("_sat") != std::string::npos) return pe::SolveResult::Sat;
  std::fprintf(stderr, "cannot determine expected result from %s\n",
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
  if (!f) {
    std::fprintf(stderr, "open(%s): failed\n", path.c_str());
    std::exit(1);
  }
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

}  // namespace

int main() {
#ifndef TESTS_DIR
#error "TESTS_DIR must be defined"
#endif
  int seq_ok = 0, seq_total = 0;
  int par_ok = 0, par_total = 0;
  int failures = 0;
  for (const auto& path : list_smt2(TESTS_DIR)) {
    std::string base = path.substr(path.find_last_of('/') + 1);
    std::string input = read_file(path);
    pe::SolveResult expected = expected_from_name(base);

    auto check = [&](const char* mode, bool parallel, int& ok, int& total) {
      ++total;
      try {
        pe::SolveResult got = pe::solve_with_mode(input, parallel);
        if (got == expected) {
          ++ok;
        } else {
          std::fprintf(stderr, "FAIL %s [%s]: expected %s got %s\n",
                       base.c_str(), mode, pe::to_string(expected),
                       pe::to_string(got));
          ++failures;
        }
      } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s [%s]: exception %s\n",
                     base.c_str(), mode, e.what());
        ++failures;
      }
    };
    check("seq", false, seq_ok, seq_total);
    check("par", true,  par_ok, par_total);
  }
  std::printf("regression_test: sequential=%d/%d parallel=%d/%d\n",
              seq_ok, seq_total, par_ok, par_total);
  return failures == 0 ? 0 : 1;
}
