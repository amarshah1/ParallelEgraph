#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "parallel_egraph/smtlib.hpp"
#include "parallel_egraph/solve.hpp"

namespace {

int usage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s [--parallel|-p] [--timing|-t] [--profile-solve] [--debug-parse] <smt2-file>\n",
               prog);
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  bool parallel = false;
  bool timing = false;
  bool profile_solve = false;
  bool debug_parse = false;
  const char* file_arg = nullptr;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--parallel" || arg == "-p") {
      parallel = true;
    } else if (arg == "--timing" || arg == "-t") {
      timing = true;
    } else if (arg == "--profile-solve") {
      profile_solve = true;
    } else if (arg == "--debug-parse") {
      debug_parse = true;
    } else if (file_arg == nullptr) {
      file_arg = argv[i];
    } else {
      std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
      return usage(argv[0]);
    }
  }

  if (file_arg == nullptr) return usage(argv[0]);

  std::ifstream f(file_arg);
  if (!f) {
    std::fprintf(stderr, "Error reading file %s\n", file_arg);
    return 1;
  }
  std::stringstream buf;
  buf << f.rdbuf();
  std::string input = buf.str();

  try {
    if (debug_parse) {
      pe::Script s = pe::parse_smtlib(input);
      std::size_t asserts = 0;
      for (const auto& c : s.commands) {
        if (c.kind == pe::Command::Kind::Assert) ++asserts;
      }
      std::printf("parsed: commands=%zu asserts=%zu\n", s.commands.size(),
                  asserts);
      return 0;
    }
    if (timing || profile_solve) {
      auto [result, t] = pe::solve_timed(input, parallel);
      std::printf("%s\n", pe::to_string(result));
      std::fprintf(stderr,
                   "timing: parse=%.6f build=%.6f merge=%.6f rebuild=%.6f "
                   "check=%.6f solve=%.6f total=%.6f\n",
                   t.parse_s, t.build_s, t.merge_s, t.rebuild_s, t.check_s,
                   t.solve_s, t.total_s);
    } else {
      auto result = pe::solve_with_mode(input, parallel);
      std::printf("%s\n", pe::to_string(result));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}
