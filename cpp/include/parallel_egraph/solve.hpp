#pragma once
// Parse + solve (entrypoint invoked from main.cpp).

#include <string>

namespace pe {

enum class SolveResult { Sat, Unsat };

struct SolveTimings {
  double parse_s = 0.0;
  double build_s = 0.0;
  double merge_s = 0.0;
  double rebuild_s = 0.0;
  double check_s = 0.0;
  double total_s = 0.0;
  double solve_s = 0.0;
};

SolveResult solve_with_mode(const std::string& input, bool parallel);
std::pair<SolveResult, SolveTimings> solve_timed(const std::string& input,
                                                 bool parallel);

void begin_solve_region();
void end_solve_region();

const char* to_string(SolveResult r);

}  // namespace pe
