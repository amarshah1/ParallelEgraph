#pragma once
// SAT-driven QF_UF solver: CaDiCaL handles the Boolean skeleton via the
// IPASIR-UP API; our existing EGraph plays the role of the QF_UF theory
// propagator. See cpp/SAT_INTEGRATION.md for design.

#include <string>
#include <utility>

#include "parallel_egraph/solve.hpp"  // SolveResult, SolveTimings

namespace pe {

// Drives the full DPLL(T) loop. Returns the same (result, timings) as
// solve_timed, with timings.merge_s = sum of theory merge work,
// timings.rebuild_s = sum of in-propagator rebuild work, and the
// existing parse/build/check fields preserved.
std::pair<SolveResult, SolveTimings> sat_solve_timed(const std::string& input,
                                                     bool parallel);

}  // namespace pe
