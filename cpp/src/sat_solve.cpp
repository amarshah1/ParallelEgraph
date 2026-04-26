// SAT-driven QF_UF solver. See cpp/SAT_INTEGRATION.md for design overview.
//
// Pipeline:
//   1. parse_smtlib (Boolean fragment supported by smtlib.cpp)
//   2. cnf_encode: assign every (= ti tj) atom a CaDiCaL var, Tseitin-encode
//      the Boolean structure above the atoms
//   3. CaDiCaL.solve() with our UfPropagator attached via IPASIR-UP

#include "parallel_egraph/sat_solve.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cadical.hpp>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/smtlib.hpp"

namespace pe {

// Stub: filled in next.
std::pair<SolveResult, SolveTimings> sat_solve_timed(const std::string& input,
                                                     bool parallel) {
  (void)input;
  (void)parallel;
  throw std::runtime_error("sat_solve_timed: not yet implemented");
}

}  // namespace pe
