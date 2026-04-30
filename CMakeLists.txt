cmake_minimum_required(VERSION 3.20)
project(parallel_egraph_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# ---- parlay ----
include(FetchContent)
FetchContent_Declare(
  parlaylib
  GIT_REPOSITORY https://github.com/cmuparlay/parlaylib.git
  GIT_TAG        master
)
set(PARLAY_TEST OFF CACHE BOOL "" FORCE)
set(PARLAY_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PARLAY_BENCHMARK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(parlaylib)

# ---- ankerl::unordered_dense ----
# Modern flat hashmap (robin-hood, single header) used by the sequential
# Nelson baseline in place of std::unordered_map. ~3x faster on the
# insert+lookup workload of the signature table, and avoids std's
# chained-bucket allocator pressure. Used only by sequential_close_nelson;
# the parallel path uses parlay's sort-based group_by_key_ordered instead.
FetchContent_Declare(
  unordered_dense
  GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
  GIT_TAG        v4.4.0
)
FetchContent_MakeAvailable(unordered_dense)

# ---- allocator: jemalloc ----
# Replaces glibc malloc. Removes per-arena lock contention that throttles
# parallel_close on wide-shallow workloads (≈1.76× faster on synthetic_bench
# quintic n=23 at T=192, ~1.20× on closure_compare_bench 16xl-d3 at T=192).
option(USE_JEMALLOC "Link binaries against jemalloc" ON)
if(USE_JEMALLOC)
  find_library(JEMALLOC_LIB jemalloc)
  if(NOT JEMALLOC_LIB)
    message(FATAL_ERROR
      "libjemalloc not found. Install with: "
      "sudo apt-get install libjemalloc-dev  (or set -DUSE_JEMALLOC=OFF)")
  endif()
  message(STATUS "Linking with jemalloc: ${JEMALLOC_LIB}")
endif()

# ---- semisort group-by variant ----
# Default: parlay::group_by_key_ordered (integer_sort) with a 32-bit
# secondary hash carried in each CanonEntry — fastest, with collisions
# made effectively impossible by 96-bit combined entropy and a defensive
# slow path for the (statistically unreachable) collision case.
# Set PE_GROUPBY_HASH=ON to build with the legacy parlay::group_by_key
# (open-addressing) + sigs_equal path, which is deterministically correct
# under any hash function. CanonEntry stays at 16 bytes either way (the
# trailing 4 bytes are h2 in the default build, idx in the legacy build).
option(PE_GROUPBY_HASH "Use deterministic hash + sigs_equal group-by" OFF)

# ---- core library ----
add_library(parallel_egraph_core STATIC
  src/unionfind.cpp
  src/egraph.cpp
  src/smtlib.cpp
)
target_include_directories(parallel_egraph_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(parallel_egraph_core PUBLIC parlay unordered_dense::unordered_dense ${JEMALLOC_LIB})
if(PE_GROUPBY_HASH)
  target_compile_definitions(parallel_egraph_core PUBLIC PE_GROUPBY_HASH)
endif()

# ---- CLI: egraph-cc <file.smt2> -> sat / unsat ----
add_executable(egraph-cc src/main.cpp)
target_link_libraries(egraph-cc PRIVATE parallel_egraph_core)

# ---- benchmark ----
add_executable(closure_compare_bench bench/closure_compare.cpp)
target_link_libraries(closure_compare_bench PRIVATE parallel_egraph_core)

add_executable(component_bench bench/component_bench.cpp)
target_link_libraries(component_bench PRIVATE parallel_egraph_core)

add_executable(smt_bench bench/smt_bench.cpp)
target_link_libraries(smt_bench PRIVATE parallel_egraph_core)

add_executable(synthetic_bench bench/synthetic_bench.cpp)
target_link_libraries(synthetic_bench PRIVATE parallel_egraph_core)

add_executable(irregular_bench bench/irregular_bench.cpp)
target_link_libraries(irregular_bench PRIVATE parallel_egraph_core)

# ---- tests ----
enable_testing()

add_executable(unionfind_test tests/unionfind_test.cpp)
target_link_libraries(unionfind_test PRIVATE parallel_egraph_core)
add_test(NAME unionfind COMMAND unionfind_test)

add_executable(regression_test tests/regression_test.cpp)
target_link_libraries(regression_test PRIVATE parallel_egraph_core)
target_compile_definitions(regression_test PRIVATE
  REGRESSION_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples/regression"
)
add_test(NAME regression COMMAND regression_test)
