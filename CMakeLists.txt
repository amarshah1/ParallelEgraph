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
# parallel_parents on wide-shallow workloads (≈1.76× faster on synthetic_bench
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
# ---- core library ----
# Semisort equality is verified structurally via sigs_equal (children-walk
# comparison resolved through the UF) — sound under any hash function.
add_library(parallel_egraph_core STATIC
  src/unionfind.cpp
  src/egraph.cpp
  src/egraph_filter.cpp
  src/smtlib.cpp
  src/semisort.cpp
)
target_include_directories(parallel_egraph_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(parallel_egraph_core PUBLIC parlay unordered_dense::unordered_dense ${JEMALLOC_LIB})

# ---- CLI: egraph-cc <file.smt2> -> sat / unsat ----
add_executable(egraph-cc src/main.cpp)
target_link_libraries(egraph-cc PRIVATE parallel_egraph_core)

# ---- benchmarks ----
add_executable(closure_compare_bench bench/closure_compare.cpp)
target_link_libraries(closure_compare_bench PRIVATE parallel_egraph_core)

add_executable(component_bench bench/component_bench.cpp)
target_link_libraries(component_bench PRIVATE parallel_egraph_core)

add_executable(synthetic_bench bench/synthetic_bench.cpp)
target_link_libraries(synthetic_bench PRIVATE parallel_egraph_core)

add_executable(irregular_bench bench/irregular_bench.cpp)
target_link_libraries(irregular_bench PRIVATE parallel_egraph_core)

add_executable(smt_bench bench/smt_bench.cpp)
target_link_libraries(smt_bench PRIVATE parallel_egraph_core)

add_executable(gates_bench bench/gates_bench.cpp)
target_link_libraries(gates_bench PRIVATE parallel_egraph_core)

# ---- tests ----
enable_testing()

add_executable(unionfind_test tests/unionfind_test.cpp)
target_link_libraries(unionfind_test PRIVATE parallel_egraph_core)
add_test(NAME unionfind COMMAND unionfind_test)

add_executable(closure_test tests/closure_test.cpp)
target_link_libraries(closure_test PRIVATE parallel_egraph_core)
add_test(NAME closure COMMAND closure_test)
