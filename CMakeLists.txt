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

# ---- core library ----
add_library(parallel_egraph_core STATIC
  src/unionfind.cpp
  src/egraph.cpp
  src/smtlib.cpp
)
target_include_directories(parallel_egraph_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(parallel_egraph_core PUBLIC parlay)

# ---- CLI: egraph-cc <file.smt2> -> sat / unsat ----
add_executable(egraph-cc src/main.cpp)
target_link_libraries(egraph-cc PRIVATE parallel_egraph_core)

# ---- benchmark ----
add_executable(closure_compare_bench bench/closure_compare.cpp)
target_link_libraries(closure_compare_bench PRIVATE parallel_egraph_core)

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
