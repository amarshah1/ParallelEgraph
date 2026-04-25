# CMake generated Testfile for 
# Source directory: /Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp
# Build directory: /Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[unionfind]=] "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/unionfind_test")
set_tests_properties([=[unionfind]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/CMakeLists.txt;51;add_test;/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/CMakeLists.txt;0;")
add_test([=[regression]=] "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/regression_test")
set_tests_properties([=[regression]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/CMakeLists.txt;58;add_test;/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/CMakeLists.txt;0;")
add_test([=[parallel_close]=] "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/parallel_close_test")
set_tests_properties([=[parallel_close]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/CMakeLists.txt;65;add_test;/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/CMakeLists.txt;0;")
subdirs("_deps/parlaylib-build")
