# CMake generated Testfile for 
# Source directory: /Users/zacharykent/ParallelEgraph/cpp
# Build directory: /Users/zacharykent/ParallelEgraph/cpp/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[unionfind]=] "/Users/zacharykent/ParallelEgraph/cpp/build/unionfind_test")
set_tests_properties([=[unionfind]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/zacharykent/ParallelEgraph/cpp/CMakeLists.txt;51;add_test;/Users/zacharykent/ParallelEgraph/cpp/CMakeLists.txt;0;")
add_test([=[regression]=] "/Users/zacharykent/ParallelEgraph/cpp/build/regression_test")
set_tests_properties([=[regression]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/zacharykent/ParallelEgraph/cpp/CMakeLists.txt;58;add_test;/Users/zacharykent/ParallelEgraph/cpp/CMakeLists.txt;0;")
add_test([=[parallel_close]=] "/Users/zacharykent/ParallelEgraph/cpp/build/parallel_close_test")
set_tests_properties([=[parallel_close]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/zacharykent/ParallelEgraph/cpp/CMakeLists.txt;65;add_test;/Users/zacharykent/ParallelEgraph/cpp/CMakeLists.txt;0;")
subdirs("_deps/parlaylib-build")
