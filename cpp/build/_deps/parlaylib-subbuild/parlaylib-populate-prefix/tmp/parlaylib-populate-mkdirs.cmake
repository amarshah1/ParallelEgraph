# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-src"
  "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-build"
  "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix"
  "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/tmp"
  "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src/parlaylib-populate-stamp"
  "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src"
  "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src/parlaylib-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src/parlaylib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/zacharykent/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src/parlaylib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
