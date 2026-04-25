# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-src")
  file(MAKE_DIRECTORY "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-src")
endif()
file(MAKE_DIRECTORY
  "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-build"
  "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix"
  "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/tmp"
  "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src/parlaylib-populate-stamp"
  "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src"
  "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src/parlaylib-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src/parlaylib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/amarshah/Desktop/Classes/Spring_2026/parallel/project/ParallelEgraph/cpp/build/_deps/parlaylib-subbuild/parlaylib-populate-prefix/src/parlaylib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
