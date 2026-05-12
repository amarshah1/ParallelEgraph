# Reproducible build for the FMCAD 2026 artifact.
#
# Build:
#   docker build -t parallel-egraph .
# Quick smoke test (~3 min):
#   docker run --rm -it parallel-egraph ./bench/scripts/kick_the_tires.sh
# Full reproduction (timing depends on host cores; ~30-90 min):
#   docker run --rm -it -v "$PWD/results:/work/bench/results" parallel-egraph \
#     ./bench/scripts/run_artifact.sh

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      ca-certificates \
      libjemalloc-dev \
      numactl \
      python3 \
      python3-pip \
      python3-pandas \
      python3-matplotlib \
      python3-numpy \
      bash gawk coreutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work

# Initialize the miter-cc-benchmarks submodule (gates phase). When
# building from a `git archive` tarball the submodule won't be present,
# so we fetch it explicitly. The build still succeeds without it; the
# gates phase will simply report no .gates files matched.
RUN if [ -d .git ] && [ -z "$(ls miter-cc-benchmarks 2>/dev/null)" ]; then \
      git submodule update --init --recursive miter-cc-benchmarks || true; \
    fi

RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j --target \
         closure_test \
         unionfind_test \
         closure_compare_bench \
         synthetic_bench \
         gates_bench

# Default to an interactive shell so reviewers can poke around.
CMD ["/bin/bash"]
