# Reproducible build for the FMCAD 2025 artifact.
#
# Build:
#   docker build -t parallel-egraph .
# Quick smoke test (~3 min):
#   docker run --rm -it parallel-egraph ./bench/scripts/kick_the_tires.sh
# Full reproduction (timing depends on host cores; ~30-90 min):
#   docker run --rm -it -v "$PWD/results:/work/bench/results" parallel-egraph \
#     ./bench/scripts/run_artifact.sh

FROM ubuntu:24.04

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
      bash awk gawk coreutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work

# Initialize the cc-benchmarks submodule if not already populated.
# When building from a `git archive` tarball the submodule won't be
# present, so we fetch it explicitly. The build will still succeed
# without it; eggcc-specific benches will skip.
RUN if [ -d .git ] && [ -z "$(ls cc-benchmarks 2>/dev/null)" ]; then \
      git submodule update --init --recursive cc-benchmarks || true; \
    fi

RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j --target \
         closure_test \
         unionfind_test \
         closure_compare_bench \
         synthetic_bench \
         smt_bench

# Default to an interactive shell so reviewers can poke around.
CMD ["/bin/bash"]
