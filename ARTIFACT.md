# Artifact for FMCAD 2026 — A Parallel E-Graph Congruence Closure

This artifact accompanies the FMCAD 2026 paper on parallel congruence
closure for e-graphs. The implementation, benchmark drivers, and the
reproduction script that regenerates every table in the paper are
included.

- **License**: MIT (see `LICENSE`).
- **Project README** (overview): `README.md`.
- **Design notes** (algorithms, invariants): `DESIGN.md`.

This `ae` branch is the **artifact-evaluation branch**: it contains
exactly the code paths exercised by the paper's reproduction
command — no ablations, no in-progress variants. The main development
branch has many more algorithm variants (sequential simple/topo/dst,
parallel topo/naive/filter-min/filter-hybrid); see the `main` branch
if you want to inspect those.

---

## Algorithms

Three algorithms, all proven sound by `tests/closure_test.cpp` (7 cases,
all must pass on a clean build):

| Tag | Method | Role |
|---|---|---|
| `nelson_simple_inline` | `sequential_close_simple_inline` | Sequential CC baseline — per-arity inline-storage hashtables (`SigK<1..4>`) plus a bump-allocated `SigBump` fallback for arity > 4. Worklist seeded with every non-leaf; pop one, canonicalize, hashcons or union. |
| `par_parents` | `parallel_parents` | Parallel BSP closure. Builds an inverted child→parent index at construction; each round walks the frontier through `parents_`, semisorts CanonEntries by signature hash, and `dnc_union`s per non-singleton bucket. |
| `par_filter` | `parallel_filter` | Parallel filter-mode closure. Per-class round stamps in `last_marked_`. Each round filters "dirty" terms via `last_marked_[find_root(child)] ∈ {R-1, R}`, semisorts dirty by canonical sig, dnc_unions buckets, restamps. Loops until dirty is empty. |

---

## Resource requirements

- **Disk**: ~3 GB total (~1 GB miter-cc-benchmarks gates corpus, fetched
  on first run + ~500 MB build artifacts + ~500 MB results).
- **RAM**: ~32 GB peak. The XL ladder's `32XL` rung builds a 64M-node
  e-graph (50K leaves × 16 fns × 64M nodes / 6.4M merges); peak working
  set hits roughly 8 bytes × 64M ≈ 0.5 GB UF + ~5 GB parents_
  + intermediate semisort buffers. Smaller machines should restrict the
  XL ladder via `--xl-labels XL,2XL,4XL` (skips the deepest rungs).
- **Cores**: any. The reproduction script sweeps T = 1, 2, 4, …, 192
  by default; cells with T > the machine's logical-core count are still
  measured but oversubscribed.
- **OS**: Linux preferred (Ubuntu 26.04 on the AWS c8i.metal-48xl instance
  used for the paper's measurements — Intel Xeon 6975P-C, 96 physical cores
  × 2 SMT = 192 threads, single socket, NUMA SNC=3, 480 MiB shared L3).
  macOS builds and runs; `numactl -i all` is Linux-only and auto-detected.

---

## What this artifact contains

| Path | What |
|---|---|
| `src/`, `include/parallel_egraph/` | Library + algorithm implementations (3 algos retained on this branch). |
| `tests/closure_test.cpp` | 7-case correctness suite. All must pass. |
| `tests/unionfind_test.cpp` | Lock-free union-find correctness. |
| `bench/closure_compare.cpp` | Synthetic uniform-DAG bench, six baked-in workloads + `PE_BENCH_CUSTOM`. |
| `bench/synthetic_bench.cpp` | Five DAG families × parameterized n, used for the cube-decomposition study. |
| `bench/gates_bench.cpp` | `.gates` driver for the miter-cc-benchmarks corpus. |
| `compare_topo_vs_async.py` | Driver that orchestrates the random / cube_decomp / gates phases across thread counts. |
| `bench/scripts/kick_the_tires.sh` | ~3-minute smoke test. |
| `bench/scripts/run_artifact.sh` | Full reproduction wrapper around `compare_topo_vs_async.py`. |
| `Dockerfile` | Reproducible build environment. |
| `miter-cc-benchmarks/` | Git submodule (miter-cc gates corpus for the gates phase). |

---

## Quickstart (no Docker)

```bash
# 1. Fetch the miter-cc-benchmarks corpus (used by the gates phase).
#    If this artifact was extracted from a zip, the empty
#    `miter-cc-benchmarks/` placeholder needs to be replaced by a
#    direct clone (there's no .git/ to drive submodule init from):
[ -d .git ] \
  && git submodule update --init --recursive miter-cc-benchmarks \
  || { rmdir miter-cc-benchmarks 2>/dev/null; \
       git clone --depth 1 \
         https://github.com/amarshah1/miter-cc-benchmarks.git \
         miter-cc-benchmarks; }
# (Step 1 is optional. If you skip it, the python driver will auto-fetch
#  the corpus the first time the gates phase needs it.)

# 2. Build (~2 minutes).
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Smoke test (~3 minutes).
./bench/scripts/kick_the_tires.sh

# 4. Full reproduction (~30-90 minutes).
./bench/scripts/run_artifact.sh
```

Expected smoke-test final line: `kick-the-tires complete.` with
`7/7 closure tests passed`.

## Quickstart (Docker)

Docker is an alternative path for reviewers who'd rather not install
the Linux dependencies (`build-essential`, `cmake`, `libjemalloc-dev`,
`numactl`, `python3`) directly on the host. If you've never used
Docker before:

**1. Install Docker.** On Linux, follow the official engine
instructions at <https://docs.docker.com/engine/install/>. On macOS
or Windows, install Docker Desktop from
<https://www.docker.com/products/docker-desktop/> and launch it once.
You can confirm Docker is working with:

```bash
docker run --rm hello-world
```

If you see "Hello from Docker!" then Docker is ready.

**2. Build the image.** From this artifact's top-level directory
(the one containing the `Dockerfile`):

```bash
docker build -t parallel-egraph .
```

This downloads an Ubuntu 26.04 base image, installs the build
dependencies, and compiles the C++ binaries. First build takes
roughly 3-5 minutes; later builds reuse cached layers. The final
image is about 1 GB.

**3. Run the smoke test inside the image.** No host side-effects:
the `--rm` flag removes the container after it exits.

```bash
docker run --rm parallel-egraph ./bench/scripts/kick_the_tires.sh
```

Expected last line: `kick-the-tires complete.`

**4. Run the full reproduction.** Mount a host directory so the CSV
outputs (about ~300 MB) persist after the container exits:

```bash
mkdir -p results
docker run --rm -v "$PWD/results:/work/runs" parallel-egraph \
  ./bench/scripts/run_artifact.sh
```

If your host has only a single NUMA node (laptops, Docker Desktop),
pass `--no-numactl` to skip the numactl auto-detect:

```bash
docker run --rm -v "$PWD/results:/work/runs" parallel-egraph \
  python3 compare_topo_vs_async.py --skip synthetic --no-numactl \
  --algos nelson_simple_inline,par_parents,par_filter
```

After the run finishes, the `results/` directory on your host holds
the CSVs (`random.csv`, `cube_decomp.csv`, `gates.csv`) and the raw
trace logs.

---

## Reproduction command (verbatim from the paper)

```bash
python3 compare_topo_vs_async.py \
  --skip synthetic \
  --threads-sweep 1,2,4,8,16,32,64,96,128,192 \
  --warmup 1 \
  --random-modes xl,default \
  --trials 5 \
  --algos nelson_simple_inline,par_parents,par_filter \
  > run.log
```

`run_artifact.sh` is a thin wrapper around this command. Knobs:

| Env | Meaning |
|---|---|
| `THREADS=1,2,...,192` | thread-sweep (default `1,2,4,8,16,32,64,96,128,192`, matching the paper invocation) |
| `TRIALS=N` | measured trials per cell |
| `WARMUP=N` | warmup invocations per cell |
| `EXTRA="--shutdown-after"` | appended to the python args (unattended overnight runs) |

Phase outputs go to `runs/topo_vs_async_<timestamp>/`:

| File | What |
|---|---|
| `random.csv` | XL ladder (XL, 2XL, …, 32XL) + 6 default workloads × T sweep |
| `cube_decomp.csv` | (d, k) sweep over cube workloads |
| `gates.csv` | per-.gates-file timings |
| `traces/...` | raw stdout/stderr per (phase, T, cell) |

---

## Reproduction time

| Phase | 12-core laptop | c8i.metal-48xl (192 threads) |
|---|---|---|
| Build | 2 min | 1 min |
| Correctness tests | <1 sec | <1 sec |
| Random (default + XL) | 20 min | 12 min |
| Cube decomp | 15 min | 10 min |
| Gates | 25 min | 15 min |
| **Total** | **~60 min** | **~40 min** |

The XL ladder's `32XL` rung (64M nodes) dominates the random phase; cap
via `--xl-labels XL,2XL,4XL,8XL` if you want a faster run.
