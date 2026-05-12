# Artifact for FMCAD 2025 — A Parallel E-Graph Congruence Closure

This artifact accompanies the FMCAD 2025 paper on parallel congruence
closure for e-graphs. The implementation, the benchmark drivers, and
the reproduction scripts that regenerate every table in the paper are
included.

- **License**: MIT (see `LICENSE`).
- **Project README** (overview): `README.md`.
- **Design notes** (algorithms, invariants): `DESIGN.md`.

---

## Algorithm landscape

The paper compares a family of closure algorithms. The artifact ships
all of them so the reader can verify both the **headline results** and
the **ablations** (intermediate variants that demonstrate where the
soundness/performance trade-offs live).

### Sequential

| Tag | Method | Sound? | Role |
|---|---|---|---|
| `nelson_seq` | `sequential_close_nelson` | yes | Nelson baseline |
| `nelson_topo` | `sequential_close_topo` | **no** | Illustrative single-pass; fails the `seq_topo_adversarial` test by design. Shown to demonstrate the cost of skipping iteration |
| `nelson_topo_iter` | `sequential_close_topo_iter` | yes | **Headline sequential** (iterated single-pass to fixed point) |
| `nelson_dst`, `nelson_simple*` | various | yes | Ablations |

### Parallel

| Tag | Method | Sound? | Role |
|---|---|---|---|
| `par_parents` | `parallel_parents` | yes | BSP-with-parents baseline (the algorithm the paper supersedes) |
| `par_topo_iter` | `parallel_topo_iter` | yes | Depth-stratified semisort, iterated to fixed point |
| `par_filter` | `parallel_filter` | yes | **Headline parallel** (filter-style, lock-free) |
| `par_filter_min_id` | `parallel_filter_min_id` | yes | Variant of `par_filter` using min-id tie-break |
| `par_filter_gbk`, `par_filter_hybrid`, `par_naive` | various | yes | Ablations |

> The two illustrative-only tests `par_topo_cross_depth` and
> `seq_topo_adversarial` exist specifically to demonstrate the
> unsoundness of the non-iterated single-pass variants. The kick-the-tires
> script tolerates exactly those two failures and rejects any others.

---

## Resource requirements

- **Disk**: ~3 GB total (2 GB submodule + 500 MB build + 500 MB results).
- **RAM**: ~8 GB peak (the largest synthetic workload, quintic-20,
  builds a 12.8M-class e-graph).
- **Cores**: any. The artifact runs sensibly on a 4-core laptop or a
  144-core server; reproduction time is the only thing that varies.
- **OS**: Linux preferred (Ubuntu 22.04+ tested). macOS builds and runs;
  `numactl -i all` is Linux-only and auto-detected/skipped on macOS.

---

## What this artifact contains

| Path | What |
|---|---|
| `src/`, `include/parallel_egraph/` | Library and algorithm implementations. |
| `tests/closure_test.cpp` | 20-case correctness suite. 18 must pass; the 2 failures are the documented illustrative-unsound variants above. |
| `bench/closure_compare.cpp` | Synthetic uniform-DAG benchmark (six workloads). |
| `bench/synthetic_bench.cpp` | Five DAG families × n ∈ {5,10,20}. |
| `bench/smt_bench.cpp` | SMT-LIB driver used for the 507-file eggcc sweep. |
| `bench/scripts/kick_the_tires.sh` | ~3-minute smoke test. |
| `bench/scripts/run_artifact.sh` | Full reproduction (~30-90 min wall time). |
| `bench/scripts/aggregate_artifact.sh` | Parses CSVs into the paper's claim table. |
| `Dockerfile` | Reproducible build environment. |
| `cc-benchmarks/` | Git submodule of the eggcc QF_UF corpus (507 ground SMT-LIB files). |

---

## Quickstart (no Docker)

```bash
# 1. Fetch the eggcc corpus submodule (skip if cc-benchmarks/ already populated).
git submodule update --init --recursive cc-benchmarks

# 2. Build (~2 minutes).
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Smoke test (~3 minutes).
./bench/scripts/kick_the_tires.sh

# 4. Full reproduction (~30-90 minutes).
./bench/scripts/run_artifact.sh
```

Expected `kick_the_tires.sh` final line: `kick-the-tires complete.`

Expected `closure_test` outcome: `18/20 closure tests passed` (the two
failures are `par_topo_cross_depth` and `seq_topo_adversarial` —
documented above).

## Quickstart (Docker)

```bash
docker build -t parallel-egraph .

docker run --rm -it parallel-egraph ./bench/scripts/kick_the_tires.sh

mkdir -p results
docker run --rm -it -v "$PWD/results:/work/bench/results" parallel-egraph \
  ./bench/scripts/run_artifact.sh
```

The Dockerfile installs `numactl`, `jemalloc`, GCC/CMake, and the
Python plotting dependencies. No internet access is required at runtime.

---

## Reproduction time

`run_artifact.sh` runs four phases in sequence; the eggcc sweep
dominates.

| Phase | 12-core laptop | 144-core server |
|---|---|---|
| Build | 2 min | 1 min |
| Correctness suite | <1 sec | <1 sec |
| T1 closure_compare | 1 min | 30 sec |
| T2 synthetic_bench | 18 min | 12 min |
| T3 eggcc sweep (507 files) | 20 min | 12 min |
| T4 strong-scaling (T=1..max) | 5 min | 8 min |
| **Total** | **~45 min** | **~33 min** |

Knobs: `T_MAX=64 ./bench/scripts/run_artifact.sh` caps thread count;
`NUMACTL= ./bench/scripts/run_artifact.sh` disables numactl
interleaving (default auto-detects multi-NUMA-node and enables it).

---

## What gets produced

All outputs land in `bench/results/artifact/`:

| File | Maps to paper |
|---|---|
| `build_info.txt` | machine info, NUMA topology, git SHA |
| `correctness.log` | `18/20 closure tests passed` — the soundness claim |
| `T1_closure_compare.csv` | Table 1 (synthetic uniform DAGs) |
| `T2_synthetic.csv` | Table 2 (five DAG families) |
| `T3_eggcc.csv` | Table 3 (eggcc QF_UF, 507 files) |
| `T4_scaling.csv` | Strong-scaling figure (par_filter across T) |
| `summary.txt` | `aggregate_artifact.sh` output — claim ratios per workload |

---

## Headline claims and how to verify them

After running `run_artifact.sh`, the `summary.txt` it produces maps each
of the following claims to a CSV cell. **All claims are ratios** between
algorithm wallclock times; absolute milliseconds vary up to 2× across
machines, but the ratios are stable to within ±20%.

| Claim | Verifier (in `summary.txt`) |
|---|---|
| **Soundness**: every *sound* algorithm agrees with Nelson on 18 / 20 hand-built test cases. The 2 expected failures are documented unsound variants kept for ablation. | `correctness.log` reports `18/20 closure tests passed`; the two failing tests are exactly `par_topo_cross_depth` and `seq_topo_adversarial`. |
| **C1**: `nelson_topo_iter` is ≥ 2× faster than `nelson_seq` across every closure_compare workload. | T1 — `nelson_seq / nelson_topo_iter` column ≥ 2 for every row. |
| **C2**: `par_filter` is ≥ 3× faster than `nelson_topo_iter` on the largest synthetic workloads (quartic-20, quintic-20). | T2 — those two rows show `nelson_topo_iter / par_filter` ≥ 3. |
| **C3**: On the eggcc corpus (507 ground QF_UF files), `par_filter` aggregate beats `nelson_topo_iter`; on the ≥100K-class subset (the workloads where parallelism actually helps) the ratio is ≥ 1.4× with > 80% win rate. | T3 summary: aggregate ratio ≥ 1.2×; ≥100K bucket ratio ≥ 1.4×, wins ≥ 80%. |
| **C4**: Strong scaling on closure_compare *large* is monotone until atomic-UF / memory-bandwidth saturation (~T = 64–128 on multi-socket NUMA). | T4 — for workload `large`, `par_filter` wall time decreases monotonically through T = 64. |

---

## Reusing the artifact beyond the paper

The benchmark drivers accept several environment variables for ad-hoc
sweeps without rebuilding:

| Env var | Meaning |
|---|---|
| `PARLAY_NUM_THREADS=N` | run parallel code with N workers |
| `PE_BENCH_ALGOS=alg1,alg2,...` | (closure_compare) restrict to a subset of algorithms |
| `PE_BENCH_ONLY=<workload>` | restrict to one workload |
| `PE_BENCH_CUSTOM=L,F,N,M,D` | inject a custom (leaves, fns, nodes, merges, depth) workload |
| `PE_BENCH_FORMAT=csv` + `PE_BENCH_HEADER=1` | machine-readable output |
| `PE_SYNTH_FAMILIES=cube,grid` | (synthetic_bench) restrict families |
| `PE_SYNTH_NS=5,10,20` | (synthetic_bench) override sizes |
| `PE_SMT_TRIALS=N` `PE_SMT_WARMUP=N` | (smt_bench) trial counts |
| `PE_TRACE=1` | dump per-round timings to stderr |

For new SMT-LIB inputs of your own:

```bash
./build/smt_bench path/to/your.smt2
./build/smt_bench /path/to/dir
```

Output CSV matches the schema used in T3, so you can pipe it through
`bench/scripts/aggregate_artifact.sh` directly.

---

## Algorithm pointers (for reviewers inspecting the source)

- **`nelson_topo_iter`** (sound sequential headline) — `src/egraph.cpp`
  `sequential_close_topo_iter`. Wraps the single-pass topo walk in an
  outer loop that iterates until no new unions are introduced.
- **`parallel_filter`** (sound parallel headline) — `src/egraph_filter.cpp`
  `parallel_filter`. Filter-style closure on `last_marked_` round
  stamps; each parlay::parallel_for round filters the dirty set down
  by canonicalizing then unioning.
- **`parallel_topo_iter`** — `src/egraph.cpp` `parallel_topo_iter`.
  Depth-stratified BSP (children-before-parents) with an outer
  fixpoint loop that re-runs rounds whose UF state changed.
- **The Lock-free UF** — `src/unionfind.cpp` (Jayanti & Tarjan Listing 3).

`DESIGN.md` has the full write-up.

---

## Submission checklist (for ourselves, before uploading to EasyChair)

- [x] LICENSE (MIT) at repo root
- [x] README + ARTIFACT.md
- [x] Dockerfile (Ubuntu 24.04 base, no internet at runtime)
- [x] kick_the_tires.sh and run_artifact.sh
- [x] aggregate_artifact.sh maps each paper claim to a CSV cell
- [x] Reproducible build (FetchContent for parlay; jemalloc, numactl pinned via apt)
- [ ] Compute SHA256 of the final archive (`shasum -a 256 artifact.zip`)
- [ ] Host the .zip at a stable URL (institutional page is fine; DOI required for camera-ready)
- [ ] EasyChair submission: link, SHA256, tested-on description ("Ubuntu 24.04 + GCC 13.2 + 144-core EPYC; also tested macOS 14 + M-series")
