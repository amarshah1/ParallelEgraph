# Running benchmarks on a cloud machine

End-to-end recipe for running `run_all_benchmarks.py` (the four-phase
sweep) and `plot_results.py` (figures from the resulting CSVs) on a
fresh Linux box.

## What gets run

`run_all_benchmarks.py` drives four benchmark suites into one
timestamped folder under `runs/`:

| phase         | binary                  | what it sweeps                                            |
| ------------- | ----------------------- | --------------------------------------------------------- |
| `random`      | `closure_compare_bench` | 6 baked-in random DAGs at every thread count              |
| `synthetic`   | `synthetic_bench`       | (family, n, threads) for chain/grid/cube/quartic/quintic  |
| `cube_decomp` | `synthetic_bench`       | cube only, swept over d ∈ {2,3,4,5} × k ∈ {5,55,105,155}  |
| `egg`         | `egraph-cc`             | every `.smt2` in `cc-benchmarks/smt-grounded` (submodule) |

All four use a 1 warmup + 5 measured trials policy. `PE_TRACE=1` is set
on every invocation so per-round semisort/consolidate timings land
alongside the master CSV.

## 1. System packages

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
    git build-essential cmake \
    python3 python3-venv python3-pip \
    libjemalloc-dev numactl
```

Amazon Linux / RHEL: swap `apt-get` for `dnf` and `libjemalloc-dev` for
`jemalloc-devel`.

`libjemalloc` is required because `CMakeLists.txt` defaults
`USE_JEMALLOC=ON`. To skip it:

```bash
cmake -B build -S . -DUSE_JEMALLOC=OFF
```

(once, before running the orchestrator — the cmake cache persists).

`numactl` is optional but recommended on multi-socket boxes; the
orchestrator auto-prepends `numactl -i all` when it's on PATH.

## 2. Clone the repos

`cc-benchmarks` is a git submodule of this repo, so it lives at
`ParallelEgraph/cc-benchmarks/`. A plain `git clone` leaves the
submodule directory empty; populate it in one of two ways:

**Option A — clone with submodules** (one step, downloads everything):

```bash
cd ~
git clone --recurse-submodules <YOUR_PARALLELEGRAPH_REMOTE> ParallelEgraph
cd ParallelEgraph
```

**Option B — clone first, init the submodule later** (smaller initial
clone; useful if you only run the random/synthetic/cube_decomp phases):

```bash
cd ~
git clone <YOUR_PARALLELEGRAPH_REMOTE> ParallelEgraph
cd ParallelEgraph
git submodule update --init --recursive cc-benchmarks
```

`run_all_benchmarks.py` runs `git submodule update --init --recursive
cc-benchmarks` automatically the first time the egg phase needs the
data, so you can skip the explicit init step if you don't mind the
implicit network call on first run.

## 3. Python env (for plotting only)

Benchmark drivers themselves are stdlib-only. Plotting needs matplotlib:

```bash
python3 -m venv .venv
.venv/bin/pip install matplotlib
```

## 4. Run the benchmarks

The first invocation builds the C++ targets via cmake (Release). All
four phases share `closure_compare_bench`, `synthetic_bench`, and
`egraph-cc`.

Default thread sweep is `1,2,4,8,16,32,64,128`. Override to match the
machine:

```bash
python3 run_all_benchmarks.py \
    --threads-sweep 1,2,4,8,16,32,64,128,192 \
    --egg-timeout 120
```

This is a long run. Daemonize with `nohup` so it survives an SSH
disconnect:

```bash
nohup python3 run_all_benchmarks.py \
    --threads-sweep 1,2,4,8,16,32,64,128,192 \
    --egg-timeout 120 \
    > run.log 2>&1 &
disown
tail -f run.log
```

Output lands in `runs/<YYYYMMDD_HHMMSS>/`. Watch progress live:

```bash
RUN=$(ls -t runs/ | head -1)
tail -f run.log
# or watch CSVs grow as invocations complete:
tail -f runs/$RUN/synthetic.csv
tail -f runs/$RUN/egg.csv
```

### Layout of a finished run

```
runs/<ts>/
  random.csv            random_trace.csv      random_traces/T<thr>.log
  synthetic.csv         synthetic_trace.csv   synthetic_traces/<family>/<family>_n<N>_T<thr>.log
  cube_decomp.csv       cube_decomp_trace.csv cube_decomp_traces/d<d>_k<k>_T<thr>.log
  egg.csv               egg_trace.csv         egg_traces/<basename>_T<thr>.log
```

`*.csv`         — per-trial wallclock + (for egg) phase timings.
`*_trace.csv`   — per-round consolidate/frontier/semisort breakdowns,
parsed from the `PE_TRACE=1` stderr stream.
`*_traces/*.log`— raw stderr per invocation (for sanity-checking).

### Cheaper first passes

The egg phase is the longest (605 files × N threads × 6 invocations).
Trim it via:

- `--skip egg` — drop the egg phase entirely.
- `--egg-pattern 'demo.*'` — basename glob filter.
- `--egg-timeout 30` — tighter wall budget per file.
- `--threads-sweep 1,2,4,8,16,32` — drop the largest counts.

Skip flags are repeatable: `--skip random --skip cube_decomp`.

## 5. Plot

Once the run finishes:

```bash
RUN=$(ls -t runs/ | head -1)
.venv/bin/python3 plot_results.py runs/$RUN
```

Figures land in `runs/$RUN/figs/`:

| figure                           | what it shows                                                                          |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| `<phase>_<workload>_T<thr>.png`  | (A) per-round stacked bars: consolidate / frontier_build / semi:keyed / semi:group_by / semi:per_group / semi:canon+dedup |
| `random_speedup.png`             | (B) log-log speedup vs nelson_seq, color = workload                                    |
| `synthetic_speedup.png`          | (B) log-log speedup vs nelson_seq, color = (family, n), linestyle = d                  |
| `cube_decomp_speedup.png`        | (B) log-log speedup vs nelson_seq, color = (cube, k), linestyle = d                    |
| `egg_speedup.png`                | (B) self-relative speedup (T1/T) on top-10 longest-close egg files                     |
| `<phase>_rounds_T<thr>.png`      | (C) per-round time lines at T=1 and T=max                                              |

All values are MEDIAN across the 5 trials.

`--top-n 10` controls how many egg files participate in the egg figures
(top by close time at T=max).

### Pulling figures back to your laptop

```bash
# from your laptop:
RUN=20260430_215551    # whatever ls -t showed
scp -r user@host:~/ParallelEgraph/runs/$RUN/figs/ ./figs_$RUN/
```

## Common gotchas

**`libjemalloc not found` during cmake configure** — install
`libjemalloc-dev` (Ubuntu) / `jemalloc-devel` (RHEL), or run cmake once
with `-DUSE_JEMALLOC=OFF` before invoking the orchestrator.

**egg phase says "no .smt2 files matched"** — the cc-benchmarks
submodule isn't populated. Run `git submodule update --init --recursive
cc-benchmarks` (or re-clone the parent with `--recurse-submodules`).
The orchestrator tries to do this automatically; if it fails, the
network is likely blocked.

**`run_all_benchmarks.py` says "(c) cube_decomp" but I see no
data** — `cube_decomp` was added recently; older orchestrator copies
won't have it. Pull latest, and confirm `--skip` doesn't include it.

**Egg phase prints `TIMEOUT` / `ERROR` rows** — expected for a few
heavy SMT files at low thread counts. `egg.csv` records them; figures
skip them when computing medians. Bump `--egg-timeout` if too many
files hit the wall budget.

**Plots have only the "linear (ideal)" line** — likely no matching
nelson_seq baseline in the data. Check the CSV's `algorithm` column
contains `nelson_seq` rows. The orchestrator runs nelson once per
workload (at the first thread count) — `--skip-nelson` style env vars
should not be set.

**Disk usage** — per-config trace logs add up: ~600 egg files × 9
thread counts ≈ 5400 trace files. Each is small (KB), total typically
<100 MB. Compress old runs with `tar -czf runs/<ts>.tgz runs/<ts>/`.
