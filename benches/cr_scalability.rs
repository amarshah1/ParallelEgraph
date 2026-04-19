//! Scalability benchmark for collect_reduce.
//!
//! Measures throughput (Melem/sec) while varying:
//!   1. Thread count      — 1, 2, 4, 8, ... up to available cores
//!   2. Input size         — 10K to 1M elements
//!   3. Bucket count       — few (4) vs many (1024)
//!   4. Key distribution   — uniform, zipfian (skewed), single-key (extreme)
//!
//! For each configuration, the benchmark computes a sequential baseline (1 thread)
//! and reports the parallel speedup factor.
//!
//! Usage:
//!   cargo bench --bench cr_scalability                     # all experiments
//!   cargo bench --bench cr_scalability -- thread-scaling   # just thread sweep
//!   cargo bench --bench cr_scalability -- size-scaling     # just size sweep
//!   cargo bench --bench cr_scalability -- distribution     # just distribution sweep
//!   cargo bench --bench cr_scalability -- union-find       # collect_reduce driving union-find

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use parallel_egraph::collect_reduce::{
    collect_reduce, collect_reduce_par_sort, collect_reduce_par_sort_with_scratch,
    collect_reduce_radix, collect_reduce_radix_with_scratch, collect_reduce_with_scratch,
    CollectReduceHelper, CrScratch,
};
use parallel_egraph::unionfind::ConcurrentUnionFind;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const TRIALS: usize = 5;
const WARMUP_TRIALS: usize = 1;

/// Size-adaptive trial count. Small inputs run more trials so fixed-cost
/// overheads (rayon spawn, first-touch page faults, allocator warm-up)
/// fall into the noise under best-of-K.
fn trials_for(n: usize) -> usize {
    match n {
        n if n <= 1_000_000 => 50,
        n if n <= 5_000_000 => 20,
        n if n <= 20_000_000 => 10,
        _ => TRIALS,
    }
}

fn warmups_for(n: usize) -> usize {
    if n <= 5_000_000 { 5 } else { WARMUP_TRIALS }
}

// ---------------------------------------------------------------------------
// Test element and helpers
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct KV {
    key: usize,
    val: u64,
}

struct SumHelper {
    sums: Vec<AtomicUsize>,
}

impl SumHelper {
    fn new(num_buckets: usize) -> Self {
        SumHelper {
            sums: (0..num_buckets).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    fn reset(&self) {
        for s in &self.sums {
            s.store(0, Ordering::Relaxed);
        }
    }
}

impl CollectReduceHelper<KV> for SumHelper {
    fn get_key(&self, elem: &KV) -> usize {
        elem.key
    }

    fn apply(&self, elem: &KV) {
        self.sums[elem.key].fetch_add(elem.val as usize, Ordering::Relaxed);
    }

    fn combine(&self, elems: &[KV]) {
        let mut local = vec![0usize; self.sums.len()];
        for elem in elems {
            local[elem.key] += elem.val as usize;
        }
        for (k, &v) in local.iter().enumerate() {
            if v > 0 {
                self.sums[k].fetch_add(v, Ordering::Relaxed);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// No-contention helper: each element writes to a distinct output slot.
// `elem.val` is populated as a unique index in [0, n), so apply/combine
// perform only disjoint raw writes — no atomics, no shared cache lines.
// Isolates sort+iterate throughput from any concurrent-data-structure cost.
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct SendPtr(*mut u64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

struct PerSlotHelper {
    out: SendPtr,
    len: usize,
}

impl PerSlotHelper {
    fn new(n: usize) -> (Self, Vec<u64>) {
        let mut buf: Vec<u64> = vec![0; n];
        let ptr = SendPtr(buf.as_mut_ptr());
        (PerSlotHelper { out: ptr, len: n }, buf)
    }
}

impl CollectReduceHelper<KV> for PerSlotHelper {
    fn get_key(&self, elem: &KV) -> usize {
        elem.key
    }

    fn apply(&self, elem: &KV) {
        let idx = elem.val as usize;
        debug_assert!(idx < self.len);
        unsafe { std::ptr::write(self.out.0.add(idx), elem.val) };
    }

    fn combine(&self, elems: &[KV]) {
        for e in elems {
            self.apply(e);
        }
    }
}

/// Generate data where `val` is a unique index in [0, n) so PerSlotHelper
/// writes never collide. Keys still drawn from the requested distribution.
fn generate_data_unique_val(
    n: usize,
    num_buckets: usize,
    dist: &Distribution,
    seed: u64,
) -> Vec<KV> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let keys: Vec<usize> = match dist {
        Distribution::Uniform => (0..n).map(|_| rng.gen_range(0..num_buckets)).collect(),
        Distribution::Zipfian(z) => {
            let zipf = Zipfian::new(num_buckets as u32, *z);
            (0..n).map(|_| zipf.next(&mut rng) as usize).collect()
        }
        Distribution::SingleKey(k) => vec![*k; n],
    };
    keys.into_iter()
        .enumerate()
        .map(|(i, key)| KV { key, val: i as u64 })
        .collect()
}

// ---------------------------------------------------------------------------
// Zipfian sampler (same as uf_ycsb)
// ---------------------------------------------------------------------------

struct Zipfian {
    n: u32,
    alpha: f64,
    eta: f64,
    zetan: f64,
    half_pow_z: f64,
}

impl Zipfian {
    fn new(n: u32, z: f64) -> Self {
        let zetan = Self::zeta(n as u64, z);
        let zeta2 = 1.0 + (0.5_f64).powf(z);
        let alpha = 1.0 / (1.0 - z);
        let half_pow_z = (0.5_f64).powf(z);
        let eta = (1.0 - (2.0 / n as f64).powf(1.0 - z)) / (1.0 - zeta2 / zetan);
        Zipfian { n, alpha, eta, zetan, half_pow_z }
    }

    fn zeta(n: u64, z: f64) -> f64 {
        (1..=n).map(|i| 1.0 / (i as f64).powf(z)).sum()
    }

    fn next(&self, rng: &mut impl Rng) -> u32 {
        let u: f64 = rng.gen();
        let uz = u * self.zetan;
        if uz < 1.0 { return 0; }
        if uz < 1.0 + self.half_pow_z { return 1; }
        let v = (self.n as f64 * (self.eta * u - self.eta + 1.0).powf(self.alpha)) as u32;
        v.min(self.n - 1)
    }
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

enum Distribution {
    Uniform,
    Zipfian(f64),
    SingleKey(usize),
}

impl std::fmt::Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Distribution::Uniform => write!(f, "uniform"),
            Distribution::Zipfian(z) => write!(f, "zipf({z:.2})"),
            Distribution::SingleKey(k) => write!(f, "single({k})"),
        }
    }
}

fn generate_data(n: usize, num_buckets: usize, dist: &Distribution, seed: u64) -> Vec<KV> {
    let mut rng = SmallRng::seed_from_u64(seed);
    match dist {
        Distribution::Uniform => (0..n)
            .map(|_| KV {
                key: rng.gen_range(0..num_buckets),
                val: rng.gen_range(1..100),
            })
            .collect(),
        Distribution::Zipfian(z) => {
            let zipf = Zipfian::new(num_buckets as u32, *z);
            (0..n)
                .map(|_| KV {
                    key: zipf.next(&mut rng) as usize,
                    val: rng.gen_range(1..100),
                })
                .collect()
        }
        Distribution::SingleKey(k) => (0..n)
            .map(|_| KV { key: *k, val: rng.gen_range(1..100) })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

fn bench_collect_reduce(data: &[KV], num_buckets: usize, threads: usize) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    let helper = SumHelper::new(num_buckets);

    // Warmup
    for _ in 0..WARMUP_TRIALS {
        helper.reset();
        pool.install(|| collect_reduce(data, &helper, num_buckets));
    }

    // Timed trials
    let mut best = Duration::MAX;
    for _ in 0..TRIALS {
        helper.reset();
        let start = Instant::now();
        pool.install(|| collect_reduce(data, &helper, num_buckets));
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

fn throughput_melem(n: usize, dur: Duration) -> f64 {
    n as f64 / dur.as_secs_f64() / 1e6
}

// ---------------------------------------------------------------------------
// Pretty table printer
// ---------------------------------------------------------------------------

struct Table {
    headers: Vec<String>,
    widths: Vec<usize>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn new(headers: &[&str]) -> Self {
        let widths = headers.iter().map(|h| h.len()).collect();
        Table {
            headers: headers.iter().map(|s| s.to_string()).collect(),
            widths,
            rows: Vec::new(),
        }
    }

    fn add_row(&mut self, cells: &[String]) {
        for (i, cell) in cells.iter().enumerate() {
            if i < self.widths.len() {
                self.widths[i] = self.widths[i].max(cell.len());
            }
        }
        self.rows.push(cells.to_vec());
    }

    fn print(&self) {
        let header_line: String = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:>w$}", h, w = self.widths[i]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {}", header_line);
        let sep: String = self
            .widths
            .iter()
            .map(|&w| "-".repeat(w))
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {}", sep);
        for row in &self.rows {
            let line: String = row
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{:>w$}", c, w = self.widths[i]))
                .collect::<Vec<_>>()
                .join("  ");
            println!("  {}", line);
        }
    }
}

// ---------------------------------------------------------------------------
// Thread count helpers
// ---------------------------------------------------------------------------

fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

fn thread_counts(max: usize) -> Vec<usize> {
    let mut v = vec![1];
    let mut t = 2;
    while t <= max {
        v.push(t);
        t *= 2;
    }
    if *v.last().unwrap() != max && max > 1 {
        v.push(max);
    }
    v
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

/// Experiment 1: Thread scaling — fixed size/buckets/distribution, vary threads.
fn experiment_thread_scaling() {
    println!();
    println!("=== Thread Scaling (collect_reduce) ===");

    let max_t = available_threads();
    let configs: Vec<(&str, usize, usize, Distribution)> = vec![
        ("uniform_1M_256b", 1_000_000, 256, Distribution::Uniform),
        ("uniform_1M_1024b", 1_000_000, 1024, Distribution::Uniform),
        ("zipf_1M_256b", 1_000_000, 256, Distribution::Zipfian(0.99)),
        ("single_1M_256b", 1_000_000, 256, Distribution::SingleKey(0)),
        ("uniform_100K_4b", 100_000, 4, Distribution::Uniform),
    ];

    let mut table = Table::new(&[
        "config", "n", "buckets", "dist", "threads",
        "Melem/s", "time_ms", "speedup",
    ]);

    for (name, n, num_buckets, dist) in &configs {
        let data = generate_data(*n, *num_buckets, dist, 42);
        let mut baseline_dur = Duration::MAX;

        for &threads in &thread_counts(max_t) {
            let dur = bench_collect_reduce(&data, *num_buckets, threads);
            if threads == 1 {
                baseline_dur = dur;
            }
            let tp = throughput_melem(*n, dur);
            let speedup = baseline_dur.as_secs_f64() / dur.as_secs_f64();

            table.add_row(&[
                name.to_string(),
                format!("{}", n),
                format!("{}", num_buckets),
                format!("{}", dist),
                format!("{}", threads),
                format!("{:.1}", tp),
                format!("{:.2}", dur.as_secs_f64() * 1000.0),
                format!("{:.2}x", speedup),
            ]);
        }
    }
    table.print();
}

/// Experiment 2: Size scaling — fixed threads, vary input size.
fn experiment_size_scaling() {
    println!();
    println!("=== Size Scaling (collect_reduce) ===");

    let max_t = available_threads();
    let sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000];
    let thread_list = [1, max_t / 2, max_t].into_iter()
        .filter(|&t| t >= 1)
        .collect::<Vec<_>>();
    // deduplicate
    let mut thread_list_dedup = vec![];
    for t in thread_list {
        if !thread_list_dedup.contains(&t) {
            thread_list_dedup.push(t);
        }
    }

    let mut table = Table::new(&[
        "n", "buckets", "threads", "Melem/s", "time_ms", "speedup_vs_1t",
    ]);

    let num_buckets = 256;
    for &n in &sizes {
        let data = generate_data(n, num_buckets, &Distribution::Uniform, 42);
        let baseline_dur = bench_collect_reduce(&data, num_buckets, 1);

        for &threads in &thread_list_dedup {
            let dur = bench_collect_reduce(&data, num_buckets, threads);
            let tp = throughput_melem(n, dur);
            let speedup = baseline_dur.as_secs_f64() / dur.as_secs_f64();

            table.add_row(&[
                format!("{}", n),
                format!("{}", num_buckets),
                format!("{}", threads),
                format!("{:.1}", tp),
                format!("{:.2}", dur.as_secs_f64() * 1000.0),
                format!("{:.2}x", speedup),
            ]);
        }
    }
    table.print();
}

/// Experiment 3: Distribution sweep — fixed size/threads, vary key distribution.
fn experiment_distribution() {
    println!();
    println!("=== Distribution Sweep (collect_reduce) ===");

    let max_t = available_threads();
    let dists: Vec<(&str, Distribution)> = vec![
        ("uniform", Distribution::Uniform),
        ("zipf_0.50", Distribution::Zipfian(0.50)),
        ("zipf_0.75", Distribution::Zipfian(0.75)),
        ("zipf_0.99", Distribution::Zipfian(0.99)),
        ("single_key", Distribution::SingleKey(0)),
    ];
    let n = 1_000_000;
    let num_buckets = 256;

    let mut table = Table::new(&[
        "dist", "n", "buckets", "threads",
        "Melem/s", "time_ms", "speedup",
    ]);

    for (name, dist) in &dists {
        let data = generate_data(n, num_buckets, dist, 42);
        let mut baseline_dur = Duration::MAX;

        for &threads in &thread_counts(max_t) {
            let dur = bench_collect_reduce(&data, num_buckets, threads);
            if threads == 1 {
                baseline_dur = dur;
            }
            let tp = throughput_melem(n, dur);
            let speedup = baseline_dur.as_secs_f64() / dur.as_secs_f64();

            table.add_row(&[
                name.to_string(),
                format!("{}", n),
                format!("{}", num_buckets),
                format!("{}", threads),
                format!("{:.1}", tp),
                format!("{:.2}", dur.as_secs_f64() * 1000.0),
                format!("{:.2}x", speedup),
            ]);
        }
    }
    table.print();
}

// ---------------------------------------------------------------------------
// Union-find via collect_reduce
// ---------------------------------------------------------------------------

/// Element: a union pair (a, b) keyed by find(a) so collect_reduce groups
/// operations touching the same equivalence class together.
#[derive(Copy, Clone)]
struct UnionPair {
    a: u32,
    b: u32,
}

/// Helper that performs union-find unions. The key is `find(a)` so that
/// pairs touching the same root are sorted into the same cache-local block.
struct UnionHelper<'a> {
    uf: &'a ConcurrentUnionFind,
}

impl<'a> CollectReduceHelper<UnionPair> for UnionHelper<'a> {
    fn get_key(&self, elem: &UnionPair) -> usize {
        self.uf.find_root(elem.a) as usize
    }

    fn apply(&self, elem: &UnionPair) {
        self.uf.union(elem.a, elem.b);
    }

    fn combine(&self, elems: &[UnionPair]) {
        for elem in elems {
            self.uf.union(elem.a, elem.b);
        }
    }
}

/// Generate union pairs. `uf_size` is the number of UF elements.
/// Keys are drawn from `dist` over `[0, uf_size)`.
fn generate_union_pairs(
    n: usize,
    uf_size: usize,
    dist: &Distribution,
    seed: u64,
) -> Vec<UnionPair> {
    let mut rng = SmallRng::seed_from_u64(seed);
    match dist {
        Distribution::Uniform => (0..n)
            .map(|_| UnionPair {
                a: rng.gen_range(0..uf_size as u32),
                b: rng.gen_range(0..uf_size as u32),
            })
            .collect(),
        Distribution::Zipfian(z) => {
            let zipf = Zipfian::new(uf_size as u32, *z);
            (0..n)
                .map(|_| UnionPair {
                    a: zipf.next(&mut rng),
                    b: rng.gen_range(0..uf_size as u32),
                })
                .collect()
        }
        Distribution::SingleKey(k) => (0..n)
            .map(|_| UnionPair {
                a: *k as u32,
                b: rng.gen_range(0..uf_size as u32),
            })
            .collect(),
    }
}

/// Benchmark: collect_reduce driving union-find operations.
/// Returns best-of-TRIALS wall time.
fn bench_cr_union_find(
    pairs: &[UnionPair],
    uf_size: usize,
    threads: usize,
) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    // Warmup
    for _ in 0..WARMUP_TRIALS {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        pool.install(|| collect_reduce(pairs, &helper, uf_size));
    }

    let mut best = Duration::MAX;
    for _ in 0..TRIALS {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        let start = Instant::now();
        pool.install(|| collect_reduce(pairs, &helper, uf_size));
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

/// Benchmark: collect_reduce_radix driving union-find operations.
fn bench_cr_radix_union_find(
    pairs: &[UnionPair],
    uf_size: usize,
    threads: usize,
) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    // Warmup
    for _ in 0..WARMUP_TRIALS {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        pool.install(|| collect_reduce_radix(pairs, &helper, uf_size));
    }

    let mut best = Duration::MAX;
    for _ in 0..TRIALS {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        let start = Instant::now();
        pool.install(|| collect_reduce_radix(pairs, &helper, uf_size));
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

/// Benchmark: collect_reduce_par_sort driving union-find operations.
fn bench_cr_par_sort_union_find(
    pairs: &[UnionPair],
    uf_size: usize,
    threads: usize,
) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    // Warmup
    for _ in 0..WARMUP_TRIALS {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        pool.install(|| collect_reduce_par_sort(pairs, &helper, uf_size));
    }

    let mut best = Duration::MAX;
    for _ in 0..TRIALS {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        let start = Instant::now();
        pool.install(|| collect_reduce_par_sort(pairs, &helper, uf_size));
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

/// Scratch-hoisted variant: semi_sort with pre-allocated CrScratch.
/// Adaptive trial count so small inputs get more samples.
fn bench_cr_union_find_hot(
    pairs: &[UnionPair],
    uf_size: usize,
    threads: usize,
) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let n = pairs.len();
    let warmups = warmups_for(n);
    let trials = trials_for(n);

    let mut scratch: CrScratch<UnionPair> = CrScratch::new();

    for _ in 0..warmups {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        pool.install(|| collect_reduce_with_scratch(pairs, &helper, uf_size, &mut scratch));
    }

    let mut best = Duration::MAX;
    for _ in 0..trials {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        let start = Instant::now();
        pool.install(|| collect_reduce_with_scratch(pairs, &helper, uf_size, &mut scratch));
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

fn bench_cr_radix_union_find_hot(
    pairs: &[UnionPair],
    uf_size: usize,
    threads: usize,
) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let n = pairs.len();
    let warmups = warmups_for(n);
    let trials = trials_for(n);

    let mut scratch: CrScratch<UnionPair> = CrScratch::new();

    for _ in 0..warmups {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        pool.install(|| collect_reduce_radix_with_scratch(pairs, &helper, uf_size, &mut scratch));
    }

    let mut best = Duration::MAX;
    for _ in 0..trials {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        let start = Instant::now();
        pool.install(|| collect_reduce_radix_with_scratch(pairs, &helper, uf_size, &mut scratch));
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

fn bench_cr_par_sort_union_find_hot(
    pairs: &[UnionPair],
    uf_size: usize,
    threads: usize,
) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let n = pairs.len();
    let warmups = warmups_for(n);
    let trials = trials_for(n);

    let mut scratch: CrScratch<UnionPair> = CrScratch::new();

    for _ in 0..warmups {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        pool.install(|| collect_reduce_par_sort_with_scratch(pairs, &helper, uf_size, &mut scratch));
    }

    let mut best = Duration::MAX;
    for _ in 0..trials {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let helper = UnionHelper { uf: &uf };
        let start = Instant::now();
        pool.install(|| collect_reduce_par_sort_with_scratch(pairs, &helper, uf_size, &mut scratch));
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

/// Baseline: sequential iteration applying unions in order (no sorting, 1 thread).
fn bench_seq_union(pairs: &[UnionPair], uf_size: usize) -> Duration {
    // Warmup
    for _ in 0..WARMUP_TRIALS {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        for p in pairs {
            uf.union(p.a, p.b);
        }
    }

    let mut best = Duration::MAX;
    for _ in 0..TRIALS {
        let uf = ConcurrentUnionFind::with_size(uf_size);
        let start = Instant::now();
        for p in pairs {
            uf.union(p.a, p.b);
        }
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

/// Experiment 4: semi_sort (collect_reduce) vs par_sort_unstable vs sequential.
///
/// Compares three strategies for applying union-find operations:
///   - semi_sort: custom counting sort with heavy-hitter detection
///   - par_sort:  rayon's par_sort_unstable_by_key
///   - seq:       plain sequential loop (1-thread baseline)
fn experiment_union_find() {
    println!();
    println!("=== Union-Find: semi_sort vs par_sort vs sequential ===");

    let max_t = available_threads();
    let threads = max_t;
    let sizes: Vec<(&str, usize, usize, Distribution)> = vec![
        ("uniform_1M", 1_000_000, 100_000, Distribution::Uniform),
        ("uniform_5M", 5_000_000, 500_000, Distribution::Uniform),
        ("uniform_10M", 10_000_000, 1_000_000, Distribution::Uniform),
        ("uniform_50M", 50_000_000, 5_000_000, Distribution::Uniform),
        ("uniform_100M", 100_000_000, 10_000_000, Distribution::Uniform),
        ("zipf_1M", 1_000_000, 100_000, Distribution::Zipfian(0.99)),
        ("zipf_5M", 5_000_000, 500_000, Distribution::Zipfian(0.99)),
        ("zipf_10M", 10_000_000, 1_000_000, Distribution::Zipfian(0.99)),
        ("zipf_50M", 50_000_000, 5_000_000, Distribution::Zipfian(0.99)),
        ("zipf_100M", 100_000_000, 10_000_000, Distribution::Zipfian(0.99)),
    ];

    let mut table = Table::new(&[
        "config", "pairs", "uf_size", "dist", "threads",
        "semi_ms", "radix_ms", "psort_ms",
        "semi_Mp/s", "radix_Mp/s", "psort_Mp/s",
        "radix_vs_semi",
    ]);

    for (name, n_pairs, uf_size, dist) in &sizes {
        let pairs = generate_union_pairs(*n_pairs, *uf_size, dist, 42);

        let semi_dur = bench_cr_union_find(&pairs, *uf_size, threads);
        let radix_dur = bench_cr_radix_union_find(&pairs, *uf_size, threads);
        let psort_dur = bench_cr_par_sort_union_find(&pairs, *uf_size, threads);

        let semi_tp = *n_pairs as f64 / semi_dur.as_secs_f64() / 1e6;
        let radix_tp = *n_pairs as f64 / radix_dur.as_secs_f64() / 1e6;
        let psort_tp = *n_pairs as f64 / psort_dur.as_secs_f64() / 1e6;
        let radix_vs_semi = semi_dur.as_secs_f64() / radix_dur.as_secs_f64();

        table.add_row(&[
            name.to_string(),
            format!("{}", n_pairs),
            format!("{}", uf_size),
            format!("{}", dist),
            format!("{}", threads),
            format!("{:.2}", semi_dur.as_secs_f64() * 1000.0),
            format!("{:.2}", radix_dur.as_secs_f64() * 1000.0),
            format!("{:.2}", psort_dur.as_secs_f64() * 1000.0),
            format!("{:.1}", semi_tp),
            format!("{:.1}", radix_tp),
            format!("{:.1}", psort_tp),
            format!("{:.2}x", radix_vs_semi),
        ]);
    }
    table.print();
}

/// Experiment 5: semi_sort size x threads matrix.
///
/// For each size, sweep thread counts. Skips low thread counts for large
/// sizes to keep runtime manageable. Reports time and per-size speedup.
fn experiment_size_x_threads() {
    println!();
    println!("=== semi_sort vs radix: size x threads (uniform, scratch reused, adaptive trials) ===");

    let max_t = available_threads();

    // Pairs of (n_pairs, uf_size). min_threads is computed dynamically so
    // that the slowest single-config runtime stays roughly constant across
    // sizes: we allow 1 thread for the smallest size and scale min_threads
    // linearly with n_pairs so that n_pairs/min_threads ≈ constant.
    let sizes: Vec<(usize, usize)> = vec![
        (1_000_000, 100_000),
        (5_000_000, 500_000),
        (10_000_000, 1_000_000),
    ];
    // Budget-based: estimate single-threaded wall time and only require more
    // threads when it would exceed 15 seconds. Assumes ~75 M pairs/sec at 1T.
    let runs = (TRIALS + WARMUP_TRIALS) as f64;
    let compute_min_threads = |n_pairs: usize| -> usize {
        let secs_at_1t = (n_pairs as f64 / 75e6) * runs;
        (secs_at_1t / 15.0).ceil().max(1.0) as usize
    };

    let mut table = Table::new(&[
        "pairs", "uf_size", "threads",
        "semi_ms", "radix_ms",
        "semi_Mp/s", "radix_Mp/s",
        "semi_spd", "radix_spd",
        "radix_vs_semi",
    ]);

    for &(n_pairs, uf_size) in &sizes {
        let min_t = compute_min_threads(n_pairs);
        let pairs = generate_union_pairs(n_pairs, uf_size, &Distribution::Uniform, 42);
        let thread_list: Vec<usize> = thread_counts(max_t)
            .into_iter()
            .filter(|&t| t >= min_t)
            .collect();
        let mut semi_baseline = Duration::MAX;
        let mut radix_baseline = Duration::MAX;

        for &threads in &thread_list {
            let semi_dur = bench_cr_union_find_hot(&pairs, uf_size, threads);
            let radix_dur = bench_cr_radix_union_find_hot(&pairs, uf_size, threads);
            if threads == *thread_list.first().unwrap() {
                semi_baseline = semi_dur;
                radix_baseline = radix_dur;
            }

            let semi_tp = n_pairs as f64 / semi_dur.as_secs_f64() / 1e6;
            let radix_tp = n_pairs as f64 / radix_dur.as_secs_f64() / 1e6;
            let semi_spd = semi_baseline.as_secs_f64() / semi_dur.as_secs_f64();
            let radix_spd = radix_baseline.as_secs_f64() / radix_dur.as_secs_f64();
            let radix_vs_semi = semi_dur.as_secs_f64() / radix_dur.as_secs_f64();

            table.add_row(&[
                format!("{}", n_pairs),
                format!("{}", uf_size),
                format!("{}", threads),
                format!("{:.2}", semi_dur.as_secs_f64() * 1000.0),
                format!("{:.2}", radix_dur.as_secs_f64() * 1000.0),
                format!("{:.1}", semi_tp),
                format!("{:.1}", radix_tp),
                format!("{:.2}x", semi_spd),
                format!("{:.2}x", radix_spd),
                format!("{:.2}x", radix_vs_semi),
            ]);
        }
    }
    table.print();
}

/// Benchmark a single (variant, thread-count, size) point with the
/// no-contention helper. `variant` picks the collect_reduce entry point.
fn bench_no_contention(
    data: &[KV],
    num_buckets: usize,
    threads: usize,
    variant: &str,
) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    let n = data.len();
    let run = |helper: &PerSlotHelper| match variant {
        "semi" => collect_reduce(data, helper, num_buckets),
        "radix" => collect_reduce_radix(data, helper, num_buckets),
        "psort" => collect_reduce_par_sort(data, helper, num_buckets),
        _ => unreachable!(),
    };

    // Warmup
    for _ in 0..WARMUP_TRIALS {
        let (helper, _buf) = PerSlotHelper::new(n);
        pool.install(|| run(&helper));
    }

    let mut best = Duration::MAX;
    for _ in 0..TRIALS {
        let (helper, _buf) = PerSlotHelper::new(n);
        let start = Instant::now();
        pool.install(|| run(&helper));
        let elapsed = start.elapsed();
        if elapsed < best {
            best = elapsed;
        }
    }
    best
}

/// Experiment: scaling of each sort variant with thread count, measured with
/// a no-contention helper (each element writes to a distinct output slot).
/// Isolates the sort-primitive's own parallel scalability from UF CAS cost.
fn experiment_no_contention_scaling() {
    println!();
    println!("=== No-Contention Scaling (per-slot writes, unique idx) ===");

    let max_t = available_threads();

    // (n, num_buckets). Keep num_buckets moderate so semi_sort takes the
    // main path (not collect_reduce_few) and radix has work to do.
    let sizes: Vec<(usize, usize)> = vec![
        (1_000_000, 1_000_000),
        (10_000_000, 1_000_000),
        (100_000_000, 10_000_000),
    ];

    for &(n, num_buckets) in &sizes {
        println!();
        println!("-- n = {}, num_buckets = {}, dist = uniform --", n, num_buckets);
        let data = generate_data_unique_val(n, num_buckets, &Distribution::Uniform, 42);

        let mut table = Table::new(&[
            "threads",
            "semi_ms", "semi_Mp/s", "semi_speedup",
            "radix_ms", "radix_Mp/s", "radix_speedup",
            "psort_ms", "psort_Mp/s", "psort_speedup",
        ]);

        let mut semi_base = Duration::MAX;
        let mut radix_base = Duration::MAX;
        let mut psort_base = Duration::MAX;

        for &threads in &thread_counts(max_t) {
            let semi = bench_no_contention(&data, num_buckets, threads, "semi");
            let radix = bench_no_contention(&data, num_buckets, threads, "radix");
            let psort = bench_no_contention(&data, num_buckets, threads, "psort");

            if threads == 1 {
                semi_base = semi;
                radix_base = radix;
                psort_base = psort;
            }

            let mp = |d: Duration| n as f64 / d.as_secs_f64() / 1e6;
            let sp = |base: Duration, d: Duration| base.as_secs_f64() / d.as_secs_f64();

            table.add_row(&[
                format!("{}", threads),
                format!("{:.2}", semi.as_secs_f64() * 1000.0),
                format!("{:.1}", mp(semi)),
                format!("{:.2}x", sp(semi_base, semi)),
                format!("{:.2}", radix.as_secs_f64() * 1000.0),
                format!("{:.1}", mp(radix)),
                format!("{:.2}x", sp(radix_base, radix)),
                format!("{:.2}", psort.as_secs_f64() * 1000.0),
                format!("{:.1}", mp(psort)),
                format!("{:.2}x", sp(psort_base, psort)),
            ]);
        }
        table.print();
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let experiment = args.get(1).map(|s| s.as_str());

    println!(
        "collect_reduce scalability benchmark  |  max_threads={}  trials={}",
        available_threads(),
        TRIALS,
    );

    match experiment {
        Some("thread-scaling") | Some("thread_scaling") => experiment_thread_scaling(),
        Some("size-scaling") | Some("size_scaling") => experiment_size_scaling(),
        Some("distribution") => experiment_distribution(),
        Some("union-find") | Some("union_find") => experiment_union_find(),
        Some("size-x-threads") | Some("size_x_threads") => experiment_size_x_threads(),
        Some("no-contention") | Some("no_contention") => experiment_no_contention_scaling(),
        _ => {
            experiment_thread_scaling();
            experiment_size_scaling();
            experiment_distribution();
            experiment_union_find();
            experiment_size_x_threads();
        }
    }
    println!();
}
