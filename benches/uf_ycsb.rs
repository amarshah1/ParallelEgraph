//! YCSB-style microbenchmark for ConcurrentUnionFind.
//!
//! Measures throughput (Mops/sec) while varying the 5 YCSB-inspired parameters:
//!   1. Thread count      — scalability under concurrency
//!   2. Update rate        — find-only (0%) to union-only (100%)
//!   3. Key distribution   — uniform vs zipfian
//!   4. Zipfian skew (z)  — contention: 0.01 ~ uniform, 0.99 = extreme hotspot
//!   5. Data structure size — 1 K to 1 M elements
//!
//! Workload presets (adapted from YCSB A-E):
//!   A  Update-heavy   50% union / 50% find,  zipfian z=0.99
//!   B  Read-mostly     5% union / 95% find,  zipfian z=0.99
//!   C  Read-only       0% union / 100% find, zipfian z=0.99
//!   D  Read-latest     5% union / 95% find,  uniform
//!   E  Write-only    100% union / 0% find,   zipfian z=0.99
//!
//! Usage:
//!   cargo bench --bench uf_ycsb                     # all experiments
//!   cargo bench --bench uf_ycsb -- scalability      # just thread scaling
//!   cargo bench --bench uf_ycsb -- contention       # just zipfian sweep
//!   cargo bench --bench uf_ycsb -- sizing           # just size sweep
//!   cargo bench --bench uf_ycsb -- update-mix       # just update rate sweep

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use parallel_egraph::unionfind::ConcurrentUnionFind;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const DEFAULT_SIZE: u32 = 100_000;
const OPS_PER_THREAD: u32 = 500_000;
const TRIALS: usize = 3;
const MAX_THREADS: usize = 8;
/// Fraction of elements pre-merged before the timed phase so trees are
/// non-trivial (find has real work to do, roots are shared).
const WARMUP_MERGE_FRAC: f64 = 0.3;

// ---------------------------------------------------------------------------
// Zipfian distribution (YCSB-style, Gray et al. 1994)
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
        assert!(
            z > 0.0 && z < 1.0,
            "z must be in (0,1); use uniform for no skew"
        );
        let zetan = Self::zeta(n as u64, z);
        let zeta2 = 1.0 + (0.5_f64).powf(z);
        let alpha = 1.0 / (1.0 - z);
        let half_pow_z = (0.5_f64).powf(z);
        let eta = (1.0 - (2.0 / n as f64).powf(1.0 - z)) / (1.0 - zeta2 / zetan);
        Zipfian {
            n,
            alpha,
            eta,
            zetan,
            half_pow_z,
        }
    }

    fn zeta(n: u64, z: f64) -> f64 {
        (1..=n).map(|i| 1.0 / (i as f64).powf(z)).sum()
    }

    fn next(&self, rng: &mut impl Rng) -> u32 {
        let u: f64 = rng.gen();
        let uz = u * self.zetan;
        if uz < 1.0 {
            return 0;
        }
        if uz < 1.0 + self.half_pow_z {
            return 1;
        }
        let v = (self.n as f64 * (self.eta * u - self.eta + 1.0).powf(self.alpha)) as u32;
        v.min(self.n - 1)
    }
}

// ---------------------------------------------------------------------------
// Key sampler (wraps uniform / zipfian)
// ---------------------------------------------------------------------------

enum Sampler {
    Uniform(u32),
    Zipf(Zipfian),
}

impl Sampler {
    fn sample(&self, rng: &mut impl Rng) -> u32 {
        match self {
            Sampler::Uniform(n) => rng.gen_range(0..*n),
            Sampler::Zipf(z) => z.next(rng),
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark config & runner
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Config {
    size: u32,
    ops_per_thread: u32,
    update_rate: f64,     // 0.0 = all finds, 1.0 = all unions
    z: Option<f64>,       // None = uniform distribution
    threads: usize,
}

/// Create a UF of the given size and pre-merge WARMUP_MERGE_FRAC of elements
/// so trees are non-trivial before the timed phase.
fn create_uf(size: u32) -> ConcurrentUnionFind {
    let mut uf = ConcurrentUnionFind::new();
    for _ in 0..size {
        uf.make_set();
    }
    let mut rng = SmallRng::seed_from_u64(42);
    let warmup = (size as f64 * WARMUP_MERGE_FRAC) as u32;
    for _ in 0..warmup {
        let a = rng.gen_range(0..size);
        let b = rng.gen_range(0..size);
        uf.union(a, b);
    }
    uf
}

/// Run one trial: all threads start together via barrier, each does
/// ops_per_thread operations, returns wall-clock duration.
fn run_trial(uf: &ConcurrentUnionFind, cfg: &Config) -> Duration {
    let barrier = Arc::new(Barrier::new(cfg.threads + 1));
    let mut wall_start = Instant::now();

    std::thread::scope(|s| {
        for tid in 0..cfg.threads {
            let b = Arc::clone(&barrier);
            let ops = cfg.ops_per_thread;
            let size = cfg.size;
            let ur = cfg.update_rate;
            let z = cfg.z;
            s.spawn(move || {
                let mut rng = SmallRng::seed_from_u64(tid as u64 * 0xCAFE + 0xBABE);
                let sampler = match z {
                    Some(z) => Sampler::Zipf(Zipfian::new(size, z)),
                    None => Sampler::Uniform(size),
                };

                b.wait(); // synchronized start

                for _ in 0..ops {
                    let key = sampler.sample(&mut rng);
                    if ur >= 1.0 || (ur > 0.0 && rng.gen_bool(ur)) {
                        let key2 = sampler.sample(&mut rng);
                        uf.union(key, key2);
                    } else {
                        std::hint::black_box(uf.find(key));
                    }
                }
            });
        }
        barrier.wait();
        wall_start = Instant::now();
    });

    wall_start.elapsed()
}

/// Run multiple trials, return sorted throughputs (Mops/sec).
fn bench(cfg: &Config) -> Vec<f64> {
    let mut throughputs = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let uf = create_uf(cfg.size);
        let dur = run_trial(&uf, cfg);
        let total_ops = cfg.ops_per_thread as u64 * cfg.threads as u64;
        throughputs.push(total_ops as f64 / dur.as_secs_f64() / 1e6);
    }
    throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    throughputs
}

fn median(v: &[f64]) -> f64 {
    v[v.len() / 2]
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
        // header
        let header_line: String = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:>w$}", h, w = self.widths[i]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {}", header_line);

        // separator
        let sep: String = self
            .widths
            .iter()
            .map(|&w| "-".repeat(w))
            .collect::<Vec<_>>()
            .join("  ");
        println!("  {}", sep);

        // rows
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

fn thread_counts() -> Vec<usize> {
    let mut v = vec![1];
    let mut t = 2;
    while t <= MAX_THREADS {
        v.push(t);
        t *= 2;
    }
    if *v.last().unwrap() != MAX_THREADS && MAX_THREADS > 1 {
        v.push(MAX_THREADS);
    }
    v
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

/// Experiment 1: Thread scalability across YCSB workloads A-E.
fn experiment_scalability() {
    println!();
    println!("=== Scalability (vary threads, per workload) ===");

    let workloads: Vec<(&str, f64, Option<f64>)> = vec![
        ("A_update_heavy", 0.50, Some(0.99)),
        ("B_read_mostly", 0.05, Some(0.99)),
        ("C_read_only", 0.00, Some(0.99)),
        ("D_uniform_read", 0.05, None),
        ("E_write_only", 1.00, Some(0.99)),
    ];

    let mut table = Table::new(&[
        "workload", "size", "threads", "update%", "dist", "z",
        "med Mops", "min Mops", "max Mops",
    ]);

    for (name, ur, z) in &workloads {
        for &threads in &thread_counts() {
            let cfg = Config {
                size: DEFAULT_SIZE,
                ops_per_thread: OPS_PER_THREAD,
                update_rate: *ur,
                z: *z,
                threads,
            };
            let tp = bench(&cfg);
            let dist_str = if z.is_some() { "zipfian" } else { "uniform" };
            let z_str = z.map_or("-".into(), |v| format!("{:.2}", v));
            table.add_row(&[
                name.to_string(),
                format!("{}", cfg.size),
                format!("{}", threads),
                format!("{:.0}", ur * 100.0),
                dist_str.into(),
                z_str,
                format!("{:.1}", median(&tp)),
                format!("{:.1}", tp.first().unwrap()),
                format!("{:.1}", tp.last().unwrap()),
            ]);
        }
    }
    table.print();
}

/// Experiment 2: Contention sweep — vary zipfian z (skew).
/// Higher z = more contention on hot keys.
fn experiment_contention() {
    println!();
    println!("=== Contention (vary z, workload A, all thread counts) ===");

    let zs = [0.01, 0.25, 0.50, 0.75, 0.99];

    let mut table = Table::new(&[
        "z", "threads", "update%", "size",
        "med Mops", "min Mops", "max Mops",
    ]);

    for &z in &zs {
        for &threads in &thread_counts() {
            let cfg = Config {
                size: DEFAULT_SIZE,
                ops_per_thread: OPS_PER_THREAD,
                update_rate: 0.50,
                z: Some(z),
                threads,
            };
            let tp = bench(&cfg);
            table.add_row(&[
                format!("{:.2}", z),
                format!("{}", threads),
                "50".into(),
                format!("{}", cfg.size),
                format!("{:.1}", median(&tp)),
                format!("{:.1}", tp.first().unwrap()),
                format!("{:.1}", tp.last().unwrap()),
            ]);
        }
    }
    table.print();
}

/// Experiment 3: Data structure size sweep.
fn experiment_sizing() {
    println!();
    println!("=== Sizing (vary n, workload A, all thread counts) ===");

    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    let mut table = Table::new(&[
        "size", "threads", "update%", "z",
        "med Mops", "min Mops", "max Mops",
    ]);

    for &size in &sizes {
        for &threads in &thread_counts() {
            let cfg = Config {
                size,
                ops_per_thread: OPS_PER_THREAD,
                update_rate: 0.50,
                z: Some(0.99),
                threads,
            };
            let tp = bench(&cfg);
            table.add_row(&[
                format!("{}", size),
                format!("{}", threads),
                "50".into(),
                "0.99".into(),
                format!("{:.1}", median(&tp)),
                format!("{:.1}", tp.first().unwrap()),
                format!("{:.1}", tp.last().unwrap()),
            ]);
        }
    }
    table.print();
}

/// Experiment 4: Update-rate sweep (read/write mix).
fn experiment_update_mix() {
    println!();
    println!("=== Update Mix (vary update%, workload A, all thread counts) ===");

    let rates = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00];

    let mut table = Table::new(&[
        "update%", "threads", "size", "z",
        "med Mops", "min Mops", "max Mops",
    ]);

    for &ur in &rates {
        for &threads in &thread_counts() {
            let cfg = Config {
                size: DEFAULT_SIZE,
                ops_per_thread: OPS_PER_THREAD,
                update_rate: ur,
                z: Some(0.99),
                threads,
            };
            let tp = bench(&cfg);
            table.add_row(&[
                format!("{:.0}", ur * 100.0),
                format!("{}", threads),
                format!("{}", cfg.size),
                "0.99".into(),
                format!("{:.1}", median(&tp)),
                format!("{:.1}", tp.first().unwrap()),
                format!("{:.1}", tp.last().unwrap()),
            ]);
        }
    }
    table.print();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let experiment = args.get(1).map(|s| s.as_str());

    println!(
        "YCSB-UF benchmark  |  max_threads={}  ops/thread={}  trials={}  default_size={}",
        MAX_THREADS,
        OPS_PER_THREAD,
        TRIALS,
        DEFAULT_SIZE,
    );

    match experiment {
        Some("scalability") => experiment_scalability(),
        Some("contention") => experiment_contention(),
        Some("sizing") => experiment_sizing(),
        Some("update-mix") | Some("update_mix") => experiment_update_mix(),
        _ => {
            experiment_scalability();
            experiment_contention();
            experiment_sizing();
            experiment_update_mix();
        }
    }
    println!();
}
