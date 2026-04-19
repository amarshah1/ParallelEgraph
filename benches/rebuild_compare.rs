//! Compare parallel vs sequential congruence closure on synthetic workloads.
//!
//! Workload: N "x" leaves, N "y" leaves, plus F function nodes of arity 2
//! over random (x_i, y_j) pairs across several op names. Then assert K random
//! equalities among the x's (and y's). Closure propagates congruence across
//! all function nodes sharing now-equivalent children.
//!
//! For each trial the e-graph is built from scratch (outside the timed region)
//! and only the close() call is timed.

use std::time::Instant;

use parallel_egraph::egraph::{EGraph, ENode};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::ThreadPool;

const TRIALS: usize = 11;
const WARMUP: usize = 3;

#[derive(Clone, Copy)]
struct Workload {
    name: &'static str,
    n_leaves: usize,
    n_fns: usize,
    n_nodes: usize,
    n_merges: usize,
    /// DAG depth. 1 = flat (all functions take leaves as children). >1 splits
    /// n_nodes across `depth` levels, with each level's functions consuming
    /// the previous level's ids. Deeper DAGs produce multi-round closure.
    depth: usize,
}

const WORKLOADS: &[Workload] = &[
    Workload { name: "small",   n_leaves: 1_000,   n_fns: 4,  n_nodes: 10_000,    n_merges: 2_000,   depth: 1 },
    Workload { name: "medium",  n_leaves: 10_000,  n_fns: 8,  n_nodes: 200_000,   n_merges: 20_000,  depth: 1 },
    Workload { name: "large",   n_leaves: 50_000,  n_fns: 16, n_nodes: 1_000_000, n_merges: 100_000, depth: 1 },
    Workload { name: "deep-s",  n_leaves: 1_000,   n_fns: 4,  n_nodes: 10_000,    n_merges: 2_000,   depth: 3 },
    Workload { name: "deep-m",  n_leaves: 10_000,  n_fns: 8,  n_nodes: 200_000,   n_merges: 20_000,  depth: 3 },
    Workload { name: "deep-l",  n_leaves: 50_000,  n_fns: 16, n_nodes: 1_000_000, n_merges: 100_000, depth: 3 },
];

fn build(w: Workload) -> (EGraph, Vec<(u32, u32)>) {
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE ^ w.n_nodes as u64);
    let mut eg = EGraph::new_parallel();

    let x_ids: Vec<u32> = (0..w.n_leaves)
        .map(|i| eg.add(ENode::leaf(format!("x{i}"))))
        .collect();
    let y_ids: Vec<u32> = (0..w.n_leaves)
        .map(|i| eg.add(ENode::leaf(format!("y{i}"))))
        .collect();

    // Build levels. Level 0 = leaves (x_ids ++ y_ids). Levels 1..depth each
    // contain roughly n_nodes/depth function nodes whose children are drawn
    // uniformly from the previous level.
    let depth = w.depth.max(1);
    let per_level = w.n_nodes / depth;
    let mut prev_level: Vec<u32> = x_ids.iter().chain(y_ids.iter()).copied().collect();
    for lvl in 0..depth {
        let count = if lvl + 1 == depth {
            w.n_nodes - per_level * (depth - 1)
        } else {
            per_level
        };
        let mut this_level: Vec<u32> = Vec::with_capacity(count);
        for _ in 0..count {
            let k = rng.gen_range(0..w.n_fns);
            let a = prev_level[rng.gen_range(0..prev_level.len())];
            let b = prev_level[rng.gen_range(0..prev_level.len())];
            let id = eg.add(ENode::new(format!("f{lvl}_{k}"), vec![a, b]));
            this_level.push(id);
        }
        prev_level = this_level;
    }

    let mut equalities: Vec<(u32, u32)> = Vec::with_capacity(w.n_merges);
    for _ in 0..(w.n_merges / 2) {
        let i = rng.gen_range(0..w.n_leaves);
        let j = rng.gen_range(0..w.n_leaves);
        equalities.push((x_ids[i], x_ids[j]));
    }
    for _ in 0..(w.n_merges - w.n_merges / 2) {
        let i = rng.gen_range(0..w.n_leaves);
        let j = rng.gen_range(0..w.n_leaves);
        equalities.push((y_ids[i], y_ids[j]));
    }

    (eg, equalities)
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench_nelson(w: Workload) -> Vec<f64> {
    for _ in 0..WARMUP {
        let (mut eg, eqs) = build(w);
        eg.sequential_close_nelson(&eqs);
    }
    let mut times = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let (mut eg, eqs) = build(w);
        let t = Instant::now();
        eg.sequential_close_nelson(&eqs);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times
}

fn bench_parallel<F>(pool: &ThreadPool, w: Workload, close: F) -> Vec<f64>
where
    F: Fn(&mut EGraph, &[(u32, u32)]) + Sync,
{
    for _ in 0..WARMUP {
        let (mut eg, eqs) = pool.install(|| build(w));
        pool.install(|| close(&mut eg, &eqs));
    }
    let mut times = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let (mut eg, eqs) = pool.install(|| build(w));
        let t = Instant::now();
        pool.install(|| close(&mut eg, &eqs));
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times
}

fn main() {
    let max_threads = rayon::current_num_threads();
    let pool_par = rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build()
        .unwrap();

    println!(
        "close_compare  trials={}  warmup={}  par_threads={}",
        TRIALS, WARMUP, max_threads
    );
    println!(
        "{:<8} {:>8} {:>10} {:>9} | {:>11} | {:>11} {:>11}",
        "name",
        "leaves",
        "nodes",
        "merges",
        "nelson_seq",
        "par_close",
        "par_spd",
    );

    for &w in WORKLOADS {
        let nel = bench_nelson(w);
        let par = bench_parallel(&pool_par, w, |eg, u| eg.parallel_close(u));

        let mn = median(&nel);
        let mp = median(&par);
        println!(
            "{:<8} {:>8} {:>10} {:>9} | {:>9.2}ms | {:>9.2}ms {:>9.2}x",
            w.name,
            w.n_leaves,
            w.n_nodes,
            w.n_merges,
            mn,
            mp,
            mn / mp,
        );
    }
}
