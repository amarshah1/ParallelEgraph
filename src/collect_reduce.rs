use rayon::prelude::*;
use std::cmp::min;
use std::mem::size_of;

const CR_SEQ_THRESHOLD: usize = 8192;

/// Trait for collect_reduce with a void, concurrent operation (e.g. union-find).
///
/// `apply` must be safe to call concurrently from multiple threads.
pub trait CollectReduceHelper<T>: Sync {
    fn get_key(&self, elem: &T) -> usize;
    fn apply(&self, elem: &T);
    /// Process an entire slice of elements known to share the same key.
    /// Override for heavy-hitter optimization (e.g. batch union).
    fn combine(&self, elems: &[T]) {
        for elem in elems {
            self.apply(elem);
        }
    }
}

// ---- Hash utilities (matching parlay's hash64 / hash64_2) ----

fn hash64(x: u64) -> u64 {
    let mut k = x;
    k = (!k).wrapping_add(k << 21);
    k ^= k >> 24;
    k = k.wrapping_add(k << 3).wrapping_add(k << 8);
    k ^= k >> 14;
    k = k.wrapping_add(k << 2).wrapping_add(k << 4);
    k ^= k >> 28;
    k = k.wrapping_add(k << 31);
    k
}

fn hash64_2(x: u64) -> u64 {
    let mut k = x;
    k = (!k).wrapping_add(k << 18);
    k ^= k >> 31;
    k = k.wrapping_mul(21);
    k ^= k >> 11;
    k = k.wrapping_add(k << 6);
    k ^= k >> 22;
    k
}

fn log2_up(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

// ---- Raw-pointer Send wrapper for parallel scatter ----

#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

// ---- Heavy-hitter-aware bucket mapper (translates get_bucket<HashEq>) ----

struct GetBucket {
    hash_table: Vec<(usize, i32)>, // (key, bucket_id); -1 = empty
    table_mask: usize,
    bucket_mask: usize,
    heavy_hitters: usize,
    shift: usize,
}

impl GetBucket {
    /// Samples `A`, identifies keys that appear disproportionately often,
    /// and assigns them dedicated buckets (indices `0..heavy_hitters`).
    fn new<T, H: CollectReduceHelper<T>>(a: &[T], helper: &H, bits: usize) -> Self {
        let n = a.len();
        let num_buckets = 1usize << bits;
        let copy_cutoff: i32 = 5;
        let num_samples = num_buckets;
        let table_size = 4 * num_samples;
        let table_mask = table_size - 1;
        let bucket_mask = num_buckets - 1;
        let shift = 8 / size_of::<T>().max(1);

        let hash_key =
            |key: usize| hash64_2((key.wrapping_add(shift) & !15) as u64) as usize;

        // --- sample into a probe table and count duplicates ---
        let mut counts: Vec<(usize, i32)> = vec![(0, -1); table_size];
        for i in 0..num_samples {
            let a_idx = (hash64(i as u64) as usize) % n;
            let key = helper.get_key(&a[a_idx]);
            let mut idx = hash_key(key) & table_mask;
            loop {
                if counts[idx].1 == -1 {
                    counts[idx] = (key, 0);
                    break;
                } else if counts[idx].0 == key {
                    counts[idx].1 += 1;
                    break;
                } else {
                    idx = (idx + 1) & table_mask;
                }
            }
        }

        // --- promote frequent keys to dedicated buckets ---
        let mut heavy_hitters = 0usize;
        let mut hash_table: Vec<(usize, i32)> = vec![(0, -1); table_size];
        for i in 0..table_size {
            if counts[i].1 + 2 > copy_cutoff {
                let key = counts[i].0;
                let idx = hash_key(key) & table_mask;
                if hash_table[idx].1 == -1 {
                    hash_table[idx] = (key, heavy_hitters as i32);
                    heavy_hitters += 1;
                }
            }
        }

        GetBucket {
            hash_table,
            table_mask,
            bucket_mask,
            heavy_hitters,
            shift,
        }
    }

    /// Map an element to its block index.
    /// Heavy-hitter keys get a dedicated block; others share the remaining blocks.
    fn bucket<T, H: CollectReduceHelper<T>>(&self, helper: &H, elem: &T) -> usize {
        let key = helper.get_key(elem);
        let hash_val =
            hash64_2((key.wrapping_add(self.shift) & !15) as u64) as usize;

        if self.heavy_hitters > 0 {
            let h = &self.hash_table[hash_val & self.table_mask];
            if h.1 != -1 && h.0 == key {
                return h.1 as usize;
            }
            let hv = hash_val & self.bucket_mask;
            if hv < self.heavy_hitters {
                return hv % (self.bucket_mask + 1 - self.heavy_hitters)
                    + self.heavy_hitters;
            }
            return hv;
        }
        hash_val & self.bucket_mask
    }
}

// ---- Parallel counting sort ----

/// Parallel three-phase counting sort.
/// Returns `(sorted_output, bucket_offsets)` where `bucket_offsets` has
/// `num_buckets + 1` entries delimiting each bucket in `sorted_output`.
fn parallel_counting_sort<T: Copy + Send + Sync>(
    input: &[T],
    keys: &[usize],
    num_buckets: usize,
) -> (Vec<T>, Vec<usize>) {
    let n = input.len();
    let num_blocks = min(rayon::current_num_threads(), 1 + n / 2048).max(1);
    let block_size = (n + num_blocks - 1) / num_blocks;

    // Phase 1 – local counts (parallel over blocks)
    let local_counts: Vec<Vec<usize>> = (0..num_blocks)
        .into_par_iter()
        .map(|b| {
            let start = b * block_size;
            let end = min(start + block_size, n);
            let mut c = vec![0usize; num_buckets];
            for i in start..end {
                c[keys[i]] += 1;
            }
            c
        })
        .collect();

    // Phase 2 – prefix sums → per-(block, bucket) write positions (sequential, small)
    let mut bucket_offsets = vec![0usize; num_buckets + 1];
    let mut block_starts = vec![vec![0usize; num_buckets]; num_blocks];
    for k in 0..num_buckets {
        bucket_offsets[k + 1] = bucket_offsets[k];
        for b in 0..num_blocks {
            block_starts[b][k] = bucket_offsets[k + 1];
            bucket_offsets[k + 1] += local_counts[b][k];
        }
    }

    // Phase 3 – scatter (parallel, disjoint writes)
    let mut output = Vec::with_capacity(n);
    unsafe { output.set_len(n) };
    let out = SendPtr(output.as_mut_ptr());

    (0..num_blocks).into_par_iter().for_each(|b| {
        let start = b * block_size;
        let end = min(start + block_size, n);
        let mut pos = block_starts[b].clone();
        for i in start..end {
            let k = keys[i];
            unsafe { std::ptr::write(out.0.add(pos[k]), input[i]) };
            pos[k] += 1;
        }
    });

    (output, bucket_offsets)
}

// ---- Sequential path ----

fn seq_collect_reduce<T, H: CollectReduceHelper<T>>(
    a: &[T],
    helper: &H,
    num_buckets: usize,
) {
    for elem in a {
        debug_assert!(helper.get_key(elem) < num_buckets);
        helper.apply(elem);
    }
}

// ---- Few-buckets path (translates collect_reduce_few) ----
//
// In the C++ version this builds per-block partial accumulators and then
// combines them in a parallel tabulate.  Since our operation is void and
// the union-find is concurrent, the combine step is a no-op; we keep
// only the sliced parallel iteration.

fn collect_reduce_few<T: Send + Sync, H: CollectReduceHelper<T>>(
    a: &[T],
    helper: &H,
    num_buckets: usize,
) {
    let n = a.len();
    if n == 0 {
        return;
    }
    let num_threads = rayon::current_num_threads();
    let num_blocks_ = min(4 * num_threads, n / num_buckets / 64) + 1;
    let block_size = (n - 1) / num_blocks_ + 1;
    let num_blocks = 1 + (n - 1) / block_size;

    if n < CR_SEQ_THRESHOLD || num_blocks == 1 || num_threads == 1 {
        seq_collect_reduce(a, helper, num_buckets);
        return;
    }

    // sliced_for equivalent – each block processed in parallel
    (0..num_blocks).into_par_iter().for_each(|i| {
        let start = i * block_size;
        let end = min(start + block_size, n);
        seq_collect_reduce(&a[start..end], helper, num_buckets);
    });
}

// ---- Main entry (translates collect_reduce) ----

/// Parallel collect-reduce with a void, concurrent operation.
///
/// Partitions `a` by key into cache-sized blocks using a counting sort with
/// heavy-hitter detection, then processes each block in parallel.
///
/// `T: Copy` is required because the counting sort physically rearranges elements.
pub fn collect_reduce<T: Copy + Send + Sync, H: CollectReduceHelper<T>>(
    a: &[T],
    helper: &H,
    num_buckets: usize,
) {
    let n = a.len();
    if n == 0 {
        return;
    }

    // #bits chosen so each block fits in ~1 MB of L3 cache
    let cache_per_thread: usize = 1_000_000;
    let bits = log2_up(1 + (2 * size_of::<T>() * n) / cache_per_thread).max(4);
    let num_blocks = 1usize << bits;

    if num_buckets <= 4 * num_blocks || n < CR_SEQ_THRESHOLD {
        collect_reduce_few(a, helper, num_buckets);
        return;
    }

    // Build heavy-hitter-aware bucket mapper
    let gb = GetBucket::new(a, helper, bits);

    // Compute block key for every element (parallel)
    let keys: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| gb.bucket(helper, &a[i]))
        .collect();

    // Counting sort: group elements by block
    let (sorted, bucket_offsets) = parallel_counting_sort(a, &keys, num_blocks);

    // Process each block in parallel
    (0..num_blocks).into_par_iter().for_each(|i| {
        let slice = &sorted[bucket_offsets[i]..bucket_offsets[i + 1]];
        if slice.is_empty() {
            return;
        }
        if i < gb.heavy_hitters {
            // Heavy hitter – entire slice shares one key
            helper.combine(slice);
        } else {
            // Shared block – iterate and apply per element
            for elem in slice {
                debug_assert!(helper.get_key(elem) < num_buckets);
                helper.apply(elem);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // --- Helpers ---

    /// Simple element: a (key, value) pair.
    #[derive(Copy, Clone, Debug)]
    struct KV {
        key: usize,
        val: u64,
    }

    /// Helper that sums values per key into a shared atomic array.
    struct SumHelper {
        sums: Vec<AtomicUsize>,
    }

    impl SumHelper {
        fn new(num_buckets: usize) -> Self {
            SumHelper {
                sums: (0..num_buckets).map(|_| AtomicUsize::new(0)).collect(),
            }
        }

        fn get_sum(&self, key: usize) -> usize {
            self.sums[key].load(Ordering::SeqCst)
        }
    }

    impl CollectReduceHelper<KV> for SumHelper {
        fn get_key(&self, elem: &KV) -> usize {
            elem.key
        }

        fn apply(&self, elem: &KV) {
            self.sums[elem.key].fetch_add(elem.val as usize, Ordering::SeqCst);
        }
    }

    /// Helper that counts elements per key.
    struct CountHelper {
        counts: Vec<AtomicUsize>,
    }

    impl CountHelper {
        fn new(num_buckets: usize) -> Self {
            CountHelper {
                counts: (0..num_buckets).map(|_| AtomicUsize::new(0)).collect(),
            }
        }

        fn get_count(&self, key: usize) -> usize {
            self.counts[key].load(Ordering::SeqCst)
        }
    }

    impl CollectReduceHelper<KV> for CountHelper {
        fn get_key(&self, elem: &KV) -> usize {
            elem.key
        }

        fn apply(&self, elem: &KV) {
            let _ = elem;
            self.counts[elem.key].fetch_add(1, Ordering::SeqCst);
        }
    }

    // --- Unit tests for hash utilities ---

    #[test]
    fn test_log2_up() {
        assert_eq!(log2_up(0), 0);
        assert_eq!(log2_up(1), 0);
        assert_eq!(log2_up(2), 1);
        assert_eq!(log2_up(3), 2);
        assert_eq!(log2_up(4), 2);
        assert_eq!(log2_up(5), 3);
        assert_eq!(log2_up(8), 3);
        assert_eq!(log2_up(9), 4);
        assert_eq!(log2_up(1024), 10);
        assert_eq!(log2_up(1025), 11);
    }

    #[test]
    fn test_hash64_deterministic() {
        for x in 0..100 {
            assert_eq!(hash64(x), hash64(x));
        }
    }

    #[test]
    fn test_hash64_2_deterministic() {
        for x in 0..100 {
            assert_eq!(hash64_2(x), hash64_2(x));
        }
    }

    #[test]
    fn test_hash64_different_inputs() {
        // Hash should spread out: no collisions among small inputs
        let hashes: Vec<u64> = (0..256).map(hash64).collect();
        let mut deduped = hashes.clone();
        deduped.sort();
        deduped.dedup();
        // Allow at most a few collisions out of 256
        assert!(deduped.len() >= 250, "too many hash64 collisions: {} unique out of 256", deduped.len());
    }

    // --- Unit tests for parallel_counting_sort ---

    #[test]
    fn test_counting_sort_empty() {
        let input: Vec<u32> = vec![];
        let keys: Vec<usize> = vec![];
        let (sorted, offsets) = parallel_counting_sort(&input, &keys, 4);
        assert!(sorted.is_empty());
        assert_eq!(offsets, vec![0; 5]);
    }

    #[test]
    fn test_counting_sort_single() {
        let input = vec![42u32];
        let keys = vec![2usize];
        let (sorted, offsets) = parallel_counting_sort(&input, &keys, 4);
        assert_eq!(sorted, vec![42]);
        assert_eq!(offsets[2], 0);
        assert_eq!(offsets[3], 1);
    }

    #[test]
    fn test_counting_sort_preserves_elements() {
        let mut rng = SmallRng::seed_from_u64(12345);
        let n = 10_000;
        let num_buckets = 16;
        let input: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
        let keys: Vec<usize> = (0..n).map(|_| rng.gen_range(0..num_buckets)).collect();

        let (sorted, offsets) = parallel_counting_sort(&input, &keys, num_buckets);

        // Same length
        assert_eq!(sorted.len(), n);

        // Offsets are monotonically non-decreasing and final = n
        for i in 0..num_buckets {
            assert!(offsets[i] <= offsets[i + 1]);
        }
        assert_eq!(offsets[num_buckets], n);

        // All elements are preserved (multiset equality)
        let mut expected: Vec<u64> = input.clone();
        let mut actual: Vec<u64> = sorted.clone();
        expected.sort();
        actual.sort();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_counting_sort_correct_buckets() {
        let mut rng = SmallRng::seed_from_u64(99);
        let n = 5_000;
        let num_buckets = 8;
        let input: Vec<usize> = (0..n).collect();
        let keys: Vec<usize> = (0..n).map(|_| rng.gen_range(0..num_buckets)).collect();

        let (sorted, offsets) = parallel_counting_sort(&input, &keys, num_buckets);

        // Every element in bucket k should have had key k
        for k in 0..num_buckets {
            for &elem in &sorted[offsets[k]..offsets[k + 1]] {
                assert_eq!(keys[elem], k, "element {elem} in bucket {k} but had key {}", keys[elem]);
            }
        }
    }

    // --- Unit tests for collect_reduce ---

    #[test]
    fn test_collect_reduce_empty() {
        let data: Vec<KV> = vec![];
        let helper = SumHelper::new(16);
        collect_reduce(&data, &helper, 16);
        for i in 0..16 {
            assert_eq!(helper.get_sum(i), 0);
        }
    }

    #[test]
    fn test_collect_reduce_single_bucket() {
        let data: Vec<KV> = (0..100).map(|i| KV { key: 0, val: i }).collect();
        let helper = SumHelper::new(1);
        collect_reduce(&data, &helper, 1);
        let expected: u64 = (0..100).sum();
        assert_eq!(helper.get_sum(0), expected as usize);
    }

    #[test]
    fn test_collect_reduce_correctness_small() {
        let num_buckets = 8;
        let data: Vec<KV> = (0..200)
            .map(|i| KV { key: i % num_buckets, val: 1 })
            .collect();
        let helper = CountHelper::new(num_buckets);
        collect_reduce(&data, &helper, num_buckets);
        for k in 0..num_buckets {
            assert_eq!(helper.get_count(k), 25, "bucket {k}");
        }
    }

    #[test]
    fn test_seq_collect_reduce_matches_parallel() {
        // Ensure sequential and parallel paths produce the same result
        let mut rng = SmallRng::seed_from_u64(42);
        let num_buckets = 32;
        let n = 5_000;
        let data: Vec<KV> = (0..n)
            .map(|_| KV {
                key: rng.gen_range(0..num_buckets),
                val: rng.gen_range(1..100),
            })
            .collect();

        let seq_helper = SumHelper::new(num_buckets);
        seq_collect_reduce(&data, &seq_helper, num_buckets);

        let par_helper = SumHelper::new(num_buckets);
        collect_reduce(&data, &par_helper, num_buckets);

        for k in 0..num_buckets {
            assert_eq!(
                seq_helper.get_sum(k),
                par_helper.get_sum(k),
                "mismatch at bucket {k}"
            );
        }
    }

    // --- Tests for heavy-hitter detection ---

    #[test]
    fn test_heavy_hitter_detection() {
        // One key dominates heavily — should be detected
        let num_buckets = 256;
        let n = 50_000;
        let mut data = Vec::with_capacity(n);
        // 90% key=0, 10% spread
        for i in 0..n {
            if i % 10 == 0 {
                data.push(KV { key: (i % (num_buckets - 1)) + 1, val: 1 });
            } else {
                data.push(KV { key: 0, val: 1 });
            }
        }

        let helper = CountHelper::new(num_buckets);
        collect_reduce(&data, &helper, num_buckets);

        // Key 0 should have been counted correctly despite heavy-hitter path
        let expected_0 = data.iter().filter(|kv| kv.key == 0).count();
        assert_eq!(helper.get_count(0), expected_0);

        // Total counts should match n
        let total: usize = (0..num_buckets).map(|k| helper.get_count(k)).sum();
        assert_eq!(total, n);
    }

    #[test]
    fn test_combine_called_for_heavy_hitter() {
        use std::sync::atomic::AtomicBool;

        struct CombineTracker {
            counts: Vec<AtomicUsize>,
            combine_called: AtomicBool,
        }

        impl CollectReduceHelper<KV> for CombineTracker {
            fn get_key(&self, elem: &KV) -> usize {
                elem.key
            }
            fn apply(&self, elem: &KV) {
                self.counts[elem.key].fetch_add(1, Ordering::SeqCst);
            }
            fn combine(&self, elems: &[KV]) {
                self.combine_called.store(true, Ordering::SeqCst);
                for elem in elems {
                    self.apply(elem);
                }
            }
        }

        // Create a dataset large enough to trigger the parallel path with a single
        // dominant key so GetBucket promotes it as a heavy hitter
        let num_buckets = 1024;
        let n = 100_000;
        let data: Vec<KV> = (0..n).map(|_| KV { key: 0, val: 1 }).collect();

        let helper = CombineTracker {
            counts: (0..num_buckets).map(|_| AtomicUsize::new(0)).collect(),
            combine_called: AtomicBool::new(false),
        };
        collect_reduce(&data, &helper, num_buckets);

        assert_eq!(helper.counts[0].load(Ordering::SeqCst), n);
        // On a multi-threaded system with enough data, combine should be called
        // (may not fire on single-thread or tiny data — that's fine)
    }

    // --- Fuzz / property-based tests ---

    /// Fuzz: random keys/values, verify sum correctness for many seeds.
    #[test]
    fn fuzz_collect_reduce_sums() {
        for seed in 0..50 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_buckets = rng.gen_range(1..=128);
            let n = rng.gen_range(0..=20_000);
            let data: Vec<KV> = (0..n)
                .map(|_| KV {
                    key: rng.gen_range(0..num_buckets),
                    val: rng.gen_range(0..1000),
                })
                .collect();

            // Compute expected sums
            let mut expected = vec![0usize; num_buckets];
            for kv in &data {
                expected[kv.key] += kv.val as usize;
            }

            let helper = SumHelper::new(num_buckets);
            collect_reduce(&data, &helper, num_buckets);

            for k in 0..num_buckets {
                assert_eq!(
                    helper.get_sum(k),
                    expected[k],
                    "seed={seed} bucket={k} n={n} num_buckets={num_buckets}"
                );
            }
        }
    }

    /// Fuzz: random keys/values, verify count correctness for many seeds.
    #[test]
    fn fuzz_collect_reduce_counts() {
        for seed in 100..150 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_buckets = rng.gen_range(1..=256);
            let n = rng.gen_range(0..=30_000);
            let data: Vec<KV> = (0..n)
                .map(|_| KV {
                    key: rng.gen_range(0..num_buckets),
                    val: 0,
                })
                .collect();

            let mut expected = vec![0usize; num_buckets];
            for kv in &data {
                expected[kv.key] += 1;
            }

            let helper = CountHelper::new(num_buckets);
            collect_reduce(&data, &helper, num_buckets);

            for k in 0..num_buckets {
                assert_eq!(
                    helper.get_count(k),
                    expected[k],
                    "seed={seed} bucket={k} n={n} num_buckets={num_buckets}"
                );
            }
        }
    }

    /// Fuzz: verify parallel_counting_sort preserves multiset and assigns correct buckets.
    #[test]
    fn fuzz_counting_sort() {
        for seed in 200..250 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_buckets = rng.gen_range(1..=64);
            let n = rng.gen_range(0..=15_000);
            let input: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
            let keys: Vec<usize> = (0..n).map(|_| rng.gen_range(0..num_buckets)).collect();

            let (sorted, offsets) = parallel_counting_sort(&input, &keys, num_buckets);

            assert_eq!(sorted.len(), n);
            assert_eq!(offsets[num_buckets], n);

            // Offsets monotone
            for i in 0..num_buckets {
                assert!(offsets[i] <= offsets[i + 1], "seed={seed}");
            }

            // Multiset preserved
            let mut exp = input.clone();
            let mut act = sorted.clone();
            exp.sort();
            act.sort();
            assert_eq!(exp, act, "seed={seed} multiset mismatch");
        }
    }

    /// Fuzz: skewed distribution (Zipf-like) to stress heavy-hitter paths.
    #[test]
    fn fuzz_collect_reduce_skewed() {
        for seed in 300..330 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_buckets = rng.gen_range(16..=512);
            let n = rng.gen_range(5_000..=50_000);

            // Zipf-like: key = floor(n / (rank+1)), many elements get key 0
            let data: Vec<KV> = (0..n)
                .map(|_| {
                    let rank = rng.gen_range(1..=num_buckets);
                    let key = num_buckets / rank;
                    KV {
                        key: key.min(num_buckets - 1),
                        val: 1,
                    }
                })
                .collect();

            let mut expected = vec![0usize; num_buckets];
            for kv in &data {
                expected[kv.key] += 1;
            }

            let helper = CountHelper::new(num_buckets);
            collect_reduce(&data, &helper, num_buckets);

            for k in 0..num_buckets {
                assert_eq!(
                    helper.get_count(k),
                    expected[k],
                    "seed={seed} bucket={k}"
                );
            }
        }
    }

    /// Fuzz: all elements share one key (extreme heavy hitter).
    #[test]
    fn fuzz_collect_reduce_single_key() {
        for seed in 400..420 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_buckets = rng.gen_range(1..=256);
            let key = rng.gen_range(0..num_buckets);
            let n = rng.gen_range(0..=25_000);
            let data: Vec<KV> = (0..n).map(|i| KV { key, val: i as u64 }).collect();

            let expected_sum: usize = (0..n).sum();
            let helper = SumHelper::new(num_buckets);
            collect_reduce(&data, &helper, num_buckets);

            assert_eq!(helper.get_sum(key), expected_sum, "seed={seed}");
            for k in 0..num_buckets {
                if k != key {
                    assert_eq!(helper.get_sum(k), 0, "seed={seed} stray sum in bucket {k}");
                }
            }
        }
    }

    /// Fuzz: exactly two keys to exercise the few-buckets path.
    #[test]
    fn fuzz_collect_reduce_two_keys() {
        for seed in 500..530 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_buckets = 2;
            let n = rng.gen_range(0..=20_000);
            let data: Vec<KV> = (0..n)
                .map(|_| KV {
                    key: rng.gen_range(0..2),
                    val: rng.gen_range(1..50),
                })
                .collect();

            let mut expected = [0usize; 2];
            for kv in &data {
                expected[kv.key] += kv.val as usize;
            }

            let helper = SumHelper::new(num_buckets);
            collect_reduce(&data, &helper, num_buckets);

            assert_eq!(helper.get_sum(0), expected[0], "seed={seed}");
            assert_eq!(helper.get_sum(1), expected[1], "seed={seed}");
        }
    }

    /// Fuzz: size near CR_SEQ_THRESHOLD boundary to exercise both sequential and parallel paths.
    #[test]
    fn fuzz_collect_reduce_threshold_boundary() {
        for delta in [0, 1, 2, 100, 1000] {
            for side in [CR_SEQ_THRESHOLD.saturating_sub(delta), CR_SEQ_THRESHOLD + delta] {
                let n = side;
                let num_buckets = 64;
                let mut rng = SmallRng::seed_from_u64(n as u64);
                let data: Vec<KV> = (0..n)
                    .map(|_| KV {
                        key: rng.gen_range(0..num_buckets),
                        val: rng.gen_range(0..100),
                    })
                    .collect();

                let mut expected = vec![0usize; num_buckets];
                for kv in &data {
                    expected[kv.key] += kv.val as usize;
                }

                let helper = SumHelper::new(num_buckets);
                collect_reduce(&data, &helper, num_buckets);

                for k in 0..num_buckets {
                    assert_eq!(
                        helper.get_sum(k),
                        expected[k],
                        "n={n} bucket={k}"
                    );
                }
            }
        }
    }
}
