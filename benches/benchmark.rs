use ndarray::{Array1, Array2};
use pkboost::PKBoostBuilder;
use std::time::Instant;

struct BenchmarkResult {
    n_samples: usize,
    n_features: usize,
    n_trees: usize,
    max_depth: usize,
    fit_duration_secs: f64,
    ms_per_tree: f64,
    sample_trees_per_sec: f64,
    #[allow(dead_code)]
    predict_duration_secs: f64,
}

fn generate_dataset(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    println!("Generating synthetic data: {} rows x {} features...", n_samples, n_features);
    let t0 = Instant::now();

    // Fast pseudo-random generation using 64-bit LCG
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        let mut s = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add((j as u64).wrapping_mul(1442695040888963407))
            .wrapping_add(1);
        s = (s ^ (s >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94d049bb133111eb);
        s ^= s >> 31;
        let u = (s as f64) / (u64::MAX as f64);
        (u - 0.5) * 4.0 // ~Uniform(-2.0, 2.0)
    });

    let y = Array1::from_shape_fn(n_samples, |i| {
        // ~20% positive class based on first 3 features
        let s = x[[i, 0]] + x[[i, 1]] * 0.5 - x[[i, 2]] * 0.8;
        if s > 0.84 { 1.0 } else { 0.0 }
    });

    let pos_ratio = y.iter().filter(|&&v| v > 0.5).count() as f64 / n_samples as f64;
    println!(
        "Data ready in {:.2}s. Shape: ({}, {}), Positive ratio: {:.3}",
        t0.elapsed().as_secs_f64(),
        x.nrows(),
        x.ncols(),
        pos_ratio
    );

    (x, y)
}

fn run_single_benchmark(
    n_samples: usize,
    n_features: usize,
    n_trees: usize,
    max_depth: usize,
    subsample: f64,
    colsample: f64,
) -> BenchmarkResult {
    println!("\n{}", "=".repeat(65));
    println!(
        "BENCHMARK: {:>9} samples x {:>2} features | {:>3} trees, depth {} | sub: {:.2}, col: {:.2}",
        n_samples, n_features, n_trees, max_depth, subsample, colsample
    );
    println!("{}", "=".repeat(65));

    let (x, y) = generate_dataset(n_samples, n_features);

    let mut model = PKBoostBuilder::new()
        .n_estimators(n_trees)
        .max_depth(max_depth)
        .learning_rate(0.1)
        .subsample(subsample)
        .colsample_bytree(colsample)
        .build_with_data(x.view(), y.view());

    println!("Starting model training...");
    let t_fit_start = Instant::now();
    model.fit(x.view(), y.view(), None, false).expect("Training failed");
    let fit_duration = t_fit_start.elapsed().as_secs_f64();

    let ms_per_tree = (fit_duration / n_trees as f64) * 1000.0;
    let sample_trees_per_sec = (n_samples as f64 * n_trees as f64) / fit_duration;
    let rows_per_sec = n_samples as f64 / fit_duration;

    println!("Testing predict_proba on {} samples...", n_samples.min(100_000));
    let test_samples = n_samples.min(100_000);
    let x_test = x.slice(ndarray::s![0..test_samples, ..]);
    let t_pred_start = Instant::now();
    let _preds = model.predict_proba(x_test).expect("Prediction failed");
    let pred_duration = t_pred_start.elapsed().as_secs_f64();

    println!("\nRESULTS:");
    println!("  Total fit time: {:.3}s", fit_duration);
    println!("  Per-tree time : {:.2} ms/tree", ms_per_tree);
    println!("  Throughput    : {:>11.0} sample-trees/sec ({:>9.0} rows/s)", sample_trees_per_sec, rows_per_sec);
    println!("  Predict ({}k) : {:.3}s ({:>9.0} rows/s)", test_samples / 1000, pred_duration, test_samples as f64 / pred_duration);

    BenchmarkResult {
        n_samples,
        n_features,
        n_trees,
        max_depth,
        fit_duration_secs: fit_duration,
        ms_per_tree,
        sample_trees_per_sec,
        predict_duration_secs: pred_duration,
    }
}

fn main() {
    println!("=================================================================");
    println!("        PKBOOST NATIVE RUST PERFORMANCE BENCHMARK SUITE          ");
    println!("=================================================================");

    let num_cpus = num_cpus::get();
    let num_threads = rayon::current_num_threads();
    println!("Hardware: {} physical CPUs, {} Rayon worker threads", num_cpus, num_threads);

    // Warmup
    println!("\n--- WARMUP PHASE ---");
    let _ = run_single_benchmark(20_000, 30, 20, 6, 0.8, 0.8);

    // 200,000 Rows Benchmark
    println!("\n--- BASELINE: 200,000 ROWS ---");
    let res_200k = run_single_benchmark(200_000, 30, 100, 6, 0.8, 0.8);

    // 1,000,000 Rows Benchmark
    println!("\n--- BASELINE: 1,000,000 ROWS ---");
    let res_1m = run_single_benchmark(1_000_000, 30, 100, 6, 0.8, 0.8);

    // Summary Table
    println!("\n\n{}", "=".repeat(75));
    println!("                     NATIVE RUST BENCHMARK BASELINE                      ");
    println!("{}", "=".repeat(75));
    println!(
        "{:<12} | {:<8} | {:<5} | {:<5} | {:<12} | {:<10} | {:<14}",
        "Samples", "Features", "Trees", "Depth", "Fit Time (s)", "ms / tree", "Throughput (st/s)"
    );
    println!("{}", "-".repeat(75));
    for res in &[&res_200k, &res_1m] {
        println!(
            "{:<12} | {:<8} | {:<5} | {:<5} | {:<12.3} | {:<10.2} | {:<14.0}",
            format!("{:>9}", res.n_samples),
            res.n_features,
            res.n_trees,
            res.max_depth,
            res.fit_duration_secs,
            res.ms_per_tree,
            res.sample_trees_per_sec
        );
    }
    println!("{}", "-".repeat(75));
    let ratio = res_1m.fit_duration_secs / res_200k.fit_duration_secs;
    println!("Data Scaling Factor : 5.00x (200k -> 1M rows)");
    println!("Time Scaling Factor : {:.2}x ({:.2}s -> {:.2}s)", ratio, res_200k.fit_duration_secs, res_1m.fit_duration_secs);
    if ratio <= 5.5 {
        println!("Cache Scaling Status: Near-Linear O(N) scaling verified.");
    } else {
        println!("Cache Scaling Status: Sub-optimal scaling (super-linear blowup: {:.2}x).", ratio);
    }
    println!("{}", "=".repeat(75));
}
