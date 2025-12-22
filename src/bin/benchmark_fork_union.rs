// ForkUnion vs Rayon performance comparison benchmark
// Uses synthetic data to measure histogram building performance

use mimalloc::MiMalloc;
use rand::Rng;
use std::time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    println!("{}", "=".repeat(80));
    println!("ForkUnion Integration Performance Benchmark");
    println!("{}", "=".repeat(80));

    // Generate synthetic data
    let n_samples = 100_000;
    let n_features = 30;

    println!(
        "\nGenerating {} samples with {} features...",
        n_samples, n_features
    );
    let mut rng = rand::thread_rng();

    let x: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| (0..n_features).map(|_| rng.gen_range(0.0..100.0)).collect())
        .collect();
    let y: Vec<f64> = (0..n_samples)
        .map(|_| if rng.gen_bool(0.3) { 1.0 } else { 0.0 })
        .collect();

    println!(
        "Data generated: {} samples, {} features",
        x.len(),
        x[0].len()
    );

    let n_positives = y.iter().filter(|&&v| v > 0.5).count();
    println!(
        "Class distribution: {}% positive",
        n_positives * 100 / n_samples
    );

    println!("\n{}", "-".repeat(80));
    println!("Training PKBoost with ForkUnion integration...");
    println!("{}", "-".repeat(80));

    // Build model
    let mut model = pkboost::OptimizedPKBoostShannon::auto(&x, &y);
    model.n_estimators = 500; // Reasonable number for benchmarking
    model.max_depth = 6;
    model.early_stopping_rounds = 50;

    // Time the training
    let start = Instant::now();
    model.fit(&x, &y, None, true).expect("Training failed");
    let train_time = start.elapsed().as_secs_f64();

    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK RESULTS");
    println!("{}", "=".repeat(80));
    println!("Training time: {:.2}s", train_time);
    println!("Samples: {}", n_samples);
    println!("Features: {}", n_features);
    println!("Trees: {}", model.n_estimators);
    println!(
        "Throughput: {:.0} samples/sec",
        n_samples as f64 / train_time
    );
    println!("{}", "=".repeat(80));

    // Quick inference benchmark
    let inf_start = Instant::now();
    let _preds = model.predict_proba(&x).expect("Prediction failed");
    let inf_time = inf_start.elapsed().as_secs_f64();

    println!(
        "\nInference time: {:.4}s ({:.0} samples/sec)",
        inf_time,
        n_samples as f64 / inf_time
    );
}
