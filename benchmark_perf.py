import time
import numpy as np
import pkboost

def run_benchmark(n_samples, n_features, n_trees, max_depth, subsample=0.8, colsample=0.8):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {n_samples:,} rows x {n_features} features, {n_trees} trees, depth {max_depth}")
    print(f"subsample: {subsample}, colsample: {colsample}")
    print(f"{'='*60}")

    print("Generating synthetic dataset...")
    np.random.seed(42)
    # Generate continuous features in standard normal
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    # Binary classification target with ~20% positives
    linear_comb = np.sum(X[:, :5], axis=1)
    y = (linear_comb > np.percentile(linear_comb, 80)).astype(np.float64)
    print(f"Dataset generated: shape {X.shape}, positive ratio: {np.mean(y):.3f}")

    clf = pkboost.PKBoostClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=subsample,
        colsample_bytree=colsample,
    )

    print("Starting training...")
    t0 = time.perf_counter()
    clf.fit(X, y, verbose=False)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    throughput = (n_samples * n_trees) / elapsed
    print(f"\nRESULTS:")
    print(f"  Total fit time: {elapsed:.3f} seconds")
    print(f"  Per-tree time : {elapsed / n_trees * 1000:.2f} ms / tree")
    print(f"  Throughput    : {throughput:,.0f} sample-trees / second ({n_samples / elapsed:,.0f} rows/s)")
    return elapsed

if __name__ == "__main__":
    # Warmup / quick smoke test
    print("Running warmup (20k rows)...")
    run_benchmark(n_samples=20_000, n_features=30, n_trees=20, max_depth=6)

    # 200,000 rows test (matches previous benchmark)
    print("\nRunning 200k rows benchmark (matches previous profiling baseline: ~7.0s)...")
    time_200k = run_benchmark(n_samples=200_000, n_features=30, n_trees=100, max_depth=6)

    # 1,000,000 rows test (matches previous benchmark baseline: ~70s)
    print("\nRunning 1M rows benchmark (matches previous profiling baseline: ~70.0s)...")
    time_1m = run_benchmark(n_samples=1_000_000, n_features=30, n_trees=100, max_depth=6)

    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"200k rows (100 trees, depth 6): {time_200k:.2f}s (Baseline was ~7.0s)")
    print(f"1M   rows (100 trees, depth 6): {time_1m:.2f}s (Baseline was ~70.0s)")
    ratio = time_1m / time_200k
    print(f"Scaling ratio (1M / 200k = 5x data): {ratio:.2f}x time")
    if ratio < 6.0:
        print("-> Near-linear cache scaling achieved (no longer 10x cache blowup)!")
    print("="*60)
