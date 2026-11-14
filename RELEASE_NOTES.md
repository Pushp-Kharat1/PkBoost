# Release Notes - PKBoost v2.2.1

## Summary

This release resolves **GitHub Issue #13** and introduces a comprehensive suite of hardware-aligned Tier-1 performance optimizations that dramatically accelerate model training and inference. Across 20,000 to 1,000,000 sample datasets, PKBoost now achieves up to a **5.27× speedup**, saturating all CPU execution ports with zero heap churn and near-linear $O(N)$ cache scaling.

---

## Performance & Architecture Breakthroughs (Tier 1)

1. **Layer-Wise Breadth-First Tree Parallelism (`src/tree.rs`)**:
   - Converted the sequential node-by-node queue into layer-by-layer parallel dispatch (`current_level.into_par_iter()`).
   - Eliminates single-threaded execution on all deep nodes (depths 2–6), saturating all available Rayon worker threads throughout tree construction.

2. **Dual-Bank (2-Way Sub-Accumulator) Histograms (`src/optimized_data.rs`)**:
   - Deployed dual sub-accumulators (`bins_0` and `bins_1`) in the 8× unrolled histogram hot loop.
   - Alternating samples into dual banks completely eliminates CPU Read-After-Write (RAW) pipeline dependency stalls on identical bins while maintaining bit-for-bit mathematical identity.

3. **Compact `u8` Bin Representation in `TransposedData` (`src/optimized_data.rs`)**:
   - Replaced 2-byte `i16` storage with 1-byte `u8` features.
   - Halves feature matrix memory from 60 MB to **30 MB** on 1M rows, enabling the entire feature set to fit comfortably within modern CPU L3 caches (32 MB) and cutting DRAM memory bus traffic in half.

4. **Columnar Direct Transform in `HistogramBuilder` (`src/histogram_builder.rs`)**:
   - Implemented `transform_transposed_u8(...)` which transforms raw float inputs directly into transposed `(n_features, n_samples)` layout column-by-column.
   - Keeps feature bin edges permanently pinned inside each core's L1 cache during binning, eliminates temporary row-major matrices, and avoids the memory copy of array transposition.

5. **Contiguous Interleaved `[f64; 4]` Histograms & AVX Subtraction (`src/optimized_data.rs`)**:
   - Stored histogram bins contiguously as `Vec<[f64; 4]>` (32 bytes per bin).
   - Completely eliminated `split_interleaved` and replaced separate vector allocations with single-instruction 256-bit AVX subtraction (`vsubpd`), eliminating over **756,000 heap allocations per 100 trees**.

6. **Zero-Allocation Parallel Loss Computation (`src/loss.rs`, `src/model.rs`, `src/regression.rs`)**:
   - Pre-allocated gradient and hessian buffers outside the boosting loop and populated them via in-place Rayon iterators (`compute_gradient_hessian_into`).
   - Eliminates 32 MB of heap traffic per tree iteration (**3.2 GB per 100 trees**).

7. **Fast Monotonic Bernoulli Subsampling (`src/model.rs`, `src/regression.rs`)**:
   - Replaced 1M-element Fisher-Yates array shuffling and sorting with an $O(N)$ sorted Bernoulli sampler.
   - Reduces subsampling time from 15–20 ms/tree to <0.5 ms/tree.

8. **Precomputed Node Sums (`src/tree.rs`)**:
   - Sums (`g_total`, `h_total`, `y_total`, `n_total`, `parent_entropy`) are now computed once per node at dispatch time, eliminating 29 redundant passes per node across features.

9. **Static Compile-Time Shannon Entropy Table (`src/entropy_table.rs`, `src/metrics.rs`)**:
   - Replaced dynamic heap-allocated `LazyLock<Vec<f64>>` with an 8 KB static table in `.rodata`.
   - Eliminates atomic synchronization checks on lookup and fits entirely inside L1 data cache.

10. **Parallel Node-Stats Gather (`src/tree.rs`)**:
    - Dispatches node statistics gathering across threads for any node with $\ge 8,192$ samples.

---

## Native Rust Benchmark Results (`benches/benchmark.rs`)

*Hardware: 4 physical CPUs, 4 Rayon worker threads, Windows MSVC release mode*

| Benchmark Suite | Configuration | Original Baseline | Post-Optimization | Measured Speedup | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Warmup Phase** | 20k rows × 30 feats, 20 trees | **0.754s** (37.7 ms/tree) | **0.143s** (7.17 ms/tree) | **5.27× faster** | 2,788,459 st/s |
| **Medium Dataset** | 200k rows × 30 feats, 100 trees | **18.383s** (183.8 ms/tree) | **5.683s** (56.8 ms/tree) | **3.23× faster** | 3,519,017 st/s |
| **Large Dataset** | 1,000,000 rows × 30 feats, 100 trees | **61.769s** (617.7 ms/tree) | **31.195s** (311.9 ms/tree) | **1.98× faster** | 3,205,641 st/s |
| **Inference (100k)**| `predict_proba` throughput | **0.538s** (185,873 rows/s) | **0.289s** (345,894 rows/s) | **1.86× faster** | 345,894 rows/s |

- **Scaling Verification**: 5.0× data increase (200k $\rightarrow$ 1M rows) results in a **5.49× runtime factor**, confirming near-linear $O(N)$ execution scaling and L3 cache containment.
- **Test Integrity**: All 33 Rust unit tests and 15 Python pytest/sklearn integration tests pass bit-for-bit.

---

## Bug Fixes & Usability (GitHub Issue #13)

- **Suppressed Unsolicited Auto-Tuner & Hardware Printouts**:
  - Removed unconditional `println!` statements in `auto_tune_principled()` (`src/auto_tuner.rs`) and `detect_hardware()` (`src/adaptive_parallel.rs`).
- **Clean & Consolidated Verbose Summary**:
  - Merged duplicate "Auto-Parameter Selection" and "Auto-Tuner" output blocks into a single structured summary shown only when `verbose = true`.
- **Always Show Final Training Results**:
  - `=== Training Complete ===` (best score, best iteration, total time, final tree count) is now always printed regardless of `verbose` setting.
