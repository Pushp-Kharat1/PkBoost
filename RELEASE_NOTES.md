# Release Notes - PKBoost v2.2.1

## Summary

This release resolves **GitHub Issue #13** and introduces major algorithmic and cache locality optimizations that eliminate the $10\times$ cache-thrashing bottleneck observed on large datasets ($1\text{M}+$ rows), achieving near-linear $O(N)$ execution scaling.

---

## Performance & Memory Optimizations

1. **Row-Index Sorting After Subsampling (`src/model.rs`, `src/regression.rs`)**:
   - Sampled row indices are now sorted in-place (`sort_unstable()`) immediately after subsampling.
   - Restores sequential memory access patterns across all feature columns during histogram construction at every tree depth.
   - Eliminates random DRAM thrashing on large datasets ($1\text{M}+$ rows), bringing 5× data scaling from $10\times$ runtime down to a near-linear **$5.05\times$**.

2. **Leaf Assignment Reuse (`src/tree.rs`)**:
   - Implemented `fit_optimized_with_predictions(...)`.
   - In-bag training samples ($80\%\text{--}100\%$) directly receive their terminal leaf values during tree partitioning, completely bypassing the redundant full-tree traversal pass ($N \times \text{depth}$ pointer hops) after every tree.
   - Out-of-bag samples (when `subsample < 1.0`) are processed with fast vectorized prediction.

3. **Branchless Partitioning (`src/tree.rs`)**:
   - Replaced conditional branching in `partition_into_optimized` with direct pointer writes (`left_ptr` / `right_ptr`) and `set_len()`.
   - Eliminates branch mispredictions during row splits.

4. **Dynamic `IndexPool` Buffer Recycling (`src/tree.rs`)**:
   - Removed the hardcoded `<= 8192` capacity ceiling that caused large partition buffers ($N \ge 10^4$) to be dropped and repeatedly reallocated at every node split.

5. **Eliminated Redundant 60 MB Transpose Clone (`src/model.rs`)**:
   - Removed `binned_data_cache.clone()`, eliminating 60+ MB of duplicate RAM allocation on 1M rows.

6. **Fused Prediction Update and Clamp (`src/model.rs`)**:
   - Combined learning rate application and $[-10, 10]$ log-odds clamping into a single parallel Rayon pass over predictions.

### Benchmark Results (30 Features, 100 Trees, Depth 6)
- **200,000 rows**: 11.69s (1,711,047 sample-trees / sec)
- **1,000,000 rows**: 58.97s (1,695,726 sample-trees / sec)
- **Scaling (5× data)**: **5.05×** time (vs 10× baseline blowup)

---

## Bug Fixes & Usability (GitHub Issue #13)

- **Suppressed Unsolicited Auto-Tuner & Hardware Printouts**:
  - Removed unconditional `println!` statements in `auto_tune_principled()` (`src/auto_tuner.rs`) and `detect_hardware()` (`src/adaptive_parallel.rs`).
- **Clean & Consolidated Verbose Summary**:
  - Merged duplicate "Auto-Parameter Selection" and "Auto-Tuner" output blocks into a single structured summary shown only when `verbose = true`.
- **Always Show Final Training Results**:
  - `=== Training Complete ===` (best score, best iteration, total time, final tree count) is now always printed regardless of `verbose` setting.

---

## Build & Packaging Improvements

- **Maturin Packaging (`pyproject.toml`)**:
  - Removed obsolete `python-source = "."` setting, enabling clean release wheel packaging (`maturin build --release`) and `pip install`.
- **Dependencies (`Cargo.toml`, `src/lib.rs`)**:
  - Commented out unused external C dependencies (`mimalloc`, `fork_union`) to ensure pure-Rust compatibility across MSVC and GNU toolchains.
