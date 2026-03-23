# PkBoost Autoresearch — SVP Company Scoring

## Goal
Maximize **PR AUC** on the Series A model benchmark (see `benchmark.py`).

The task: predict which seed-stage startups will raise a Series A within 4–7 months.
- ~3400:1 class imbalance (0.03% positive rate)
- ~80 features: founder background, funding history, traction metrics
- Temporal data: quarterly snapshots 2019–2023

PR AUC is the right metric here. With 3400:1 imbalance, ROC AUC is nearly
insensitive to improvements that matter operationally (top-100 precision).

## Files you may modify
- `src/loss.rs` — loss functions, gradients, hessians
- `src/model.rs` — training loop, subsampling, hyperparameter defaults
- `src/tree.rs` — tree building, leaf computation, regularization
- `src/histogram_builder.rs` — feature binning strategy

## Hard constraints
1. Do NOT change the signature or semantics of `predict_contributions()` — production
   relies on it for per-company heat score interpretation
2. Do NOT change JSON serialization format without `#[serde(default)]` on new fields
   (old models must still load)
3. Changes must pass `cargo check --lib` before full build
4. `focal_gamma` parameter must remain exposed (it's the first deployed experiment)
5. Do NOT modify `src/python_bindings.rs` or `pkboost_sklearn/` — only Rust core

## Key context

**Loss function** (`src/loss.rs`):
- `OptimizedShannonLoss` — weighted log-loss with `scale_pos_weight`
- `focal_gamma` is now implemented: focal loss down-weights easy negatives

**Tree building** (`src/tree.rs`):
- `OptimizedTreeShannon` — SoA layout, flat vecs, binned features (i16)
- Newton step: leaf_value = -sum(grad) / (sum(hess) + reg_lambda)

**Model** (`src/model.rs`):
- `OptimizedPKBoostShannon` — holds trees + hyperparams + histogram_builder
- Uses f32 gradients internally (cast back from f64 computation)
- PR AUC tracked for early stopping

**Histogram** (`src/histogram_builder.rs`):
- Bins continuous features into i16 using equal-width binning
- Default: 32 bins per feature

## High-value hypotheses to explore

1. **Focal gamma tuning** — focal_gamma=0 (current) vs 0.5, 1.0, 2.0
2. **Leaf Laplace smoothing** — with 3400:1 imbalance, minority-class leaves
   are estimated from very few samples; add additive smoothing to leaf values
3. **Recency weighting** — startup fundraising patterns shift over time (ZIRP/post-ZIRP);
   down-weight samples from quarters >12 months ago
4. **Histogram bin allocation** — more bins for log-transformed skewed features;
   32 bins may be too coarse for funding_amount distributions
5. **Class-balanced subsampling** — always include all positives in each tree's
   subsample, fill remainder with negatives (vs current random subsample)
6. **Asymmetric learning rate** — larger Newton step for rare positive-class nodes

## Experiment protocol
For each proposed change:
1. State hypothesis and expected direction of effect
2. Make ONE focused change
3. Ensure `cargo check --lib` passes
4. Run `benchmark.py` and compare PR AUC to baseline
5. Keep if PR AUC improves, revert if not
6. Record result in experiment_log.jsonl
