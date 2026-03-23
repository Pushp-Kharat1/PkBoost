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
- `src/metrics.rs` — PR-AUC / ROC-AUC computation used for early stopping signal

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

**Metrics / Early stopping** (`src/metrics.rs`):
- `calculate_pr_auc` / `calculate_roc_auc` used for validation-set early stopping
- Benchmark now passes a validation split (Q2–Q4 2022) as `eval_set`
- Early stopping fires after `early_stopping_rounds=30` without improvement
- PR-AUC uses left-Riemann sum (no interpolation between thresholds)
- `AUCCalculator` re-sorts scores on every call; sorted indices are cached between calls
- Shannon entropy uses a 10,000-bin lookup table (`ENTROPY_LUT_SIZE`)
- The benchmark reports both `val_pr_auc` (early-stop signal) and `pr_auc` (test set, primary metric)

## Already tried — do NOT re-propose these

The following have been benchmarked exhaustively and did not improve PR AUC.
Proposing variants of these is a waste of time:

- **Focal gamma tuning** — gamma=0.5, 1.0, 2.0 all tried repeatedly. No improvement.
- **Histogram bin count** — 32→64 and 32→256 both tried. No improvement.
- **Laplace smoothing on leaf values** — multiple formulations tried. No improvement.
- **Prediction clamp widening** — [-10,10]→[-15,15] tried. No improvement.
- **Early stopping changes** — eval frequency, window size tweaks tried. No improvement.
- **Subsampling ratio variants** — adjusting neg_to_pos ratio tried. No improvement.

## Fresh hypotheses to explore (not yet tried)

Think beyond the obvious. The kept improvements show the model benefits from
changes that address the fundamental imbalance and gradient dynamics.

1. **Recency weighting** — ZIRP ended in 2022; down-weight samples from pre-2022
   quarters (e.g. `sample_weight *= 0.95^quarters_ago`). Implement in the gradient
   calculation by multiplying grad/hess by a time-decay factor.

2. **Asymmetric regularization** — apply weaker `reg_lambda` to positive-class nodes
   (small hessian sum → over-regularized) and stronger to negative-class nodes.
   In `tree.rs`: if node is majority-positive, use `reg_lambda * 0.1`.

3. **Gradient clipping** — clip extreme gradients to prevent runaway updates on
   high-weight positive samples. In `loss.rs`, clamp grad to `[-10*weight, 10*weight]`.

4. **Per-feature bin allocation** — in `histogram_builder.rs`, use more bins (64)
   for features with high variance (log-transformed funding amounts) and fewer (16)
   for near-binary features. Use variance of the feature to decide.

5. **Shrinkage on small leaves** — in `tree.rs`, if a leaf covers < N samples,
   shrink its value toward zero: `value *= n_samples / (n_samples + shrinkage_threshold)`.
   Different from Laplace — multiplicative, and threshold should be ~50 samples.

6. **Minimum positive count per split** — refuse splits where either child has fewer
   than K positive samples (e.g. K=2). Adds to `min_child_weight` logic in tree.rs.

7. **Base score warm start** — currently base_score is log(pos_rate/(1-pos_rate)).
   Try initializing to a slightly more optimistic value (e.g. multiply by 0.9)
   to reduce early over-confidence on negatives.

8. **Better PR-AUC interpolation for early stopping** (`src/metrics.rs`) — the
   current implementation uses a left-Riemann sum which underestimates area when
   the curve has large recall jumps. Use linear interpolation between consecutive
   (recall, precision) points: `auc += 0.5 * (p_prev + p_curr) * (r_curr - r_prev)`.
   A more accurate early-stopping signal should let the model train longer where useful.

9. **Entropy LUT resolution** (`src/metrics.rs`) — Shannon entropy uses 10,000 bins
   for the lookup table. At 3400:1 imbalance the positive ratio is ~0.00029, landing
   in bin 2-3. Increase to 100,000 bins for better resolution at extreme imbalance,
   or switch to direct computation when `p < 0.001`.

10. **Early stopping on smoothed PR-AUC** (`src/model.rs`) — the current smoothing
    window averages the last 3 validation PR-AUC values. With 3400:1 imbalance the
    signal is noisy; try a window of 5 or use exponential moving average instead of
    a sliding window mean.

## Experiment protocol
For each proposed change:
1. State hypothesis and expected direction of effect
2. Make ONE focused change
3. Ensure `cargo check --lib` passes
4. Run `benchmark.py` and compare PR AUC to baseline
5. Keep if PR AUC improves, revert if not
6. Record result in experiment_log.jsonl
