// Regression tree with variance reduction (equivalent to information gain for continuous targets)

use crate::optimized_data::CachedHistogram;
use crate::tree::{HistSplitResult, TreeParams};

// Calculate variance reduction for regression (analogous to information gain)
pub fn calculate_variance_reduction(
    parent_var: f64,
    n_left: f64,
    var_left: f64,
    n_right: f64,
    var_right: f64,
    n_total: f64,
) -> f64 {
    parent_var - (n_left * var_left + n_right * var_right) / n_total
}

// Calculate variance from y values
pub fn calculate_variance(y_sum: f64, y_sq_sum: f64, n: f64) -> f64 {
    if n < 1.0 {
        return 0.0;
    }
    let mean = y_sum / n;
    (y_sq_sum / n) - mean.powi(2)
}

// Find best split with variance reduction (for regression)
pub fn find_best_split_regression(
    hist: &CachedHistogram,
    params: &TreeParams,
    depth: i32,
) -> HistSplitResult {
    let bins = &hist.bins;

    let g_total = hist.sums.grad as f64;
    let h_total = hist.sums.hess as f64;
    let y_total = hist.sums.y as f64;
    let n_total = hist.sums.count as f64;

    // We still need y_sq_total for variance calculation
    let mut y_sq_total = 0.0;
    for b in bins {
        y_sq_total += (b.y as f64) * (b.y as f64);
    }

    if n_total < params.min_samples_split as f64 {
        return HistSplitResult::default();
    }

    let parent_var = calculate_variance(y_total, y_sq_total, n_total);
    let parent_score = g_total * g_total / (h_total + params.reg_lambda);

    // Adaptive weight: use variance reduction more at shallow depths
    let adaptive_weight = params.mi_weight * (-0.1 * depth as f64).exp();

    let mut best_split = HistSplitResult::default();
    let mut gl = 0.0;
    let mut hl = 0.0;
    let mut y_left = 0.0;
    let mut y_sq_left = 0.0;
    let mut n_left = 0.0;

    let n_splits = bins.len().saturating_sub(1);

    for i in 0..n_splits {
        let b = unsafe { bins.get_unchecked(i) };
        gl += b.grad as f64;
        hl += b.hess as f64;
        y_left += b.y as f64;
        y_sq_left += (b.y as f64) * (b.y as f64);
        n_left += b.count as f64;

        if n_left < 1.0 || hl < params.min_child_weight {
            continue;
        }

        let gr = g_total - gl;
        let hr = h_total - hl;
        let n_right = n_total - n_left;
        let y_right = y_total - y_left;
        let y_sq_right = y_sq_total - y_sq_left;

        if n_right < 1.0 || hr < params.min_child_weight {
            continue;
        }

        // Newton gain (gradient-based)
        let newton_gain = 0.5
            * (gl.powi(2) / (hl + params.reg_lambda) + gr.powi(2) / (hr + params.reg_lambda)
                - parent_score)
            - params.gamma;

        // Variance reduction (information gain for regression)
        let var_left = calculate_variance(y_left, y_sq_left, n_left);
        let var_right = calculate_variance(y_right, y_sq_right, n_right);
        let var_reduction =
            calculate_variance_reduction(parent_var, n_left, var_left, n_right, var_right, n_total);

        // Combined gain: Newton + Variance Reduction
        let combined_gain = newton_gain + adaptive_weight * var_reduction;

        if combined_gain > best_split.best_gain {
            best_split.best_gain = combined_gain;
            best_split.best_bin_idx = i as u8;
        }
    }

    best_split
}
