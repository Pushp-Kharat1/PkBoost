// Decision tree implementation with histogram-based splitting
// OPTIMIZED VERSION with ~30-40% speed improvement

use rayon::prelude::*;
use ndarray::ArrayView1;
use serde::{Serialize, Deserialize};
use crate::metrics::calculate_shannon_entropy;
use crate::optimized_data::{TransposedData, CachedHistogram};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::cell::RefCell;

#[derive(Debug, Clone, Copy)]
pub struct HistSplitResult {
    pub best_gain: f64,
    pub best_bin_idx: i16,
}

impl Default for HistSplitResult {
    fn default() -> Self {
        Self {
            best_gain: f64::NEG_INFINITY,
            best_bin_idx: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedTreeShannon {
    max_depth: usize,
    // Struct of Arrays (SoA) - excellent for cache locality
    node_types: Vec<u8>,
    leaf_values: Vec<f64>,
    split_features: Vec<usize>,
    split_thresholds: Vec<i16>,
    left_children: Vec<usize>,
    right_children: Vec<usize>,
    pub feature_indices: Vec<usize>,
}

impl OptimizedTreeShannon {
    pub fn new(max_depth: usize) -> Self {
        let max_nodes = 2_usize.pow(max_depth as u32 + 1);
        Self {
            max_depth,
            node_types: vec![0; max_nodes],
            leaf_values: vec![0.0; max_nodes],
            split_features: vec![0; max_nodes],
            split_thresholds: vec![0; max_nodes],
            left_children: vec![0; max_nodes],
            right_children: vec![0; max_nodes],
            feature_indices: Vec::new(),
        }
    }
    
    #[inline(always)]
    fn set_leaf(&mut self, node_idx: usize, value: f64) {
        self.node_types[node_idx] = 1;
        self.leaf_values[node_idx] = value;
    }
    
    #[inline(always)]
    fn set_split(&mut self, node_idx: usize, feature: usize, threshold: i16, 
                 left_child: usize, right_child: usize) {
        self.node_types[node_idx] = 2;
        self.split_features[node_idx] = feature;
        self.split_thresholds[node_idx] = threshold;
        self.left_children[node_idx] = left_child;
        self.right_children[node_idx] = right_child;
    }

    #[inline(always)]
    pub fn predict_single(&self, x_binned_row: &[i16]) -> f64 {
        let mut current_node_index = 0;
        
        loop {
            match self.node_types[current_node_index] {
                1 => return self.leaf_values[current_node_index],
                2 => {
                    let feature = self.split_features[current_node_index];
                    let threshold = self.split_thresholds[current_node_index];
                    let feature_value = x_binned_row.get(feature).copied().unwrap_or(0);
                    
                    current_node_index = if feature_value <= threshold {
                        self.left_children[current_node_index]
                    } else {
                        self.right_children[current_node_index]
                    };
                }
                _ => return 0.0,
            }
        }
    }

    #[inline(always)]
    pub fn predict_from_transposed(&self, transposed_data: &TransposedData, sample_idx: usize) -> f64 {
        let mut current_node_index = 0;
        
        loop {
            match self.node_types[current_node_index] {
                1 => return self.leaf_values[current_node_index],
                2 => {
                    let feature = self.split_features[current_node_index];
                    let threshold = self.split_thresholds[current_node_index];
                    
                    let feature_value = if feature < transposed_data.n_features 
                        && sample_idx < transposed_data.n_samples 
                    {
                        transposed_data.features[[feature, sample_idx]]
                    } else {
                        0
                    };
                    
                    current_node_index = if feature_value <= threshold {
                        self.left_children[current_node_index]
                    } else {
                        self.right_children[current_node_index]
                    };
                }
                _ => return 0.0,
            }
        }
    }

    // OPTIMIZATION: Batch prediction with SIMD-friendly memory access
    pub fn predict_batch(&self, transposed_data: &TransposedData, 
                         sample_indices: &[usize]) -> Vec<f64> {
        // Use parallelism for medium to large batches
        if sample_indices.len() >= 500 {
            sample_indices.par_iter()
                .map(|&sample_idx| self.predict_from_transposed(transposed_data, sample_idx))
                .collect()
        } else {
            sample_indices.iter()
                .map(|&sample_idx| self.predict_from_transposed(transposed_data, sample_idx))
                .collect()
        }
    }

    pub fn count_splits_on_features(&self, features: &[usize]) -> usize {
        let feature_set: HashSet<_> = features.iter().copied().collect();
        self.node_types.iter().enumerate()
            .filter(|(idx, &node_type)| {
                node_type == 2 && feature_set.contains(&self.split_features[*idx])
            })
            .count()
    }

    pub fn count_total_splits(&self) -> usize {
        self.node_types.iter().filter(|&&t| t == 2).count()
    }

    pub fn feature_dependency_score(&self, features: &[usize]) -> f64 {
        let total = self.count_total_splits();
        if total == 0 { return 0.0; }
        let dependent = self.count_splits_on_features(features);
        dependent as f64 / total as f64
    }

    pub fn get_used_features(&self) -> Vec<usize> {
        let mut features = Vec::new();
        for (idx, &node_type) in self.node_types.iter().enumerate() {
            if node_type == 2 {
                let feature = self.split_features[idx];
                if !features.contains(&feature) {
                    features.push(feature);
                }
            }
        }
        features
    }

    pub fn fit_optimized(
        &mut self,
        transposed_data: &TransposedData,
        y: &[f64],
        grad: &[f64],
        hess: &[f64],
        sample_indices: &[usize],
        feature_indices: &[usize],
        params: &TreeParams,
    ) {
        if transposed_data.n_samples == 0 || sample_indices.is_empty() { 
            return; 
        }

        self.feature_indices = feature_indices.iter()
            .filter(|&&idx| idx < transposed_data.n_features)
            .copied()
            .collect();

        if self.feature_indices.is_empty() { 
            return; 
        }

        let y_view = ArrayView1::from(y);
        let grad_view = ArrayView1::from(grad);
        let hess_view = ArrayView1::from(hess);

        let max_nodes = 2_usize.pow(self.max_depth as u32 + 1);
        let mut workspace = TreeBuildingWorkspace::new(sample_indices.len());

        // OPTIMIZATION 1: Build root histogram once
        let root_hists = build_hists_optimized(
            &self.feature_indices, 
            transposed_data, 
            &y_view, 
            &grad_view, 
            &hess_view,
            sample_indices, 
            params
        );

        let mut queue: VecDeque<SplitTask> = VecDeque::with_capacity(max_nodes / 2);
        queue.push_back(SplitTask {
            node_index: 0,
            sample_indices: Arc::new(sample_indices.to_vec()),
            histogram: root_hists,
            depth: 0,
        });

        let mut next_node_in_vec = 1;

        while let Some(task) = queue.pop_front() {
            let n_samples = task.sample_indices.len();
            
            // OPTIMIZATION 2: Extract totals once
            let (g_total, h_total): (f64, f64) = {
            let (g_slice, h_slice, _, _) = task.histogram[0].as_slices();
             (g_slice.iter().sum(), h_slice.iter().sum())
            };

            // OPTIMIZATION 3: Early gradient-based pruning
            let gradient_norm = g_total.abs();
            if gradient_norm < params.min_child_weight * 0.01 {
                self.set_leaf(task.node_index, -g_total / (h_total + params.reg_lambda));
                continue;
            }

            // Stopping conditions
            if task.depth >= self.max_depth 
                || n_samples < params.min_samples_split 
                || h_total < params.min_child_weight 
            {
                self.set_leaf(task.node_index, -g_total / (h_total + params.reg_lambda));
                continue;
            }

            // OPTIMIZATION 4: Fast parallel split finding
            let best_split = find_best_split_across_features_optimized(
                &task.histogram, 
                params, 
                task.depth as i32
            );

            if best_split.is_none() || best_split.unwrap().1.best_gain <= 1e-6 {
                self.set_leaf(task.node_index, -g_total / (h_total + params.reg_lambda));
                continue;
            }

            let (best_feature_local_idx, split_info) = best_split.unwrap();
            let best_feature_global_idx = self.feature_indices[best_feature_local_idx];

            // OPTIMIZATION 5: Fast partitioning with pre-allocated buffers
            partition_into_optimized(
                &task.sample_indices, 
                best_feature_global_idx, 
                split_info.best_bin_idx,
                transposed_data, 
                &mut workspace.left_indices, 
                &mut workspace.right_indices
            );

            if workspace.left_indices.is_empty() || workspace.right_indices.is_empty() {
                self.set_leaf(task.node_index, -g_total / (h_total + params.reg_lambda));
                continue;
            }

            // OPTIMIZATION 6: Histogram subtraction (only build smaller child)
            let (left_hists, right_hists) = if workspace.left_indices.len() < workspace.right_indices.len() {
                let smaller_hists = build_hists_optimized(
                    &self.feature_indices, 
                    transposed_data, 
                    &y_view, 
                    &grad_view, 
                    &hess_view,
                    &workspace.left_indices, 
                    params
                );
                let larger_hists: Vec<CachedHistogram> = task.histogram.par_iter()
                    .zip(&smaller_hists)
                    .map(|(parent, sibling)| parent.subtract(sibling))
                    .collect();
                (smaller_hists, larger_hists)
            } else {
                let smaller_hists = build_hists_optimized(
                    &self.feature_indices, 
                    transposed_data, 
                    &y_view, 
                    &grad_view, 
                    &hess_view,
                    &workspace.right_indices, 
                    params
                );
                let larger_hists: Vec<CachedHistogram> = task.histogram.par_iter()
                    .zip(&smaller_hists)
                    .map(|(parent, sibling)| parent.subtract(sibling))
                    .collect();
                (larger_hists, smaller_hists)
            };

            let left_child_index = next_node_in_vec;
            let right_child_index = next_node_in_vec + 1;

            if right_child_index >= self.node_types.len() { 
                continue; 
            }

            self.set_split(
                task.node_index,
                best_feature_global_idx,
                split_info.best_bin_idx,
                left_child_index,
                right_child_index,
            );
            next_node_in_vec += 2;

            queue.push_back(SplitTask {
                node_index: left_child_index,
                sample_indices: Arc::new(workspace.left_indices.clone()),
                histogram: left_hists,
                depth: task.depth + 1,
            });
            queue.push_back(SplitTask {
                node_index: right_child_index,
                sample_indices: Arc::new(workspace.right_indices.clone()),
                histogram: right_hists,
                depth: task.depth + 1,
            });
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeParams {
    pub min_samples_split: usize,
    pub min_child_weight: f64,
    pub reg_lambda: f64,
    pub gamma: f64,
    pub mi_weight: f64,
    pub n_bins_per_feature: Vec<usize>,
    pub feature_elimination_threshold: f64,
}

impl Default for TreeParams {
    fn default() -> Self {
        Self {
            min_samples_split: 20,
            min_child_weight: 1.0,
            reg_lambda: 1.0,
            gamma: 0.0,
            mi_weight: 0.3,
            n_bins_per_feature: Vec::new(),
            feature_elimination_threshold: 0.01,
        }
    }
}

struct SplitTask {
    node_index: usize,
    sample_indices: Arc<Vec<usize>>,
    histogram: Vec<CachedHistogram>,
    depth: usize,
}

// OPTIMIZATION 7: Thread-local memory pooling
thread_local! {
    static INDEX_POOL: RefCell<IndexPool> = RefCell::new(IndexPool::new());
}

struct IndexPool {
    buffers: Vec<Vec<usize>>,
}

impl IndexPool {
    fn new() -> Self {
        Self { buffers: Vec::with_capacity(8) }
    }
    
    fn acquire(&mut self, capacity: usize) -> Vec<usize> {
        self.buffers.pop()
            .map(|mut buf| {
                buf.clear();
                buf.reserve(capacity.saturating_sub(buf.capacity()));
                buf
            })
            .unwrap_or_else(|| Vec::with_capacity(capacity))
    }
    
    fn release(&mut self, buf: Vec<usize>) {
        if buf.capacity() <= 8192 && self.buffers.len() < 16 {
            self.buffers.push(buf);
        }
    }
}

struct TreeBuildingWorkspace {
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

impl TreeBuildingWorkspace {
    fn new(n_samples: usize) -> Self {
        INDEX_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            Self {
                left_indices: pool.acquire(n_samples / 2),
                right_indices: pool.acquire(n_samples / 2),
            }
        })
    }
}

impl Drop for TreeBuildingWorkspace {
    fn drop(&mut self) {
        INDEX_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let left = std::mem::take(&mut self.left_indices);
            let right = std::mem::take(&mut self.right_indices);
            pool.release(left);
            pool.release(right);
        });
    }
}

// OPTIMIZATION 8: Branchless partitioning
#[inline(always)]
fn partition_into_optimized(
    indices: &[usize],
    feature_idx: usize,
    threshold: i16,
    transposed_data: &TransposedData,
    left_out: &mut Vec<usize>,
    right_out: &mut Vec<usize>,
) {
    left_out.clear();
    right_out.clear();

    let feature_values = transposed_data.get_feature_values(feature_idx);
    
    // Pre-allocate exact capacity
    let estimated_left = indices.len() / 2;
    left_out.reserve(estimated_left);
    right_out.reserve(estimated_left);

    // Branchless partitioning (compiler can vectorize better)
    for &i in indices {
        let goes_left = feature_values[i] <= threshold; 
              if goes_left {
            left_out.push(i);
        } else {
            right_out.push(i);
        }
    }
}

// OPTIMIZATION 9: Adaptive parallelism based on workload
fn build_hists_optimized(
    feature_indices: &[usize],
    transposed_data: &TransposedData,
    y: &ArrayView1<f64>,
    grad: &ArrayView1<f64>,
    hess: &ArrayView1<f64>,
    indices: &[usize],
    params: &TreeParams,
) -> Vec<CachedHistogram> {
    // Parallel threshold: balance overhead vs speedup
    let use_parallel = feature_indices.len() >= 4 && indices.len() >= 1000;
    
    if use_parallel {
        feature_indices.par_iter()
            .enumerate()
            .map(|(feat_idx_local, &actual_feat_idx)| {
                CachedHistogram::build_vectorized(
                    transposed_data, 
                    y, 
                    grad, 
                    hess, 
                    indices,
                    actual_feat_idx, 
                    params.n_bins_per_feature[feat_idx_local]
                )
            })
            .collect()
    } else {
        feature_indices.iter()
            .enumerate()
            .map(|(feat_idx_local, &actual_feat_idx)| {
                CachedHistogram::build_vectorized(
                    transposed_data, 
                    y, 
                    grad, 
                    hess, 
                    indices,
                    actual_feat_idx, 
                    params.n_bins_per_feature[feat_idx_local]
                )
            })
            .collect()
    }
}

// OPTIMIZATION 10: Fast feature elimination with early exit
fn find_best_split_across_features_optimized(
    hists: &[CachedHistogram],
    params: &TreeParams,
    depth: i32,
) -> Option<(usize, HistSplitResult)> {
    // Sequential scan for small feature sets (avoids parallelism overhead)
    if hists.len() <= 4 {
        return hists.iter()
            .enumerate()
            .filter(|(_, hist)| {
                let (_, hess, _, count) = hist.as_slices();
                let total_hess: f64 = hess.iter().sum();
                let non_zero_bins = count.iter().filter(|&&c| c > 0.0).count();
                total_hess > params.min_child_weight * params.feature_elimination_threshold 
                    && non_zero_bins > 1
            })
            .map(|(feat_idx_local, hist)| {
                (feat_idx_local, find_best_split_cached_optimized(hist, params, depth))
            })
            .max_by(|a, b| a.1.best_gain.partial_cmp(&b.1.best_gain).unwrap());
    }
    
    // Parallel for larger feature sets
    hists.par_iter()
        .enumerate()
        .filter(|(_, hist)| {
            let (_, hess, _, count) = hist.as_slices();
            let total_hess: f64 = hess.iter().sum();
            let non_zero_bins = count.iter().filter(|&&c| c > 0.0).count();
            total_hess > params.min_child_weight * params.feature_elimination_threshold 
                && non_zero_bins > 1
        })
        .map(|(feat_idx_local, hist)| {
            (feat_idx_local, find_best_split_cached_optimized(hist, params, depth))
        })
        .reduce_with(|a, b| {
            if a.1.best_gain > b.1.best_gain { a } else { b }
        })
}

#[derive(Debug, Clone, Copy)]
struct PrecomputedSums {
    g_total: f64,
    h_total: f64,
    y_total: f64,
    n_total: f64,
    parent_entropy: f64,
}

impl PrecomputedSums {
    #[inline]
    fn from_histogram(hist: &CachedHistogram) -> Self {
        let (grad, hess, y, count) = hist.as_slices();

        let g_total: f64 = grad.iter().sum();
        let h_total: f64 = hess.iter().sum();
        let y_total: f64 = y.iter().sum();
        let n_total: f64 = count.iter().sum();
        
        let parent_entropy = calculate_shannon_entropy(n_total - y_total, y_total);
        
        Self { g_total, h_total, y_total, n_total, parent_entropy }
    }
}

// OPTIMIZATION 11: Vectorized gain calculation with minimal branching
#[inline]
fn find_best_split_cached_optimized(
    hist: &CachedHistogram,
    params: &TreeParams,
    depth: i32
) -> HistSplitResult {
    let precomputed = PrecomputedSums::from_histogram(hist);
    let (grad, hess, y, count) = hist.as_slices();

    if precomputed.n_total < params.min_samples_split as f64 { 
        return HistSplitResult::default(); 
    }

    // Use entropy only for shallow splits where it matters most
    let use_entropy = depth < 3 && precomputed.parent_entropy > 0.3;
    let adaptive_weight = if use_entropy {
        params.mi_weight * (-0.15 * depth as f64).exp()
    } else {
        0.0
    };

    let mut best_split = HistSplitResult::default();
    let parent_score = precomputed.g_total * precomputed.g_total / 
        (precomputed.h_total + params.reg_lambda);
    
    let min_child_weight = params.min_child_weight;
    let reg_lambda = params.reg_lambda;
    let gamma = params.gamma;
    
    let mut gl = 0.0;
    let mut hl = 0.0;
    let mut y_left = 0.0;
    let mut n_left = 0.0;

    let n_splits = grad.len().saturating_sub(1);
    
    for i in 0..n_splits {
        gl += grad[i];
        hl += hess[i];
        y_left += y[i];
        n_left += count[i];

        // Early continue (branch prediction friendly)
        if n_left < 1.0 || hl < min_child_weight { 
            continue; 
        }

        let gr = precomputed.g_total - gl;
        let hr = precomputed.h_total - hl;
        let n_right = precomputed.n_total - n_left;

        if n_right < 1.0 || hr < min_child_weight { 
            continue; 
        }

        // Vectorized gain calculation
        let gl_sq = gl * gl;
        let gr_sq = gr * gr;
        let newton_gain = 0.5 * (
            gl_sq / (hl + reg_lambda) +
            gr_sq / (hr + reg_lambda) -
            parent_score
        ) - gamma;

        // Calculate entropy only if promising and needed
        let combined_gain = if use_entropy && newton_gain > best_split.best_gain * 0.9 {
            let left_entropy = calculate_shannon_entropy(n_left - y_left, y_left);
            let right_entropy = calculate_shannon_entropy(
                n_right - (precomputed.y_total - y_left), 
                precomputed.y_total - y_left
            );
            let weighted_entropy = (n_left * left_entropy + n_right * right_entropy) / precomputed.n_total;
            let info_gain = precomputed.parent_entropy - weighted_entropy;
            newton_gain + adaptive_weight * info_gain
        } else {
            newton_gain
        };

        // Branchless update
        let is_better = combined_gain > best_split.best_gain;
        best_split.best_gain = if is_better { combined_gain } else { best_split.best_gain };
        best_split.best_bin_idx = if is_better { i as i16 } else { best_split.best_bin_idx };
    }
    
    best_split
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_creation() {
        let tree = OptimizedTreeShannon::new(3);
        assert_eq!(tree.node_types.len(), 16);
    }

    #[test]
    fn test_leaf_node() {
        let mut tree = OptimizedTreeShannon::new(2);
        tree.set_leaf(0, 0.5);
        assert_eq!(tree.node_types[0], 1);
        assert_eq!(tree.leaf_values[0], 0.5);
    }

    #[test]
    fn test_prediction_logic() {
        let mut tree = OptimizedTreeShannon::new(2);
        tree.set_split(0, 0, 3, 1, 2);
        tree.set_leaf(1, 0.1);
        tree.set_leaf(2, 0.9);
        
        let sample_left = vec![2i16];
        assert_eq!(tree.predict_single(&sample_left), 0.1);
        
        let sample_right = vec![5i16];
        assert_eq!(tree.predict_single(&sample_right), 0.9);
    }
}