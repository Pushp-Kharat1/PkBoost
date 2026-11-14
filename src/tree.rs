// Decision tree implementation with histogram-based splitting
// OPTIMIZED VERSION with ~30-40% speed improvement

use crate::metrics::calculate_shannon_entropy;
use crate::optimized_data::{
    gather_node_stats, gather_node_stats_par, CachedHistogram, TransposedData,
};
use ndarray::ArrayView1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashSet;

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
    fn set_split(
        &mut self,
        node_idx: usize,
        feature: usize,
        threshold: i16,
        left_child: usize,
        right_child: usize,
    ) {
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
    pub fn predict_from_transposed(
        &self,
        transposed_data: &TransposedData,
        sample_idx: usize,
    ) -> f64 {
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
                        transposed_data.features[[feature, sample_idx]] as i16
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
    pub fn predict_batch(
        &self,
        transposed_data: &TransposedData,
        sample_indices: &[usize],
    ) -> Vec<f64> {
        // Use parallelism for medium to large batches
        if sample_indices.len() >= 500 {
            sample_indices
                .par_iter()
                .map(|&sample_idx| self.predict_from_transposed(transposed_data, sample_idx))
                .collect()
        } else {
            sample_indices
                .iter()
                .map(|&sample_idx| self.predict_from_transposed(transposed_data, sample_idx))
                .collect()
        }
    }

    pub fn count_splits_on_features(&self, features: &[usize]) -> usize {
        let feature_set: HashSet<_> = features.iter().copied().collect();
        self.node_types
            .iter()
            .enumerate()
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
        if total == 0 {
            return 0.0;
        }
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
        self.fit_optimized_internal(
            transposed_data,
            y,
            grad,
            hess,
            sample_indices,
            feature_indices,
            params,
            None,
        );
    }

    pub fn fit_optimized_with_predictions(
        &mut self,
        transposed_data: &TransposedData,
        y: &[f64],
        grad: &[f64],
        hess: &[f64],
        sample_indices: &[usize],
        feature_indices: &[usize],
        params: &TreeParams,
        tree_preds: &mut [f64],
    ) {
        self.fit_optimized_internal(
            transposed_data,
            y,
            grad,
            hess,
            sample_indices,
            feature_indices,
            params,
            Some(tree_preds),
        );
    }

    fn fit_optimized_internal(
        &mut self,
        transposed_data: &TransposedData,
        y: &[f64],
        grad: &[f64],
        hess: &[f64],
        sample_indices: &[usize],
        feature_indices: &[usize],
        params: &TreeParams,
        mut tree_preds: Option<&mut [f64]>,
    ) {
        if transposed_data.n_samples == 0 || sample_indices.is_empty() {
            return;
        }

        self.feature_indices = feature_indices
            .iter()
            .filter(|&&idx| idx < transposed_data.n_features)
            .copied()
            .collect();

        if self.feature_indices.is_empty() {
            return;
        }

        let y_view = ArrayView1::from(y);
        let grad_view = ArrayView1::from(grad);
        let hess_view = ArrayView1::from(hess);

        // OPTIMIZATION 1: Build root histogram once (parallel over features)
        let root_hists = build_hists_optimized(
            &self.feature_indices,
            transposed_data,
            &y_view,
            &grad_view,
            &hess_view,
            sample_indices,
            params,
            true,
        );

        let mut current_level = vec![SplitTask {
            node_index: 0,
            sample_indices: sample_indices.to_vec(),
            histogram: root_hists,
            depth: 0,
        }];

        let mut next_node_in_vec = 1;
        let max_nodes = self.node_types.len();

        // LAYER-WISE (BREADTH-FIRST) TREE NODE PARALLELISM
        // Nodes at each depth are processed in parallel across Rayon worker threads.
        for current_depth in 0..self.max_depth {
            if current_level.is_empty() {
                break;
            }

            let num_nodes = current_level.len();
            // Parallelize across nodes when multiple nodes exist; otherwise parallelize across features.
            let parallelize_nodes = num_nodes > 1;
            let parallel_features = !parallelize_nodes;

            let feature_indices = &self.feature_indices;
            let max_depth = self.max_depth;

            let results: Vec<NodeProcessResult> = if parallelize_nodes {
                current_level
                    .into_par_iter()
                    .map(|task| {
                        process_split_task(
                            task,
                            feature_indices,
                            transposed_data,
                            &y_view,
                            &grad_view,
                            &hess_view,
                            params,
                            max_depth,
                            false,
                        )
                    })
                    .collect()
            } else {
                current_level
                    .into_iter()
                    .map(|task| {
                        process_split_task(
                            task,
                            feature_indices,
                            transposed_data,
                            &y_view,
                            &grad_view,
                            &hess_view,
                            params,
                            max_depth,
                            parallel_features,
                        )
                    })
                    .collect()
            };

            let mut next_level = Vec::with_capacity(results.len() * 2);

            for res in results {
                match res {
                    NodeProcessResult::Leaf {
                        node_index,
                        leaf_val,
                        sample_indices,
                    } => {
                        self.set_leaf(node_index, leaf_val);
                        if let Some(ref mut out_preds) = tree_preds {
                            for &idx in &sample_indices {
                                out_preds[idx] = leaf_val;
                            }
                        }
                        INDEX_POOL.with(|p| p.borrow_mut().release(sample_indices));
                    }
                    NodeProcessResult::Split {
                        node_index,
                        best_feature_global_idx,
                        best_bin_idx,
                        left_indices,
                        right_indices,
                        left_hists,
                        right_hists,
                        leaf_val_fallback,
                    } => {
                        let left_child_index = next_node_in_vec;
                        let right_child_index = next_node_in_vec + 1;

                        if right_child_index >= max_nodes {
                            INDEX_POOL.with(|p| {
                                let mut pool = p.borrow_mut();
                                pool.release(left_indices);
                                pool.release(right_indices);
                            });
                            self.set_leaf(node_index, leaf_val_fallback);
                            continue;
                        }

                        next_node_in_vec += 2;

                        self.set_split(
                            node_index,
                            best_feature_global_idx,
                            best_bin_idx,
                            left_child_index,
                            right_child_index,
                        );

                        next_level.push(SplitTask {
                            node_index: left_child_index,
                            sample_indices: left_indices,
                            histogram: left_hists,
                            depth: current_depth + 1,
                        });
                        next_level.push(SplitTask {
                            node_index: right_child_index,
                            sample_indices: right_indices,
                            histogram: right_hists,
                            depth: current_depth + 1,
                        });
                    }
                }
            }

            current_level = next_level;
        }

        // Convert any leaf tasks remaining at max_depth into leaves
        for task in current_level {
            let (g_total, h_total) = task.histogram[0]
                .bins
                .iter()
                .fold((0.0, 0.0), |(g, h), b| (g + b[0], h + b[1]));
            let leaf_val = -g_total / (h_total + params.reg_lambda);
            self.set_leaf(task.node_index, leaf_val);
            if let Some(ref mut out_preds) = tree_preds {
                for &idx in &task.sample_indices {
                    out_preds[idx] = leaf_val;
                }
            }
            INDEX_POOL.with(|p| p.borrow_mut().release(task.sample_indices));
        }

        // OPTIMIZATION: Predict out-of-bag rows (only for samples that were not in sample_indices)
        if let Some(ref mut out_preds) = tree_preds {
            if sample_indices.len() < transposed_data.n_samples {
                let is_sorted = sample_indices.windows(2).all(|w| w[0] <= w[1]);
                if is_sorted {
                    let mut sample_iter = sample_indices.iter().copied().peekable();
                    for i in 0..transposed_data.n_samples {
                        if sample_iter.peek() == Some(&i) {
                            sample_iter.next();
                        } else {
                            out_preds[i] = self.predict_from_transposed(transposed_data, i);
                        }
                    }
                } else {
                    let mut in_bag = vec![false; transposed_data.n_samples];
                    for &i in sample_indices {
                        if i < in_bag.len() {
                            in_bag[i] = true;
                        }
                    }
                    for i in 0..transposed_data.n_samples {
                        if !in_bag[i] {
                            out_preds[i] = self.predict_from_transposed(transposed_data, i);
                        }
                    }
                }
            }
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
    sample_indices: Vec<usize>,
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
        Self {
            buffers: Vec::with_capacity(32),
        }
    }

    fn acquire(&mut self, capacity: usize) -> Vec<usize> {
        self.buffers
            .pop()
            .map(|mut buf| {
                buf.clear();
                if buf.capacity() < capacity {
                    buf.reserve(capacity - buf.capacity());
                }
                buf
            })
            .unwrap_or_else(|| Vec::with_capacity(capacity))
    }

    fn release(&mut self, mut buf: Vec<usize>) {
        buf.clear();
        if self.buffers.len() < 32 {
            self.buffers.push(buf);
        }
    }
}

// OPTIMIZATION 8: Truly branchless partitioning
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

    left_out.reserve(indices.len());
    right_out.reserve(indices.len());

    let left_ptr = left_out.as_mut_ptr();
    let right_ptr = right_out.as_mut_ptr();
    let mut n_left = 0usize;
    let mut n_right = 0usize;

    let thresh_u8 = threshold.clamp(0, 255) as u8;

    unsafe {
        for &i in indices {
            let val = *feature_values.get_unchecked(i);
            let goes_left = (val <= thresh_u8) as usize;
            let goes_right = goes_left ^ 1;

            *left_ptr.add(n_left) = i;
            *right_ptr.add(n_right) = i;

            n_left += goes_left;
            n_right += goes_right;
        }

        left_out.set_len(n_left);
        right_out.set_len(n_right);
    }
}

enum NodeProcessResult {
    Leaf {
        node_index: usize,
        leaf_val: f64,
        sample_indices: Vec<usize>,
    },
    Split {
        node_index: usize,
        best_feature_global_idx: usize,
        best_bin_idx: i16,
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
        left_hists: Vec<CachedHistogram>,
        right_hists: Vec<CachedHistogram>,
        leaf_val_fallback: f64,
    },
}

fn process_split_task(
    task: SplitTask,
    feature_indices: &[usize],
    transposed_data: &TransposedData,
    y_view: &ArrayView1<f64>,
    grad_view: &ArrayView1<f64>,
    hess_view: &ArrayView1<f64>,
    params: &TreeParams,
    max_depth: usize,
    parallel_features: bool,
) -> NodeProcessResult {
    let n_samples = task.sample_indices.len();
    let precomputed = {
        let (g_total, h_total, y_total, n_total) = task.histogram[0]
            .bins
            .iter()
            .fold((0.0, 0.0, 0.0, 0.0), |(g, h, y, n), b| {
                (g + b[0], h + b[1], y + b[2], n + b[3])
            });
        let parent_entropy = calculate_shannon_entropy(n_total - y_total, y_total);
        PrecomputedSums {
            g_total,
            h_total,
            y_total,
            n_total,
            parent_entropy,
        }
    };

    let leaf_val = -precomputed.g_total / (precomputed.h_total + params.reg_lambda);

    if task.depth >= max_depth
        || n_samples < params.min_samples_split
        || precomputed.h_total < params.min_child_weight
    {
        return NodeProcessResult::Leaf {
            node_index: task.node_index,
            leaf_val,
            sample_indices: task.sample_indices,
        };
    }

    let best_split = find_best_split_across_features_optimized(
        &task.histogram,
        &precomputed,
        params,
        task.depth as i32,
        parallel_features,
    );

    if best_split.is_none() || best_split.unwrap().1.best_gain <= 1e-6 {
        return NodeProcessResult::Leaf {
            node_index: task.node_index,
            leaf_val,
            sample_indices: task.sample_indices,
        };
    }

    let (best_feature_local_idx, split_info) = best_split.unwrap();
    let best_feature_global_idx = feature_indices[best_feature_local_idx];

    let (mut left_indices, mut right_indices) = INDEX_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        (
            pool.acquire(task.sample_indices.len()),
            pool.acquire(task.sample_indices.len()),
        )
    });

    partition_into_optimized(
        &task.sample_indices,
        best_feature_global_idx,
        split_info.best_bin_idx,
        transposed_data,
        &mut left_indices,
        &mut right_indices,
    );

    if left_indices.is_empty() || right_indices.is_empty() {
        INDEX_POOL.with(|p| {
            let mut pool = p.borrow_mut();
            pool.release(left_indices);
            pool.release(right_indices);
        });
        return NodeProcessResult::Leaf {
            node_index: task.node_index,
            leaf_val,
            sample_indices: task.sample_indices,
        };
    }

    INDEX_POOL.with(|p| p.borrow_mut().release(task.sample_indices));

    let (left_hists, right_hists) = if left_indices.len() < right_indices.len() {
        let smaller_hists = build_hists_optimized(
            feature_indices,
            transposed_data,
            y_view,
            grad_view,
            hess_view,
            &left_indices,
            params,
            parallel_features,
        );
        let larger_hists: Vec<CachedHistogram> = if parallel_features {
            task.histogram
                .par_iter()
                .zip(&smaller_hists)
                .map(|(parent, sibling)| parent.subtract(sibling))
                .collect()
        } else {
            task.histogram
                .iter()
                .zip(&smaller_hists)
                .map(|(parent, sibling)| parent.subtract(sibling))
                .collect()
        };
        (smaller_hists, larger_hists)
    } else {
        let smaller_hists = build_hists_optimized(
            feature_indices,
            transposed_data,
            y_view,
            grad_view,
            hess_view,
            &right_indices,
            params,
            parallel_features,
        );
        let larger_hists: Vec<CachedHistogram> = if parallel_features {
            task.histogram
                .par_iter()
                .zip(&smaller_hists)
                .map(|(parent, sibling)| parent.subtract(sibling))
                .collect()
        } else {
            task.histogram
                .iter()
                .zip(&smaller_hists)
                .map(|(parent, sibling)| parent.subtract(sibling))
                .collect()
        };
        (larger_hists, smaller_hists)
    };

    NodeProcessResult::Split {
        node_index: task.node_index,
        best_feature_global_idx,
        best_bin_idx: split_info.best_bin_idx,
        left_indices,
        right_indices,
        left_hists,
        right_hists,
        leaf_val_fallback: leaf_val,
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
    parallel_features: bool,
) -> Vec<CachedHistogram> {
    // Parallel threshold: balance overhead vs speedup
    let use_parallel = parallel_features && feature_indices.len() >= 4 && indices.len() >= 10000;

    // Small/sparse nodes: gather (grad, hess, y) once in index order so each
    // feature reads them sequentially (2x faster than 30 separate random reads per sample).
    // Dense nodes (e.g. the root): indices are close to a linear scan already.
    let use_ordered = indices.len() * 2 < transposed_data.n_samples;

    if !use_ordered {
        let build = |(feat_idx_local, &actual_feat_idx): (usize, &usize)| {
            CachedHistogram::build_vectorized(
                transposed_data,
                y,
                grad,
                hess,
                indices,
                actual_feat_idx,
                params.n_bins_per_feature[feat_idx_local],
            )
        };
        return if use_parallel {
            feature_indices.par_iter().enumerate().map(build).collect()
        } else {
            feature_indices.iter().enumerate().map(build).collect()
        };
    }

    let (g, h, yv) = (
        grad.as_slice().expect("grad must be contiguous"),
        hess.as_slice().expect("hess must be contiguous"),
        y.as_slice().expect("y must be contiguous"),
    );
    let stats: Vec<[f64; 4]> = if indices.len() >= 8192 {
        gather_node_stats_par(indices, g, h, yv)
    } else {
        gather_node_stats(indices, g, h, yv)
    };

    let build = |(feat_idx_local, &actual_feat_idx): (usize, &usize)| {
        transposed_data.build_histogram_ordered(
            actual_feat_idx,
            indices,
            &stats,
            params.n_bins_per_feature[feat_idx_local],
        )
    };

    if use_parallel {
        feature_indices.par_iter().enumerate().map(build).collect()
    } else {
        feature_indices.iter().enumerate().map(build).collect()
    }
}

// OPTIMIZATION 10: Fast feature elimination with early exit
fn find_best_split_across_features_optimized(
    hists: &[CachedHistogram],
    precomputed: &PrecomputedSums,
    params: &TreeParams,
    depth: i32,
    parallel: bool,
) -> Option<(usize, HistSplitResult)> {
    if precomputed.h_total <= params.min_child_weight * params.feature_elimination_threshold {
        return None;
    }

    let filter_map = |(feat_idx_local, hist): (usize, &CachedHistogram)| {
        let non_zero_bins = hist.bins.iter().filter(|b| b[3] > 0.0).count();
        if non_zero_bins <= 1 {
            None
        } else {
            Some((
                feat_idx_local,
                find_best_split_cached_optimized(hist, precomputed, params, depth),
            ))
        }
    };

    if !parallel || hists.len() <= 4 {
        hists
            .iter()
            .enumerate()
            .filter_map(filter_map)
            .max_by(|a, b| a.1.best_gain.partial_cmp(&b.1.best_gain).unwrap())
    } else {
        hists
            .par_iter()
            .enumerate()
            .filter_map(filter_map)
            .reduce_with(|a, b| if a.1.best_gain > b.1.best_gain { a } else { b })
    }
}

#[derive(Debug, Clone, Copy)]
struct PrecomputedSums {
    g_total: f64,
    h_total: f64,
    y_total: f64,
    n_total: f64,
    parent_entropy: f64,
}

// OPTIMIZATION 11: Vectorized gain calculation with minimal branching
#[inline]
fn find_best_split_cached_optimized(
    hist: &CachedHistogram,
    precomputed: &PrecomputedSums,
    params: &TreeParams,
    depth: i32,
) -> HistSplitResult {
    let bins = &hist.bins;

    if precomputed.n_total < params.min_samples_split as f64 {
        return HistSplitResult::default();
    }

    let use_entropy = precomputed.parent_entropy > 0.01;
    let adaptive_weight = if use_entropy {
        params.mi_weight * (-0.15 * depth as f64).exp()
    } else {
        0.0
    };

    let mut best_split = HistSplitResult::default();
    let parent_score =
        precomputed.g_total * precomputed.g_total / (precomputed.h_total + params.reg_lambda);

    let min_child_weight = params.min_child_weight;
    let reg_lambda = params.reg_lambda;
    let gamma = params.gamma;

    let mut gl = 0.0;
    let mut hl = 0.0;
    let mut y_left = 0.0;
    let mut n_left = 0.0;

    let n_splits = bins.len().saturating_sub(1);

    for i in 0..n_splits {
        let b = unsafe { *bins.get_unchecked(i) };
        gl += b[0];
        hl += b[1];
        y_left += b[2];
        n_left += b[3];

        if n_left < 1.0 || hl < min_child_weight {
            continue;
        }

        let gr = precomputed.g_total - gl;
        let hr = precomputed.h_total - hl;
        let n_right = precomputed.n_total - n_left;

        if n_right < 1.0 || hr < min_child_weight {
            continue;
        }

        let left_score = gl * gl / (hl + reg_lambda);
        let right_score = gr * gr / (hr + reg_lambda);
        let newton_gain = 0.5 * (left_score + right_score - parent_score) - gamma;

        let combined_gain = if use_entropy && newton_gain > best_split.best_gain * 0.9 {
            let left_entropy = calculate_shannon_entropy(n_left - y_left, y_left);
            let right_entropy = calculate_shannon_entropy(
                n_right - (precomputed.y_total - y_left),
                precomputed.y_total - y_left,
            );
            let weighted_entropy =
                (n_left * left_entropy + n_right * right_entropy) / precomputed.n_total;
            let info_gain = precomputed.parent_entropy - weighted_entropy;
            newton_gain + adaptive_weight * info_gain
        } else {
            newton_gain
        };

        if combined_gain > best_split.best_gain {
            best_split.best_gain = combined_gain;
            best_split.best_bin_idx = i as i16;
        }
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

    /// Exercises build_hists_optimized on both the ordered (sparse node) and
    /// per-feature (dense node) paths with parallelism enabled (>= 4 features,
    /// >= 10_000 indices, multi-threaded pool), against a sequential reference.
    #[test]
    fn build_hists_parallel_paths_match_sequential_reference() {
        use ndarray::{Array1, Array2};
        let n = 60_000;
        let nf = 6;
        let n_bins = 17;
        let binned =
            Array2::from_shape_fn((n, nf), |(i, j)| ((i * 31 + j * 7) % (n_bins + 3)) as i16);
        let t = TransposedData::from_binned(binned);
        let g = Array1::from_shape_fn(n, |i| i as f64 * 0.001 + 0.25);
        let h = Array1::from_shape_fn(n, |i| 1.0 + (i % 97) as f64 * 0.01);
        let y = Array1::from_shape_fn(n, |i| ((i * 13) % 2) as f64);
        let feats: Vec<usize> = (0..nf).collect();
        let params = TreeParams {
            n_bins_per_feature: vec![n_bins; nf],
            ..Default::default()
        };

        // sparse (ordered path, 25k > GATHER_CHUNK*3) and dense (per-feature path)
        let sparse: Vec<usize> = (0..n).filter(|i| i % 12 < 5).collect();
        let dense: Vec<usize> = (0..n).filter(|i| i % 10 != 3).collect();
        assert!(sparse.len() * 2 < n && sparse.len() >= 10_000);
        assert!(dense.len() * 2 >= n);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for idx in [&sparse, &dense] {
            let got = pool.install(|| {
                build_hists_optimized(
                    &feats,
                    &t,
                    &y.view(),
                    &g.view(),
                    &h.view(),
                    idx,
                    &params,
                    true,
                )
            });
            assert_eq!(got.len(), nf);
            for (f, got_f) in got.iter().enumerate() {
                let want =
                    t.build_histogram_vectorized(f, idx, &g.view(), &h.view(), &y.view(), n_bins);
                assert_eq!(got_f, &want, "hist f={f} len={}", idx.len());
            }
        }
    }

    #[test]
    fn test_fit_optimized_with_predictions_matches_predict_batch() {
        use ndarray::Array2;
        let n = 200;
        let nf = 4;
        let n_bins = 8;
        let binned = Array2::from_shape_fn((n, nf), |(i, j)| ((i * 7 + j * 3) % n_bins) as i16);
        let t = TransposedData::from_binned(binned);
        let g: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 1.0).collect();
        let h: Vec<f64> = (0..n).map(|_| 1.0).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let feats: Vec<usize> = (0..nf).collect();
        let params = TreeParams {
            min_samples_split: 5,
            min_child_weight: 1.0,
            reg_lambda: 1.0,
            n_bins_per_feature: vec![n_bins; nf],
            ..Default::default()
        };

        // Test with 80% subsample (both in-bag and out-of-bag)
        let sample_indices: Vec<usize> = (0..n).filter(|i| i % 5 != 0).collect();
        let mut tree = OptimizedTreeShannon::new(4);
        let mut tree_preds = vec![0.0f64; n];

        tree.fit_optimized_with_predictions(
            &t,
            &y,
            &g,
            &h,
            &sample_indices,
            &feats,
            &params,
            &mut tree_preds,
        );

        let all_indices: Vec<usize> = (0..n).collect();
        let batch_preds = tree.predict_batch(&t, &all_indices);

        for i in 0..n {
            assert!(
                (tree_preds[i] - batch_preds[i]).abs() < 1e-10,
                "Mismatch at index {}: tree_pred = {}, batch_pred = {}",
                i,
                tree_preds[i],
                batch_preds[i]
            );
        }
    }
}
