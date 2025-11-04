// Decision tree implementation with histogram-based splitting
// Uses Shannon entropy + gradient info for splits

use rayon::prelude::*;
use ndarray::ArrayView1;
use crate::metrics::calculate_shannon_entropy;
use crate::optimized_data::{TransposedData, CachedHistogram};
use crate::precision::AdaptiveCompute;
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct HistSplitResult {
    pub best_gain: f64,
    pub best_bin_idx: i32,
}

impl Default for HistSplitResult {
    fn default() -> Self {
        Self {
            best_gain: f64::NEG_INFINITY,
            best_bin_idx: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Node {
    Uninitialized,
    Leaf { value: f64 },  // leaf nodes store prediction values
    Split {
        feature: usize,  // which feature to split on
        threshold: i32,  // binned threshold value
        left_child: usize,  // index in nodes array
        right_child: usize,
    },
}

#[derive(Debug, Clone)]
pub struct OptimizedTreeShannon {
    max_depth: usize,
    nodes: Vec<Node>,  // flat array representation of tree
    pub feature_indices: Vec<usize>,  // which features this tree uses
}

#[derive(Debug, Clone)]
pub struct TreeParams {
    pub min_samples_split: usize,
    pub min_child_weight: f64,
    pub reg_lambda: f64,
    pub gamma: f64,
    pub mi_weight: f64,
    pub n_bins_per_feature: Vec<usize>,
}

struct TreeBuildingWorkspace {
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

impl TreeBuildingWorkspace {
    fn new(n_samples: usize) -> Self {
        Self {
            left_indices: Vec::with_capacity(n_samples),
            right_indices: Vec::with_capacity(n_samples),
        }
    }
}

struct SplitTask {
    node_index: usize,
    sample_indices: Arc<Vec<usize>>,
    histogram: Vec<CachedHistogram>,
    depth: usize,
}

impl OptimizedTreeShannon {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            nodes: Vec::new(),
            feature_indices: Vec::new(),
        }
    }

    // main tree building function - uses BFS with histogram subtraction trick
    pub fn fit_optimized(
        &mut self,
        transposed_data: &TransposedData,
        y: &[f64],
        grad: &[f64],  // gradients from loss function
        hess: &[f64],  // hessians (second derivatives)
        sample_indices: &[usize],  // which samples to use (subsampling)
        feature_indices: &[usize],  // which features to consider
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
        self.nodes = vec![Node::Uninitialized; max_nodes];  // preallocate

        let mut workspace = TreeBuildingWorkspace::new(sample_indices.len());

        // build histograms for root node
        let root_hists = build_hists(
            &self.feature_indices, 
            transposed_data, 
            &y_view, 
            &grad_view, 
            &hess_view,
            sample_indices, 
            params
        );

        // BFS queue for building tree level by level
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
            let (g_total, h_total): (f64, f64) = {
                let (g_slice, h_slice, _, _) = task.histogram[0].as_slices();
                (g_slice.iter().sum(), h_slice.iter().sum())
            };

            // Check stopping conditions
            if task.depth >= self.max_depth 
                || n_samples < params.min_samples_split 
                || h_total < params.min_child_weight 
            {
                self.nodes[task.node_index] = Node::Leaf { 
                    value: -g_total / (h_total + params.reg_lambda) 
                };
                continue;
            }

            // find best split across all features
            let best_split = find_best_split_across_features(
                &task.histogram, 
                params, 
                task.depth as i32
            );

            if best_split.is_none() || best_split.unwrap().1.best_gain <= 1e-6 {
                self.nodes[task.node_index] = Node::Leaf { 
                    value: -g_total / (h_total + params.reg_lambda) 
                };
                continue;
            }

            let (best_feature_local_idx, split_info) = best_split.unwrap();
            let best_feature_global_idx = self.feature_indices[best_feature_local_idx];

            // split samples into left and right children
            partition_into(
                &task.sample_indices, 
                best_feature_global_idx, 
                split_info.best_bin_idx,
                transposed_data, 
                &mut workspace.left_indices, 
                &mut workspace.right_indices
            );

            if workspace.left_indices.is_empty() || workspace.right_indices.is_empty() {
                self.nodes[task.node_index] = Node::Leaf { 
                    value: -g_total / (h_total + params.reg_lambda) 
                };
                continue;
            }

            // Build histograms for children (histogram subtraction trick)
            // histogram subtraction trick - only build hist for smaller child
            let (left_hists, right_hists) = if workspace.left_indices.len() < workspace.right_indices.len() {
                let smaller_hists = build_hists(
                    &self.feature_indices, 
                    transposed_data, 
                    &y_view, 
                    &grad_view, 
                    &hess_view,
                    &workspace.left_indices, 
                    params
                );
                let larger_hists = subtract_hists(&task.histogram, &smaller_hists);
                (smaller_hists, larger_hists)
            } else {
                let smaller_hists = build_hists(
                    &self.feature_indices, 
                    transposed_data, 
                    &y_view, 
                    &grad_view, 
                    &hess_view,
                    &workspace.right_indices, 
                    params
                );
                let larger_hists = subtract_hists(&task.histogram, &smaller_hists);
                (larger_hists, smaller_hists)
            };

            let left_child_index = next_node_in_vec;
            let right_child_index = next_node_in_vec + 1;

            if right_child_index >= self.nodes.len() { 
                continue; 
            }

            self.nodes[task.node_index] = Node::Split {
                feature: best_feature_global_idx,
                threshold: split_info.best_bin_idx,
                left_child: left_child_index,
                right_child: right_child_index,
            };
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

    pub fn predict_single(&self, x_binned_row: &[i32]) -> f64 {
        let mut current_node_index = 0;
        loop {
            match self.nodes.get(current_node_index) {
                Some(Node::Leaf { value }) => return *value,
                Some(Node::Split { feature, threshold, left_child, right_child }) => {
                    let feature_value = x_binned_row.get(*feature).copied().unwrap_or(0);
                    if feature_value <= *threshold {
                        current_node_index = *left_child;
                    } else {
                        current_node_index = *right_child;
                    }
                }
                _ => return 0.0,
            }
        }
    }

    pub fn predict_from_transposed(&self, transposed_data: &TransposedData, sample_idx: usize) -> f64 {
        let mut current_node_index = 0;
        loop {
            match self.nodes.get(current_node_index) {
                Some(Node::Leaf { value }) => return *value,
                Some(Node::Split { feature, threshold, left_child, right_child }) => {
                    let feature_value = if *feature < transposed_data.n_features 
                        && sample_idx < transposed_data.n_samples 
                    {
                        transposed_data.features[[*feature, sample_idx]]
                    } else {
                        0
                    };
                    if feature_value <= *threshold {
                        current_node_index = *left_child;
                    } else {
                        current_node_index = *right_child;
                    }
                }
                _ => return 0.0,
            }
        }
    }

    pub fn count_splits_on_features(&self, features: &[usize]) -> usize {
        let feature_set: HashSet<_> = features.iter().copied().collect();
        self.nodes.iter().filter(|node| {
            if let Node::Split { feature, .. } = node {
                feature_set.contains(feature)
            } else { 
                false 
            }
        }).count()
    }

    pub fn count_total_splits(&self) -> usize {
        self.nodes.iter()
            .filter(|node| matches!(node, Node::Split { .. }))
            .count()
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
        for node in &self.nodes {
            if let Node::Split { feature, .. } = node {
                if !features.contains(feature) {
                    features.push(*feature);
                }
            }
        }
        features
    }
}

fn partition_into(
    indices: &[usize],
    feature_idx: usize,
    threshold: i32,
    transposed_data: &TransposedData,
    left_out: &mut Vec<usize>,
    right_out: &mut Vec<usize>,
) {
    left_out.clear();
    right_out.clear();

    let feature_values = transposed_data.get_feature_values(feature_idx);

    for &i in indices {
        if feature_values[i] <= threshold {
            left_out.push(i);
        } else {
            right_out.push(i);
        }
    }
}

fn build_hists(
    feature_indices: &[usize],
    transposed_data: &TransposedData,
    y: &ArrayView1<f64>,
    grad: &ArrayView1<f64>,
    hess: &ArrayView1<f64>,
    indices: &[usize],
    params: &TreeParams,
) -> Vec<CachedHistogram> {
    // OPTIMIZATION 3: Only parallelize when beneficial (saves 15-25% for small feature sets)
    let use_parallel = feature_indices.len() > 20 || indices.len() > 5000;
    
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

fn subtract_hists(
    parent_hists: &[CachedHistogram],
    sibling_hists: &[CachedHistogram],
) -> Vec<CachedHistogram> {
    parent_hists.par_iter()
        .zip(sibling_hists)
        .map(|(parent, sibling)| parent.subtract(sibling))
        .collect()
}

fn find_best_split_across_features(
    hists: &[CachedHistogram],
    params: &TreeParams,
    depth: i32,
) -> Option<(usize, HistSplitResult)> {
    hists.par_iter()
        .enumerate()
        .map(|(feat_idx_local, hist)| {
            (feat_idx_local, find_best_split_cached(hist, params, depth))
        })
        .reduce_with(|a, b| {
            if a.1.best_gain > b.1.best_gain { 
                a 
            } else { 
                b 
            }
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
    fn from_histogram(hist: &CachedHistogram) -> Self {
        let (grad, hess, y, count) = hist.as_slices();

        // Use progressive precision for gradient/hessian accumulation
        let g_total = AdaptiveCompute::gradient_accumulate_progressive(grad);
        let h_total = AdaptiveCompute::gradient_accumulate_progressive(hess);
        let y_total: f64 = y.iter().sum();
        let n_total: f64 = count.iter().sum();
        
        let parent_entropy = calculate_shannon_entropy(n_total - y_total, y_total);
        
        Self {
            g_total,
            h_total,
            y_total,
            n_total,
            parent_entropy,
        }
    }
}

// find best split point for a single feature using its histogram
fn find_best_split_cached(
    hist: &CachedHistogram,
    params: &TreeParams,
    depth: i32
) -> HistSplitResult {
    let precomputed = PrecomputedSums::from_histogram(hist);
    let (grad, hess, y, count) = hist.as_slices();

    if precomputed.n_total < params.min_samples_split as f64 { 
        return HistSplitResult::default(); 
    }

    // OPTIMIZATION 1: Skip entropy at deep levels (saves 10-15%)
    let use_entropy = depth < 4 && precomputed.parent_entropy > 0.5;
    let adaptive_weight = if use_entropy {
        params.mi_weight * (-0.1 * depth as f64).exp()
    } else {
        0.0
    };

    let mut best_split = HistSplitResult::default();
    let parent_score = precomputed.g_total * precomputed.g_total / (precomputed.h_total + params.reg_lambda);

    let mut gl = 0.0;
    let mut hl = 0.0;
    let mut y_left = 0.0;
    let mut n_left = 0.0;

    for i in 0..(grad.len().saturating_sub(1)) {
        gl += grad[i];
        hl += hess[i];
        y_left += y[i];
        n_left += count[i];

        if n_left < 1.0 || hl < params.min_child_weight { 
            continue; 
        }

        let gr = precomputed.g_total - gl;
        let hr = precomputed.h_total - hl;
        let n_right = precomputed.n_total - n_left;

        if n_right < 1.0 || hr < params.min_child_weight { 
            continue; 
        }

        // standard xgboost gain formula
        let newton_gain = 0.5 * (
            gl.powi(2) / (hl + params.reg_lambda) +
            gr.powi(2) / (hr + params.reg_lambda) -
            parent_score
        ) - params.gamma;

        // OPTIMIZATION 1: Only calculate entropy if needed and gain is promising
        let combined_gain = if use_entropy && newton_gain > best_split.best_gain * 0.5 {
            let left_entropy = calculate_shannon_entropy(n_left - y_left, y_left);
            let right_entropy = calculate_shannon_entropy(n_right - (precomputed.y_total - y_left), precomputed.y_total - y_left);
            let weighted_entropy = (n_left * left_entropy + n_right * right_entropy) / precomputed.n_total;
            let info_gain = precomputed.parent_entropy - weighted_entropy;
            newton_gain + adaptive_weight * info_gain
        } else {
            newton_gain
        };

        if combined_gain > best_split.best_gain {
            best_split.best_gain = combined_gain;
            best_split.best_bin_idx = i as i32;
        }
    }
    
    best_split
}
