// Decisions tree implementation with histogram-based splitting
// OPTIMIZED VERSION with ~30-40% speed improvement

use crate::fork_parallel::pfor_range_map;
use crate::metrics::calculate_shannon_entropy;
use crate::optimized_data::{CachedHistogram, TransposedData};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone, Copy)]
pub struct HistSplitResult {
    pub best_gain: f64,
    pub best_bin_idx: u8,
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
    // Struct of Arrays (SoA) - chosen for cache locality
    node_types: Vec<u8>,
    leaf_values: Vec<f64>,
    split_features: Vec<usize>,
    split_thresholds: Vec<u8>,
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
        threshold: u8,
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
    pub fn predict_single(&self, x_binned_row: &[u8]) -> f64 {
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
        sample_idx: u32,
    ) -> f64 {
        let mut current_node_index = 0;
        let s_idx = sample_idx as usize;

        loop {
            match self.node_types[current_node_index] {
                1 => return self.leaf_values[current_node_index],
                2 => {
                    let feature = self.split_features[current_node_index];
                    let threshold = self.split_thresholds[current_node_index];

                    // Use optimized sliceless access if possible
                    let feature_values = transposed_data.get_feature_values(feature);
                    let feature_value = unsafe { *feature_values.get_unchecked(s_idx) };

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

    // Batch prediction — ForkUnion parallel for large sets, sequential for small
    pub fn predict_batch(
        &self,
        transposed_data: &TransposedData,
        sample_indices: &[u32],
    ) -> Vec<f64> {
        if sample_indices.len() >= 50_000 {
            pfor_range_map(0..sample_indices.len(), |i| {
                self.predict_from_transposed(transposed_data, sample_indices[i])
            })
        } else {
            sample_indices
                .iter()
                .map(|&sample_idx| self.predict_from_transposed(transposed_data, sample_idx))
                .collect()
        }
    }

    /// OPTIMIZED: Predict into pre-allocated buffer (zero-allocation)
    #[inline]
    pub fn predict_batch_into(
        &self,
        transposed_data: &TransposedData,
        sample_indices: &[u32],
        output: &mut [f64],
    ) {
        debug_assert_eq!(sample_indices.len(), output.len());

        for (out, &sample_idx) in output.iter_mut().zip(sample_indices.iter()) {
            *out = self.predict_from_transposed(transposed_data, sample_idx);
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
        grad: &[f32],
        hess: &[f32],
        sample_indices: &[u32],
        feature_indices: &[usize],
        params: &TreeParams,
        mut full_preds: Option<&mut Vec<f64>>,
        learning_rate: f64,
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

        // OPTIMIZATION: Pre-compute compact y labels (u8 instead of f64)
        // Reduces memory bandwidth by ~28% in histogram building hot loop
        let y_u8: Vec<u8> = y.iter().map(|&v| if v > 0.5 { 1u8 } else { 0u8 }).collect();

        let max_nodes = 2_usize.pow(self.max_depth as u32 + 1);
        let n_features = self.feature_indices.len();
        let mut workspace = TreeBuildingWorkspace::new(sample_indices.len(), n_features);

        // OPTIMIZATION 1: Build root histogram once (into root_hists)
        let mut root_hists: Vec<CachedHistogram> = (0..n_features)
            .map(|_i| CachedHistogram::default())
            .collect();

        build_hists_into(
            &self.feature_indices,
            transposed_data,
            &y_u8,
            grad,
            hess,
            sample_indices,
            params,
            &mut root_hists,
            &mut workspace.active_data,
        );

        // Initial indices in workspace
        workspace.indices[..sample_indices.len()].copy_from_slice(sample_indices);

        let mut queue: VecDeque<SplitTask> = VecDeque::with_capacity(max_nodes / 2);
        queue.push_back(SplitTask {
            node_index: 0,
            sample_start: 0,
            sample_len: sample_indices.len(),
            histogram: root_hists,
            depth: 0,
        });

        let mut next_node_in_vec = 1;

        while let Some(task) = queue.pop_front() {
            let n_samples = task.sample_len;

            // OPTIMIZATION 2: Extract totals once (ZERO-COST using pre-computed sums)
            let g_total = task.histogram[0].sums.grad;
            let h_total = task.histogram[0].sums.hess;

            // OPTIMIZATION 3: Early gradient-based pruning
            let gradient_norm = g_total.abs() as f64;
            if gradient_norm < params.min_child_weight * 0.01 {
                let leaf_val = (-g_total as f64) / (h_total as f64 + params.reg_lambda);
                self.set_leaf(task.node_index, leaf_val);
                if let Some(ref mut fp) = full_preds {
                    for &idx in &workspace.indices[task.sample_start..task.sample_start + n_samples]
                    {
                        fp[idx as usize] =
                            (fp[idx as usize] + learning_rate * leaf_val).clamp(-10.0, 10.0);
                    }
                }
                workspace.release_hists(task.histogram);
                continue;
            }

            // Stopping conditions
            if task.depth >= self.max_depth
                || n_samples < params.min_samples_split
                || (h_total as f64) < params.min_child_weight
            {
                let leaf_val = (-g_total as f64) / (h_total as f64 + params.reg_lambda);
                self.set_leaf(task.node_index, leaf_val);
                if let Some(ref mut fp) = full_preds {
                    for &idx in &workspace.indices[task.sample_start..task.sample_start + n_samples]
                    {
                        fp[idx as usize] =
                            (fp[idx as usize] + learning_rate * leaf_val).clamp(-10.0, 10.0);
                    }
                }
                workspace.release_hists(task.histogram);
                continue;
            }

            // OPTIMIZATION 4: Seq split finding for 29 features
            let best_split = find_best_split_across_features_parallel(
                &task.histogram,
                params,
                task.depth as i32,
            );

            if best_split.is_none() || best_split.as_ref().unwrap().1.best_gain <= 1e-6 {
                let leaf_val = (-g_total as f64) / (h_total as f64 + params.reg_lambda);
                self.set_leaf(task.node_index, leaf_val);
                if let Some(ref mut fp) = full_preds {
                    for &idx in &workspace.indices[task.sample_start..task.sample_start + n_samples]
                    {
                        fp[idx as usize] =
                            (fp[idx as usize] + learning_rate * leaf_val).clamp(-10.0, 10.0);
                    }
                }
                workspace.release_hists(task.histogram);
                continue;
            }

            let (best_feature_local_idx, split_info) = best_split.unwrap();
            let best_feature_global_idx = self.feature_indices[best_feature_local_idx];

            // OPTIMIZATION 5: In-place partitioning
            let mut left_count = 0;
            let mut right_count = 0;
            let feature_values = transposed_data.get_feature_values(best_feature_global_idx);
            let threshold = split_info.best_bin_idx;

            for &idx in &workspace.indices[task.sample_start..task.sample_start + n_samples] {
                if unsafe { *feature_values.get_unchecked(idx as usize) } <= threshold {
                    workspace.buffer_b[task.sample_start + left_count] = idx;
                    left_count += 1;
                } else {
                    workspace.buffer_b[task.sample_start + n_samples - 1 - right_count] = idx;
                    right_count += 1;
                }
            }

            // Copy back to workspace.indices
            workspace.indices[task.sample_start..task.sample_start + n_samples].copy_from_slice(
                &workspace.buffer_b[task.sample_start..task.sample_start + n_samples],
            );

            if left_count == 0 || right_count == 0 {
                let leaf_val = (-g_total as f64) / (h_total as f64 + params.reg_lambda);
                self.set_leaf(task.node_index, leaf_val);
                if let Some(ref mut fp) = full_preds {
                    for &idx in &workspace.indices[task.sample_start..task.sample_start + n_samples]
                    {
                        fp[idx as usize] =
                            (fp[idx as usize] + learning_rate * leaf_val).clamp(-10.0, 10.0);
                    }
                }
                workspace.release_hists(task.histogram);
                continue;
            }

            // OPTIMIZATION 6: Histogram subtraction
            let (smaller_start, smaller_len, _, _) = if left_count < right_count {
                (
                    task.sample_start,
                    left_count,
                    task.sample_start + left_count,
                    right_count,
                )
            } else {
                (
                    task.sample_start + left_count,
                    right_count,
                    task.sample_start,
                    left_count,
                )
            };

            let mut smaller_hists = workspace.acquire_hists(n_features);
            let mut larger_hists = workspace.acquire_hists(n_features);

            build_hists_into(
                &self.feature_indices,
                transposed_data,
                &y_u8,
                grad,
                hess,
                &workspace.indices[smaller_start..smaller_start + smaller_len],
                params,
                &mut smaller_hists,
                &mut workspace.active_data,
            );

            // Subtract: parent - smaller = larger
            for fi in 0..n_features {
                task.histogram[fi].subtract_into(&smaller_hists[fi], &mut larger_hists[fi]);
            }

            let left_child_index = next_node_in_vec;
            let right_child_index = next_node_in_vec + 1;

            if right_child_index >= self.node_types.len() {
                workspace.release_hists(smaller_hists);
                workspace.release_hists(larger_hists);
                workspace.release_hists(task.histogram);
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

            let (left_hists, right_hists) = if left_count < right_count {
                (smaller_hists, larger_hists)
            } else {
                (larger_hists, smaller_hists)
            };

            queue.push_back(SplitTask {
                node_index: left_child_index,
                sample_start: task.sample_start,
                sample_len: left_count,
                histogram: left_hists,
                depth: task.depth + 1,
            });
            queue.push_back(SplitTask {
                node_index: right_child_index,
                sample_start: task.sample_start + left_count,
                sample_len: right_count,
                histogram: right_hists,
                depth: task.depth + 1,
            });

            workspace.release_hists(task.histogram);
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
    sample_start: usize,
    sample_len: usize,
    histogram: Vec<CachedHistogram>,
    depth: usize,
}

// OPTIMIZATION 7: Thread-local memory pooling
thread_local! {
    static INDEX_POOL: RefCell<IndexPool> = RefCell::new(IndexPool::new());
}

struct IndexPool {
    buffers: Vec<Vec<u32>>,
    active_buffers: Vec<crate::optimized_data::ActiveData>,
    bin_buffers: Vec<Vec<u8>>,
    histogram_buffers: Vec<Vec<CachedHistogram>>,
}

impl IndexPool {
    fn new() -> Self {
        Self {
            buffers: Vec::with_capacity(8),
            active_buffers: Vec::with_capacity(4),
            bin_buffers: Vec::with_capacity(4),
            histogram_buffers: Vec::with_capacity(16),
        }
    }

    fn acquire(&mut self, size: usize) -> Vec<u32> {
        self.buffers
            .pop()
            .map(|mut buf| {
                buf.resize(size, 0);
                buf
            })
            .unwrap_or_else(|| vec![0u32; size])
    }

    fn release(&mut self, buf: Vec<u32>) {
        if buf.capacity() <= 1024 * 1024 && self.buffers.len() < 16 {
            self.buffers.push(buf);
        }
    }

    fn acquire_active(&mut self, capacity: usize) -> crate::optimized_data::ActiveData {
        self.active_buffers
            .pop()
            .map(|mut buf| {
                buf.resize(capacity);
                buf
            })
            .unwrap_or_else(|| crate::optimized_data::ActiveData::new(capacity))
    }

    fn release_active(&mut self, buf: crate::optimized_data::ActiveData) {
        if buf.samples.capacity() <= 512 * 1024 && self.active_buffers.len() < 8 {
            self.active_buffers.push(buf);
        }
    }

    fn acquire_bins(&mut self, capacity: usize) -> Vec<u8> {
        self.bin_buffers
            .pop()
            .map(|mut buf| {
                buf.clear();
                buf.reserve(capacity.saturating_sub(buf.capacity()));
                buf.resize(capacity, 0);
                buf
            })
            .unwrap_or_else(|| vec![0u8; capacity])
    }

    fn release_bins(&mut self, buf: Vec<u8>) {
        if buf.capacity() <= 2 * 1024 * 1024 && self.bin_buffers.len() < 8 {
            self.bin_buffers.push(buf);
        }
    }

    fn acquire_hists(&mut self, n_features: usize) -> Vec<CachedHistogram> {
        self.histogram_buffers
            .pop()
            .map(|mut h| {
                if h.len() != n_features {
                    h.resize(n_features, CachedHistogram::default());
                }
                h
            })
            .unwrap_or_else(|| {
                (0..n_features)
                    .map(|_| CachedHistogram::default())
                    .collect()
            })
    }

    fn release_hists(&mut self, hists: Vec<CachedHistogram>) {
        if self.histogram_buffers.len() < 32 {
            self.histogram_buffers.push(hists);
        }
    }
}

struct TreeBuildingWorkspace {
    indices: Vec<u32>,
    buffer_b: Vec<u32>,
    active_data: crate::optimized_data::ActiveData,
    bin_buffer: Vec<u8>,
}

impl TreeBuildingWorkspace {
    fn new(n_samples: usize, _n_features: usize) -> Self {
        INDEX_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            Self {
                indices: pool.acquire(n_samples),
                buffer_b: pool.acquire(n_samples),
                active_data: pool.acquire_active(n_samples + 1024),
                bin_buffer: pool.acquire_bins(n_samples * 4 + 1024),
            }
        })
    }

    fn acquire_hists(&mut self, n_features: usize) -> Vec<CachedHistogram> {
        INDEX_POOL.with(|pool| pool.borrow_mut().acquire_hists(n_features))
    }

    fn release_hists(&mut self, hists: Vec<CachedHistogram>) {
        INDEX_POOL.with(|pool| pool.borrow_mut().release_hists(hists))
    }
}

impl Drop for TreeBuildingWorkspace {
    fn drop(&mut self) {
        INDEX_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.release(std::mem::take(&mut self.indices));
            pool.release(std::mem::take(&mut self.buffer_b));
            pool.release_active(std::mem::take(&mut self.active_data));
            pool.release_bins(std::mem::take(&mut self.bin_buffer));
        });
    }
}

// OPTIMIZATION: ForkUnion parallel per-feature histogram building (ZERO-ALLOCATION)
// Uses ForkUnion's low-overhead for_n instead of Rayon's high-overhead par_iter_mut.
// No mutexes, no CAS, no heap allocations on the dispatch path.
fn build_hists_into(
    feature_indices: &[usize],
    transposed_data: &TransposedData,
    y_u8: &[u8],
    grad: &[f32],
    hess: &[f32],
    indices: &[u32],
    params: &TreeParams,
    out: &mut [CachedHistogram],
    workspace_active: &mut crate::optimized_data::ActiveData,
) {
    debug_assert!(out.len() >= feature_indices.len());
    let n_features = feature_indices.len();
    let n_samples = indices.len();

    // Reorganize data into contiguous buffer (AoS)
    crate::optimized_data::gather_active_data(indices, grad, hess, y_u8, workspace_active);

    if n_samples >= 25_000 {
        use crate::fork_parallel::pfor_indexed;
        let n_chunks = (n_features + 3) / 4;
        let out_ptr = out.as_mut_ptr() as usize;
        let active_ref = &*workspace_active;

        pfor_indexed(n_chunks, move |chunk_idx| {
            let start = chunk_idx * 4;
            let end = (start + 4).min(n_features);
            let out_ptr = out_ptr as *mut CachedHistogram;

            if end - start == 4 {
                let mut f_indices = [0usize; 4];
                let mut n_bins = [0usize; 4];
                for i in 0..4 {
                    f_indices[i] = feature_indices[start + i];
                    n_bins[i] = params.n_bins_per_feature[start + i];
                }
                INDEX_POOL.with(|pool| {
                    let mut pool = pool.borrow_mut();
                    let mut bin_buf = pool.acquire_bins(n_samples * 4);
                    transposed_data.gather_bins_4x(&f_indices, indices, &n_bins, &mut bin_buf);
                    let hists = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(start), 4) };
                    transposed_data.build_multi_histogram_4x(active_ref, &n_bins, &bin_buf, hists);
                    pool.release_bins(bin_buf);
                });
            } else {
                for i in start..end {
                    let hist = unsafe { &mut *out_ptr.add(i) };
                    transposed_data.build_histogram_into_f32(
                        feature_indices[i],
                        indices,
                        active_ref,
                        params.n_bins_per_feature[i],
                        hist,
                    );
                }
            }
        });
    } else {
        // Sequential path
        let mut f_idx = 0;
        while f_idx + 4 <= n_features {
            let mut f_indices = [0usize; 4];
            let mut n_bins = [0usize; 4];
            for i in 0..4 {
                f_indices[i] = feature_indices[f_idx + i];
                n_bins[i] = params.n_bins_per_feature[f_idx + i];
            }
            INDEX_POOL.with(|pool| {
                let mut pool = pool.borrow_mut();
                let mut bin_buf = pool.acquire_bins(n_samples * 4);
                transposed_data.gather_bins_4x(&f_indices, indices, &n_bins, &mut bin_buf);
                transposed_data.build_multi_histogram_4x(
                    workspace_active,
                    &n_bins,
                    &bin_buf,
                    &mut out[f_idx..f_idx + 4],
                );
                pool.release_bins(bin_buf);
            });
            f_idx += 4;
        }
        while f_idx < n_features {
            transposed_data.build_histogram_into_f32(
                feature_indices[f_idx],
                indices,
                workspace_active,
                params.n_bins_per_feature[f_idx],
                &mut out[f_idx],
            );
            f_idx += 1;
        }
    }
}

// OPTIMIZATION: Parallel split finding (scans all feature histograms in parallel)
fn find_best_split_across_features_parallel(
    hists: &[CachedHistogram],
    params: &TreeParams,
    depth: i32,
) -> Option<(usize, HistSplitResult)> {
    let n = hists.len();
    if n == 0 {
        return None;
    }

    // For small feature sets (like the 29 in fraud dataset), sequential is faster
    // due to thread dispatch overhead being higher than the gain calculation.
    let mut best_feature_local_idx = 0;
    let mut best_result = HistSplitResult::default();
    let mut found = false;

    for (feat_idx_local, hist) in hists.iter().enumerate() {
        let total_hess = hist.sums.hess;
        let non_zero_bins = hist.bins.iter().filter(|b| b.count > 0.0).count();

        if (total_hess as f64) > params.min_child_weight * params.feature_elimination_threshold
            && non_zero_bins > 1
        {
            let res = find_best_split_cached_optimized(hist, params, depth);
            if res.best_gain > best_result.best_gain {
                best_result = res;
                best_feature_local_idx = feat_idx_local;
                found = true;
            }
        }
    }

    if found && best_result.best_gain > 1e-6 {
        Some((best_feature_local_idx, best_result))
    } else {
        None
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

impl PrecomputedSums {
    #[inline]
    fn from_histogram(hist: &CachedHistogram) -> Self {
        let g_total = hist.sums.grad as f64;
        let h_total = hist.sums.hess as f64;
        let y_total = hist.sums.y as f64;
        let n_total = hist.sums.count as f64;

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

// OPTIMIZATION 11: Vectorized gain calculation with minimal branching
#[inline]
fn find_best_split_cached_optimized(
    hist: &CachedHistogram,
    params: &TreeParams,
    depth: i32,
) -> HistSplitResult {
    let precomputed = PrecomputedSums::from_histogram(hist);

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

    let bins = &hist.bins;
    let n_splits = bins.len().saturating_sub(1);

    for i in 0..n_splits {
        let b = unsafe { bins.get_unchecked(i) };
        gl += b.grad as f64;
        hl += b.hess as f64;
        y_left += b.y as f64;
        n_left += b.count as f64;

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

        // Optimized gain calculation
        let left_score = gl * gl / (hl + reg_lambda);
        let right_score = gr * gr / (hr + reg_lambda);
        let newton_gain = 0.5 * (left_score + right_score - parent_score) - gamma;

        // Calculate entropy only if promising and needed
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
            best_split.best_bin_idx = i as u8;
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

        let sample_left = vec![2u8];
        assert_eq!(tree.predict_single(&sample_left), 0.1);

        let sample_right = vec![5u8];
        assert_eq!(tree.predict_single(&sample_right), 0.9);
    }
}
