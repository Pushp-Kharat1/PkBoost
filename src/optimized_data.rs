use ndarray::{Array2, ArrayView1, ArrayView2};

#[derive(Debug, Clone)]
pub struct TransposedData {
    pub features: Array2<u8>, // Transposed: (n_features, n_samples), u8 for minimal bandwidth
    pub n_samples: usize,
    pub n_features: usize,
}

impl TransposedData {
    /// Create TransposedData from binned Array2<u8> (zero-copy path from histogram_builder)
    /// Input: (n_samples, n_features), Output: transposed to (n_features, n_samples)
    pub fn from_binned(binned: Array2<u8>) -> Self {
        let (n_samples, n_features) = binned.dim();
        if n_samples == 0 || n_features == 0 {
            return Self {
                features: Array2::zeros((0, 0)),
                n_samples: 0,
                n_features: 0,
            };
        }

        // Transpose: (n_samples, n_features) -> (n_features, n_samples)
        // Use from_shape_vec to create contiguous standard layout for .as_slice()
        let transposed = binned.t();
        let features = Array2::from_shape_vec(
            (n_features, n_samples),
            transposed.iter().cloned().collect(),
        )
        .expect("Failed to create feature array from binned data");

        Self {
            features,
            n_samples,
            n_features,
        }
    }

    /// Create TransposedData from binned ArrayView2<u8> (avoids extra allocation)
    pub fn from_binned_view(binned: ArrayView2<'_, u8>) -> Self {
        let (n_samples, n_features) = binned.dim();
        if n_samples == 0 || n_features == 0 {
            return Self {
                features: Array2::zeros((0, 0)),
                n_samples: 0,
                n_features: 0,
            };
        }

        // Transpose and create contiguous standard layout
        let transposed = binned.t();
        let features = Array2::from_shape_vec(
            (n_features, n_samples),
            transposed.iter().cloned().collect(),
        )
        .expect("Failed to create feature array from binned data");

        Self {
            features,
            n_samples,
            n_features,
        }
    }

    /// Legacy: Create from Vec<Vec<i32>> (deprecated, use from_binned instead)
    #[deprecated(note = "Use from_binned() with Array2<u8> for zero-copy performance")]
    pub fn from_rows(rows: &[Vec<i32>]) -> Self {
        if rows.is_empty() {
            return Self {
                features: Array2::zeros((0, 0)),
                n_samples: 0,
                n_features: 0,
            };
        }

        const BLOCK_SIZE: usize = 64;

        let n_samples = rows.len();
        let n_features = rows[0].len();
        let mut features = Array2::zeros((n_features, n_samples));

        // Block-based transposition for better cache locality
        for feature_block_start in (0..n_features).step_by(BLOCK_SIZE) {
            let feature_block_end = (feature_block_start + BLOCK_SIZE).min(n_features);

            for sample_block_start in (0..n_samples).step_by(BLOCK_SIZE) {
                let sample_block_end = (sample_block_start + BLOCK_SIZE).min(n_samples);

                for sample_idx in sample_block_start..sample_block_end {
                    let row = &rows[sample_idx];
                    for feature_idx in feature_block_start..feature_block_end {
                        if feature_idx < row.len() {
                            features[[feature_idx, sample_idx]] = row[feature_idx] as u8;
                        }
                    }
                }
            }
        }

        Self {
            features,
            n_samples,
            n_features,
        }
    }

    pub fn get_feature_values(&self, feature_idx: usize) -> &[u8] {
        // With standard row-major layout, each row is contiguous in memory
        // Slice directly from the underlying storage
        let start = feature_idx * self.n_samples;
        let end = start + self.n_samples;
        &self.features.as_slice().unwrap_or_else(|| {
            // This should not happen with our current implementation
            // since we ensure contiguity in from_binned methods
            panic!("TransposedData features array is not contiguous. This indicates a bug in array creation.")
        })[start..end]
    }

    #[inline]
    pub fn build_histogram_vectorized(
        &self,
        feature_idx: usize,
        indices: &[usize],
        grad: &[f32],
        hess: &[f32],
        y_u8: &[u8],
        n_bins: usize,
    ) -> CachedHistogram {
        if n_bins == 0 {
            return CachedHistogram::new(vec![], vec![], vec![], vec![]);
        }

        let mut hist_grad = vec![0.0; n_bins];
        let mut hist_hess = vec![0.0; n_bins];
        let mut hist_y = vec![0.0; n_bins];
        let mut hist_count = vec![0.0; n_bins];

        let feature_values = self.get_feature_values(feature_idx);

        let len = indices.len();
        let max_bin = n_bins - 1;

        debug_assert!(
            indices.iter().all(|&idx| idx < self.n_samples),
            "Invalid sample index found"
        );

        // OPTIMIZATION: 8x unrolled loop with separated fetch/update phases
        let mut i = 0;

        while i + 8 <= len {
            unsafe {
                // Phase 1: Fetch all indices
                let idx0 = *indices.get_unchecked(i);
                let idx1 = *indices.get_unchecked(i + 1);
                let idx2 = *indices.get_unchecked(i + 2);
                let idx3 = *indices.get_unchecked(i + 3);
                let idx4 = *indices.get_unchecked(i + 4);
                let idx5 = *indices.get_unchecked(i + 5);
                let idx6 = *indices.get_unchecked(i + 6);
                let idx7 = *indices.get_unchecked(i + 7);
                // Phase 2: Compute all bins
                let bin0 = (*feature_values.get_unchecked(idx0) as usize).min(max_bin);
                let bin1 = (*feature_values.get_unchecked(idx1) as usize).min(max_bin);
                let bin2 = (*feature_values.get_unchecked(idx2) as usize).min(max_bin);
                let bin3 = (*feature_values.get_unchecked(idx3) as usize).min(max_bin);
                let bin4 = (*feature_values.get_unchecked(idx4) as usize).min(max_bin);
                let bin5 = (*feature_values.get_unchecked(idx5) as usize).min(max_bin);
                let bin6 = (*feature_values.get_unchecked(idx6) as usize).min(max_bin);
                let bin7 = (*feature_values.get_unchecked(idx7) as usize).min(max_bin);
                // Phase 3: Fetch all grad/hess/y values (f32 grad/hess, u8 y)
                let g0 = *grad.get_unchecked(idx0);
                let h0 = *hess.get_unchecked(idx0);
                let y0 = *y_u8.get_unchecked(idx0);
                let g1 = *grad.get_unchecked(idx1);
                let h1 = *hess.get_unchecked(idx1);
                let y1 = *y_u8.get_unchecked(idx1);
                let g2 = *grad.get_unchecked(idx2);
                let h2 = *hess.get_unchecked(idx2);
                let y2 = *y_u8.get_unchecked(idx2);
                let g3 = *grad.get_unchecked(idx3);
                let h3 = *hess.get_unchecked(idx3);
                let y3 = *y_u8.get_unchecked(idx3);
                let g4 = *grad.get_unchecked(idx4);
                let h4 = *hess.get_unchecked(idx4);
                let y4 = *y_u8.get_unchecked(idx4);
                let g5 = *grad.get_unchecked(idx5);
                let h5 = *hess.get_unchecked(idx5);
                let y5 = *y_u8.get_unchecked(idx5);
                let g6 = *grad.get_unchecked(idx6);
                let h6 = *hess.get_unchecked(idx6);
                let y6 = *y_u8.get_unchecked(idx6);
                let g7 = *grad.get_unchecked(idx7);
                let h7 = *hess.get_unchecked(idx7);
                let y7 = *y_u8.get_unchecked(idx7);
                // Phase 4: Accumulate (f32→f64 cast on accumulate, bins stay f64)
                *hist_grad.get_unchecked_mut(bin0) += g0 as f64;
                *hist_hess.get_unchecked_mut(bin0) += h0 as f64;
                *hist_y.get_unchecked_mut(bin0) += y0 as f64;
                *hist_count.get_unchecked_mut(bin0) += 1.0;
                *hist_grad.get_unchecked_mut(bin1) += g1 as f64;
                *hist_hess.get_unchecked_mut(bin1) += h1 as f64;
                *hist_y.get_unchecked_mut(bin1) += y1 as f64;
                *hist_count.get_unchecked_mut(bin1) += 1.0;
                *hist_grad.get_unchecked_mut(bin2) += g2 as f64;
                *hist_hess.get_unchecked_mut(bin2) += h2 as f64;
                *hist_y.get_unchecked_mut(bin2) += y2 as f64;
                *hist_count.get_unchecked_mut(bin2) += 1.0;
                *hist_grad.get_unchecked_mut(bin3) += g3 as f64;
                *hist_hess.get_unchecked_mut(bin3) += h3 as f64;
                *hist_y.get_unchecked_mut(bin3) += y3 as f64;
                *hist_count.get_unchecked_mut(bin3) += 1.0;
                *hist_grad.get_unchecked_mut(bin4) += g4 as f64;
                *hist_hess.get_unchecked_mut(bin4) += h4 as f64;
                *hist_y.get_unchecked_mut(bin4) += y4 as f64;
                *hist_count.get_unchecked_mut(bin4) += 1.0;
                *hist_grad.get_unchecked_mut(bin5) += g5 as f64;
                *hist_hess.get_unchecked_mut(bin5) += h5 as f64;
                *hist_y.get_unchecked_mut(bin5) += y5 as f64;
                *hist_count.get_unchecked_mut(bin5) += 1.0;
                *hist_grad.get_unchecked_mut(bin6) += g6 as f64;
                *hist_hess.get_unchecked_mut(bin6) += h6 as f64;
                *hist_y.get_unchecked_mut(bin6) += y6 as f64;
                *hist_count.get_unchecked_mut(bin6) += 1.0;
                *hist_grad.get_unchecked_mut(bin7) += g7 as f64;
                *hist_hess.get_unchecked_mut(bin7) += h7 as f64;
                *hist_y.get_unchecked_mut(bin7) += y7 as f64;
                *hist_count.get_unchecked_mut(bin7) += 1.0;
            }
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            unsafe {
                let idx = *indices.get_unchecked(i);
                let bin = (*feature_values.get_unchecked(idx) as usize).min(max_bin);
                *hist_grad.get_unchecked_mut(bin) += *grad.get_unchecked(idx) as f64;
                *hist_hess.get_unchecked_mut(bin) += *hess.get_unchecked(idx) as f64;
                *hist_y.get_unchecked_mut(bin) += *y_u8.get_unchecked(idx) as f64;
                *hist_count.get_unchecked_mut(bin) += 1.0;
            }
            i += 1;
        }

        CachedHistogram::new(hist_grad, hist_hess, hist_y, hist_count)
    }

    /// OPTIMIZED: Build histogram into pre-allocated CachedHistogram buffers
    /// Avoids allocating 4 new Vecs per call (millions of saved allocations)
    #[inline]
    pub fn build_histogram_into(
        &self,
        feature_idx: usize,
        indices: &[usize],
        grad: &ArrayView1<f64>,
        hess: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        n_bins: usize,
        out: &mut CachedHistogram,
    ) {
        if n_bins == 0 {
            return;
        }

        out.reset(n_bins);

        let hist_grad = &mut out.grad;
        let hist_hess = &mut out.hess;
        let hist_y = &mut out.y;
        let hist_count = &mut out.count;

        let feature_values = self.get_feature_values(feature_idx);
        let grad_slice = grad.as_slice().expect("Gradient array must be contiguous");
        let hess_slice = hess.as_slice().expect("Hessian array must be contiguous");
        let y_slice = y.as_slice().expect("Target array must be contiguous");

        let len = indices.len();
        let max_bin = n_bins - 1;

        // 8x unrolled loop with separated fetch/update phases
        let mut i = 0;

        while i + 8 <= len {
            unsafe {
                let idx0 = *indices.get_unchecked(i);
                let idx1 = *indices.get_unchecked(i + 1);
                let idx2 = *indices.get_unchecked(i + 2);
                let idx3 = *indices.get_unchecked(i + 3);
                let idx4 = *indices.get_unchecked(i + 4);
                let idx5 = *indices.get_unchecked(i + 5);
                let idx6 = *indices.get_unchecked(i + 6);
                let idx7 = *indices.get_unchecked(i + 7);
                let bin0 = (*feature_values.get_unchecked(idx0) as usize).min(max_bin);
                let bin1 = (*feature_values.get_unchecked(idx1) as usize).min(max_bin);
                let bin2 = (*feature_values.get_unchecked(idx2) as usize).min(max_bin);
                let bin3 = (*feature_values.get_unchecked(idx3) as usize).min(max_bin);
                let bin4 = (*feature_values.get_unchecked(idx4) as usize).min(max_bin);
                let bin5 = (*feature_values.get_unchecked(idx5) as usize).min(max_bin);
                let bin6 = (*feature_values.get_unchecked(idx6) as usize).min(max_bin);
                let bin7 = (*feature_values.get_unchecked(idx7) as usize).min(max_bin);
                let g0 = *grad_slice.get_unchecked(idx0);
                let h0 = *hess_slice.get_unchecked(idx0);
                let y0 = *y_slice.get_unchecked(idx0);
                let g1 = *grad_slice.get_unchecked(idx1);
                let h1 = *hess_slice.get_unchecked(idx1);
                let y1 = *y_slice.get_unchecked(idx1);
                let g2 = *grad_slice.get_unchecked(idx2);
                let h2 = *hess_slice.get_unchecked(idx2);
                let y2 = *y_slice.get_unchecked(idx2);
                let g3 = *grad_slice.get_unchecked(idx3);
                let h3 = *hess_slice.get_unchecked(idx3);
                let y3 = *y_slice.get_unchecked(idx3);
                let g4 = *grad_slice.get_unchecked(idx4);
                let h4 = *hess_slice.get_unchecked(idx4);
                let y4 = *y_slice.get_unchecked(idx4);
                let g5 = *grad_slice.get_unchecked(idx5);
                let h5 = *hess_slice.get_unchecked(idx5);
                let y5 = *y_slice.get_unchecked(idx5);
                let g6 = *grad_slice.get_unchecked(idx6);
                let h6 = *hess_slice.get_unchecked(idx6);
                let y6 = *y_slice.get_unchecked(idx6);
                let g7 = *grad_slice.get_unchecked(idx7);
                let h7 = *hess_slice.get_unchecked(idx7);
                let y7 = *y_slice.get_unchecked(idx7);
                *hist_grad.get_unchecked_mut(bin0) += g0;
                *hist_hess.get_unchecked_mut(bin0) += h0;
                *hist_y.get_unchecked_mut(bin0) += y0;
                *hist_count.get_unchecked_mut(bin0) += 1.0;
                *hist_grad.get_unchecked_mut(bin1) += g1;
                *hist_hess.get_unchecked_mut(bin1) += h1;
                *hist_y.get_unchecked_mut(bin1) += y1;
                *hist_count.get_unchecked_mut(bin1) += 1.0;
                *hist_grad.get_unchecked_mut(bin2) += g2;
                *hist_hess.get_unchecked_mut(bin2) += h2;
                *hist_y.get_unchecked_mut(bin2) += y2;
                *hist_count.get_unchecked_mut(bin2) += 1.0;
                *hist_grad.get_unchecked_mut(bin3) += g3;
                *hist_hess.get_unchecked_mut(bin3) += h3;
                *hist_y.get_unchecked_mut(bin3) += y3;
                *hist_count.get_unchecked_mut(bin3) += 1.0;
                *hist_grad.get_unchecked_mut(bin4) += g4;
                *hist_hess.get_unchecked_mut(bin4) += h4;
                *hist_y.get_unchecked_mut(bin4) += y4;
                *hist_count.get_unchecked_mut(bin4) += 1.0;
                *hist_grad.get_unchecked_mut(bin5) += g5;
                *hist_hess.get_unchecked_mut(bin5) += h5;
                *hist_y.get_unchecked_mut(bin5) += y5;
                *hist_count.get_unchecked_mut(bin5) += 1.0;
                *hist_grad.get_unchecked_mut(bin6) += g6;
                *hist_hess.get_unchecked_mut(bin6) += h6;
                *hist_y.get_unchecked_mut(bin6) += y6;
                *hist_count.get_unchecked_mut(bin6) += 1.0;
                *hist_grad.get_unchecked_mut(bin7) += g7;
                *hist_hess.get_unchecked_mut(bin7) += h7;
                *hist_y.get_unchecked_mut(bin7) += y7;
                *hist_count.get_unchecked_mut(bin7) += 1.0;
            }
            i += 8;
        }

        while i < len {
            unsafe {
                let idx = *indices.get_unchecked(i);
                let bin = (*feature_values.get_unchecked(idx) as usize).min(max_bin);
                *hist_grad.get_unchecked_mut(bin) += *grad_slice.get_unchecked(idx);
                *hist_hess.get_unchecked_mut(bin) += *hess_slice.get_unchecked(idx);
                *hist_y.get_unchecked_mut(bin) += *y_slice.get_unchecked(idx);
                *hist_count.get_unchecked_mut(bin) += 1.0;
            }
            i += 1;
        }
    }

    /// Zero-allocation histogram build for f32 grad/hess + u8 y hot path.
    /// Resets and fills a pre-allocated CachedHistogram, avoiding 4 Vec allocs per call.
    #[inline]
    pub fn build_histogram_into_f32(
        &self,
        feature_idx: usize,
        indices: &[usize],
        grad: &[f32],
        hess: &[f32],
        y_u8: &[u8],
        n_bins: usize,
        out: &mut CachedHistogram,
    ) {
        if n_bins == 0 {
            return;
        }

        out.reset(n_bins);

        let hist_grad = &mut out.grad;
        let hist_hess = &mut out.hess;
        let hist_y = &mut out.y;
        let hist_count = &mut out.count;

        let feature_values = self.get_feature_values(feature_idx);
        let len = indices.len();
        let max_bin = n_bins - 1;

        // 8x unrolled loop (same as build_histogram_vectorized but into pre-allocated buffers)
        let mut i = 0;

        while i + 8 <= len {
            unsafe {
                let idx0 = *indices.get_unchecked(i);
                let idx1 = *indices.get_unchecked(i + 1);
                let idx2 = *indices.get_unchecked(i + 2);
                let idx3 = *indices.get_unchecked(i + 3);
                let idx4 = *indices.get_unchecked(i + 4);
                let idx5 = *indices.get_unchecked(i + 5);
                let idx6 = *indices.get_unchecked(i + 6);
                let idx7 = *indices.get_unchecked(i + 7);
                let bin0 = (*feature_values.get_unchecked(idx0) as usize).min(max_bin);
                let bin1 = (*feature_values.get_unchecked(idx1) as usize).min(max_bin);
                let bin2 = (*feature_values.get_unchecked(idx2) as usize).min(max_bin);
                let bin3 = (*feature_values.get_unchecked(idx3) as usize).min(max_bin);
                let bin4 = (*feature_values.get_unchecked(idx4) as usize).min(max_bin);
                let bin5 = (*feature_values.get_unchecked(idx5) as usize).min(max_bin);
                let bin6 = (*feature_values.get_unchecked(idx6) as usize).min(max_bin);
                let bin7 = (*feature_values.get_unchecked(idx7) as usize).min(max_bin);
                let g0 = *grad.get_unchecked(idx0);
                let h0 = *hess.get_unchecked(idx0);
                let y0 = *y_u8.get_unchecked(idx0);
                let g1 = *grad.get_unchecked(idx1);
                let h1 = *hess.get_unchecked(idx1);
                let y1 = *y_u8.get_unchecked(idx1);
                let g2 = *grad.get_unchecked(idx2);
                let h2 = *hess.get_unchecked(idx2);
                let y2 = *y_u8.get_unchecked(idx2);
                let g3 = *grad.get_unchecked(idx3);
                let h3 = *hess.get_unchecked(idx3);
                let y3 = *y_u8.get_unchecked(idx3);
                let g4 = *grad.get_unchecked(idx4);
                let h4 = *hess.get_unchecked(idx4);
                let y4 = *y_u8.get_unchecked(idx4);
                let g5 = *grad.get_unchecked(idx5);
                let h5 = *hess.get_unchecked(idx5);
                let y5 = *y_u8.get_unchecked(idx5);
                let g6 = *grad.get_unchecked(idx6);
                let h6 = *hess.get_unchecked(idx6);
                let y6 = *y_u8.get_unchecked(idx6);
                let g7 = *grad.get_unchecked(idx7);
                let h7 = *hess.get_unchecked(idx7);
                let y7 = *y_u8.get_unchecked(idx7);
                *hist_grad.get_unchecked_mut(bin0) += g0 as f64;
                *hist_hess.get_unchecked_mut(bin0) += h0 as f64;
                *hist_y.get_unchecked_mut(bin0) += y0 as f64;
                *hist_count.get_unchecked_mut(bin0) += 1.0;
                *hist_grad.get_unchecked_mut(bin1) += g1 as f64;
                *hist_hess.get_unchecked_mut(bin1) += h1 as f64;
                *hist_y.get_unchecked_mut(bin1) += y1 as f64;
                *hist_count.get_unchecked_mut(bin1) += 1.0;
                *hist_grad.get_unchecked_mut(bin2) += g2 as f64;
                *hist_hess.get_unchecked_mut(bin2) += h2 as f64;
                *hist_y.get_unchecked_mut(bin2) += y2 as f64;
                *hist_count.get_unchecked_mut(bin2) += 1.0;
                *hist_grad.get_unchecked_mut(bin3) += g3 as f64;
                *hist_hess.get_unchecked_mut(bin3) += h3 as f64;
                *hist_y.get_unchecked_mut(bin3) += y3 as f64;
                *hist_count.get_unchecked_mut(bin3) += 1.0;
                *hist_grad.get_unchecked_mut(bin4) += g4 as f64;
                *hist_hess.get_unchecked_mut(bin4) += h4 as f64;
                *hist_y.get_unchecked_mut(bin4) += y4 as f64;
                *hist_count.get_unchecked_mut(bin4) += 1.0;
                *hist_grad.get_unchecked_mut(bin5) += g5 as f64;
                *hist_hess.get_unchecked_mut(bin5) += h5 as f64;
                *hist_y.get_unchecked_mut(bin5) += y5 as f64;
                *hist_count.get_unchecked_mut(bin5) += 1.0;
                *hist_grad.get_unchecked_mut(bin6) += g6 as f64;
                *hist_hess.get_unchecked_mut(bin6) += h6 as f64;
                *hist_y.get_unchecked_mut(bin6) += y6 as f64;
                *hist_count.get_unchecked_mut(bin6) += 1.0;
                *hist_grad.get_unchecked_mut(bin7) += g7 as f64;
                *hist_hess.get_unchecked_mut(bin7) += h7 as f64;
                *hist_y.get_unchecked_mut(bin7) += y7 as f64;
                *hist_count.get_unchecked_mut(bin7) += 1.0;
            }
            i += 8;
        }

        while i < len {
            unsafe {
                let idx = *indices.get_unchecked(i);
                let bin = (*feature_values.get_unchecked(idx) as usize).min(max_bin);
                *hist_grad.get_unchecked_mut(bin) += *grad.get_unchecked(idx) as f64;
                *hist_hess.get_unchecked_mut(bin) += *hess.get_unchecked(idx) as f64;
                *hist_y.get_unchecked_mut(bin) += *y_u8.get_unchecked(idx) as f64;
                *hist_count.get_unchecked_mut(bin) += 1.0;
            }
            i += 1;
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CachedHistogram {
    pub grad: Vec<f64>,
    pub hess: Vec<f64>,
    pub y: Vec<f64>,
    pub count: Vec<f64>,
}

impl CachedHistogram {
    pub fn new(grad: Vec<f64>, hess: Vec<f64>, y: Vec<f64>, count: Vec<f64>) -> Self {
        Self {
            grad,
            hess,
            y,
            count,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.grad.len()
    }

    #[inline]
    pub fn as_slices(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        (&self.grad, &self.hess, &self.y, &self.count)
    }

    /// Reset all histogram bins to zero, resizing if necessary
    #[inline]
    pub fn reset(&mut self, n_bins: usize) {
        if self.grad.len() == n_bins {
            self.grad.fill(0.0);
            self.hess.fill(0.0);
            self.y.fill(0.0);
            self.count.fill(0.0);
        } else {
            self.grad = vec![0.0; n_bins];
            self.hess = vec![0.0; n_bins];
            self.y = vec![0.0; n_bins];
            self.count = vec![0.0; n_bins];
        }
    }

    pub fn build_vectorized(
        transposed_data: &TransposedData,
        y_u8: &[u8],
        grad: &[f32],
        hess: &[f32],
        indices: &[usize],
        feature_idx: usize,
        n_bins: usize,
    ) -> Self {
        transposed_data.build_histogram_vectorized(feature_idx, indices, grad, hess, y_u8, n_bins)
    }

    /// OPTIMIZED: Build into a pre-allocated CachedHistogram buffer
    pub fn build_vectorized_into(
        transposed_data: &TransposedData,
        y: &ArrayView1<f64>,
        grad: &ArrayView1<f64>,
        hess: &ArrayView1<f64>,
        indices: &[usize],
        feature_idx: usize,
        n_bins: usize,
        out: &mut Self,
    ) {
        transposed_data.build_histogram_into(feature_idx, indices, grad, hess, y, n_bins, out);
    }

    /// Fast subtraction using unsafe array access (parent - sibling = other_child)
    #[inline]
    pub fn subtract(&self, other: &Self) -> Self {
        let n = self.grad.len();
        let mut grad_out = vec![0.0; n];
        let mut hess_out = vec![0.0; n];
        let mut y_out = vec![0.0; n];
        let mut count_out = vec![0.0; n];
        let grad_self = &self.grad;
        let hess_self = &self.hess;
        let y_self = &self.y;
        let count_self = &self.count;
        let grad_other = &other.grad;
        let hess_other = &other.hess;
        let y_other = &other.y;
        let count_other = &other.count;
        // 4x unrolled subtraction with unsafe access
        let mut i = 0;
        while i + 4 <= n {
            unsafe {
                *grad_out.get_unchecked_mut(i) =
                    *grad_self.get_unchecked(i) - *grad_other.get_unchecked(i);
                *hess_out.get_unchecked_mut(i) =
                    *hess_self.get_unchecked(i) - *hess_other.get_unchecked(i);
                *y_out.get_unchecked_mut(i) = *y_self.get_unchecked(i) - *y_other.get_unchecked(i);
                *count_out.get_unchecked_mut(i) =
                    *count_self.get_unchecked(i) - *count_other.get_unchecked(i);
                *grad_out.get_unchecked_mut(i + 1) =
                    *grad_self.get_unchecked(i + 1) - *grad_other.get_unchecked(i + 1);
                *hess_out.get_unchecked_mut(i + 1) =
                    *hess_self.get_unchecked(i + 1) - *hess_other.get_unchecked(i + 1);
                *y_out.get_unchecked_mut(i + 1) =
                    *y_self.get_unchecked(i + 1) - *y_other.get_unchecked(i + 1);
                *count_out.get_unchecked_mut(i + 1) =
                    *count_self.get_unchecked(i + 1) - *count_other.get_unchecked(i + 1);
                *grad_out.get_unchecked_mut(i + 2) =
                    *grad_self.get_unchecked(i + 2) - *grad_other.get_unchecked(i + 2);
                *hess_out.get_unchecked_mut(i + 2) =
                    *hess_self.get_unchecked(i + 2) - *hess_other.get_unchecked(i + 2);
                *y_out.get_unchecked_mut(i + 2) =
                    *y_self.get_unchecked(i + 2) - *y_other.get_unchecked(i + 2);
                *count_out.get_unchecked_mut(i + 2) =
                    *count_self.get_unchecked(i + 2) - *count_other.get_unchecked(i + 2);
                *grad_out.get_unchecked_mut(i + 3) =
                    *grad_self.get_unchecked(i + 3) - *grad_other.get_unchecked(i + 3);
                *hess_out.get_unchecked_mut(i + 3) =
                    *hess_self.get_unchecked(i + 3) - *hess_other.get_unchecked(i + 3);
                *y_out.get_unchecked_mut(i + 3) =
                    *y_self.get_unchecked(i + 3) - *y_other.get_unchecked(i + 3);
                *count_out.get_unchecked_mut(i + 3) =
                    *count_self.get_unchecked(i + 3) - *count_other.get_unchecked(i + 3);
            }
            i += 4;
        }

        // Handle remaining
        while i < n {
            unsafe {
                *grad_out.get_unchecked_mut(i) =
                    *grad_self.get_unchecked(i) - *grad_other.get_unchecked(i);
                *hess_out.get_unchecked_mut(i) =
                    *hess_self.get_unchecked(i) - *hess_other.get_unchecked(i);
                *y_out.get_unchecked_mut(i) = *y_self.get_unchecked(i) - *y_other.get_unchecked(i);
                *count_out.get_unchecked_mut(i) =
                    *count_self.get_unchecked(i) - *count_other.get_unchecked(i);
            }
            i += 1;
        }

        Self {
            grad: grad_out,
            hess: hess_out,
            y: y_out,
            count: count_out,
        }
    }

    /// Zero-allocation subtraction: writes (self - other) into `out`.
    #[inline]
    pub fn subtract_into(&self, other: &Self, out: &mut Self) {
        let n = self.grad.len();
        out.reset(n);

        let out_grad = &mut out.grad;
        let out_hess = &mut out.hess;
        let out_y = &mut out.y;
        let out_count = &mut out.count;

        let grad_self = &self.grad;
        let hess_self = &self.hess;
        let y_self = &self.y;
        let count_self = &self.count;

        let grad_other = &other.grad;
        let hess_other = &other.hess;
        let y_other = &other.y;
        let count_other = &other.count;

        let mut i = 0;
        while i + 4 <= n {
            unsafe {
                *out_grad.get_unchecked_mut(i) =
                    *grad_self.get_unchecked(i) - *grad_other.get_unchecked(i);
                *out_hess.get_unchecked_mut(i) =
                    *hess_self.get_unchecked(i) - *hess_other.get_unchecked(i);
                *out_y.get_unchecked_mut(i) = *y_self.get_unchecked(i) - *y_other.get_unchecked(i);
                *out_count.get_unchecked_mut(i) =
                    *count_self.get_unchecked(i) - *count_other.get_unchecked(i);
                *out_grad.get_unchecked_mut(i + 1) =
                    *grad_self.get_unchecked(i + 1) - *grad_other.get_unchecked(i + 1);
                *out_hess.get_unchecked_mut(i + 1) =
                    *hess_self.get_unchecked(i + 1) - *hess_other.get_unchecked(i + 1);
                *out_y.get_unchecked_mut(i + 1) =
                    *y_self.get_unchecked(i + 1) - *y_other.get_unchecked(i + 1);
                *out_count.get_unchecked_mut(i + 1) =
                    *count_self.get_unchecked(i + 1) - *count_other.get_unchecked(i + 1);
                *out_grad.get_unchecked_mut(i + 2) =
                    *grad_self.get_unchecked(i + 2) - *grad_other.get_unchecked(i + 2);
                *out_hess.get_unchecked_mut(i + 2) =
                    *hess_self.get_unchecked(i + 2) - *hess_other.get_unchecked(i + 2);
                *out_y.get_unchecked_mut(i + 2) =
                    *y_self.get_unchecked(i + 2) - *y_other.get_unchecked(i + 2);
                *out_count.get_unchecked_mut(i + 2) =
                    *count_self.get_unchecked(i + 2) - *count_other.get_unchecked(i + 2);
                *out_grad.get_unchecked_mut(i + 3) =
                    *grad_self.get_unchecked(i + 3) - *grad_other.get_unchecked(i + 3);
                *out_hess.get_unchecked_mut(i + 3) =
                    *hess_self.get_unchecked(i + 3) - *hess_other.get_unchecked(i + 3);
                *out_y.get_unchecked_mut(i + 3) =
                    *y_self.get_unchecked(i + 3) - *y_other.get_unchecked(i + 3);
                *out_count.get_unchecked_mut(i + 3) =
                    *count_self.get_unchecked(i + 3) - *count_other.get_unchecked(i + 3);
            }
            i += 4;
        }

        while i < n {
            unsafe {
                *out_grad.get_unchecked_mut(i) =
                    *grad_self.get_unchecked(i) - *grad_other.get_unchecked(i);
                *out_hess.get_unchecked_mut(i) =
                    *hess_self.get_unchecked(i) - *hess_other.get_unchecked(i);
                *out_y.get_unchecked_mut(i) = *y_self.get_unchecked(i) - *y_other.get_unchecked(i);
                *out_count.get_unchecked_mut(i) =
                    *count_self.get_unchecked(i) - *count_other.get_unchecked(i);
            }
            i += 1;
        }
    }
}
