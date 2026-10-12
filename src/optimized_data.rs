use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Map a stored bin value to a histogram slot.
///
/// The binner never emits negative values (NaNs are median-imputed before
/// binning), but if one ever appears it goes to bin 0 rather than wrapping
/// through `as usize` and silently landing in the last bin.
#[inline(always)]
fn bin_index(value: i16, max_bin: usize) -> usize {
    debug_assert!(value >= 0, "negative bin value {value}");
    (value.max(0) as usize).min(max_bin)
}

#[derive(Debug, Clone)]
pub struct TransposedData {
    pub features: Array2<i16>, // Transposed: (n_features, n_samples), i16 for cache efficiency
    pub n_samples: usize,
    pub n_features: usize,
}

impl TransposedData {
    /// Create TransposedData from binned Array2<i16> produced by histogram_builder.
    /// Input: (n_samples, n_features), Output: transposed to (n_features, n_samples).
    /// Note: this always allocates a new, feature-major copy of the data.
    pub fn from_binned(binned: Array2<i16>) -> Self {
        Self::from_binned_view(binned.view())
    }

    /// Create TransposedData from binned ArrayView2<i16> (avoids extra allocation)
    pub fn from_binned_view(binned: ArrayView2<'_, i16>) -> Self {
        let (n_samples, n_features) = binned.dim();
        if n_samples == 0 || n_features == 0 {
            return Self {
                features: Array2::zeros((0, 0)),
                n_samples: 0,
                n_features: 0,
            };
        }

        // Transpose and own
        let features = binned.t().as_standard_layout().to_owned();

        Self {
            features,
            n_samples,
            n_features,
        }
    }

    /// Legacy: Create from Vec<Vec<i32>> (deprecated, use from_binned instead)
    #[deprecated(note = "Use from_binned() with Array2<i16> for zero-copy performance")]
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
                            features[[feature_idx, sample_idx]] = row[feature_idx] as i16;
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

    pub fn get_feature_values(&self, feature_idx: usize) -> &[i16] {
        self.features.row(feature_idx).to_slice().unwrap()
    }

    #[inline]
    pub fn build_histogram_vectorized(
        &self,
        feature_idx: usize,
        indices: &[usize],
        grad: &ArrayView1<f64>,
        hess: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
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
        let grad_slice = grad.as_slice().unwrap();
        let hess_slice = hess.as_slice().unwrap();
        let y_slice = y.as_slice().unwrap();

        let len = indices.len();
        let max_bin = n_bins - 1;

        // The loops below use unchecked indexing, so these checks are what keep
        // this safe function sound in release builds. They are O(len) and cheap
        // compared to the histogram work itself.
        assert!(
            feature_idx < self.n_features,
            "feature_idx {feature_idx} out of range (n_features = {})",
            self.n_features
        );
        assert_eq!(feature_values.len(), self.n_samples);
        assert!(
            grad_slice.len() >= self.n_samples
                && hess_slice.len() >= self.n_samples
                && y_slice.len() >= self.n_samples,
            "grad/hess/y must have at least n_samples ({}) elements",
            self.n_samples
        );
        assert!(
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
                let bin0 = bin_index(*feature_values.get_unchecked(idx0), max_bin);
                let bin1 = bin_index(*feature_values.get_unchecked(idx1), max_bin);
                let bin2 = bin_index(*feature_values.get_unchecked(idx2), max_bin);
                let bin3 = bin_index(*feature_values.get_unchecked(idx3), max_bin);
                let bin4 = bin_index(*feature_values.get_unchecked(idx4), max_bin);
                let bin5 = bin_index(*feature_values.get_unchecked(idx5), max_bin);
                let bin6 = bin_index(*feature_values.get_unchecked(idx6), max_bin);
                let bin7 = bin_index(*feature_values.get_unchecked(idx7), max_bin);

                // Phase 3: Fetch all grad/hess/y values
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

                // Phase 4: Accumulate
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

        // Handle remaining elements
        while i < len {
            unsafe {
                let idx = *indices.get_unchecked(i);
                let bin = bin_index(*feature_values.get_unchecked(idx), max_bin);
                *hist_grad.get_unchecked_mut(bin) += *grad_slice.get_unchecked(idx);
                *hist_hess.get_unchecked_mut(bin) += *hess_slice.get_unchecked(idx);
                *hist_y.get_unchecked_mut(bin) += *y_slice.get_unchecked(idx);
                *hist_count.get_unchecked_mut(bin) += 1.0;
            }
            i += 1;
        }

        CachedHistogram::new(hist_grad, hist_hess, hist_y, hist_count)
    }
}

#[derive(Debug, Clone, Default)]
pub struct CachedHistogram {
    pub grad: Array1<f64>,
    pub hess: Array1<f64>,
    pub y: Array1<f64>,
    pub count: Array1<f64>,
}

impl CachedHistogram {
    pub fn new(grad: Vec<f64>, hess: Vec<f64>, y: Vec<f64>, count: Vec<f64>) -> Self {
        Self {
            grad: Array1::from(grad),
            hess: Array1::from(hess),
            y: Array1::from(y),
            count: Array1::from(count),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.grad.len()
    }

    pub fn as_slices(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        (
            self.grad.as_slice().unwrap(),
            self.hess.as_slice().unwrap(),
            self.y.as_slice().unwrap(),
            self.count.as_slice().unwrap(),
        )
    }

    pub fn build_vectorized(
        transposed_data: &TransposedData,
        y: &ArrayView1<f64>,
        grad: &ArrayView1<f64>,
        hess: &ArrayView1<f64>,
        indices: &[usize],
        feature_idx: usize,
        n_bins: usize,
    ) -> Self {
        transposed_data.build_histogram_vectorized(feature_idx, indices, grad, hess, y, n_bins)
    }

    /// Fast subtraction using unsafe array access (parent - sibling = other_child)
    #[inline]
    pub fn subtract(&self, other: &Self) -> Self {
        let n = self.grad.len();
        let mut grad_out = vec![0.0; n];
        let mut hess_out = vec![0.0; n];
        let mut y_out = vec![0.0; n];
        let mut count_out = vec![0.0; n];

        let grad_self = self.grad.as_slice().unwrap();
        let hess_self = self.hess.as_slice().unwrap();
        let y_self = self.y.as_slice().unwrap();
        let count_self = self.count.as_slice().unwrap();

        let grad_other = other.grad.as_slice().unwrap();
        let hess_other = other.hess.as_slice().unwrap();
        let y_other = other.y.as_slice().unwrap();
        let count_other = other.count.as_slice().unwrap();

        // Required for the unchecked indexing below to be sound.
        assert!(
            hess_self.len() == n
                && y_self.len() == n
                && count_self.len() == n
                && grad_other.len() == n
                && hess_other.len() == n
                && y_other.len() == n
                && count_other.len() == n,
            "histogram length mismatch in subtract"
        );

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
            grad: Array1::from(grad_out),
            hess: Array1::from(hess_out),
            y: Array1::from(y_out),
            count: Array1::from(count_out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample() -> TransposedData {
        // 4 samples x 2 features
        TransposedData::from_binned(array![[0i16, 1], [1, 1], [2, 0], [1, 0]])
    }

    #[test]
    fn transposes_layout() {
        let t = sample();
        assert_eq!((t.n_features, t.n_samples), (2, 4));
        assert_eq!(t.get_feature_values(0), &[0, 1, 2, 1]);
        assert_eq!(t.get_feature_values(1), &[1, 1, 0, 0]);
    }

    #[test]
    fn histogram_matches_naive() {
        let n = 37; // exercises both the 8x unrolled loop and the tail
        let binned = Array2::from_shape_fn((n, 1), |(i, _)| (i % 5) as i16);
        let t = TransposedData::from_binned(binned);
        let g = Array1::from_shape_fn(n, |i| i as f64 * 0.5);
        let h = Array1::from_shape_fn(n, |i| 1.0 + i as f64);
        let y = Array1::from_shape_fn(n, |i| (i % 2) as f64);
        let idx: Vec<usize> = (0..n).rev().collect();
        let hist = t.build_histogram_vectorized(0, &idx, &g.view(), &h.view(), &y.view(), 5);
        for b in 0..5 {
            let members: Vec<usize> = (0..n).filter(|i| i % 5 == b).collect();
            assert_eq!(hist.count[b], members.len() as f64);
            assert_eq!(hist.grad[b], members.iter().map(|&i| g[i]).sum::<f64>());
            assert_eq!(hist.hess[b], members.iter().map(|&i| h[i]).sum::<f64>());
            assert_eq!(hist.y[b], members.iter().map(|&i| y[i]).sum::<f64>());
        }
    }

    #[test]
    #[should_panic(expected = "Invalid sample index")]
    fn rejects_out_of_range_index() {
        let t = sample();
        let v = Array1::<f64>::zeros(4);
        t.build_histogram_vectorized(0, &[0, 4], &v.view(), &v.view(), &v.view(), 3);
    }

    #[test]
    #[should_panic(expected = "at least n_samples")]
    fn rejects_short_gradients() {
        let t = sample();
        let v = Array1::<f64>::zeros(4);
        let short = Array1::<f64>::zeros(2);
        t.build_histogram_vectorized(0, &[0], &short.view(), &v.view(), &v.view(), 3);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn subtract_rejects_mismatched_lengths() {
        let a = CachedHistogram::new(vec![1.0; 5], vec![1.0; 5], vec![1.0; 5], vec![1.0; 5]);
        let b = CachedHistogram::new(vec![1.0; 3], vec![1.0; 3], vec![1.0; 3], vec![1.0; 3]);
        a.subtract(&b);
    }

    #[test]
    fn subtract_roundtrip() {
        let p = CachedHistogram::new(vec![5.0; 6], vec![4.0; 6], vec![3.0; 6], vec![2.0; 6]);
        let c = CachedHistogram::new(vec![1.0; 6], vec![1.0; 6], vec![1.0; 6], vec![1.0; 6]);
        let d = p.subtract(&c);
        assert!(d.grad.iter().all(|&x| x == 4.0) && d.count.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn negative_bin_goes_to_zero_in_release() {
        assert_eq!(bin_index(3, 4), 3);
        assert_eq!(bin_index(9, 4), 4);
        if !cfg!(debug_assertions) {
            assert_eq!(bin_index(-1, 4), 0);
        }
    }
}
