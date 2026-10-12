use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Map a stored bin value to a histogram slot.
///
/// Negative values are normalized to 0 when `TransposedData` is constructed,
/// so the hot loop only needs a clamp to `max_bin`. Going through `u16`
/// avoids sign extension; if a negative value is written into the public
/// `features` field afterwards it lands in the last bin. Either way the
/// result is always `<= max_bin`, which is what memory safety relies on.
#[inline(always)]
fn bin_index(value: i16, max_bin: usize) -> usize {
    (value as u16 as usize).min(max_bin)
}

/// Convert a legacy `i32` bin to the stored `i16` form with the same
/// normalization as the other constructors: negatives become 0, and values
/// above `i16::MAX` saturate to `i16::MAX` (which `bin_index` then clamps to
/// the last histogram bin) instead of wrapping.
#[inline]
fn saturate_bin(value: i32) -> i16 {
    value.clamp(0, i16::MAX as i32) as i16
}

/// Gather per-sample `[grad, hess, y, 1.0]` for one tree node, in `indices`
/// order, so every feature's histogram can read them sequentially instead of
/// re-gathering three random-access arrays per feature. Bounds-checked: this
/// runs once per node, not once per feature.
pub fn gather_node_stats(
    indices: &[usize],
    grad: &[f64],
    hess: &[f64],
    y: &[f64],
) -> Vec<[f64; 4]> {
    indices
        .iter()
        .map(|&i| [grad[i], hess[i], y[i], 1.0])
        .collect()
}

#[inline]
fn split_interleaved(bins: Vec<[f64; 4]>) -> CachedHistogram {
    let n = bins.len();
    let (mut g, mut h, mut y, mut c) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for b in bins {
        g.push(b[0]);
        h.push(b[1]);
        y.push(b[2]);
        c.push(b[3]);
    }
    CachedHistogram::new(g, h, y, c)
}

#[cold]
#[inline(never)]
fn invalid_sample_index(idx: usize, n_samples: usize) -> ! {
    panic!("Invalid sample index found: {idx} >= n_samples ({n_samples})")
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
    /// Allocates exactly one feature-major buffer; the input is consumed.
    pub fn from_binned(binned: Array2<i16>) -> Self {
        // reversed_axes() is a free metadata swap on the owned array; the
        // single copy happens in as_standard_layout(). into_owned() then moves
        // that buffer out of the Cow instead of cloning it again.
        Self::from_feature_major(binned.reversed_axes().as_standard_layout().into_owned())
    }

    /// Create TransposedData from binned ArrayView2<i16> (avoids extra allocation)
    /// One allocation for the owned output; previously `.to_owned()` on the
    /// Cow returned by `as_standard_layout()` cloned the reordered data a
    /// second time.
    pub fn from_binned_view(binned: ArrayView2<'_, i16>) -> Self {
        Self::from_feature_major(binned.t().as_standard_layout().into_owned())
    }

    /// `features` is already (n_features, n_samples) in standard layout.
    /// Empty inputs keep their real dimensions, e.g. (n_features, 0).
    fn from_feature_major(mut features: Array2<i16>) -> Self {
        debug_assert!(features.is_standard_layout());
        // The binner never emits negatives (NaNs are median-imputed), but
        // normalize defensively so a stray negative counts as bin 0. This runs
        // once per dataset instead of once per sample per histogram.
        features.mapv_inplace(|v| v.max(0));
        let (n_features, n_samples) = features.dim();
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
                            features[[feature_idx, sample_idx]] =
                                saturate_bin(row[feature_idx]);
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

        assert!(
            feature_idx < self.n_features,
            "feature_idx {feature_idx} out of range (n_features = {})",
            self.n_features
        );

        let feature_values = self.get_feature_values(feature_idx);
        let grad_slice = grad.as_slice().unwrap();
        let hess_slice = hess.as_slice().unwrap();
        let y_slice = y.as_slice().unwrap();

        let max_bin = n_bins - 1;

        // The loops below use unchecked indexing, so these checks are what keep
        // this safe function sound in release builds. They are O(len) and cheap
        // compared to the histogram work itself.
        assert_eq!(feature_values.len(), self.n_samples);
        assert!(
            grad_slice.len() >= self.n_samples
                && hess_slice.len() >= self.n_samples
                && y_slice.len() >= self.n_samples,
            "grad/hess/y must have at least n_samples ({}) elements",
            self.n_samples
        );

        // Interleaved [grad, hess, y, count] per bin: one cache line per
        // update instead of four separate arrays. Per-bin summation order is
        // unchanged, so results are bit-identical to the old layout.
        let mut bins = vec![[0.0f64; 4]; n_bins];
        let n_samples = self.n_samples;

        // 8x unrolled. Index validation is fused into the loop (one
        // predictable branch per 8 samples) instead of a separate pass.
        let mut chunks = indices.chunks_exact(8);
        for chunk in &mut chunks {
            let max_idx = chunk.iter().copied().fold(0, usize::max);
            if max_idx >= n_samples {
                invalid_sample_index(max_idx, n_samples);
            }
            for &idx in chunk {
                // SAFETY: idx < n_samples, and feature_values, grad, hess and
                // y all have at least n_samples elements (asserted above).
                // bin_index() always returns <= max_bin < bins.len().
                unsafe {
                    let bin = bin_index(*feature_values.get_unchecked(idx), max_bin);
                    let e = bins.get_unchecked_mut(bin);
                    e[0] += *grad_slice.get_unchecked(idx);
                    e[1] += *hess_slice.get_unchecked(idx);
                    e[2] += *y_slice.get_unchecked(idx);
                    e[3] += 1.0;
                }
            }
        }
        for &idx in chunks.remainder() {
            if idx >= n_samples {
                invalid_sample_index(idx, n_samples);
            }
            let e = &mut bins[bin_index(feature_values[idx], max_bin)];
            e[0] += grad_slice[idx];
            e[1] += hess_slice[idx];
            e[2] += y_slice[idx];
            e[3] += 1.0;
        }

        split_interleaved(bins)
    }

    /// Build one feature's histogram from stats pre-gathered by
    /// [`gather_node_stats`] for the same `indices`. Use this when building
    /// histograms for many features over the same node: the random-access
    /// reads drop from four per sample per feature to one.
    #[inline]
    pub fn build_histogram_ordered(
        &self,
        feature_idx: usize,
        indices: &[usize],
        stats: &[[f64; 4]],
        n_bins: usize,
    ) -> CachedHistogram {
        if n_bins == 0 {
            return CachedHistogram::new(vec![], vec![], vec![], vec![]);
        }
        assert!(
            feature_idx < self.n_features,
            "feature_idx {feature_idx} out of range (n_features = {})",
            self.n_features
        );
        assert_eq!(
            stats.len(),
            indices.len(),
            "stats must be gathered for the same indices"
        );

        let feature_values = self.get_feature_values(feature_idx);
        assert_eq!(feature_values.len(), self.n_samples);
        let n_samples = self.n_samples;
        let max_bin = n_bins - 1;
        let mut bins = vec![[0.0f64; 4]; n_bins];

        let mut idx_chunks = indices.chunks_exact(8);
        let mut stat_chunks = stats.chunks_exact(8);
        for (ic, sc) in (&mut idx_chunks).zip(&mut stat_chunks) {
            let max_idx = ic.iter().copied().fold(0, usize::max);
            if max_idx >= n_samples {
                invalid_sample_index(max_idx, n_samples);
            }
            for (&idx, st) in ic.iter().zip(sc) {
                // SAFETY: idx < n_samples == feature_values.len(), and
                // bin_index() always returns <= max_bin < bins.len().
                unsafe {
                    let bin = bin_index(*feature_values.get_unchecked(idx), max_bin);
                    let e = bins.get_unchecked_mut(bin);
                    e[0] += st[0];
                    e[1] += st[1];
                    e[2] += st[2];
                    e[3] += st[3];
                }
            }
        }
        for (&idx, st) in idx_chunks.remainder().iter().zip(stat_chunks.remainder()) {
            if idx >= n_samples {
                invalid_sample_index(idx, n_samples);
            }
            let e = &mut bins[bin_index(feature_values[idx], max_bin)];
            e[0] += st[0];
            e[1] += st[1];
            e[2] += st[2];
            e[3] += st[3];
        }

        split_interleaved(bins)
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

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.grad.is_empty()
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
    fn bin_index_clamps_to_max_bin() {
        assert_eq!(bin_index(3, 4), 3);
        assert_eq!(bin_index(4, 4), 4);
        assert_eq!(bin_index(9, 4), 4);
        assert_eq!(bin_index(i16::MAX, 4), 4);
        // Negatives never reach here via constructors, but must still be in range.
        assert!(bin_index(-1, 4) <= 4 && bin_index(i16::MIN, 4) <= 4);
    }

    #[test]
    fn constructors_normalize_negative_bins() {
        let b = array![[-1i16, 3], [i16::MIN, -7], [2, 0]];
        let a = TransposedData::from_binned(b.clone());
        let v = TransposedData::from_binned_view(b.view());
        assert_eq!(a.features, array![[0i16, 0, 2], [3, 0, 0]]);
        assert_eq!(a.features, v.features);
    }

    #[test]
    fn rejects_bad_index_in_unrolled_and_tail_loops() {
        let n = 20;
        let t = TransposedData::from_binned(Array2::<i16>::zeros((n, 1)));
        let v = Array1::<f64>::zeros(n);
        // position 3 is inside the 8x block, position 17 is in the tail
        for pos in [3usize, 17] {
            let mut idx: Vec<usize> = (0..18).collect();
            idx[pos] = n;
            let r = std::panic::catch_unwind(|| {
                t.build_histogram_vectorized(0, &idx, &v.view(), &v.view(), &v.view(), 2)
            });
            assert!(r.is_err(), "bad index at {pos} accepted");
        }
    }

    #[test]
    fn histogram_negative_and_overflow_bins() {
        // bins: -1 -> 0, 7 -> 2 (n_bins = 3), 1 -> 1
        let t = TransposedData::from_binned(array![[-1i16], [7], [1], [-5]]);
        let g = array![1.0, 10.0, 100.0, 1000.0];
        let ones = Array1::<f64>::ones(4);
        let hist = t.build_histogram_vectorized(0, &[0, 1, 2, 3], &g.view(), &ones.view(), &ones.view(), 3);
        assert_eq!(hist.grad.to_vec(), vec![1001.0, 100.0, 10.0]);
        assert_eq!(hist.count.to_vec(), vec![2.0, 1.0, 1.0]);
    }

    #[test]
    fn zero_samples_keeps_feature_count() {
        let t = TransposedData::from_binned(Array2::<i16>::zeros((0, 5)));
        assert_eq!((t.n_features, t.n_samples), (5, 0));
        assert_eq!(t.features.dim(), (5, 0));
        assert!(t.get_feature_values(4).is_empty());
        let e = Array1::<f64>::zeros(0);
        let hist = t.build_histogram_vectorized(2, &[], &e.view(), &e.view(), &e.view(), 4);
        assert_eq!(hist.count.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn zero_features_keeps_sample_count() {
        let t = TransposedData::from_binned_view(Array2::<i16>::zeros((7, 0)).view());
        assert_eq!((t.n_features, t.n_samples), (0, 7));
        assert_eq!(t.features.dim(), (0, 7));
    }

    #[test]
    fn owned_and_view_paths_agree() {
        let b = Array2::from_shape_fn((13, 6), |(i, j)| (i * 7 + j) as i16);
        let a = TransposedData::from_binned(b.clone());
        let v = TransposedData::from_binned_view(b.view());
        assert_eq!(a.features, v.features);
        assert_eq!(a.features, b.t());
        assert!(a.features.is_standard_layout());
    }

    #[test]
    #[should_panic(expected = "feature_idx 2 out of range")]
    fn rejects_invalid_feature_index() {
        let t = sample();
        let v = Array1::<f64>::zeros(4);
        t.build_histogram_vectorized(2, &[0], &v.view(), &v.view(), &v.view(), 3);
    }

    #[test]
    fn rejects_each_short_input() {
        let t = sample();
        let ok = Array1::<f64>::zeros(4);
        let short = Array1::<f64>::zeros(3);
        for which in 0..3 {
            let (g, h, y) = match which {
                0 => (&short, &ok, &ok),
                1 => (&ok, &short, &ok),
                _ => (&ok, &ok, &short),
            };
            let r = std::panic::catch_unwind(|| {
                t.build_histogram_vectorized(0, &[0], &g.view(), &h.view(), &y.view(), 3)
            });
            assert!(r.is_err(), "short input #{which} was accepted");
        }
    }

    #[test]
    fn subtract_rejects_each_mismatched_field() {
        let full = || CachedHistogram::new(vec![1.0; 5], vec![1.0; 5], vec![1.0; 5], vec![1.0; 5]);
        let short = Array1::<f64>::zeros(3);
        for side in 0..2 {
            for field in 0..4 {
                let mut bad = full();
                match field {
                    0 => bad.grad = short.clone(),
                    1 => bad.hess = short.clone(),
                    2 => bad.y = short.clone(),
                    _ => bad.count = short.clone(),
                }
                let good = full();
                let r = std::panic::catch_unwind(|| {
                    if side == 0 { bad.subtract(&good) } else { good.subtract(&bad) }
                });
                assert!(r.is_err(), "mismatch side={side} field={field} was accepted");
            }
        }
    }

    #[test]
    #[allow(deprecated)]
    fn from_rows_normalizes_like_other_constructors() {
        let rows = vec![vec![-1, 3], vec![i32::MIN, 40_000], vec![2, i16::MAX as i32]];
        let t = TransposedData::from_rows(&rows);
        assert_eq!(t.features, array![[0i16, 0, 2], [3, i16::MAX, i16::MAX]]);
        let from_binned = TransposedData::from_binned(array![[-1i16, 3], [i16::MIN, 0], [2, 0]]);
        assert_eq!(t.features.row(0), from_binned.features.row(0));
        // negative -> bin 0, saturated -> last bin
        let ones = Array1::<f64>::ones(3);
        let h0 = t.build_histogram_vectorized(0, &[0, 1, 2], &ones.view(), &ones.view(), &ones.view(), 3);
        assert_eq!(h0.count.to_vec(), vec![2.0, 0.0, 1.0]);
        let h1 = t.build_histogram_vectorized(1, &[0, 1, 2], &ones.view(), &ones.view(), &ones.view(), 4);
        assert_eq!(h1.count.to_vec(), vec![0.0, 0.0, 0.0, 3.0]);
    }

    fn random_case(n: usize, nb: usize, seed: u64) -> (TransposedData, Array1<f64>, Array1<f64>, Array1<f64>, Vec<usize>) {
        let mut s = seed;
        let mut next = move || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as usize };
        let t = TransposedData::from_binned(Array2::from_shape_fn((n, 3), |_| (next() % (nb + 2)) as i16));
        let g = Array1::from_shape_fn(n, |_| (next() % 1000) as f64 / 7.0 - 50.0);
        let h = Array1::from_shape_fn(n, |_| (next() % 1000) as f64 / 13.0);
        let y = Array1::from_shape_fn(n, |_| (next() % 2) as f64);
        let idx: Vec<usize> = (0..n).filter(|_| next() % 3 != 0).collect();
        (t, g, h, y, idx)
    }

    #[test]
    fn ordered_matches_vectorized_bit_for_bit() {
        for &(n, nb) in &[(0usize, 4usize), (5, 1), (37, 5), (1000, 32), (4099, 2)] {
            let (t, g, h, y, idx) = random_case(n, nb, n as u64 + 1);
            let stats = gather_node_stats(&idx, g.as_slice().unwrap(), h.as_slice().unwrap(), y.as_slice().unwrap());
            for f in 0..3 {
                let a = t.build_histogram_vectorized(f, &idx, &g.view(), &h.view(), &y.view(), nb);
                let b = t.build_histogram_ordered(f, &idx, &stats, nb);
                assert_eq!(a.grad, b.grad);
                assert_eq!(a.hess, b.hess);
                assert_eq!(a.y, b.y);
                assert_eq!(a.count, b.count);
            }
        }
    }

    #[test]
    fn ordered_rejects_bad_inputs() {
        let (t, g, h, y, idx) = random_case(50, 4, 9);
        let stats = gather_node_stats(&idx, g.as_slice().unwrap(), h.as_slice().unwrap(), y.as_slice().unwrap());
        assert!(std::panic::catch_unwind(|| t.build_histogram_ordered(3, &idx, &stats, 4)).is_err());
        assert!(std::panic::catch_unwind(|| t.build_histogram_ordered(0, &idx[1..], &stats, 4)).is_err());
        for pos in [2usize, idx.len() - 1] {
            let mut bad = idx.clone();
            bad[pos] = 50;
            assert!(std::panic::catch_unwind(|| t.build_histogram_ordered(0, &bad, &stats, 4)).is_err());
        }
        // gather itself is bounds-checked
        assert!(std::panic::catch_unwind(|| gather_node_stats(&[50], g.as_slice().unwrap(), h.as_slice().unwrap(), y.as_slice().unwrap())).is_err());
    }

}
