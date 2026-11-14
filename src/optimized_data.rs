use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Map a stored bin value to a histogram slot.
#[inline(always)]
fn bin_index(value: u8, max_bin: usize) -> usize {
    (value as usize).min(max_bin)
}

/// Convert a legacy `i32` bin to the stored `u8` form.
#[inline]
fn saturate_bin(value: i32) -> u8 {
    value.clamp(0, 255) as u8
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

/// Chunk size for [`gather_node_stats_par`].
pub const GATHER_CHUNK: usize = 8192;

/// Parallel version of [`gather_node_stats`]. Output position `k` always holds
/// the stats for `indices[k]`: each output chunk is paired with the input
/// chunk at the same offset via `zip` and written in place, so correctness
/// does not depend on the order in which Rayon collects or flattens results.
pub fn gather_node_stats_par(
    indices: &[usize],
    grad: &[f64],
    hess: &[f64],
    y: &[f64],
) -> Vec<[f64; 4]> {
    let mut out = vec![[0.0f64; 4]; indices.len()];
    out.par_chunks_mut(GATHER_CHUNK)
        .zip(indices.par_chunks(GATHER_CHUNK))
        .for_each(|(dst, idx)| {
            for (d, &i) in dst.iter_mut().zip(idx) {
                *d = [grad[i], hess[i], y[i], 1.0];
            }
        });
    out
}

#[cold]
#[inline(never)]
fn invalid_sample_index(idx: usize, n_samples: usize) -> ! {
    panic!("Invalid sample index found: {idx} >= n_samples ({n_samples})")
}

#[derive(Debug, Clone)]
pub struct TransposedData {
    pub features: Array2<u8>, // Transposed: (n_features, n_samples), u8 for cache efficiency
    pub n_samples: usize,
    pub n_features: usize,
}

impl TransposedData {
    /// Create TransposedData from already-transposed Array2<u8> produced by histogram_builder.
    /// Input: (n_features, n_samples). Zero copies!
    pub fn from_binned_transposed_u8(features: Array2<u8>) -> Self {
        let (n_features, n_samples) = features.dim();
        Self {
            features,
            n_samples,
            n_features,
        }
    }

    /// Create TransposedData from row-major binned Array2<u8>.
    /// Input: (n_samples, n_features), Output: transposed to (n_features, n_samples).
    pub fn from_binned_u8(binned: Array2<u8>) -> Self {
        Self::from_feature_major(binned.reversed_axes().as_standard_layout().into_owned())
    }

    /// Create TransposedData from binned Array2<i16> produced by histogram_builder.
    /// Input: (n_samples, n_features), Output: transposed to (n_features, n_samples) as u8.
    pub fn from_binned(binned: Array2<i16>) -> Self {
        let binned_u8 = binned.mapv(|v| v.clamp(0, 255) as u8);
        Self::from_feature_major(binned_u8.reversed_axes().as_standard_layout().into_owned())
    }

    /// Create TransposedData from binned ArrayView2<i16> (avoids extra allocation)
    pub fn from_binned_view(binned: ArrayView2<'_, i16>) -> Self {
        let binned_u8 = binned.map(|&v| v.clamp(0, 255) as u8);
        Self::from_feature_major(binned_u8.t().as_standard_layout().into_owned())
    }

    /// Create TransposedData from binned ArrayView2<u8>
    pub fn from_binned_view_u8(binned: ArrayView2<'_, u8>) -> Self {
        Self::from_feature_major(binned.t().as_standard_layout().into_owned())
    }

    /// `features` is already (n_features, n_samples) in standard layout.
    /// Empty inputs keep their real dimensions, e.g. (n_features, 0).
    fn from_feature_major(features: Array2<u8>) -> Self {
        debug_assert!(features.is_standard_layout());
        let (n_features, n_samples) = features.dim();
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
                            features[[feature_idx, sample_idx]] = saturate_bin(row[feature_idx]);
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
            return CachedHistogram::from_bins(vec![]);
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

        // Interleaved [grad, hess, y, count] per bin: the four accumulators
        // for a bin are adjacent in memory (32 bytes) instead of spread across
        // four separate arrays. Per-bin summation order is
        // unchanged, so results are bit-identical to the old layout.
        let mut bins_0 = vec![[0.0f64; 4]; n_bins];
        let mut bins_1 = vec![[0.0f64; 4]; n_bins];
        let n_samples = self.n_samples;

        // 8x unrolled with dual sub-accumulators (bins_0 and bins_1) to eliminate
        // Read-After-Write (RAW) pipeline dependency stalls on identical bins.
        let mut chunks = indices.chunks_exact(8);
        for chunk in &mut chunks {
            let max_idx = chunk.iter().copied().fold(0, usize::max);
            if max_idx >= n_samples {
                invalid_sample_index(max_idx, n_samples);
            }
            unsafe {
                let idx0 = *chunk.get_unchecked(0);
                let idx1 = *chunk.get_unchecked(1);
                let bin0 = bin_index(*feature_values.get_unchecked(idx0), max_bin);
                let bin1 = bin_index(*feature_values.get_unchecked(idx1), max_bin);
                let e0 = bins_0.get_unchecked_mut(bin0);
                let e1 = bins_1.get_unchecked_mut(bin1);
                e0[0] += *grad_slice.get_unchecked(idx0);
                e0[1] += *hess_slice.get_unchecked(idx0);
                e0[2] += *y_slice.get_unchecked(idx0);
                e0[3] += 1.0;
                e1[0] += *grad_slice.get_unchecked(idx1);
                e1[1] += *hess_slice.get_unchecked(idx1);
                e1[2] += *y_slice.get_unchecked(idx1);
                e1[3] += 1.0;

                let idx2 = *chunk.get_unchecked(2);
                let idx3 = *chunk.get_unchecked(3);
                let bin2 = bin_index(*feature_values.get_unchecked(idx2), max_bin);
                let bin3 = bin_index(*feature_values.get_unchecked(idx3), max_bin);
                let e2 = bins_0.get_unchecked_mut(bin2);
                let e3 = bins_1.get_unchecked_mut(bin3);
                e2[0] += *grad_slice.get_unchecked(idx2);
                e2[1] += *hess_slice.get_unchecked(idx2);
                e2[2] += *y_slice.get_unchecked(idx2);
                e2[3] += 1.0;
                e3[0] += *grad_slice.get_unchecked(idx3);
                e3[1] += *hess_slice.get_unchecked(idx3);
                e3[2] += *y_slice.get_unchecked(idx3);
                e3[3] += 1.0;

                let idx4 = *chunk.get_unchecked(4);
                let idx5 = *chunk.get_unchecked(5);
                let bin4 = bin_index(*feature_values.get_unchecked(idx4), max_bin);
                let bin5 = bin_index(*feature_values.get_unchecked(idx5), max_bin);
                let e4 = bins_0.get_unchecked_mut(bin4);
                let e5 = bins_1.get_unchecked_mut(bin5);
                e4[0] += *grad_slice.get_unchecked(idx4);
                e4[1] += *hess_slice.get_unchecked(idx4);
                e4[2] += *y_slice.get_unchecked(idx4);
                e4[3] += 1.0;
                e5[0] += *grad_slice.get_unchecked(idx5);
                e5[1] += *hess_slice.get_unchecked(idx5);
                e5[2] += *y_slice.get_unchecked(idx5);
                e5[3] += 1.0;

                let idx6 = *chunk.get_unchecked(6);
                let idx7 = *chunk.get_unchecked(7);
                let bin6 = bin_index(*feature_values.get_unchecked(idx6), max_bin);
                let bin7 = bin_index(*feature_values.get_unchecked(idx7), max_bin);
                let e6 = bins_0.get_unchecked_mut(bin6);
                let e7 = bins_1.get_unchecked_mut(bin7);
                e6[0] += *grad_slice.get_unchecked(idx6);
                e6[1] += *hess_slice.get_unchecked(idx6);
                e6[2] += *y_slice.get_unchecked(idx6);
                e6[3] += 1.0;
                e7[0] += *grad_slice.get_unchecked(idx7);
                e7[1] += *hess_slice.get_unchecked(idx7);
                e7[2] += *y_slice.get_unchecked(idx7);
                e7[3] += 1.0;
            }
        }
        for &idx in chunks.remainder() {
            if idx >= n_samples {
                invalid_sample_index(idx, n_samples);
            }
            let e = &mut bins_0[bin_index(feature_values[idx], max_bin)];
            e[0] += grad_slice[idx];
            e[1] += hess_slice[idx];
            e[2] += y_slice[idx];
            e[3] += 1.0;
        }

        // Merge sub-accumulators (only n_bins operations)
        for i in 0..n_bins {
            bins_0[i][0] += bins_1[i][0];
            bins_0[i][1] += bins_1[i][1];
            bins_0[i][2] += bins_1[i][2];
            bins_0[i][3] += bins_1[i][3];
        }

        CachedHistogram::from_bins(bins_0)
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
            return CachedHistogram::from_bins(vec![]);
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
        let mut bins_0 = vec![[0.0f64; 4]; n_bins];
        let mut bins_1 = vec![[0.0f64; 4]; n_bins];

        let mut idx_chunks = indices.chunks_exact(8);
        let mut stat_chunks = stats.chunks_exact(8);
        for (ic, sc) in (&mut idx_chunks).zip(&mut stat_chunks) {
            let max_idx = ic.iter().copied().fold(0, usize::max);
            if max_idx >= n_samples {
                invalid_sample_index(max_idx, n_samples);
            }
            unsafe {
                let idx0 = *ic.get_unchecked(0);
                let idx1 = *ic.get_unchecked(1);
                let bin0 = bin_index(*feature_values.get_unchecked(idx0), max_bin);
                let bin1 = bin_index(*feature_values.get_unchecked(idx1), max_bin);
                let st0 = sc.get_unchecked(0);
                let st1 = sc.get_unchecked(1);
                let e0 = bins_0.get_unchecked_mut(bin0);
                let e1 = bins_1.get_unchecked_mut(bin1);
                e0[0] += st0[0];
                e0[1] += st0[1];
                e0[2] += st0[2];
                e0[3] += st0[3];
                e1[0] += st1[0];
                e1[1] += st1[1];
                e1[2] += st1[2];
                e1[3] += st1[3];

                let idx2 = *ic.get_unchecked(2);
                let idx3 = *ic.get_unchecked(3);
                let bin2 = bin_index(*feature_values.get_unchecked(idx2), max_bin);
                let bin3 = bin_index(*feature_values.get_unchecked(idx3), max_bin);
                let st2 = sc.get_unchecked(2);
                let st3 = sc.get_unchecked(3);
                let e2 = bins_0.get_unchecked_mut(bin2);
                let e3 = bins_1.get_unchecked_mut(bin3);
                e2[0] += st2[0];
                e2[1] += st2[1];
                e2[2] += st2[2];
                e2[3] += st2[3];
                e3[0] += st3[0];
                e3[1] += st3[1];
                e3[2] += st3[2];
                e3[3] += st3[3];

                let idx4 = *ic.get_unchecked(4);
                let idx5 = *ic.get_unchecked(5);
                let bin4 = bin_index(*feature_values.get_unchecked(idx4), max_bin);
                let bin5 = bin_index(*feature_values.get_unchecked(idx5), max_bin);
                let st4 = sc.get_unchecked(4);
                let st5 = sc.get_unchecked(5);
                let e4 = bins_0.get_unchecked_mut(bin4);
                let e5 = bins_1.get_unchecked_mut(bin5);
                e4[0] += st4[0];
                e4[1] += st4[1];
                e4[2] += st4[2];
                e4[3] += st4[3];
                e5[0] += st5[0];
                e5[1] += st5[1];
                e5[2] += st5[2];
                e5[3] += st5[3];

                let idx6 = *ic.get_unchecked(6);
                let idx7 = *ic.get_unchecked(7);
                let bin6 = bin_index(*feature_values.get_unchecked(idx6), max_bin);
                let bin7 = bin_index(*feature_values.get_unchecked(idx7), max_bin);
                let st6 = sc.get_unchecked(6);
                let st7 = sc.get_unchecked(7);
                let e6 = bins_0.get_unchecked_mut(bin6);
                let e7 = bins_1.get_unchecked_mut(bin7);
                e6[0] += st6[0];
                e6[1] += st6[1];
                e6[2] += st6[2];
                e6[3] += st6[3];
                e7[0] += st7[0];
                e7[1] += st7[1];
                e7[2] += st7[2];
                e7[3] += st7[3];
            }
        }
        for (&idx, st) in idx_chunks.remainder().iter().zip(stat_chunks.remainder()) {
            if idx >= n_samples {
                invalid_sample_index(idx, n_samples);
            }
            let e = &mut bins_0[bin_index(feature_values[idx], max_bin)];
            e[0] += st[0];
            e[1] += st[1];
            e[2] += st[2];
            e[3] += st[3];
        }

        // Merge sub-accumulators (only n_bins operations)
        for i in 0..n_bins {
            bins_0[i][0] += bins_1[i][0];
            bins_0[i][1] += bins_1[i][1];
            bins_0[i][2] += bins_1[i][2];
            bins_0[i][3] += bins_1[i][3];
        }

        CachedHistogram::from_bins(bins_0)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CachedHistogram {
    pub bins: Vec<[f64; 4]>,
}

impl CachedHistogram {
    pub fn new(grad: Vec<f64>, hess: Vec<f64>, y: Vec<f64>, count: Vec<f64>) -> Self {
        let n = grad.len();
        assert!(
            hess.len() == n && y.len() == n && count.len() == n,
            "histogram length mismatch in CachedHistogram::new"
        );
        let mut bins = Vec::with_capacity(n);
        for i in 0..n {
            bins.push([grad[i], hess[i], y[i], count[i]]);
        }
        Self { bins }
    }

    #[inline(always)]
    pub fn from_bins(bins: Vec<[f64; 4]>) -> Self {
        Self { bins }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bins.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
    }

    #[inline(always)]
    pub fn as_bins(&self) -> &[[f64; 4]] {
        &self.bins
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

    /// Fast subtraction (parent - sibling = other_child)
    /// Operates on contiguous [f64; 4] bins (vectorized with AVX-256).
    #[inline]
    pub fn subtract(&self, other: &Self) -> Self {
        let n = self.bins.len();
        assert_eq!(n, other.bins.len(), "histogram length mismatch in subtract");

        let mut out = vec![[0.0f64; 4]; n];
        let self_slice = &self.bins;
        let other_slice = &other.bins;

        for i in 0..n {
            let s = self_slice[i];
            let o = other_slice[i];
            out[i] = [
                s[0] - o[0],
                s[1] - o[1],
                s[2] - o[2],
                s[3] - o[3],
            ];
        }

        Self { bins: out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

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
            assert_eq!(hist.bins[b][3], members.len() as f64);
            assert_eq!(hist.bins[b][0], members.iter().map(|&i| g[i]).sum::<f64>());
            assert_eq!(hist.bins[b][1], members.iter().map(|&i| h[i]).sum::<f64>());
            assert_eq!(hist.bins[b][2], members.iter().map(|&i| y[i]).sum::<f64>());
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
        assert!(d.bins.iter().all(|b| b[0] == 4.0 && b[3] == 1.0));
    }

    #[test]
    fn bin_index_clamps_to_max_bin() {
        assert_eq!(bin_index(3, 4), 3);
        assert_eq!(bin_index(4, 4), 4);
        assert_eq!(bin_index(9, 4), 4);
        assert_eq!(bin_index(255, 4), 4);
    }

    #[test]
    fn constructors_normalize_negative_bins() {
        let b = array![[-1i16, 3], [i16::MIN, -7], [2, 0]];
        let a = TransposedData::from_binned(b.clone());
        let v = TransposedData::from_binned_view(b.view());
        assert_eq!(a.features, array![[0u8, 0, 2], [3, 0, 0]]);
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
        let hist = t.build_histogram_vectorized(
            0,
            &[0, 1, 2, 3],
            &g.view(),
            &ones.view(),
            &ones.view(),
            3,
        );
        assert_eq!(hist.bins.iter().map(|b| b[0]).collect::<Vec<_>>(), vec![1001.0, 100.0, 10.0]);
        assert_eq!(hist.bins.iter().map(|b| b[3]).collect::<Vec<_>>(), vec![2.0, 1.0, 1.0]);
    }

    #[test]
    fn zero_samples_keeps_feature_count() {
        let t = TransposedData::from_binned(Array2::<i16>::zeros((0, 5)));
        assert_eq!((t.n_features, t.n_samples), (5, 0));
        assert_eq!(t.features.dim(), (5, 0));
        assert!(t.get_feature_values(4).is_empty());
        let e = Array1::<f64>::zeros(0);
        let hist = t.build_histogram_vectorized(2, &[], &e.view(), &e.view(), &e.view(), 4);
        assert_eq!(hist.bins.iter().map(|b| b[3]).collect::<Vec<_>>(), vec![0.0; 4]);
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
        assert_eq!(a.features, b.mapv(|v| v as u8).t());
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
    fn new_rejects_each_mismatched_field() {
        let full = || (vec![1.0; 5], vec![1.0; 5], vec![1.0; 5], vec![1.0; 5]);
        let short = vec![1.0; 3];
        for field in 0..4 {
            let (mut g, mut h, mut y, mut c) = full();
            match field {
                0 => g = short.clone(),
                1 => h = short.clone(),
                2 => y = short.clone(),
                _ => c = short.clone(),
            }
            let r = std::panic::catch_unwind(|| {
                CachedHistogram::new(g, h, y, c);
            });
            assert!(
                r.is_err(),
                "mismatch field={field} was accepted"
            );
        }
    }

    #[test]
    #[allow(deprecated)]
    fn from_rows_normalizes_like_other_constructors() {
        let rows = vec![
            vec![-1, 3],
            vec![i32::MIN, 40_000],
            vec![2, i16::MAX as i32],
        ];
        let t = TransposedData::from_rows(&rows);
        assert_eq!(t.features, array![[0u8, 0, 2], [3, 255, 255]]);
        let from_binned = TransposedData::from_binned(array![[-1i16, 3], [i16::MIN, 0], [2, 0]]);
        assert_eq!(t.features.row(0), from_binned.features.row(0));
        // negative -> bin 0, saturated -> last bin
        let ones = Array1::<f64>::ones(3);
        let h0 = t.build_histogram_vectorized(
            0,
            &[0, 1, 2],
            &ones.view(),
            &ones.view(),
            &ones.view(),
            3,
        );
        assert_eq!(h0.bins.iter().map(|b| b[3]).collect::<Vec<_>>(), vec![2.0, 0.0, 1.0]);
        let h1 = t.build_histogram_vectorized(
            1,
            &[0, 1, 2],
            &ones.view(),
            &ones.view(),
            &ones.view(),
            4,
        );
        assert_eq!(h1.bins.iter().map(|b| b[3]).collect::<Vec<_>>(), vec![0.0, 0.0, 0.0, 3.0]);
    }

    fn random_case(
        n: usize,
        nb: usize,
        seed: u64,
    ) -> (
        TransposedData,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Vec<usize>,
    ) {
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as usize
        };
        let t = TransposedData::from_binned(Array2::from_shape_fn((n, 3), |_| {
            (next() % (nb + 2)) as i16
        }));
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
            let stats = gather_node_stats(
                &idx,
                g.as_slice().unwrap(),
                h.as_slice().unwrap(),
                y.as_slice().unwrap(),
            );
            for f in 0..3 {
                let a = t.build_histogram_vectorized(f, &idx, &g.view(), &h.view(), &y.view(), nb);
                let b = t.build_histogram_ordered(f, &idx, &stats, nb);
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn ordered_rejects_bad_inputs() {
        let (t, g, h, y, idx) = random_case(50, 4, 9);
        let stats = gather_node_stats(
            &idx,
            g.as_slice().unwrap(),
            h.as_slice().unwrap(),
            y.as_slice().unwrap(),
        );
        assert!(
            std::panic::catch_unwind(|| t.build_histogram_ordered(3, &idx, &stats, 4)).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| t.build_histogram_ordered(0, &idx[1..], &stats, 4))
                .is_err()
        );
        for pos in [2usize, idx.len() - 1] {
            let mut bad = idx.clone();
            bad[pos] = 50;
            assert!(
                std::panic::catch_unwind(|| t.build_histogram_ordered(0, &bad, &stats, 4)).is_err()
            );
        }
        // gather itself is bounds-checked
        assert!(std::panic::catch_unwind(|| gather_node_stats(
            &[50],
            g.as_slice().unwrap(),
            h.as_slice().unwrap(),
            y.as_slice().unwrap()
        ))
        .is_err());
    }

    /// Distinct, position-dependent values so any misplaced element is caught.
    fn distinct_inputs(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let g = (0..n).map(|i| i as f64 + 0.25).collect();
        let h = (0..n).map(|i| -(i as f64) - 0.5).collect();
        let y = (0..n).map(|i| (i as f64) * 3.0 + 0.125).collect();
        (g, h, y)
    }

    #[test]
    fn parallel_gather_matches_sequential_across_chunk_boundaries() {
        let n = 100_000;
        let (g, h, y) = distinct_inputs(n);
        // A non-monotonic permutation-like order, so chunk misplacement or
        // within-chunk reordering cannot accidentally line up.
        let perm: Vec<usize> = (0..n).map(|k| (k * 7919 + 13) % n).collect();
        let c = GATHER_CHUNK;
        for len in [0, 1, c - 1, c, c + 1, 3 * c + 17, n] {
            let idx = &perm[..len];
            let seq = gather_node_stats(idx, &g, &h, &y);
            let par = pool(4).install(|| gather_node_stats_par(idx, &g, &h, &y));
            assert_eq!(par, seq, "len {len}");
            for (k, &i) in idx.iter().enumerate() {
                assert_eq!(par[k], [g[i], h[i], y[i], 1.0]);
            }
        }
    }

    #[test]
    fn parallel_gather_is_bounds_checked() {
        let (g, h, y) = distinct_inputs(10);
        let r = std::panic::catch_unwind(|| gather_node_stats_par(&[0, 10], &g, &h, &y));
        assert!(r.is_err());
    }

    pub(crate) fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }
}
