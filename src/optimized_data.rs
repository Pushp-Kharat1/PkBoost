use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SampleData {
    pub grad: f32,
    pub hess: f32,
    pub y: f32,
    pub count: f32,
}

impl SampleData {
    #[inline(always)]
    pub unsafe fn as_m128(&self) -> __m128 {
        _mm_load_ps(&self.grad as *const f32)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActiveData {
    pub samples: Vec<SampleData>,
}

impl ActiveData {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
        }
    }

    pub fn resize(&mut self, n: usize) {
        self.samples.resize(n, SampleData::default());
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    #[inline(always)]
    pub unsafe fn as_m128_ptr(&self) -> *const SampleData {
        self.samples.as_ptr()
    }
}

#[derive(Debug, Clone)]
pub struct TransposedData {
    pub features: Array2<u8>, // Transposed: (n_features, n_samples), u8 for minimal bandwidth
    pub n_samples: usize,
    pub n_features: usize,
}

impl TransposedData {
    /// Create TransposedData from binned Array2<u8> (zero-copy path from histogram_builder)
    pub fn from_binned(binned: Array2<u8>) -> Self {
        let (n_samples, n_features) = binned.dim();
        if n_samples == 0 || n_features == 0 {
            return Self {
                features: Array2::zeros((0, 0)),
                n_samples: 0,
                n_features: 0,
            };
        }

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

    pub fn from_binned_view(binned: ArrayView2<'_, u8>) -> Self {
        let (n_samples, n_features) = binned.dim();
        if n_samples == 0 || n_features == 0 {
            return Self {
                features: Array2::zeros((0, 0)),
                n_samples: 0,
                n_features: 0,
            };
        }

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

    #[inline]
    pub fn get_feature_values(&self, feature_idx: usize) -> &[u8] {
        let start = feature_idx * self.n_samples;
        let end = start + self.n_samples;
        &self
            .features
            .as_slice()
            .expect("TransposedData features array must be contiguous")[start..end]
    }

    #[inline]
    pub fn build_histogram_into_f32(
        &self,
        feature_idx: usize,
        indices: &[u32],
        active_data: &ActiveData,
        n_bins: usize,
        out: &mut CachedHistogram,
    ) {
        if n_bins == 0 {
            return;
        }

        out.reset(n_bins);
        let feature_values = self.get_feature_values(feature_idx);
        let len = indices.len();
        let max_bin = (n_bins - 1) as usize;

        let mut i = 0;
        let samples = &active_data.samples;
        let hist_bins = &mut out.bins;

        while i + 8 <= len {
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    // Software prefetch: fetch values for upcoming indices
                    // 16 iterations ahead is usually a good heuristic for modern CPUs
                    if i + 16 < len {
                        let next_idx = *indices.get_unchecked(i + 16) as usize;
                        _mm_prefetch(
                            (feature_values.get_unchecked(next_idx) as *const u8) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }

                for j in 0..8 {
                    let idx = *indices.get_unchecked(i + j) as usize;
                    let bin = (*feature_values.get_unchecked(idx) as usize).min(max_bin);
                    let sample = samples.get_unchecked(i + j);
                    let hist_bin = hist_bins.get_unchecked_mut(bin);

                    hist_bin.grad += sample.grad;
                    hist_bin.hess += sample.hess;
                    hist_bin.y += sample.y;
                    hist_bin.count += sample.count;
                }
            }
            i += 8;
        }

        while i < len {
            unsafe {
                let idx = *indices.get_unchecked(i) as usize;
                let bin = (*feature_values.get_unchecked(idx) as usize).min(max_bin);
                let sample = samples.get_unchecked(i);
                let hist_bin = hist_bins.get_unchecked_mut(bin);

                hist_bin.grad += sample.grad;
                hist_bin.hess += sample.hess;
                hist_bin.y += sample.y;
                hist_bin.count += sample.count;
            }
            i += 1;
        }

        // Final sum update
        let mut s = BinData::default();
        for b in hist_bins.iter() {
            s.grad += b.grad;
            s.hess += b.hess;
            s.y += b.y;
            s.count += b.count;
        }
        out.sums = TotalSums {
            grad: s.grad,
            hess: s.hess,
            y: s.y,
            count: s.count,
        };
    }

    /// OPTIMIZATION: Multi-feature histogram kernel (Pass 2 of Double-Pass)
    /// Processes 4 histograms using pre-gathered bin indices.
    /// This loop is 100% linear and cache-friendly.
    pub fn build_multi_histogram_4x(
        &self,
        active_data: &ActiveData,
        n_bins_per_feat: &[usize; 4],
        pre_gathered_bins: &[u8],        // [u8; 4] per sample
        outputs: &mut [CachedHistogram], // Expects exactly 4 histograms
    ) {
        debug_assert_eq!(outputs.len(), 4);
        let len = active_data.samples.len();
        debug_assert!(pre_gathered_bins.len() >= len * 4);

        for i in 0..4 {
            outputs[i].reset(n_bins_per_feat[i]);
        }

        let samples = active_data;

        let mut i = 0;
        // Hot loop: 100% linear access to samples and pre_gathered_bins
        while i + 4 <= len {
            unsafe {
                for j in 0..4 {
                    let sample = samples.as_m128_ptr().add(i + j);
                    let bins_ptr = pre_gathered_bins.as_ptr().add((i + j) * 4);

                    let bin0 = *bins_ptr as usize;
                    let bin1 = *bins_ptr.add(1) as usize;
                    let bin2 = *bins_ptr.add(2) as usize;
                    let bin3 = *bins_ptr.add(3) as usize;

                    let s_v = _mm_load_ps(sample as *const f32);

                    let b0_ptr = outputs[0].bins.as_mut_ptr().add(bin0) as *mut f32;
                    let b1_ptr = outputs[1].bins.as_mut_ptr().add(bin1) as *mut f32;
                    let b2_ptr = outputs[2].bins.as_mut_ptr().add(bin2) as *mut f32;
                    let b3_ptr = outputs[3].bins.as_mut_ptr().add(bin3) as *mut f32;

                    // SIMD Accumulation: 4 additions in 1 instruction
                    _mm_store_ps(b0_ptr, _mm_add_ps(_mm_load_ps(b0_ptr), s_v));
                    _mm_store_ps(b1_ptr, _mm_add_ps(_mm_load_ps(b1_ptr), s_v));
                    _mm_store_ps(b2_ptr, _mm_add_ps(_mm_load_ps(b2_ptr), s_v));
                    _mm_store_ps(b3_ptr, _mm_add_ps(_mm_load_ps(b3_ptr), s_v));
                }
            }
            i += 4;
        }

        // Remainder
        while i < len {
            unsafe {
                let sample = samples.samples.get_unchecked(i).as_m128();
                let bins_ptr = pre_gathered_bins.as_ptr().add(i * 4);

                let bin0 = *bins_ptr as usize;
                let bin1 = *bins_ptr.add(1) as usize;
                let bin2 = *bins_ptr.add(2) as usize;
                let bin3 = *bins_ptr.add(3) as usize;

                let b0_ptr = outputs[0].bins.as_mut_ptr().add(bin0) as *mut f32;
                let b1_ptr = outputs[1].bins.as_mut_ptr().add(bin1) as *mut f32;
                let b2_ptr = outputs[2].bins.as_mut_ptr().add(bin2) as *mut f32;
                let b3_ptr = outputs[3].bins.as_mut_ptr().add(bin3) as *mut f32;

                _mm_store_ps(b0_ptr, _mm_add_ps(_mm_load_ps(b0_ptr), sample));
                _mm_store_ps(b1_ptr, _mm_add_ps(_mm_load_ps(b1_ptr), sample));
                _mm_store_ps(b2_ptr, _mm_add_ps(_mm_load_ps(b2_ptr), sample));
                _mm_store_ps(b3_ptr, _mm_add_ps(_mm_load_ps(b3_ptr), sample));
            }
            i += 1;
        }

        // Final sum update for all 4 histograms
        for hist in outputs.iter_mut() {
            let mut s = BinData::default();
            for b in hist.bins.iter() {
                s.grad += b.grad;
                s.hess += b.hess;
                s.y += b.y;
                s.count += b.count;
            }
            hist.sums = TotalSums {
                grad: s.grad,
                hess: s.hess,
                y: s.y,
                count: s.count,
            };
        }
    }

    /// Gathers bin indices for 4 features into a compact [u8; 4] buffer.
    /// This is Pass 1 of the Double-Pass histogram strategy.
    pub fn gather_bins_4x(
        &self,
        feature_indices: &[usize; 4],
        indices: &[u32],
        n_bins_per_feat: &[usize; 4],
        out_bins: &mut [u8], // Must be at least indices.len() * 4
    ) {
        let len = indices.len();
        debug_assert!(out_bins.len() >= len * 4);

        let fv0 = self.get_feature_values(feature_indices[0]);
        let fv1 = self.get_feature_values(feature_indices[1]);
        let fv2 = self.get_feature_values(feature_indices[2]);
        let fv3 = self.get_feature_values(feature_indices[3]);

        let max0 = (n_bins_per_feat[0] - 1) as u8;
        let max1 = (n_bins_per_feat[1] - 1) as u8;
        let max2 = (n_bins_per_feat[2] - 1) as u8;
        let max3 = (n_bins_per_feat[3] - 1) as u8;

        let mut i = 0;
        while i + 4 <= len {
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if i + 16 < len {
                        let next_idx = *indices.get_unchecked(i + 16) as usize;
                        _mm_prefetch(
                            (fv0.get_unchecked(next_idx) as *const u8) as *const i8,
                            _MM_HINT_T0,
                        );
                        _mm_prefetch(
                            (fv1.get_unchecked(next_idx) as *const u8) as *const i8,
                            _MM_HINT_T0,
                        );
                        _mm_prefetch(
                            (fv2.get_unchecked(next_idx) as *const u8) as *const i8,
                            _MM_HINT_T0,
                        );
                        _mm_prefetch(
                            (fv3.get_unchecked(next_idx) as *const u8) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }

                for j in 0..4 {
                    let idx = *indices.get_unchecked(i + j) as usize;
                    let b0 = (*fv0.get_unchecked(idx)).min(max0);
                    let b1 = (*fv1.get_unchecked(idx)).min(max1);
                    let b2 = (*fv2.get_unchecked(idx)).min(max2);
                    let b3 = (*fv3.get_unchecked(idx)).min(max3);

                    let out_ptr = out_bins.as_mut_ptr().add((i + j) * 4);
                    *out_ptr = b0;
                    *out_ptr.add(1) = b1;
                    *out_ptr.add(2) = b2;
                    *out_ptr.add(3) = b3;
                }
            }
            i += 4;
        }

        while i < len {
            unsafe {
                let idx = *indices.get_unchecked(i) as usize;
                let b0 = (*fv0.get_unchecked(idx)).min(max0);
                let b1 = (*fv1.get_unchecked(idx)).min(max1);
                let b2 = (*fv2.get_unchecked(idx)).min(max2);
                let b3 = (*fv3.get_unchecked(idx)).min(max3);

                let out_ptr = out_bins.as_mut_ptr().add(i * 4);
                *out_ptr = b0;
                *out_ptr.add(1) = b1;
                *out_ptr.add(2) = b2;
                *out_ptr.add(3) = b3;
            }
            i += 1;
        }
    }
}

/// Gathers grad, hess, and y values into interleaved AoS buffers.
pub fn gather_active_data(
    indices: &[u32],
    grad: &[f32],
    hess: &[f32],
    y: &[u8],
    out: &mut ActiveData,
) {
    let n = indices.len();
    out.resize(n);

    // Parallelize larger gather operations using ForkUnion
    if n > 16384 {
        use crate::fork_parallel::pfor_indexed;
        let out_ptr = out.samples.as_mut_ptr() as usize;

        let grad_ptr = grad.as_ptr() as usize;
        let hess_ptr = hess.as_ptr() as usize;
        let y_ptr = y.as_ptr() as usize;
        let indices_ptr = indices.as_ptr() as usize;

        let n_chunks = (n / 2048).max(1).min(num_cpus::get() * 4);
        let chunk_size = (n + n_chunks - 1) / n_chunks;

        pfor_indexed(n_chunks, move |chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(n);
            if start >= end {
                return;
            }

            unsafe {
                let chunk_indices_ptr = (indices_ptr as *const u32).add(start);
                let out_samples_ptr = (out_ptr as *mut SampleData).add(start);

                // Scalar gather for now - can optimize with AVX2 later
                for j in 0..(end - start) {
                    let idx = *chunk_indices_ptr.add(j) as usize;
                    *out_samples_ptr.add(j) = SampleData {
                        grad: *(grad_ptr as *const f32).add(idx),
                        hess: *(hess_ptr as *const f32).add(idx),
                        y: *(y_ptr as *const u8).add(idx) as f32,
                        count: 1.0,
                    };
                }
            }
        });
    } else {
        // Sequential path for small nodes
        for (i, &idx) in indices.iter().enumerate() {
            let idx = idx as usize;
            unsafe {
                out.samples[i] = SampleData {
                    grad: *grad.get_unchecked(idx),
                    hess: *hess.get_unchecked(idx),
                    y: *y.get_unchecked(idx) as f32,
                    count: 1.0,
                };
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TotalSums {
    pub grad: f32,
    pub hess: f32,
    pub y: f32,
    pub count: f32,
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BinData {
    pub grad: f32,
    pub hess: f32,
    pub y: f32,
    pub count: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CachedHistogram {
    pub bins: Vec<BinData>,
    pub sums: TotalSums,
}

impl CachedHistogram {
    pub fn new(bins: Vec<BinData>) -> Self {
        let mut s = BinData::default();
        for b in bins.iter() {
            s.grad += b.grad;
            s.hess += b.hess;
            s.y += b.y;
            s.count += b.count;
        }
        let sums = TotalSums {
            grad: s.grad,
            hess: s.hess,
            y: s.y,
            count: s.count,
        };
        Self { bins, sums }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bins.len()
    }

    #[inline]
    pub fn reset(&mut self, n_bins: usize) {
        if self.bins.len() == n_bins {
            self.bins.fill(BinData::default());
        } else {
            self.bins = vec![BinData::default(); n_bins];
        }
        self.sums = TotalSums::default();
    }

    #[inline]
    pub fn subtract_into(&self, other: &Self, out: &mut Self) {
        let n = self.bins.len();
        out.reset(n);

        let mut i = 0;
        let self_bins = &self.bins;
        let other_bins = &other.bins;
        let out_bins = &mut out.bins;

        while i + 4 <= n {
            unsafe {
                let s0 = self_bins.get_unchecked(i);
                let o0 = other_bins.get_unchecked(i);
                let d0 = out_bins.get_unchecked_mut(i);
                d0.grad = s0.grad - o0.grad;
                d0.hess = s0.hess - o0.hess;
                d0.y = s0.y - o0.y;
                d0.count = s0.count - o0.count;

                let s1 = self_bins.get_unchecked(i + 1);
                let o1 = other_bins.get_unchecked(i + 1);
                let d1 = out_bins.get_unchecked_mut(i + 1);
                d1.grad = s1.grad - o1.grad;
                d1.hess = s1.hess - o1.hess;
                d1.y = s1.y - o1.y;
                d1.count = s1.count - o1.count;

                let s2 = self_bins.get_unchecked(i + 2);
                let o2 = other_bins.get_unchecked(i + 2);
                let d2 = out_bins.get_unchecked_mut(i + 2);
                d2.grad = s2.grad - o2.grad;
                d2.hess = s2.hess - o2.hess;
                d2.y = s2.y - o2.y;
                d2.count = s2.count - o2.count;

                let s3 = self_bins.get_unchecked(i + 3);
                let o3 = other_bins.get_unchecked(i + 3);
                let d3 = out_bins.get_unchecked_mut(i + 3);
                d3.grad = s3.grad - o3.grad;
                d3.hess = s3.hess - o3.hess;
                d3.y = s3.y - o3.y;
                d3.count = s3.count - o3.count;
            }
            i += 4;
        }

        while i < n {
            let s = &self_bins[i];
            let o = &other_bins[i];
            let d = &mut out_bins[i];
            d.grad = s.grad - o.grad;
            d.hess = s.hess - o.hess;
            d.y = s.y - o.y;
            d.count = s.count - o.count;
            i += 1;
        }

        out.sums.grad = self.sums.grad - other.sums.grad;
        out.sums.hess = self.sums.hess - other.sums.hess;
        out.sums.y = self.sums.y - other.sums.y;
        out.sums.count = self.sums.count - other.sums.count;
    }
}
