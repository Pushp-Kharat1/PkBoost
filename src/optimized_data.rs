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
    pub features: Array2<u8>,
    pub n_samples: usize,
    pub n_features: usize,
}

impl TransposedData {
    pub fn from_binned(binned: Array2<u8>) -> Self {
        let (n_samples, n_features) = binned.dim();
        let transposed = binned.t();
        let features = Array2::from_shape_vec(
            (n_features, n_samples),
            transposed.iter().cloned().collect(),
        )
        .unwrap();
        Self {
            features,
            n_samples,
            n_features,
        }
    }

    pub fn from_binned_view(binned: ArrayView2<'_, u8>) -> Self {
        let (n_samples, n_features) = binned.dim();
        let transposed = binned.t();
        let features = Array2::from_shape_vec(
            (n_features, n_samples),
            transposed.iter().cloned().collect(),
        )
        .unwrap();
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
        &self.features.as_slice().unwrap()[start..end]
    }

    pub fn build_histogram_into_f32(
        &self,
        feature_idx: usize,
        indices: &[u32],
        active_data: &ActiveData,
        n_bins: usize,
        out: &mut CachedHistogram,
    ) {
        out.reset(n_bins);
        let fv = self.get_feature_values(feature_idx);
        let max_b = (n_bins - 1) as usize;
        for i in 0..indices.len() {
            unsafe {
                let idx = *indices.get_unchecked(i) as usize;
                let b = (*fv.get_unchecked(idx) as usize).min(max_b);
                let s = &active_data.samples[i];
                let target = &mut out.bins[b];
                target.grad += s.grad;
                target.hess += s.hess;
                target.y += s.y;
                target.count += s.count;
            }
        }
        let mut s = BinData::default();
        for b in out.bins.iter() {
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

    pub fn gather_bins_4x(
        &self,
        f_indices: &[usize; 4],
        indices: &[u32],
        n_bins: &[usize; 4],
        out: &mut [u8],
    ) {
        let fv0 = self.get_feature_values(f_indices[0]);
        let fv1 = self.get_feature_values(f_indices[1]);
        let fv2 = self.get_feature_values(f_indices[2]);
        let fv3 = self.get_feature_values(f_indices[3]);
        let m0 = (n_bins[0] - 1) as u8;
        let m1 = (n_bins[1] - 1) as u8;
        let m2 = (n_bins[2] - 1) as u8;
        let m3 = (n_bins[3] - 1) as u8;
        for i in 0..indices.len() {
            unsafe {
                let idx = *indices.get_unchecked(i) as usize;
                let out_ptr = out.as_mut_ptr().add(i * 4);
                *out_ptr = (*fv0.get_unchecked(idx)).min(m0);
                *out_ptr.add(1) = (*fv1.get_unchecked(idx)).min(m1);
                *out_ptr.add(2) = (*fv2.get_unchecked(idx)).min(m2);
                *out_ptr.add(3) = (*fv3.get_unchecked(idx)).min(m3);
            }
        }
    }

    pub fn build_multi_histogram_4x(
        &self,
        active: &ActiveData,
        n_bins: &[usize; 4],
        pre_bins: &[u8],
        outputs: &mut [CachedHistogram],
    ) {
        for i in 0..4 {
            outputs[i].reset(n_bins[i]);
        }
        let len = active.samples.len();
        let samples_ptr = unsafe { active.as_m128_ptr() };
        for i in 0..len {
            unsafe {
                let s_v = _mm_load_ps(samples_ptr.add(i) as *const f32);
                let p = pre_bins.as_ptr().add(i * 4);
                let p0 = outputs[0].bins.as_mut_ptr().add(*p as usize) as *mut f32;
                let p1 = outputs[1].bins.as_mut_ptr().add(*p.add(1) as usize) as *mut f32;
                let p2 = outputs[2].bins.as_mut_ptr().add(*p.add(2) as usize) as *mut f32;
                let p3 = outputs[3].bins.as_mut_ptr().add(*p.add(3) as usize) as *mut f32;
                _mm_store_ps(p0, _mm_add_ps(_mm_load_ps(p0), s_v));
                _mm_store_ps(p1, _mm_add_ps(_mm_load_ps(p1), s_v));
                _mm_store_ps(p2, _mm_add_ps(_mm_load_ps(p2), s_v));
                _mm_store_ps(p3, _mm_add_ps(_mm_load_ps(p3), s_v));
            }
        }
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
}

pub fn gather_active_data(
    indices: &[u32],
    grad: &[f32],
    hess: &[f32],
    y: &[u8],
    out: &mut ActiveData,
) {
    let n = indices.len();
    out.resize(n);
    if n > 32768 {
        use crate::fork_parallel::pfor_indexed;
        let n_chunks = 16;
        let chunk_size = (n + n_chunks - 1) / n_chunks;
        let out_ptr = out.samples.as_mut_ptr() as usize;
        let g_ptr = grad.as_ptr() as usize;
        let h_ptr = hess.as_ptr() as usize;
        let y_ptr = y.as_ptr() as usize;
        let ind_ptr = indices.as_ptr() as usize;
        pfor_indexed(n_chunks, move |idx| {
            let start = idx * chunk_size;
            let end = ((idx + 1) * chunk_size).min(n);
            if start >= end {
                return;
            }
            unsafe {
                let os = (out_ptr as *mut SampleData).add(start);
                let ci = (ind_ptr as *const u32).add(start);
                for i in 0..(end - start) {
                    let idx = *ci.add(i) as usize;
                    *os.add(i) = SampleData {
                        grad: *(g_ptr as *const f32).add(idx),
                        hess: *(h_ptr as *const f32).add(idx),
                        y: *(y_ptr as *const u8).add(idx) as f32,
                        count: 1.0,
                    };
                }
            }
        });
    } else {
        for i in 0..n {
            let idx = indices[i] as usize;
            out.samples[i] = SampleData {
                grad: grad[idx],
                hess: hess[idx],
                y: y[idx] as f32,
                count: 1.0,
            };
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
        Self {
            bins,
            sums: TotalSums {
                grad: s.grad,
                hess: s.hess,
                y: s.y,
                count: s.count,
            },
        }
    }
    pub fn reset(&mut self, n: usize) {
        if self.bins.len() == n {
            self.bins.fill(BinData::default());
        } else {
            self.bins = vec![BinData::default(); n];
        }
        self.sums = TotalSums::default();
    }
    pub fn subtract_into(&self, other: &Self, out: &mut Self) {
        let n = self.bins.len();
        if out.bins.len() != n {
            out.bins.resize(n, BinData::default());
        }
        unsafe {
            let sp = self.bins.as_ptr() as *const f32;
            let op = other.bins.as_ptr() as *const f32;
            let dp = out.bins.as_mut_ptr() as *mut f32;
            for i in 0..n {
                let s = _mm_load_ps(sp.add(i * 4));
                let o = _mm_load_ps(op.add(i * 4));
                _mm_store_ps(dp.add(i * 4), _mm_sub_ps(s, o));
            }
        }
        out.sums.grad = self.sums.grad - other.sums.grad;
        out.sums.hess = self.sums.hess - other.sums.hess;
        out.sums.y = self.sums.y - other.sums.y;
        out.sums.count = self.sums.count - other.sums.count;
    }
}
