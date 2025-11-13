use simsimd::SpatialSimilarity;  // For f64-specific kernels (cosine, dot, etc.)
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug, Clone)]
pub struct TransposedData {
    pub features: Array2<i32>,
    pub n_samples: usize,
    pub n_features: usize,
}

impl TransposedData {
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
                            features[[feature_idx, sample_idx]] = row[feature_idx];
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

    pub fn get_feature_values(&self, feature_idx: usize) -> &[i32] {
        self.features.row(feature_idx).to_slice().unwrap()
    }

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
        let len = indices.len();
        
        // Scalar processing (optimal for irregular scatters; SimSIMD doesn't accelerate this directly)
        // For contiguous bins, uncomment the sorted variant below and sort indices by bin first
        for i in 0..len {
            let idx = indices[i];
            if idx < self.n_samples {
                let bin = feature_values[idx] as usize;
                if bin < n_bins {
                    hist_grad[bin] += grad[idx];
                    hist_hess[bin] += hess[idx];
                    hist_y[bin] += y[idx];
                    hist_count[bin] += 1.0;
                }
            }
        }

        // OPTIONAL: Variant for contiguous adds (requires pre-sorting indices by bin)
        // Assume indices are sorted by bin for demonstration; use radix sort for O(n) if bins < 256
        // This uses ndarray sums for grouped chunks (fast, but no SimSIMD here as it lacks simple reductions)
        /*
        use ndarray::Array1;
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_by_key(|&i| feature_values[i] as usize);  // O(n log n); use radix for better
        let mut bin_start = 0;
        while bin_start < len {
            let bin = feature_values[sorted_indices[bin_start]] as usize;
            if bin >= n_bins {
                bin_start += 1;
                continue;
            }
            let bin_end = (bin_start + 1..len).position(|j| feature_values[sorted_indices[j]] as usize != bin)
                .map_or(len, |k| bin_start + k);
            let chunk_size = bin_end - bin_start;
            if chunk_size >= 4 {  // Threshold for vectorization worth
                // Extract sub-arrays for grad/hess/y in this bin
                let grad_sub: Array1<f64> = Array1::from_iter((bin_start..bin_end).map(|k| {
                    let idx = sorted_indices[k];
                    grad[idx]
                }));
                let hess_sub: Array1<f64> = Array1::from_iter((bin_start..bin_end).map(|k| {
                    let idx = sorted_indices[k];
                    hess[idx]
                }));
                let y_sub: Array1<f64> = Array1::from_iter((bin_start..bin_end).map(|k| {
                    let idx = sorted_indices[k];
                    y[idx]
                }));
                // Use ndarray sum for fast reduction (SIMD-accelerated internally)
                hist_grad[bin] += grad_sub.sum();
                hist_hess[bin] += hess_sub.sum();
                hist_y[bin] += y_sub.sum();
                hist_count[bin] += chunk_size as f64;
            } else {
                // Scalar fallback for small chunks
                for k in bin_start..bin_end {
                    let idx = sorted_indices[k];
                    hist_grad[bin] += grad[idx];
                    hist_hess[bin] += hess[idx];
                    hist_y[bin] += y[idx];
                    hist_count[bin] += 1.0;
                }
            }
            bin_start = bin_end;
        }
        */

        CachedHistogram::new(hist_grad, hist_hess, hist_y, hist_count)
    }

    // Example: Using SimSIMD to its fullest for histogram similarity (e.g., compare grad/hess across features)
    // Dispatches to widest SIMD (AVX-512/NEON) for cosine; great for feature selection or adversarial checks
    pub fn compare_histograms_cosine(&self, hist1: &CachedHistogram, hist2: &CachedHistogram) -> Result<f64, &'static str> {
        if hist1.grad.len() != hist2.grad.len() {
            return Err("Histograms must have equal bins");
        }
        let grad1 = hist1.grad.as_slice().unwrap();
        let grad2 = hist2.grad.as_slice().unwrap();
        // Fullest extent: Cosine distance via SimSIMD (auto-dispatches to hardware SIMD); convert to similarity
        let cos_dist = f64::cosine(grad1, grad2).ok_or("Cosine computation failed")?;
        Ok(1.0 - cos_dist)
    }
}

#[derive(Debug, Clone)]
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

    pub fn subtract(&self, other: &Self) -> Self {
        let grad = &self.grad - &other.grad;
        let hess = &self.hess - &other.hess;
        let y = &self.y - &other.y;
        let count = &self.count - &other.count;

        Self { grad, hess, y, count }
    }

    pub fn as_slices(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        (
            self.grad.as_slice().unwrap(),
            self.hess.as_slice().unwrap(),
            self.y.as_slice().unwrap(),
            self.count.as_slice().unwrap(),
        )
    }
}