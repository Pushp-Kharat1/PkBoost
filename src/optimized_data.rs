use ndarray::{Array2, ArrayView1};
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
        let mut hist_grad = vec![0.0; n_bins];
        let mut hist_hess = vec![0.0; n_bins];
        let mut hist_y = vec![0.0; n_bins];
        let mut hist_count = vec![0.0; n_bins];

        let feature_values = self.get_feature_values(feature_idx);

        for &idx in indices {
            if idx >= self.n_samples {
                continue;
            }
            let bin = feature_values[idx] as usize;
            if bin >= n_bins {
                continue;
            }
            hist_grad[bin] += grad[idx];
            hist_hess[bin] += hess[idx];
            hist_y[bin] += y[idx];
            hist_count[bin] += 1.0;
        }

        CachedHistogram::new(hist_grad, hist_hess, hist_y, hist_count)
    }
}

#[derive(Debug, Clone)]
pub struct CachedHistogram {
    pub grad: Array1<f64>,
    pub hess: Array1<f64>,
    pub y: Array1<f64>,
    pub count: Array1<f64>,
}

use ndarray::Array1;

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
