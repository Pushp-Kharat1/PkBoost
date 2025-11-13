use rayon::prelude::*;
use crate::adaptive_parallel::{adaptive_par_map, ParallelComplexity};

#[derive(Debug, Clone)]
pub struct OptimizedHistogramBuilder {
    pub max_bins: usize,
    pub bin_edges: Vec<Vec<f64>>,
    pub n_bins_per_feature: Vec<usize>,
    pub medians: Vec<f64>,
}

impl OptimizedHistogramBuilder {
    pub fn new(max_bins: usize) -> Self {
        Self { 
            max_bins, 
            bin_edges: Vec::new(), 
            n_bins_per_feature: Vec::new(),
            medians: Vec::new(),
        }
    }

    // OPTIMIZATION 1: Fast median calculation with sampling for large datasets
    #[inline]
    fn calculate_median(feature_values: &mut [f64]) -> f64 {
    if feature_values.is_empty() {
        return 0.0;
    }

    if feature_values.len() > 10000 {
        let sample_size = 10000;
        let step = feature_values.len() / sample_size;
        
        let mut sample: Vec<f64> = feature_values
            .iter()
            .step_by(step)
            .take(sample_size)
            .cloned()
            .collect();
        
        let mid = sample.len() / 2;
        let is_even = sample.len() % 2 == 0;
        
        let median_val = if is_even && mid > 0 {
            // For even length, we need both middle values
            let (_, median1, right) = sample.select_nth_unstable_by(
                mid,
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            );
            let median1_val = *median1;
            
            // Find the second median in the right partition
            let (_, median2, _) = right.select_nth_unstable_by(
                0,
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            );
            (median1_val + *median2) / 2.0
        } else {
            let (_, median, _) = sample.select_nth_unstable_by(
                mid,
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            );
            *median
        };
        
        median_val
    } else {
        let mid = feature_values.len() / 2;
        let is_even = feature_values.len() % 2 == 0;
        
        let median_val = if is_even && mid > 0 {
            // For even length, we need both middle values
            let (_, median1, right) = feature_values.select_nth_unstable_by(
                mid,
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            );
            let median1_val = *median1;
            
            // Find the second median in the right partition
            let (_, median2, _) = right.select_nth_unstable_by(
                0,
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            );
            (median1_val + *median2) / 2.0
        } else {
            let (_, median, _) = feature_values.select_nth_unstable_by(
                mid,
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            );
            *median
        };
        
        median_val
    }
}

    pub fn fit(&mut self, x: &[Vec<f64>]) -> &mut Self {
        if x.is_empty() { return self; }
        let n_features = x[0].len();
        
        // OPTIMIZATION 2: Parallel feature processing with load balancing
        let results: Vec<(Vec<f64>, usize, f64)> = (0..n_features)
            .into_par_iter()
            .map(|feature_idx| {
                // Extract feature values
                let mut valid_values: Vec<f64> = x.iter()
                    .filter_map(|row| {
                        let val = row[feature_idx];
                        if !val.is_nan() { Some(val) } else { None }
                    })
                    .collect();

                if valid_values.is_empty() {
                    return (vec![0.0], 1, 0.0);
                }

                let median = Self::calculate_median(&mut valid_values);
                
                // OPTIMIZATION 3: Use pdqsort (pattern-defeating quicksort) via unstable_sort
                valid_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                
                // OPTIMIZATION 4: Fast deduplication
                valid_values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

                // OPTIMIZATION 5: Smart binning strategy
                let edges = if valid_values.len() <= self.max_bins {
                    valid_values
                } else {
                    self.create_adaptive_bins(&valid_values)
                };
                
                (edges.clone(), edges.len(), median)
            }).collect();

        // Unpack results
        for (edges, n_bins, median) in results {
            self.bin_edges.push(edges);
            self.n_bins_per_feature.push(n_bins);
            self.medians.push(median);
        }
        self
    }

    // OPTIMIZATION 6: Adaptive binning (more bins where data is dense)
    #[inline]
    fn create_adaptive_bins(&self, sorted_values: &[f64]) -> Vec<f64> {
        if sorted_values.is_empty() {
            return vec![0.0];
        }

        let n = sorted_values.len();
        let len_minus_1 = n - 1;
        let mut bins = Vec::with_capacity(self.max_bins + 1);
        
        // Non-uniform quantiles: more bins in tails (where outliers matter)
        for i in 0..=self.max_bins {
            let q = if i < self.max_bins / 4 {
                // Lower tail: finer granularity
                (i as f64 / (self.max_bins as f64 / 4.0)) * 0.10
            } else if i > 3 * self.max_bins / 4 {
                // Upper tail: finer granularity
                0.90 + ((i - 3 * self.max_bins / 4) as f64 / (self.max_bins as f64 / 4.0)) * 0.10
            } else {
                // Middle: coarser granularity
                0.10 + ((i - self.max_bins / 4) as f64 / (self.max_bins as f64 / 2.0)) * 0.80
            };
            
            let idx = (len_minus_1 as f64 * q).round() as usize;
            bins.push(sorted_values[idx.min(len_minus_1)]);
        }
        
        // Fast dedup
        bins.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        bins.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
        bins
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<i32>> {
        adaptive_par_map(x, ParallelComplexity::Medium, |row| {
            self.transform_row(row)
        })
    }
    
    // OPTIMIZATION 7: Branchless transform with SIMD-friendly access pattern
    #[inline]
    fn transform_row(&self, row: &[f64]) -> Vec<i32> {
        row.iter()
            .enumerate()
            .map(|(feature_idx, &value)| {
                let imputed_value = if value.is_nan() {
                    self.medians[feature_idx]
                } else {
                    value
                };
                
                let edges = &self.bin_edges[feature_idx];
                let bin_idx = self.find_bin_fast(edges, imputed_value);
                let n_edges = self.n_bins_per_feature[feature_idx];
                
                // Branchless clamping
                let final_bin_idx = if n_edges > 0 { 
                    bin_idx.min(n_edges - 1) 
                } else { 
                    0 
                };
                final_bin_idx as i32
            })
            .collect()
    }
    
    pub fn transform_batched(&self, x: &[Vec<f64>], batch_size: usize) -> Vec<Vec<i32>> {
        let config = crate::adaptive_parallel::get_parallel_config();
        
        if config.memory_efficient_mode && x.len() > batch_size {
            let mut results = Vec::with_capacity(x.len());
            for chunk in x.chunks(batch_size) {
                let batch_result = self.transform(chunk);
                results.extend(batch_result);
            }
            results
        } else {
            self.transform(x)
        }
    }
    
    // OPTIMIZATION 8: Hybrid binary/exponential search
    #[inline(always)]
    fn find_bin_fast(&self, edges: &[f64], value: f64) -> usize {
        if edges.is_empty() { return 0; }
        
        // OPTIMIZATION: Linear search for tiny arrays (cache-friendly)
        if edges.len() <= 8 {
            return edges.iter()
                .position(|&x| x >= value)
                .unwrap_or(edges.len() - 1);
        }
        
        // OPTIMIZATION: Exponential search for skewed distributions
        if edges.len() <= 32 {
            let mut bound = 1;
            while bound < edges.len() && edges[bound] < value {
                bound *= 2;
            }
            let start = bound / 2;
            let end = bound.min(edges.len());
            return start + edges[start..end].iter()
                .position(|&x| x >= value)
                .unwrap_or(end - start - 1);
        }
        
        // Binary search for large arrays
        let mut left = 0;
        let mut right = edges.len();
        while left < right {
            let mid = left + (right - left) / 2;
            if edges[mid] < value {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left.min(edges.len() - 1)
    }
}