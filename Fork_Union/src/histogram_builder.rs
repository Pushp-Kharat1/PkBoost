use fork_union::prelude::*;
use crate::adaptive_parallel::{adaptive_par_map, ParallelComplexity};
// SimSIMD dependency added for future SIMD optimizations
#[allow(unused_imports)]
use simsimd::SpatialSimilarity;
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
            let sample_len = sample.len();
            let (_, median, _) = sample.select_nth_unstable_by(
                mid,
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            );
            let median_val = *median;
            
            if sample_len % 2 == 0 && mid > 0 {
                let (_, second_median, _) = sample.select_nth_unstable_by(
                    mid - 1,
                    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                );
                (median_val + *second_median) / 2.0
            } else {
                median_val
            }
        } else {
            let mid = feature_values.len() / 2;
            let fv_len = feature_values.len();
            let (_, median, _) = feature_values.select_nth_unstable_by(
                mid,
                |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            );
            let median_val = *median;
            
            if fv_len % 2 == 0 && mid > 0 {
                let (_, second_median, _) = feature_values.select_nth_unstable_by(
                    mid - 1,
                    |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                );
                (median_val + *second_median) / 2.0
            } else {
                median_val
            }
        }
    }

    /// Fit histograms - CORRECTED VERSION
    pub fn fit(&mut self, x: &[Vec<f64>]) -> &mut Self {
        if x.is_empty() { return self; }
        let n_features = x[0].len();
        
        use std::sync::{Arc, Mutex};
        
        let mut pool = fork_union::spawn(fork_union::count_logical_cores());
        
        // ✅ CORRECT: Collect results with Arc<Mutex>
        let results = Arc::new(Mutex::new(Vec::new()));
        let results_clone = Arc::clone(&results);
        let max_bins = self.max_bins;
        
        (0..n_features)
            .into_par_iter()
            .with_pool(&mut pool)
            .for_each(|feature_idx| {
                let mut valid_values: Vec<f64> = x.iter()
                    .map(|row| row[feature_idx])
                    .filter(|&v| !v.is_nan())
                    .collect();

                let median = Self::calculate_median(&mut valid_values);
                valid_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                valid_values.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

                let edges = if valid_values.len() <= max_bins {
                    valid_values
                } else {
                    let mut quantiles = Vec::with_capacity(max_bins + 1);
                    if !valid_values.is_empty() {
                        let len_minus_1 = valid_values.len() - 1;
                        for i in 0..=max_bins {
                            let q = if i < max_bins / 4 {
                                (i as f64 / (max_bins as f64 / 4.0)) * 0.10
                            } else if i > 3 * max_bins / 4 {
                                0.90 + ((i - 3 * max_bins / 4) as f64 / (max_bins as f64 / 4.0)) * 0.10
                            } else {
                                0.10 + ((i - max_bins / 4) as f64 / (max_bins as f64 / 2.0)) * 0.80
                            };
                            let idx = ((len_minus_1 as f64 * q).round() as usize).min(len_minus_1);
                            quantiles.push(valid_values[idx]);
                        }
                        quantiles.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        quantiles.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
                    }
                    quantiles
                };
                
                results_clone.lock().unwrap().push((feature_idx, edges, median));
            });
        
        // ✅ CORRECT: Process results sequentially and sort by feature index
        let mut final_results = match Arc::try_unwrap(results) {
            Ok(mutex) => mutex.into_inner().unwrap(),
            Err(arc) => arc.lock().unwrap().clone(),
        };
        
        final_results.sort_by_key(|&(idx, _, _)| idx);
        
        for (_, edges, median) in final_results {
            self.bin_edges.push(edges.clone());
            self.n_bins_per_feature.push(edges.len());
            self.medians.push(median);
        }
        
        self
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<i32>> {
        adaptive_par_map(x, ParallelComplexity::Medium, |row| {
            self.transform_row(row)
        })
    }
    
    fn transform_row(&self, row: &[f64]) -> Vec<i32> {
        row.iter().enumerate().map(|(feature_idx, &value)| {
            let imputed_value = if value.is_nan() {
                self.medians[feature_idx]
            } else {
                value
            };
            
            let edges = &self.bin_edges[feature_idx];
            let bin_idx = self.find_bin_fast(edges, imputed_value);
            let n_edges = self.n_bins_per_feature[feature_idx];
            let final_bin_idx = if n_edges > 0 { bin_idx.min(n_edges - 1) } else { 0 };
            final_bin_idx as i32
        }).collect()
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
    
    /// Fast bin finding with optimized search
    fn find_bin_fast(&self, edges: &[f64], value: f64) -> usize {
        if edges.is_empty() { return 0; }
        
        // Linear search for small arrays (cache-friendly)
        if edges.len() <= 16 {
            edges.iter().position(|&x| x >= value).unwrap_or(edges.len() - 1)
        } else {
            // Binary search for larger arrays
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
            left
        }
    }
}
