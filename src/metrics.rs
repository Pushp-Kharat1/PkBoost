#[inline(always)]
pub fn calculate_shannon_entropy(count0: f64, count1: f64) -> f64 {
    let total = count0 + count1;
    if total < 1e-9 { return 0.0; }
    let p1 = count1 / total;
    
    // Use fast lookup table for binary entropy
    fast_binary_entropy_from_ratio(p1)
}

use std::sync::LazyLock;

const ENTROPY_LUT_SIZE: usize = 10000;

static ENTROPY_LOOKUP: LazyLock<Vec<f64>> = LazyLock::new(|| {
    let mut lut = vec![0.0; ENTROPY_LUT_SIZE];
    for i in 1..ENTROPY_LUT_SIZE {
        let p = i as f64 / ENTROPY_LUT_SIZE as f64;
        if p < 1.0 {
            let p_comp = 1.0 - p;
            lut[i] = if p > 1e-9 && p_comp > 1e-9 {
                -p * p.log2() - p_comp * p_comp.log2()
            } else {
                0.0
            };
        }
    }
    lut
});

#[inline]
pub fn fast_binary_entropy_from_ratio(positive_ratio: f64) -> f64 {
    if positive_ratio <= 0.0 || positive_ratio >= 1.0 {
        return 0.0;
    }
    let idx = (positive_ratio * ENTROPY_LUT_SIZE as f64) as usize;
    ENTROPY_LOOKUP[idx.min(ENTROPY_LUT_SIZE - 1)]
}

pub struct AUCCalculator {
    sorted_indices: Vec<usize>,
}

impl AUCCalculator {
    pub fn new() -> Self {
        Self {
            sorted_indices: Vec::new(),
        }
    }

    pub fn prepare(&mut self, y_scores: &[f64]) {
        let n = y_scores.len();
        
        if self.sorted_indices.len() != n {
            self.sorted_indices = (0..n).collect();
        }
        
        self.sorted_indices.sort_unstable_by(|&a, &b| {
            y_scores[b].partial_cmp(&y_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn roc_auc(&self, y_true: &[f64]) -> f64 {
        let mut tp = 0.0;
        let mut auc_numerator = 0.0;
        let total_pos: f64 = y_true.iter().sum();
        let total_neg = y_true.len() as f64 - total_pos;
        
        if total_pos < 1e-9 || total_neg < 1e-9 { 
            return 0.5; 
        }
        
        for &idx in &self.sorted_indices {
            if y_true[idx] > 0.5 {
                tp += 1.0;
            } else {
                auc_numerator += tp;
            }
        }
        
        auc_numerator / (total_pos * total_neg)
    }

    pub fn pr_auc(&self, y_true: &[f64]) -> f64 {
        let total_pos: f64 = y_true.iter().sum();
        if total_pos < 1e-9 { 
            return 0.0; 
        }
        
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut auc = 0.0;
        let mut prev_recall = 0.0;
        
        for &idx in &self.sorted_indices {
            if y_true[idx] > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            
            let recall = tp / total_pos;
            let precision = if tp + fp > 0.0 { 
                tp / (tp + fp) 
            } else { 
                1.0 
            };
            
            auc += precision * (recall - prev_recall);
            prev_recall = recall;
        }
        
        auc
    }
}

pub fn calculate_roc_auc(y_true: &[f64], y_scores: &[f64]) -> f64 {
    let mut calculator = AUCCalculator::new();
    calculator.prepare(y_scores);
    calculator.roc_auc(y_true)
}

pub fn calculate_pr_auc(y_true: &[f64], y_scores: &[f64]) -> f64 {
    let mut calculator = AUCCalculator::new();
    calculator.prepare(y_scores);
    calculator.pr_auc(y_true)
}
