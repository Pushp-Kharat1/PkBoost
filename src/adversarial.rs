// Adversarial ensemble - basically a secondary model that looks for weaknesses
// in the primary model's predictions

use crate::model::OptimizedPKBoostShannon;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Vulnerability {
    pub confidence: f64,
    pub error: f64,
    pub sample_idx: usize,
}

pub struct AdversarialEnsemble {
    pub recent_vulnerabilities: VecDeque<Vulnerability>,  // rolling window of mistakes
    pub model: OptimizedPKBoostShannon,  // small model trained on hard examples
    vulnerability_window: usize,
    #[allow(dead_code)]  
    vulnerability_threshold: f64,
    pos_class_weight: f64,  // weight for minority class
    vulnerability_ema: f64,  // exponential moving average
    ema_alpha: f64,  // smoothing factor (0.1 = slow decay)
}

impl AdversarialEnsemble {
    pub fn new(pos_ratio: f64) -> Self {
        let mut model = OptimizedPKBoostShannon::new(); 
        
        // small shallow model - just needs to find patterns in errors
        model.max_depth = 3;
        model.learning_rate = 0.1;
        model.n_estimators = 5; 
        
        let pos_class_weight = if pos_ratio > 1e-9 {
            (1.0 / pos_ratio).min(1000.0)
        } else {
            1000.0
        };
        
        Self {
            recent_vulnerabilities: VecDeque::new(),
            model,
            vulnerability_window: 5000,  // Large enough for multiple batches
            vulnerability_threshold: 0.15,
            pos_class_weight,
            vulnerability_ema: 0.0,
            ema_alpha: 0.1,  // Slow decay - keeps history
        }
    }
    
    pub fn record_vulnerability(&mut self, vuln: Vulnerability) {
        // Only record if there's actual error (threshold: 0.2)
        if vuln.error > 0.2 {
            if self.recent_vulnerabilities.len() >= self.vulnerability_window {
                self.recent_vulnerabilities.pop_front();
            }
            self.recent_vulnerabilities.push_back(vuln.clone());
            
            // Update EMA: new_ema = alpha * new_value + (1 - alpha) * old_ema
            self.vulnerability_ema = self.ema_alpha * vuln.confidence + (1.0 - self.ema_alpha) * self.vulnerability_ema;
        }
    }
    
    pub fn get_vulnerability_score(&self) -> f64 {
        // Use EMA instead of raw average - prevents score from dropping to zero
        self.vulnerability_ema
    }

    // calculate how badly the model screwed up on this sample
    pub fn find_vulnerability(
        &self,
        y_true: f64,
        primary_pred: f64,
        sample_idx: usize,
    ) -> Vulnerability {
        let confidence = (primary_pred - 0.5).abs() * 2.0;  // how sure was the model?
        let error = (y_true - primary_pred).abs();
        
        // weight errors on minority class more (but normalized to 0-1 range)
        let class_weight = if y_true > 0.5 {
            (self.pos_class_weight / 100.0).min(5.0)  // cap at 5x weight
        } else {
            1.0
        };
        
        // confident mistakes = high vulnerability (normalized to 0-1)
        let vulnerability_strength = (confidence * error * class_weight).min(1.0);
        
        Vulnerability {
            confidence: vulnerability_strength, 
            error,
            sample_idx,
        }
    }
}