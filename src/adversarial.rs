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
}

impl AdversarialEnsemble {
    pub fn new(pos_ratio: f64) -> Self {
        let mut model = OptimizedPKBoostShannon::new(); 
        
        // small shallow model - just needs to find patterns in errors
        model.max_depth = 3;
        model.learning_rate = 0.1;
        model.n_estimators = 5; 
        
        let pos_class_weight = (1.0 / pos_ratio).min(1000.0);
        
        Self {
            recent_vulnerabilities: VecDeque::new(),
            model,
            vulnerability_window: 100,
            vulnerability_threshold: 0.15,
            pos_class_weight,
        }
    }
    
    pub fn record_vulnerability(&mut self, vuln: Vulnerability) {
        // Only record if there's actual error (threshold: 0.3)
        if vuln.error > 0.3 {
            if self.recent_vulnerabilities.len() >= self.vulnerability_window {
                self.recent_vulnerabilities.pop_front();
            }
            self.recent_vulnerabilities.push_back(vuln);
        }
    }
    
    pub fn get_vulnerability_score(&self) -> f64 {
        if self.recent_vulnerabilities.is_empty() {
            return 0.0;
        }
        self.recent_vulnerabilities.iter().map(|v| v.confidence).sum::<f64>() / self.recent_vulnerabilities.len() as f64
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