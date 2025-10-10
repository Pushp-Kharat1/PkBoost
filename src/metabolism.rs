// Feature metabolism - tracks which features are still useful over time
// Features that dont get used decay and eventually "die"

#[derive(Debug, Clone)]
pub struct FeatureHealth {
    pub utility_score: f64,  // starts at 1.0, decays if not used
    pub decay_rate: f64,
    pub last_used_iteration: usize,
    pub adversarial_exposure: f64,  // how often this feature appears in vulnerable samples
    pub lifetime_usage: usize,
}

impl FeatureHealth {
    pub fn new(decay_rate: f64) -> Self {
        Self {
            utility_score: 1.0,
            decay_rate,
            last_used_iteration: 0,
            adversarial_exposure: 0.0,
            lifetime_usage: 0,
        }
    }
    
    // feature got used - reset its health to full
    pub fn update_usage(&mut self, current_iteration: usize) {
        self.utility_score = 1.0;
        self.last_used_iteration = current_iteration;
        self.lifetime_usage += 1;
    }
    
    // feature hasnt been used - decay its health
    pub fn apply_decay(&mut self, current_iteration: usize) {
        let iterations_since_use = current_iteration.saturating_sub(self.last_used_iteration);
        if iterations_since_use > 0 {
            self.utility_score *= (1.0 - self.decay_rate).powi(iterations_since_use as i32);
        }
    }
    
    pub fn is_dead(&self, threshold: f64) -> bool {
        self.utility_score < threshold
    }
    
    pub fn is_vulnerable(&self) -> bool {
        self.utility_score < 0.5 && self.adversarial_exposure > 0.3
    }
}

pub struct FeatureMetabolism {
    feature_health: Vec<FeatureHealth>,
    death_threshold: f64,
}

impl FeatureMetabolism {
    pub fn new(n_features: usize) -> Self {
        let decay_rate = 0.0005;  // slow decay - dont kill features too quickly
        Self {
            feature_health: (0..n_features)
                .map(|_| FeatureHealth::new(decay_rate))
                .collect(),
            death_threshold: 0.15,  // below this = feature is considered dead
        }
    }
    
    // update health for all features based on usage
    pub fn update(&mut self, used_features: &[usize], current_iteration: usize) {
        // refresh health for features that were used
        for &feature_idx in used_features {
            if let Some(health) = self.feature_health.get_mut(feature_idx) {
                health.update_usage(current_iteration);
            }
        }
        
        // apply decay to all features
        for health in &mut self.feature_health {
            health.apply_decay(current_iteration);
        }
    }
    
    pub fn get_dead_features(&self) -> Vec<usize> {
        self.feature_health.iter()
            .enumerate()
            .filter(|(_, h)| h.is_dead(self.death_threshold))
            .map(|(idx, _)| idx)
            .collect()
    }
    
    pub fn get_vulnerable_features(&self) -> Vec<usize> {
        self.feature_health.iter()
            .enumerate()
            .filter(|(_, h)| h.is_vulnerable())
            .map(|(idx, _)| idx)
            .collect()
    }
    
    pub fn get_healthy_features(&self) -> Vec<usize> {
        self.feature_health.iter()
            .enumerate()
            .filter(|(_, h)| h.utility_score > 0.6 && h.adversarial_exposure < 0.2)
            .map(|(idx, _)| idx)
            .collect()
    }
    
    pub fn record_adversarial_exploitation(&mut self, features: &[usize]) {
        for &feature_idx in features {
            if let Some(health) = self.feature_health.get_mut(feature_idx) {
                health.adversarial_exposure += 0.1;
                health.adversarial_exposure = health.adversarial_exposure.min(1.0);
            }
        }
    }
    
    pub fn get_feature_health(&self, feature_idx: usize) -> Option<&FeatureHealth> {
        self.feature_health.get(feature_idx)
    }
}
