// Hierarchical Adaptive Boosting (HAB) - Partition-based ensemble
// Divides feature space into specialized regions, each with its own model

use crate::model::OptimizedPKBoostShannon;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    Binary,
    MultiClass { n_classes: usize },
}

#[derive(Debug, Clone, Copy)]
pub enum PartitionMethod {
    KMeans,
    Random,
}

pub struct PartitionConfig {
    pub n_partitions: usize,
    pub specialist_estimators: usize,
    pub specialist_max_depth: usize,
    pub specialist_learning_rate: f64,
    pub task_type: TaskType,
    pub partition_method: PartitionMethod,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            n_partitions: 10,
            specialist_estimators: 200,
            specialist_max_depth: 6,
            specialist_learning_rate: 0.05,
            task_type: TaskType::Binary,
            partition_method: PartitionMethod::KMeans,
        }
    }
}

pub struct PartitionedClassifier {
    config: PartitionConfig,
    centroids: Vec<Vec<f64>>,
    specialists: Vec<OptimizedPKBoostShannon>,
    #[allow(dead_code)]
    multi_specialists: Vec<Vec<OptimizedPKBoostShannon>>,
    fitted: bool,
    // Drift tracking per partition
    partition_baseline_error: Vec<f64>,
    partition_error_ema: Vec<f64>,
    partition_sample_counts: Vec<usize>,
    // Ensemble weighting
    specialist_weights: Vec<f64>,
    use_weighted_ensemble: bool,
}

impl PartitionedClassifier {
    pub fn new(config: PartitionConfig) -> Self {
        let n = config.n_partitions;
        Self {
            config,
            centroids: Vec::new(),
            specialists: Vec::new(),
            multi_specialists: Vec::new(),
            fitted: false,
            partition_baseline_error: vec![0.0; n],
            partition_error_ema: vec![0.0; n],
            partition_sample_counts: vec![0; n],
            specialist_weights: vec![1.0; n],
            use_weighted_ensemble: true,
        }
    }
    
    // K-means clustering to partition feature space
    fn kmeans_partition(&mut self, x: &[Vec<f64>], max_iters: usize) {
        let n_features = x[0].len();
        let k = self.config.n_partitions;
        
        // Initialize centroids randomly
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.shuffle(&mut rng);
        
        self.centroids = indices[..k].iter()
            .map(|&i| x[i].clone())
            .collect();
        
        for _ in 0..max_iters {
            // Assign samples to nearest centroid
            let assignments = self.assign_to_partitions(x);
            
            // Update centroids
            let mut new_centroids = vec![vec![0.0; n_features]; k];
            let mut counts = vec![0; k];
            
            for (sample, &partition) in x.iter().zip(assignments.iter()) {
                for (j, &val) in sample.iter().enumerate() {
                    new_centroids[partition][j] += val;
                }
                counts[partition] += 1;
            }
            
            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[i] as f64;
                    }
                }
            }
            
            self.centroids = new_centroids;
        }
    }
    
    fn assign_to_partitions(&self, x: &[Vec<f64>]) -> Vec<usize> {
        use simsimd::SpatialSimilarity;
        
        x.par_iter().map(|sample| {
            // Convert to f32 for SIMD
            let sample_f32: Vec<f32> = sample.iter().map(|&v| v as f32).collect();
            
            self.centroids.iter()
                .enumerate()
                .map(|(i, centroid)| {
                    let centroid_f32: Vec<f32> = centroid.iter().map(|&v| v as f32).collect();
                    // Use SIMD squared Euclidean distance
                    let dist = f32::sqeuclidean(&sample_f32[..], &centroid_f32[..])
                        .map(|d| d as f64)
                        .unwrap_or(f64::INFINITY);
                    (i, dist)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }).collect()
    }
    
    pub fn partition_data(&mut self, x: &[Vec<f64>], _y: &[f64], verbose: bool) {
        if verbose {
            println!("Partitioning data into {} regions...", self.config.n_partitions);
        }
        
        match self.config.partition_method {
            PartitionMethod::KMeans => self.kmeans_partition(x, 10),
            PartitionMethod::Random => {
                let n_features = x[0].len();
                use rand::Rng;
                let mut rng = rand::thread_rng();
                self.centroids = (0..self.config.n_partitions)
                    .map(|_| (0..n_features).map(|_| rng.gen_range(-1.0..1.0)).collect())
                    .collect();
            }
        }
        
        if verbose {
            let assignments = self.assign_to_partitions(x);
            let mut counts = vec![0; self.config.n_partitions];
            for &p in &assignments {
                counts[p] += 1;
            }
            println!("Partition sizes: {:?}", counts);
        }
    }
    
    pub fn train_specialists(&mut self, x: &[Vec<f64>], y: &[f64], verbose: bool) -> Result<(), String> {
        self.train_specialists_with_validation(x, y, None, verbose)
    }
    
    pub fn train_specialists_with_validation(
        &mut self, 
        x: &[Vec<f64>], 
        y: &[f64],
        val_set: Option<(&[Vec<f64>], &[f64])>,
        verbose: bool
    ) -> Result<(), String> {
        let assignments = self.assign_to_partitions(x);
        
        // Group samples by partition
        let mut partition_data: HashMap<usize, (Vec<Vec<f64>>, Vec<f64>)> = HashMap::new();
        for (i, &partition) in assignments.iter().enumerate() {
            partition_data.entry(partition)
                .or_insert_with(|| (Vec::new(), Vec::new()))
                .0.push(x[i].clone());
            partition_data.entry(partition)
                .or_insert_with(|| (Vec::new(), Vec::new()))
                .1.push(y[i]);
        }
        
        if verbose {
            println!("Training {} specialists...", self.config.n_partitions);
        }
        
        // Train specialists in parallel
        let specialists: Vec<_> = (0..self.config.n_partitions).into_par_iter()
            .map(|partition_id| {
                if let Some((x_part, y_part)) = partition_data.get(&partition_id) {
                    if x_part.len() < 10 {
                        return Err(format!("Partition {} has insufficient data", partition_id));
                    }
                    
                    // Auto-tune specialist for imbalanced data
                    let mut specialist = OptimizedPKBoostShannon::auto(x_part, y_part);
                    specialist.n_estimators = self.config.specialist_estimators;
                    specialist.max_depth = self.config.specialist_max_depth;
                    specialist.learning_rate = self.config.specialist_learning_rate;
                    specialist.early_stopping_rounds = 50;
                    // OPTIMIZATION: Increase Shannon entropy weight for better minority class detection
                    specialist.mi_weight = 0.3;  // Higher than default 0.1
                    
                    specialist.fit(x_part, y_part, None, false)?;
                    Ok(specialist)
                } else {
                    Err(format!("No data for partition {}", partition_id))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        
        self.specialists = specialists;
        self.fitted = true;
        
        // Calculate weights and baseline error
        if let Some((x_val, y_val)) = val_set {
            let val_assignments = self.assign_to_partitions(x_val);
            let mut val_partition_data: HashMap<usize, (Vec<Vec<f64>>, Vec<f64>)> = HashMap::new();
            
            for (i, &partition) in val_assignments.iter().enumerate() {
                val_partition_data.entry(partition)
                    .or_insert_with(|| (Vec::new(), Vec::new()))
                    .0.push(x_val[i].clone());
                val_partition_data.entry(partition)
                    .or_insert_with(|| (Vec::new(), Vec::new()))
                    .1.push(y_val[i]);
            }
            
            // Calculate PR-AUC based weights
            for partition_id in 0..self.config.n_partitions {
                if let Some((x_part, y_part)) = val_partition_data.get(&partition_id) {
                    if x_part.len() > 5 {
                        if let Ok(probs) = self.specialists[partition_id].predict_proba(x_part) {
                            let pr_auc = crate::metrics::calculate_pr_auc(y_part, &probs);
                            self.specialist_weights[partition_id] = pr_auc.max(0.5);  // Min weight 0.5
                            
                            let errors: Vec<f64> = probs.iter().zip(y_part.iter())
                                .map(|(&prob, &true_y)| if (prob > 0.5) == (true_y > 0.5) { 0.0 } else { 1.0 })
                                .collect();
                            let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
                            self.partition_baseline_error[partition_id] = avg_error;
                            self.partition_error_ema[partition_id] = avg_error;
                        }
                    }
                }
            }
            
            if verbose {
                println!("Specialist weights (PR-AUC): {:?}", 
                    self.specialist_weights.iter().map(|w| format!("{:.3}", w)).collect::<Vec<_>>());
            }
        } else {
            // No validation set - use training error
            for partition_id in 0..self.config.n_partitions {
                if let Some((x_part, y_part)) = partition_data.get(&partition_id) {
                    if let Ok(probs) = self.specialists[partition_id].predict_proba(x_part) {
                        let errors: Vec<f64> = probs.iter().zip(y_part.iter())
                            .map(|(&prob, &true_y)| if (prob > 0.5) == (true_y > 0.5) { 0.0 } else { 1.0 })
                            .collect();
                        let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
                        self.partition_baseline_error[partition_id] = avg_error;
                        self.partition_error_ema[partition_id] = avg_error;
                    }
                }
            }
        }
        
        if verbose {
            println!("Training complete. {} specialists ready.", self.specialists.len());
        }
        
        Ok(())
    }
    
    pub fn predict_proba(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if !self.fitted {
            return Err("Model not fitted".to_string());
        }
        
        let assignments = self.assign_to_partitions(x);
        
        let n_classes = match self.config.task_type {
            TaskType::Binary => 2,
            TaskType::MultiClass { n_classes } => n_classes,
        };
        
        let mut results = vec![vec![0.0; n_classes]; x.len()];
        
        // Weighted ensemble prediction with normalization
        if self.use_weighted_ensemble {
            for (i, &partition) in assignments.iter().enumerate() {
                if partition < self.specialists.len() {
                    let specialist = &self.specialists[partition];
                    let sample = vec![x[i].clone()];
                    
                    if let Ok(probs) = specialist.predict_proba(&sample) {
                        // Use specialist prediction directly (no weighting distortion)
                        results[i][1] = probs[0];
                        results[i][0] = 1.0 - probs[0];
                    }
                }
            }
        } else {
            // Original unweighted prediction
            for (i, &partition) in assignments.iter().enumerate() {
                if partition < self.specialists.len() {
                    let specialist = &self.specialists[partition];
                    let sample = vec![x[i].clone()];
                    
                    match self.config.task_type {
                        TaskType::Binary => {
                            let probs = specialist.predict_proba(&sample)?;
                            results[i][1] = probs[0];
                            results[i][0] = 1.0 - probs[0];
                        }
                        TaskType::MultiClass { n_classes } => {
                            let probs = specialist.predict_proba(&sample)?;
                            let sum: f64 = (0..n_classes).map(|c| if c == 0 { 1.0 - probs[0] } else { probs[0] / (n_classes - 1) as f64 }).sum();
                            for c in 0..n_classes {
                                results[i][c] = if c == 0 { (1.0 - probs[0]) / sum } else { (probs[0] / (n_classes - 1) as f64) / sum };
                            }
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<usize>, String> {
        let probs = self.predict_proba(x)?;
        Ok(probs.iter().map(|p| {
            p.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }).collect())
    }
    
    // Observe batch and detect drifted partitions
    pub fn observe_batch(&mut self, x: &[Vec<f64>], y: &[f64]) -> Vec<usize> {
        if !self.fitted { return Vec::new(); }
        
        let assignments = self.assign_to_partitions(x);
        let mut drifted = Vec::new();
        
        // Group by partition
        let mut partition_samples: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (i, &p) in assignments.iter().enumerate() {
            partition_samples.entry(p).or_insert_with(Vec::new).push(i);
        }
        
        for (partition_id, indices) in partition_samples {
            if partition_id >= self.specialists.len() { continue; }
            
            let x_part: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
            let y_part: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            
            if let Ok(probs) = self.specialists[partition_id].predict_proba(&x_part) {
                let errors: Vec<f64> = probs.iter().zip(y_part.iter())
                    .map(|(&prob, &true_y)| if (prob > 0.5) == (true_y > 0.5) { 0.0 } else { 1.0 })
                    .collect();
                let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
                
                // Update EMA
                let alpha = 0.1;
                self.partition_error_ema[partition_id] = 
                    alpha * avg_error + (1.0 - alpha) * self.partition_error_ema[partition_id];
                self.partition_sample_counts[partition_id] += indices.len();
                
                // Detect drift (error increased by 30%)
                let baseline = self.partition_baseline_error[partition_id].max(0.01);
                if self.partition_error_ema[partition_id] > baseline * 1.3 {
                    drifted.push(partition_id);
                }
            }
        }
        
        drifted
    }
    
    // Retrain specific partitions
    pub fn metamorph_partitions(
        &mut self,
        partition_ids: &[usize],
        buffer_x: &[Vec<f64>],
        buffer_y: &[f64],
        verbose: bool,
    ) -> Result<(), String> {
        if verbose {
            println!("[METAMORPH] Retraining {} partitions: {:?}", partition_ids.len(), partition_ids);
        }
        
        let assignments = self.assign_to_partitions(buffer_x);
        
        for &partition_id in partition_ids {
            if partition_id >= self.specialists.len() { continue; }
            
            // Get samples for this partition
            let indices: Vec<usize> = assignments.iter()
                .enumerate()
                .filter(|(_, &p)| p == partition_id)
                .map(|(i, _)| i)
                .collect();
            
            if indices.len() < 50 {
                if verbose {
                    println!("  [WARN] Partition {} has insufficient data ({} samples)", partition_id, indices.len());
                }
                continue;
            }
            
            let x_part: Vec<Vec<f64>> = indices.iter().map(|&i| buffer_x[i].clone()).collect();
            let y_part: Vec<f64> = indices.iter().map(|&i| buffer_y[i]).collect();
            
            if verbose {
                println!("  [TRAIN] Retraining partition {} on {} samples...", partition_id, x_part.len());
            }
            
            // Retrain specialist with auto-tuning
            let mut new_specialist = OptimizedPKBoostShannon::auto(&x_part, &y_part);
            new_specialist.n_estimators = self.config.specialist_estimators;
            new_specialist.max_depth = self.config.specialist_max_depth;
            new_specialist.learning_rate = self.config.specialist_learning_rate;
            new_specialist.early_stopping_rounds = 20;
            new_specialist.mi_weight = 0.3;  // Higher Shannon entropy weight
            
            new_specialist.fit(&x_part, &y_part, None, false)?;
            self.specialists[partition_id] = new_specialist;
            
            // Reset drift tracking
            if let Ok(probs) = self.specialists[partition_id].predict_proba(&x_part) {
                let errors: Vec<f64> = probs.iter().zip(y_part.iter())
                    .map(|(&prob, &true_y)| if (prob > 0.5) == (true_y > 0.5) { 0.0 } else { 1.0 })
                    .collect();
                let avg_error = errors.iter().sum::<f64>() / errors.len() as f64;
                self.partition_baseline_error[partition_id] = avg_error;
                self.partition_error_ema[partition_id] = avg_error;
            }
            
            if verbose {
                println!("  [DONE] Partition {} retrained", partition_id);
            }
        }
        
        Ok(())
    }
}

pub struct PartitionedClassifierBuilder {
    config: PartitionConfig,
}

impl PartitionedClassifierBuilder {
    pub fn new() -> Self {
        Self {
            config: PartitionConfig::default(),
        }
    }
    
    pub fn n_partitions(mut self, n: usize) -> Self {
        self.config.n_partitions = n;
        self
    }
    
    pub fn specialist_estimators(mut self, n: usize) -> Self {
        self.config.specialist_estimators = n;
        self
    }
    
    pub fn specialist_max_depth(mut self, d: usize) -> Self {
        self.config.specialist_max_depth = d;
        self
    }
    
    pub fn task_type(mut self, t: TaskType) -> Self {
        self.config.task_type = t;
        self
    }
    
    pub fn build(self) -> PartitionedClassifier {
        PartitionedClassifier::new(self.config)
    }
}

impl Default for PartitionedClassifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}
