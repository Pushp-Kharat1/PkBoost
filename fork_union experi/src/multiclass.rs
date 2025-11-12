// Multi-class classification using One-vs-Rest with softmax
use crate::model::OptimizedPKBoostShannon;
use fork_union::prelude::*;

pub struct MultiClassPKBoost {
    classifiers: Vec<OptimizedPKBoostShannon>,
    n_classes: usize,
    fitted: bool,
}

impl MultiClassPKBoost {
    pub fn new(n_classes: usize) -> Self {
        Self {
            classifiers: Vec::new(),
            n_classes,
            fitted: false,
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &[f64], eval_set: Option<(&Vec<Vec<f64>>, &[f64])>, verbose: bool) -> Result<(), String> {
        if self.n_classes < 2 {
            return Err("n_classes must be >= 2".to_string());
        }

        if verbose {
            println!("Training {} OvR classifiers...", self.n_classes);
        }

        let mut pool = fork_union::spawn(fork_union::count_logical_cores());
        let classifiers: std::sync::Arc<Vec<fork_union::SpinMutex<OptimizedPKBoostShannon>>> = 
            std::sync::Arc::new((0..self.n_classes).map(|_| fork_union::SpinMutex::new(OptimizedPKBoostShannon::new())).collect());
        let classifiers_clone = std::sync::Arc::clone(&classifiers);
        
        (0..self.n_classes).into_par_iter().with_pool(&mut pool).for_each(|class_idx| {
            let y_binary: Vec<f64> = y.iter().map(|&label| if (label as usize) == class_idx { 1.0 } else { 0.0 }).collect();
            
            let eval_binary = eval_set.map(|(x_val, y_val)| {
                let y_val_binary: Vec<f64> = y_val.iter().map(|&label| if (label as usize) == class_idx { 1.0 } else { 0.0 }).collect();
                (x_val.clone(), y_val_binary)
            });

            let mut clf = OptimizedPKBoostShannon::auto(x, &y_binary);
            
            let eval_ref = eval_binary.as_ref().map(|(x_v, y_v)| (x_v, y_v.as_slice()));
            clf.fit(x, &y_binary, eval_ref, false).ok();
            
            if verbose {
                println!("  Class {} trained", class_idx);
            }
            *classifiers_clone[class_idx].lock() = clf;
        });
        
        self.classifiers = std::sync::Arc::try_unwrap(classifiers).ok().unwrap()
            .into_iter().map(|m| m.into_inner()).collect();

        self.fitted = true;
        if verbose {
            println!("Multi-class training complete");
        }
        Ok(())
    }

    pub fn predict_proba(&self, x: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        if !self.fitted {
            return Err("Model not fitted".to_string());
        }

        let logits: Vec<Vec<f64>> = self.classifiers.iter().map(|clf| {
            clf.predict_proba(x).unwrap_or_else(|_| vec![0.0; x.len()])
        }).collect();

        let mut probs = vec![vec![0.0; self.n_classes]; x.len()];
        
        for i in 0..x.len() {
            let sample_logits: Vec<f64> = (0..self.n_classes).map(|c| logits[c][i]).collect();
            let sample_probs = softmax(&sample_logits);
            probs[i] = sample_probs;
        }

        Ok(probs)
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Result<Vec<usize>, String> {
        let probs = self.predict_proba(x)?;
        Ok(probs.iter().map(|p| {
            p.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }).collect())
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum).collect()
}
