// Multi-class classification using One-vs-Rest with softmax
use crate::model::OptimizedPKBoostShannon;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

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

    /// Fit using ArrayView2 (zero-copy from Python)
    pub fn fit(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        eval_set: Option<(ArrayView2<'_, f64>, ArrayView1<'_, f64>)>,
        verbose: bool,
    ) -> Result<(), String> {
        if self.n_classes < 2 {
            return Err("n_classes must be >= 2".to_string());
        }

        if verbose {
            println!("Training {} OvR classifiers...", self.n_classes);
        }

        let y_slice = y.as_slice().ok_or("y must be contiguous")?;

        // We need to convert to owned data for parallel iteration since we can't share views
        let x_owned: Array2<f64> = x.to_owned();
        let eval_owned = eval_set.map(|(x_val, y_val)| (x_val.to_owned(), y_val.to_owned()));

        self.classifiers = (0..self.n_classes)
            .into_par_iter()
            .map(|class_idx| {
                let y_binary: Array1<f64> = Array1::from_iter(y_slice.iter().map(|&label| {
                    if (label as usize) == class_idx {
                        1.0
                    } else {
                        0.0
                    }
                }));

                let eval_binary = eval_owned.as_ref().map(|(x_val, y_val)| {
                    let y_val_slice = y_val.as_slice().unwrap();
                    let y_val_binary: Array1<f64> =
                        Array1::from_iter(y_val_slice.iter().map(|&label| {
                            if (label as usize) == class_idx {
                                1.0
                            } else {
                                0.0
                            }
                        }));
                    (x_val.clone(), y_val_binary)
                });

                let mut clf = OptimizedPKBoostShannon::auto(x_owned.view(), y_binary.view());

                let eval_ref = eval_binary
                    .as_ref()
                    .map(|(x_v, y_v)| (x_v.view(), y_v.view()));
                clf.fit(x_owned.view(), y_binary.view(), eval_ref, false)
                    .ok();

                if verbose {
                    println!("  Class {} trained", class_idx);
                }
                clf
            })
            .collect();

        self.fitted = true;
        if verbose {
            println!("Multi-class training complete");
        }
        Ok(())
    }

    /// Predict probabilities from ArrayView2 (zero-copy from Python)
    /// Returns Array2 where each column is class probabilities
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if !self.fitted {
            return Err("Model not fitted".to_string());
        }

        let n_samples = x.nrows();

        // Collect logits from each classifier
        let logits: Vec<Array1<f64>> = self
            .classifiers
            .par_iter()
            .map(|clf| {
                clf.predict_proba(x)
                    .unwrap_or_else(|_| Array1::zeros(n_samples))
            })
            .collect();

        // Build output array (n_samples, n_classes)
        let mut probs = Array2::zeros((n_samples, self.n_classes));

        for i in 0..n_samples {
            let sample_logits: Vec<f64> = (0..self.n_classes).map(|c| logits[c][i]).collect();
            let sample_probs = softmax(&sample_logits);
            for (c, &p) in sample_probs.iter().enumerate() {
                probs[[i, c]] = p;
            }
        }

        Ok(probs)
    }

    /// Predict class labels from ArrayView2 (zero-copy from Python)
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<usize>, String> {
        let probs = self.predict_proba(x)?;
        let predictions: Vec<usize> = probs
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();
        Ok(Array1::from(predictions))
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum).collect()
}
