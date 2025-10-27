use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, PyReadonlyArray2, PyReadonlyArray1};
use crate::model::OptimizedPKBoostShannon;
use crate::living_booster::{AdversarialLivingBooster, SystemState};

#[pyclass]
pub struct PKBoostClassifier {
    model: Option<OptimizedPKBoostShannon>,
    fitted: bool,
}

#[pymethods]
impl PKBoostClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=20,
        min_child_weight=1.0,
        reg_lambda=1.0,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0
    ))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_child_weight: f64,
        reg_lambda: f64,
        gamma: f64,
        subsample: f64,
        colsample_bytree: f64,
        scale_pos_weight: f64,
    ) -> Self {
        let mut model = OptimizedPKBoostShannon::new();
        model.n_estimators = n_estimators;
        model.learning_rate = learning_rate;
        model.max_depth = max_depth;
        model.min_samples_split = min_samples_split;
        model.min_child_weight = min_child_weight;
        model.reg_lambda = reg_lambda;
        model.gamma = gamma;
        model.subsample = subsample;
        model.colsample_bytree = colsample_bytree;
        model.scale_pos_weight = scale_pos_weight;
        
        Self {
            model: Some(model),
            fitted: false,
        }
    }

    #[staticmethod]
    fn auto() -> Self {
        Self {
            model: None,
            fitted: false,
        }
    }

    #[pyo3(signature = (x, y, x_val=None, y_val=None, verbose=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        x_val: Option<PyReadonlyArray2<f64>>,
        y_val: Option<PyReadonlyArray1<f64>>,
        verbose: Option<bool>,
    ) -> PyResult<()> {
        let x_vec: Vec<Vec<f64>> = x.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        let y_vec: Vec<f64> = y.as_array().to_vec();
        
        let eval_set = if let (Some(xv), Some(yv)) = (x_val, y_val) {
            let x_val_vec: Vec<Vec<f64>> = xv.as_array().rows()
                .into_iter().map(|row| row.to_vec()).collect();
            let y_val_vec: Vec<f64> = yv.as_array().to_vec();
            Some((x_val_vec, y_val_vec))
        } else { None };

        let verbose = verbose.unwrap_or(false);
        
        py.allow_threads(|| {
            if self.model.is_none() {
                let mut auto_model = OptimizedPKBoostShannon::auto(&x_vec, &y_vec);
                auto_model.fit(&x_vec, &y_vec, eval_set.as_ref().map(|(x, y)| (x, y.as_slice())), verbose)?;
                self.model = Some(auto_model);
                self.fitted = true;
                Ok(())
            } else if let Some(ref mut model) = self.model {
                model.fit(&x_vec, &y_vec, eval_set.as_ref().map(|(x, y)| (x, y.as_slice())), verbose)?;
                self.fitted = true;
                Ok(())
            } else {
                Err("Model not initialized".to_string())
            }
        }).map_err(|e| PyValueError::new_err(format!("Training failed: {}", e)))?;
        
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let x_vec: Vec<Vec<f64>> = x.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        
        let predictions = py.allow_threads(|| {
            self.model.as_ref()
                .ok_or("Model not initialized".to_string())
                .and_then(|m| m.predict_proba(&x_vec))
        }).map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;
        
        Ok(PyArray1::from_vec_bound(py, predictions))
    }

    #[pyo3(signature = (x, threshold=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        threshold: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let x_vec: Vec<Vec<f64>> = x.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        
        let proba = py.allow_threads(|| {
            self.model.as_ref()
                .ok_or("Model not initialized".to_string())
                .and_then(|m| m.predict_proba(&x_vec))
        }).map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;
        
        let threshold = threshold.unwrap_or(0.5);
        let predictions: Vec<i32> = proba.iter()
            .map(|&p| if p >= threshold { 1 } else { 0 })
            .collect();
        
        Ok(PyArray1::from_vec_bound(py, predictions))
    }

    fn get_feature_importance<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let importance = py.allow_threads(|| {
            if let Some(ref model) = self.model {
                let usage = model.get_feature_usage();
                let total: usize = usage.iter().sum();
                if total > 0 {
                    usage.iter().map(|&u| u as f64 / total as f64).collect()
                } else {
                    vec![0.0; usage.len()]
                }
            } else {
                vec![]
            }
        });

        Ok(PyArray1::from_vec_bound(py, importance))
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn get_n_trees(&self) -> PyResult<usize> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }
        Ok(self.model.as_ref().map(|m| m.trees.len()).unwrap_or(0))
    }
}

#[pyclass]
pub struct PKBoostAdaptive {
    booster: Option<AdversarialLivingBooster>,
    fitted: bool,
}

#[pymethods]
impl PKBoostAdaptive {
    #[new]
    fn new() -> Self {
        Self {
            booster: None,
            fitted: false,
        }
    }

    #[pyo3(signature = (x, y, x_val=None, y_val=None, verbose=None))]
    fn fit_initial<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        x_val: Option<PyReadonlyArray2<f64>>,
        y_val: Option<PyReadonlyArray1<f64>>,
        verbose: Option<bool>,
    ) -> PyResult<()> {
        let x_vec: Vec<Vec<f64>> = x.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        let y_vec: Vec<f64> = y.as_array().to_vec();
        
        let eval_set = if let (Some(xv), Some(yv)) = (x_val, y_val) {
            let x_val_vec: Vec<Vec<f64>> = xv.as_array().rows()
                .into_iter().map(|row| row.to_vec()).collect();
            let y_val_vec: Vec<f64> = yv.as_array().to_vec();
            Some((x_val_vec, y_val_vec))
        } else { None };

        let verbose = verbose.unwrap_or(false);
        
        py.allow_threads(|| {
            let mut booster = AdversarialLivingBooster::new(&x_vec, &y_vec);
            booster.fit_initial(&x_vec, &y_vec, eval_set.as_ref().map(|(x, y)| (x, y.as_slice())), verbose)?;
            self.booster = Some(booster);
            self.fitted = true;
            Ok(())
        }).map_err(|e: String| PyValueError::new_err(format!("Training failed: {}", e)))?;
        
        Ok(())
    }

    #[pyo3(signature = (x, y, verbose=None))]
    fn observe_batch<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        verbose: Option<bool>,
    ) -> PyResult<()> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }

        let x_vec: Vec<Vec<f64>> = x.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        let y_vec: Vec<f64> = y.as_array().to_vec();
        let verbose = verbose.unwrap_or(false);
        
        py.allow_threads(|| {
            self.booster.as_mut()
                .ok_or("Booster not initialized".to_string())
                .and_then(|b| b.observe_batch(&x_vec, &y_vec, verbose))
        }).map_err(|e: String| PyValueError::new_err(format!("Observation failed: {}", e)))?;
        
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }

        let x_vec: Vec<Vec<f64>> = x.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        
        let predictions = py.allow_threads(|| {
            self.booster.as_ref()
                .ok_or("Booster not initialized".to_string())
                .and_then(|b| b.predict_proba(&x_vec))
        }).map_err(|e: String| PyValueError::new_err(format!("Prediction failed: {}", e)))?;
        
        Ok(PyArray1::from_vec_bound(py, predictions))
    }

    #[pyo3(signature = (x, threshold=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        threshold: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }

        let x_vec: Vec<Vec<f64>> = x.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        
        let proba = py.allow_threads(|| {
            self.booster.as_ref()
                .ok_or("Booster not initialized".to_string())
                .and_then(|b| b.predict_proba(&x_vec))
        }).map_err(|e: String| PyValueError::new_err(format!("Prediction failed: {}", e)))?;
        
        let threshold = threshold.unwrap_or(0.5);
        let predictions: Vec<i32> = proba.iter()
            .map(|&p| if p >= threshold { 1 } else { 0 })
            .collect();
        
        Ok(PyArray1::from_vec_bound(py, predictions))
    }

    fn get_vulnerability_score(&self) -> PyResult<f64> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }
        Ok(self.booster.as_ref().map(|b| b.get_vulnerability_score()).unwrap_or(0.0))
    }

    fn get_state(&self) -> PyResult<String> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }
        let state = self.booster.as_ref().map(|b| b.get_state()).unwrap_or(SystemState::Normal);
        Ok(match state {
            SystemState::Normal => "Normal".to_string(),
            SystemState::Alert { checks_in_alert } => format!("Alert({})", checks_in_alert),
            SystemState::Metamorphosis => "Metamorphosis".to_string(),
        })
    }

    fn get_metamorphosis_count(&self) -> PyResult<usize> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }
        Ok(self.booster.as_ref().map(|b| b.get_metamorphosis_count()).unwrap_or(0))
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[pymodule]
fn pkboost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PKBoostClassifier>()?;
    m.add_class::<PKBoostAdaptive>()?;
    Ok(())
}
