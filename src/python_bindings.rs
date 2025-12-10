use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::PyBytes;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray1};
use crate::model::OptimizedPKBoostShannon;
use crate::living_booster::{AdversarialLivingBooster, SystemState};
use crate::regression::PKBoostRegressor;
use crate::multiclass::MultiClassPKBoost;

fn ensure_contiguous<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import_bound("numpy")?;
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("dtype", "float64")?;
    kwargs.set_item("order", "C")?;
    np.call_method("ascontiguousarray", (arr,), Some(&kwargs))
}

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
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        x_val: Option<&Bound<'py, PyAny>>,
        y_val: Option<&Bound<'py, PyAny>>,
        verbose: Option<bool>,
    ) -> PyResult<()> {
        // Helper to convert any array-like to PyReadonlyArray2
        let to_array2 = |arr: &Bound<'py, PyAny>| -> PyResult<PyReadonlyArray2<'py, f64>> {
            if let Ok(readonly) = arr.extract::<PyReadonlyArray2<f64>>() {
                Ok(readonly)
            } else {
                let np = py.import_bound("numpy")?;
                let converted = np.call_method1("asarray", (arr,))?;
                converted.extract::<PyReadonlyArray2<f64>>()
            }
        };
        
        // Helper to convert any array-like to PyReadonlyArray1
        let to_array1 = |arr: &Bound<'py, PyAny>| -> PyResult<PyReadonlyArray1<'py, f64>> {
            if let Ok(readonly) = arr.extract::<PyReadonlyArray1<f64>>() {
                Ok(readonly)
            } else {
                let np = py.import_bound("numpy")?;
                let converted = np.call_method1("asarray", (arr,))?;
                converted.extract::<PyReadonlyArray1<f64>>()
            }
        };
        
        let x_array = to_array2(x)?;
        let y_array = to_array1(y)?;
        
        let x_val_array = x_val.map(to_array2).transpose()?;
        let y_val_array = y_val.map(to_array1).transpose()?;
        
        // Convert to Vec format as expected by Rust model
        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        let y_vec: Vec<f64> = y_array.as_array().to_vec();
        
        let eval_set = if let (Some(xv), Some(yv)) = (x_val_array, y_val_array) {
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
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        // Convert to array
        let x_array: PyReadonlyArray2<f64> = if let Ok(readonly) = x.extract::<PyReadonlyArray2<f64>>() {
            readonly
        } else {
            let np = py.import_bound("numpy")?;
            let converted = np.call_method1("asarray", (x,))?;
            converted.extract::<PyReadonlyArray2<f64>>()?
        };

        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
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
        x: &Bound<'py, PyAny>,
        threshold: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        // Convert to array
        let x_array: PyReadonlyArray2<f64> = if let Ok(readonly) = x.extract::<PyReadonlyArray2<f64>>() {
            readonly
        } else {
            let np = py.import_bound("numpy")?;
            let converted = np.call_method1("asarray", (x,))?;
            converted.extract::<PyReadonlyArray2<f64>>()?
        };

        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
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

    /// Serialize the fitted model to bytes (JSON format).
    /// Returns bytes that can be saved to disk or transmitted.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let model = self.model.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?;

        let json_bytes = serde_json::to_vec(model)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))?;

        Ok(PyBytes::new_bound(py, &json_bytes))
    }

    /// Deserialize a model from bytes (JSON format).
    /// This is a class method that returns a new fitted PKBoostClassifier.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = data.as_bytes();

        let model: OptimizedPKBoostShannon = serde_json::from_slice(bytes)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {}", e)))?;

        Ok(Self {
            model: Some(model),
            fitted: true,
        })
    }

    /// Serialize the fitted model to a JSON string.
    fn to_json(&self) -> PyResult<String> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let model = self.model.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?;

        serde_json::to_string(model)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))
    }

    /// Deserialize a model from a JSON string.
    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let model: OptimizedPKBoostShannon = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {}", e)))?;

        Ok(Self {
            model: Some(model),
            fitted: true,
        })
    }

    /// Save the fitted model to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let model = self.model.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?;

        let json_bytes = serde_json::to_vec(model)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))?;

        std::fs::write(path, json_bytes)
            .map_err(|e| PyValueError::new_err(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load a fitted model from a file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to read file: {}", e)))?;

        let model: OptimizedPKBoostShannon = serde_json::from_slice(&bytes)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {}", e)))?;

        Ok(Self {
            model: Some(model),
            fitted: true,
        })
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
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        x_val: Option<&Bound<'py, PyAny>>,
        y_val: Option<&Bound<'py, PyAny>>,
        verbose: Option<bool>,
    ) -> PyResult<()> {
        let to_array2 = |arr: &Bound<'py, PyAny>| -> PyResult<PyReadonlyArray2<'py, f64>> {
            let contiguous = ensure_contiguous(py, arr)?;
            contiguous.extract::<PyReadonlyArray2<f64>>()
        };
        
        let to_array1 = |arr: &Bound<'py, PyAny>| -> PyResult<PyReadonlyArray1<'py, f64>> {
            let contiguous = ensure_contiguous(py, arr)?;
            contiguous.extract::<PyReadonlyArray1<f64>>()
        };
        
        let x_array = to_array2(x)?;
        let y_array = to_array1(y)?;
        
        let x_val_array = x_val.map(to_array2).transpose()?;
        let y_val_array = y_val.map(to_array1).transpose()?;
        
        // Convert to Vec format
        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        let y_vec: Vec<f64> = y_array.as_array().to_vec();
        
        let eval_set = if let (Some(xv), Some(yv)) = (x_val_array, y_val_array) {
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
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        verbose: Option<bool>,
    ) -> PyResult<()> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }

        let x_contiguous = ensure_contiguous(py, x)?;
        let y_contiguous = ensure_contiguous(py, y)?;
        
        let x_array = x_contiguous.extract::<PyReadonlyArray2<f64>>()?;
        let y_array = y_contiguous.extract::<PyReadonlyArray1<f64>>()?;

        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        let y_vec: Vec<f64> = y_array.as_array().to_vec();
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
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }

        let x_contiguous = ensure_contiguous(py, x)?;
        let x_array = x_contiguous.extract::<PyReadonlyArray2<f64>>()?;

        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
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
        x: &Bound<'py, PyAny>,
        threshold: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit_initial() first."));
        }

        let x_contiguous = ensure_contiguous(py, x)?;
        let x_array = x_contiguous.extract::<PyReadonlyArray2<f64>>()?;

        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
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

#[pyclass]
pub struct PKBoostRegressorPy {
    model: Option<PKBoostRegressor>,
    fitted: bool,
}

#[pymethods]
impl PKBoostRegressorPy {
    #[new]
    fn new() -> Self {
        Self {
            model: None,
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
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        x_val: Option<&Bound<'py, PyAny>>,
        y_val: Option<&Bound<'py, PyAny>>,
        verbose: Option<bool>,
    ) -> PyResult<()> {
        let to_array2 = |arr: &Bound<'py, PyAny>| -> PyResult<PyReadonlyArray2<'py, f64>> {
            if let Ok(readonly) = arr.extract::<PyReadonlyArray2<f64>>() {
                Ok(readonly)
            } else {
                let np = py.import_bound("numpy")?;
                let converted = np.call_method1("asarray", (arr,))?;
                converted.extract::<PyReadonlyArray2<f64>>()
            }
        };
        
        let to_array1 = |arr: &Bound<'py, PyAny>| -> PyResult<PyReadonlyArray1<'py, f64>> {
            if let Ok(readonly) = arr.extract::<PyReadonlyArray1<f64>>() {
                Ok(readonly)
            } else {
                let np = py.import_bound("numpy")?;
                let converted = np.call_method1("asarray", (arr,))?;
                converted.extract::<PyReadonlyArray1<f64>>()
            }
        };
        
        let x_array = to_array2(x)?;
        let y_array = to_array1(y)?;
        
        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        let y_vec: Vec<f64> = y_array.as_array().to_vec();
        
        let eval_set = if let (Some(xv), Some(yv)) = (x_val.map(to_array2).transpose()?, y_val.map(to_array1).transpose()?) {
            let x_val_vec: Vec<Vec<f64>> = xv.as_array().rows()
                .into_iter().map(|row| row.to_vec()).collect();
            let y_val_vec: Vec<f64> = yv.as_array().to_vec();
            Some((x_val_vec, y_val_vec))
        } else { None };

        let verbose = verbose.unwrap_or(false);
        
        py.allow_threads(|| {
            if self.model.is_none() {
                let mut auto_model = PKBoostRegressor::auto(&x_vec, &y_vec);
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

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let x_array: PyReadonlyArray2<f64> = if let Ok(readonly) = x.extract::<PyReadonlyArray2<f64>>() {
            readonly
        } else {
            let np = py.import_bound("numpy")?;
            let converted = np.call_method1("asarray", (x,))?;
            converted.extract::<PyReadonlyArray2<f64>>()?
        };

        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        
        let predictions = py.allow_threads(|| {
            self.model.as_ref()
                .ok_or("Model not initialized".to_string())
                .and_then(|m| m.predict(&x_vec))
        }).map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;
        
        Ok(PyArray1::from_vec_bound(py, predictions))
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[pyclass]
pub struct PKBoostMultiClassPy {
    model: Option<MultiClassPKBoost>,
    fitted: bool,
}

#[pymethods]
impl PKBoostMultiClassPy {
    #[new]
    fn new(n_classes: usize) -> Self {
        Self {
            model: Some(MultiClassPKBoost::new(n_classes)),
            fitted: false,
        }
    }

    #[pyo3(signature = (x, y, x_val=None, y_val=None, verbose=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        x_val: Option<&Bound<'py, PyAny>>,
        y_val: Option<&Bound<'py, PyAny>>,
        verbose: Option<bool>,
    ) -> PyResult<()> {
        let to_array2 = |arr: &Bound<'py, PyAny>| -> PyResult<PyReadonlyArray2<'py, f64>> {
            if let Ok(readonly) = arr.extract::<PyReadonlyArray2<f64>>() {
                Ok(readonly)
            } else {
                let np = py.import_bound("numpy")?;
                let converted = np.call_method1("asarray", (arr,))?;
                converted.extract::<PyReadonlyArray2<f64>>()
            }
        };
        
        let to_array1 = |arr: &Bound<'py, PyAny>| -> PyResult<PyReadonlyArray1<'py, f64>> {
            if let Ok(readonly) = arr.extract::<PyReadonlyArray1<f64>>() {
                Ok(readonly)
            } else {
                let np = py.import_bound("numpy")?;
                let converted = np.call_method1("asarray", (arr,))?;
                converted.extract::<PyReadonlyArray1<f64>>()
            }
        };
        
        let x_array = to_array2(x)?;
        let y_array = to_array1(y)?;
        
        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        let y_vec: Vec<f64> = y_array.as_array().to_vec();
        
        let eval_set = if let (Some(xv), Some(yv)) = (x_val.map(to_array2).transpose()?, y_val.map(to_array1).transpose()?) {
            let x_val_vec: Vec<Vec<f64>> = xv.as_array().rows()
                .into_iter().map(|row| row.to_vec()).collect();
            let y_val_vec: Vec<f64> = yv.as_array().to_vec();
            Some((x_val_vec, y_val_vec))
        } else { None };

        let verbose = verbose.unwrap_or(false);
        
        py.allow_threads(|| {
            if let Some(ref mut model) = self.model {
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
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let x_array: PyReadonlyArray2<f64> = if let Ok(readonly) = x.extract::<PyReadonlyArray2<f64>>() {
            readonly
        } else {
            let np = py.import_bound("numpy")?;
            let converted = np.call_method1("asarray", (x,))?;
            converted.extract::<PyReadonlyArray2<f64>>()?
        };

        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        
        let predictions = py.allow_threads(|| {
            self.model.as_ref()
                .ok_or("Model not initialized".to_string())
                .and_then(|m| m.predict_proba(&x_vec))
        }).map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;
        
        let n_samples = predictions.len();
        let n_classes = if n_samples > 0 { predictions[0].len() } else { 0 };
        let flat: Vec<f64> = predictions.into_iter().flatten().collect();
        
        Ok(PyArray2::from_vec2_bound(py, &vec![flat; 1].into_iter()
            .flat_map(|row| row.chunks(n_classes).map(|c| c.to_vec()).collect::<Vec<_>>())
            .collect::<Vec<_>>())?)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err("Model not fitted. Call fit() first."));
        }

        let x_array: PyReadonlyArray2<f64> = if let Ok(readonly) = x.extract::<PyReadonlyArray2<f64>>() {
            readonly
        } else {
            let np = py.import_bound("numpy")?;
            let converted = np.call_method1("asarray", (x,))?;
            converted.extract::<PyReadonlyArray2<f64>>()?
        };

        let x_vec: Vec<Vec<f64>> = x_array.as_array().rows()
            .into_iter().map(|row| row.to_vec()).collect();
        
        let predictions = py.allow_threads(|| {
            self.model.as_ref()
                .ok_or("Model not initialized".to_string())
                .and_then(|m| m.predict(&x_vec))
        }).map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;
        
        Ok(PyArray1::from_vec_bound(py, predictions))
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
    m.add_class::<PKBoostRegressorPy>()?;
    m.add_class::<PKBoostMultiClassPy>()?;
    Ok(())
}