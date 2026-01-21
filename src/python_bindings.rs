use crate::living_booster::{AdversarialLivingBooster, SystemState};
use crate::model::OptimizedPKBoostShannon;
use crate::multiclass::MultiClassPKBoost;
use crate::regression::PKBoostRegressor;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Helper to convert any array-like to a contiguous PyReadonlyArray2
fn to_readonly_array2<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f64>> {
    if let Ok(readonly) = arr.extract::<PyReadonlyArray2<f64>>() {
        // Check if already contiguous
        if readonly.as_array().is_standard_layout() {
            return Ok(readonly);
        }
    }
    // Make contiguous copy if needed
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    let contiguous = np.call_method("ascontiguousarray", (arr,), Some(&kwargs))?;
    contiguous
        .extract::<PyReadonlyArray2<f64>>()
        .map_err(|e| PyValueError::new_err(format!("Failed to convert to 2D array: {}", e)))
}

/// Helper to convert any array-like to a contiguous PyReadonlyArray1
fn to_readonly_array1<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    if let Ok(readonly) = arr.extract::<PyReadonlyArray1<f64>>() {
        if readonly.as_array().is_standard_layout() {
            return Ok(readonly);
        }
    }
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    let contiguous = np.call_method("ascontiguousarray", (arr,), Some(&kwargs))?;
    contiguous
        .extract::<PyReadonlyArray1<f64>>()
        .map_err(|e| PyValueError::new_err(format!("Failed to convert to 1D array: {}", e)))
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

    /// Fit the model - TRUE ZERO-COPY: borrows directly from NumPy arrays
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
        // Get readonly views - zero-copy if already contiguous f64
        let x_arr = to_readonly_array2(py, x)?;
        let y_arr = to_readonly_array1(py, y)?;
        let x_val_arr = x_val.map(|xv| to_readonly_array2(py, xv)).transpose()?;
        let y_val_arr = y_val.map(|yv| to_readonly_array1(py, yv)).transpose()?;
        let verbose = verbose.unwrap_or(false);

        // Get ArrayViews - these borrow from NumPy, true zero-copy
        let x_view = x_arr.as_array();
        let y_view = y_arr.as_array();

        // Build eval_set from views
        let eval_set = match (&x_val_arr, &y_val_arr) {
            (Some(xv), Some(yv)) => Some((xv.as_array(), yv.as_array())),
            _ => None,
        };

        // Initialize model if using auto mode
        if self.model.is_none() {
            self.model = Some(OptimizedPKBoostShannon::auto(x_view, y_view));
        }

        // Fit model - Rayon provides internal parallelism
        if let Some(ref mut model) = self.model {
            model
                .fit(x_view, y_view, eval_set, verbose)
                .map_err(|e| PyValueError::new_err(format!("Training failed: {}", e)))?;
            self.fitted = true;
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Model initialization failed"))
        }
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }

        let x_arr = to_readonly_array2(py, x)?;
        let x_view = x_arr.as_array();

        let predictions = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?
            .predict_proba(x_view)
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;

        Ok(predictions.into_pyarray(py))
    }

    #[pyo3(signature = (x, threshold=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        threshold: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }

        let x_arr = to_readonly_array2(py, x)?;
        let x_view = x_arr.as_array();
        let threshold = threshold.unwrap_or(0.5);

        let proba = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?
            .predict_proba(x_view)
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;

        let predictions: Vec<i32> = proba
            .iter()
            .map(|&p| if p >= threshold { 1 } else { 0 })
            .collect();

        Ok(PyArray1::from_vec(py, predictions))
    }

    fn get_feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }

        let importance = if let Some(ref model) = self.model {
            let usage = model.get_feature_usage();
            let total: usize = usage.iter().sum();
            if total > 0 {
                usage.iter().map(|&u| u as f64 / total as f64).collect()
            } else {
                vec![0.0; usage.len()]
            }
        } else {
            vec![]
        };

        Ok(PyArray1::from_vec(py, importance))
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn get_n_trees(&self) -> PyResult<usize> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }
        Ok(self.model.as_ref().map(|m| m.trees.len()).unwrap_or(0))
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?;
        let json_bytes = serde_json::to_vec(model)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))?;
        Ok(PyBytes::new(py, &json_bytes))
    }

    #[staticmethod]
    fn from_bytes(_py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = data.as_bytes();
        let model: OptimizedPKBoostShannon = serde_json::from_slice(bytes)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {}", e)))?;
        Ok(Self {
            model: Some(model),
            fitted: true,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?;
        serde_json::to_string(model)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))
    }

    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let model: OptimizedPKBoostShannon = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Deserialization failed: {}", e)))?;
        Ok(Self {
            model: Some(model),
            fitted: true,
        })
    }

    fn save(&self, path: &str) -> PyResult<()> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?;
        let json_bytes = serde_json::to_vec(model)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))?;
        std::fs::write(path, json_bytes)
            .map_err(|e| PyValueError::new_err(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

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
        let x_arr = to_readonly_array2(py, x)?;
        let y_arr = to_readonly_array1(py, y)?;
        let x_val_arr = x_val.map(|xv| to_readonly_array2(py, xv)).transpose()?;
        let y_val_arr = y_val.map(|yv| to_readonly_array1(py, yv)).transpose()?;
        let verbose = verbose.unwrap_or(false);

        let x_view = x_arr.as_array();
        let y_view = y_arr.as_array();
        let eval_set = match (&x_val_arr, &y_val_arr) {
            (Some(xv), Some(yv)) => Some((xv.as_array(), yv.as_array())),
            _ => None,
        };

        let mut booster = AdversarialLivingBooster::new(x_view, y_view);
        booster
            .fit_initial(x_view, y_view, eval_set, verbose)
            .map_err(|e| PyValueError::new_err(format!("Training failed: {}", e)))?;
        self.booster = Some(booster);
        self.fitted = true;
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
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit_initial() first.",
            ));
        }

        let x_arr = to_readonly_array2(py, x)?;
        let y_arr = to_readonly_array1(py, y)?;
        let verbose = verbose.unwrap_or(false);

        self.booster
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Booster not initialized"))?
            .observe_batch(x_arr.as_array(), y_arr.as_array(), verbose)
            .map_err(|e| PyValueError::new_err(format!("Observation failed: {}", e)))?;

        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit_initial() first.",
            ));
        }

        let x_arr = to_readonly_array2(py, x)?;
        let predictions = self
            .booster
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Booster not initialized"))?
            .predict_proba(x_arr.as_array())
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;

        Ok(predictions.into_pyarray(py))
    }

    #[pyo3(signature = (x, threshold=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        threshold: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit_initial() first.",
            ));
        }

        let x_arr = to_readonly_array2(py, x)?;
        let threshold = threshold.unwrap_or(0.5);

        let proba = self
            .booster
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Booster not initialized"))?
            .predict_proba(x_arr.as_array())
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;

        let predictions: Vec<i32> = proba
            .iter()
            .map(|&p| if p >= threshold { 1 } else { 0 })
            .collect();

        Ok(PyArray1::from_vec(py, predictions))
    }

    fn get_vulnerability_score(&self) -> PyResult<f64> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit_initial() first.",
            ));
        }
        Ok(self
            .booster
            .as_ref()
            .map(|b| b.get_vulnerability_score())
            .unwrap_or(0.0))
    }

    fn get_state(&self) -> PyResult<String> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit_initial() first.",
            ));
        }
        let state = self
            .booster
            .as_ref()
            .map(|b| b.get_state())
            .unwrap_or(SystemState::Normal);
        Ok(match state {
            SystemState::Normal => "Normal".to_string(),
            SystemState::Alert { checks_in_alert } => format!("Alert({})", checks_in_alert),
            SystemState::Metamorphosis => "Metamorphosis".to_string(),
        })
    }

    fn get_metamorphosis_count(&self) -> PyResult<usize> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit_initial() first.",
            ));
        }
        Ok(self
            .booster
            .as_ref()
            .map(|b| b.get_metamorphosis_count())
            .unwrap_or(0))
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[pyclass(name = "PKBoostRegressor")]
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
        let x_arr = to_readonly_array2(py, x)?;
        let y_arr = to_readonly_array1(py, y)?;
        let x_val_arr = x_val.map(|xv| to_readonly_array2(py, xv)).transpose()?;
        let y_val_arr = y_val.map(|yv| to_readonly_array1(py, yv)).transpose()?;
        let verbose = verbose.unwrap_or(false);

        let x_view = x_arr.as_array();
        let y_view = y_arr.as_array();
        let eval_set = match (&x_val_arr, &y_val_arr) {
            (Some(xv), Some(yv)) => Some((xv.as_array(), yv.as_array())),
            _ => None,
        };

        if self.model.is_none() {
            let mut auto_model = PKBoostRegressor::auto(x_view, y_view);
            auto_model
                .fit(x_view, y_view, eval_set, verbose)
                .map_err(|e| PyValueError::new_err(format!("Training failed: {}", e)))?;
            self.model = Some(auto_model);
        } else if let Some(ref mut model) = self.model {
            model
                .fit(x_view, y_view, eval_set, verbose)
                .map_err(|e| PyValueError::new_err(format!("Training failed: {}", e)))?;
        }
        self.fitted = true;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }

        let x_arr = to_readonly_array2(py, x)?;
        let predictions = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?
            .predict(x_arr.as_array())
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;

        Ok(predictions.into_pyarray(py))
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[pyclass(name = "PKBoostMultiClass")]
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
        let x_arr = to_readonly_array2(py, x)?;
        let y_arr = to_readonly_array1(py, y)?;
        let x_val_arr = x_val.map(|xv| to_readonly_array2(py, xv)).transpose()?;
        let y_val_arr = y_val.map(|yv| to_readonly_array1(py, yv)).transpose()?;
        let verbose = verbose.unwrap_or(false);

        let x_view = x_arr.as_array();
        let y_view = y_arr.as_array();
        let eval_set = match (&x_val_arr, &y_val_arr) {
            (Some(xv), Some(yv)) => Some((xv.as_array(), yv.as_array())),
            _ => None,
        };

        if let Some(ref mut model) = self.model {
            model
                .fit(x_view, y_view, eval_set, verbose)
                .map_err(|e| PyValueError::new_err(format!("Training failed: {}", e)))?;
            self.fitted = true;
        }
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }

        let x_arr = to_readonly_array2(py, x)?;
        let predictions = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?
            .predict_proba(x_arr.as_array())
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;

        Ok(predictions.into_pyarray(py))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            ));
        }

        let x_arr = to_readonly_array2(py, x)?;
        let predictions = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Model not initialized"))?
            .predict(x_arr.as_array())
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))?;

        Ok(predictions.into_pyarray(py))
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
