//! # PKBoost: Shannon-Guided Gradient Boosting
//!
//! [![Crates.io](https://img.shields.io/crates/v/pkboost.svg)](https://crates.io/crates/pkboost)
//! [![Documentation](https://docs.rs/pkboost/badge.svg)](https://docs.rs/pkboost)
//! [![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
//!
//! PKBoost (**P**erformance-Based **K**nowledge **Boost**er) is an adaptive gradient boosting
//! library built from scratch in Rust, specifically designed for **extreme class imbalance**
//! and **concept drift** scenarios.
//!
//! ## Key Features
//!
//! - **Extreme Imbalance Handling**: Outperforms XGBoost/LightGBM on datasets with <5% minority class
//! - **Drift Detection & Adaptation**: Automatically detects concept drift and triggers model adaptation
//! - **Shannon Entropy Guidance**: Splits optimized using information theory for minority class
//! - **Auto-Tuning**: No hyperparameter tuning required - auto-configures based on data
//! - **Multi-Task Support**: Binary classification, multi-class, and regression
//! - **Built-in Metrics**: PR-AUC, ROC-AUC, F1, RMSE, R², and more
//!
//! ## Quick Start
//!
//! ### Binary Classification (Recommended for Imbalanced Data)
//!
//! ```rust,no_run
//! use pkboost::{OptimizedPKBoostShannon, calculate_pr_auc, calculate_roc_auc};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Your data: Vec<Vec<f64>> for features, Vec<f64> for labels (0.0 or 1.0)
//!     let x_train: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
//!     let y_train: Vec<f64> = vec![0.0, 1.0];
//!     let x_test: Vec<Vec<f64>> = vec![vec![1.5, 2.5]];
//!     let y_test: Vec<f64> = vec![0.0];
//!
//!     // Create model with auto-tuning (recommended)
//!     let mut model = OptimizedPKBoostShannon::auto(&x_train, &y_train);
//!
//!     // Train with optional validation set for early stopping
//!     model.fit(&x_train, &y_train, None, true)?;
//!
//!     // Predict probabilities
//!     let predictions = model.predict_proba(&x_test)?;
//!
//!     // Evaluate
//!     let pr_auc = calculate_pr_auc(&y_test, &predictions);
//!     let roc_auc = calculate_roc_auc(&y_test, &predictions);
//!     println!("PR-AUC: {:.4}, ROC-AUC: {:.4}", pr_auc, roc_auc);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Multi-Class Classification
//!
//! ```rust,no_run
//! use pkboost::MultiClassPKBoost;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let x_train: Vec<Vec<f64>> = vec![/* your data */];
//!     let y_train: Vec<f64> = vec![0.0, 1.0, 2.0]; // Class labels: 0, 1, 2, ...
//!     let x_test: Vec<Vec<f64>> = vec![/* test data */];
//!
//!     // Specify number of classes
//!     let mut model = MultiClassPKBoost::new(3);
//!    
//!     // Train
//!     model.fit(&x_train, &y_train, None, true)?;
//!
//!     // Get class probabilities [n_samples, n_classes]
//!     let probs = model.predict_proba(&x_test)?;
//!    
//!     // Or get predicted class indices
//!     let predictions = model.predict(&x_test)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Regression
//!
//! ```rust,no_run
//! use pkboost::{PKBoostRegressor, calculate_rmse, calculate_r2};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let x_train: Vec<Vec<f64>> = vec![/* your data */];
//!     let y_train: Vec<f64> = vec![/* continuous targets */];
//!     let x_test: Vec<Vec<f64>> = vec![/* test data */];
//!     let y_test: Vec<f64> = vec![/* test targets */];
//!
//!     // Create regressor with auto configuration
//!     let mut model = PKBoostRegressor::auto(&x_train, &y_train);
//!    
//!     // Train
//!     model.fit(&x_train, &y_train, None, true)?;
//!
//!     // Predict
//!     let predictions = model.predict(&x_test)?;
//!
//!     // Evaluate
//!     let rmse = calculate_rmse(&y_test, &predictions);
//!     let r2 = calculate_r2(&y_test, &predictions);
//!     println!("RMSE: {:.4}, R²: {:.4}", rmse, r2);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Adaptive Model with Drift Detection
//!
//! For streaming data or scenarios where data distribution changes over time:
//!
//! ```rust,no_run
//! use pkboost::AdversarialLivingBooster;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let x_train: Vec<Vec<f64>> = vec![/* initial training data */];
//!     let y_train: Vec<f64> = vec![/* initial labels */];
//!
//!     // Create adaptive model
//!     let mut model = AdversarialLivingBooster::new(&x_train, &y_train);
//!    
//!     // Initial training
//!     model.fit_initial(&x_train, &y_train, None, true)?;
//!
//!     // As new data arrives, observe it (model adapts automatically)
//!     let x_new: Vec<Vec<f64>> = vec![/* new batch */];
//!     let y_new: Vec<f64> = vec![/* new labels */];
//!     model.observe_batch(&x_new, &y_new, true)?;
//!
//!     // Check model state
//!     println!("Vulnerability score: {:.4}", model.get_vulnerability_score());
//!     println!("Metamorphosis count: {}", model.get_metamorphosis_count());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Builder Pattern (Advanced Configuration)
//!
//! For fine-grained control over hyperparameters:
//!
//! ```rust,no_run
//! use pkboost::OptimizedPKBoostShannon;
//!
//! let model = OptimizedPKBoostShannon::builder()
//!     .n_estimators(200)
//!     .learning_rate(0.05)
//!     .max_depth(6)
//!     .min_samples_split(10)
//!     .reg_lambda(1.0)
//!     .gamma(0.1)
//!     .subsample(0.8)
//!     .colsample_bytree(0.8)
//!     .early_stopping_rounds(20)
//!     .histogram_bins(32)
//!     .mi_weight(0.1)           // Mutual information weight for imbalance
//!     .scale_pos_weight(5.0)    // Weight for positive class
//!     .build();
//! ```
//!
//! ## Core Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`OptimizedPKBoostShannon`] | Binary classification with Shannon entropy guidance |
//! | [`MultiClassPKBoost`] | Multi-class classification via One-vs-Rest |
//! | [`PKBoostRegressor`] | Regression with MSE, Huber, or Poisson loss |
//! | [`AdversarialLivingBooster`] | Adaptive model with drift detection |
//!
//! ## Metrics
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`calculate_pr_auc`] | Precision-Recall AUC (best for imbalanced data) |
//! | [`calculate_roc_auc`] | Receiver Operating Characteristic AUC |
//! | [`calculate_rmse`] | Root Mean Squared Error |
//! | [`calculate_mae`] | Mean Absolute Error |
//! | [`calculate_r2`] | R² coefficient of determination |
//!
//! ## Model Serialization
//!
//! PKBoost models implement `serde::Serialize` and `serde::Deserialize`:
//!
//! ```rust,no_run
//! use pkboost::OptimizedPKBoostShannon;
//!
//! // Save model
//! let model = OptimizedPKBoostShannon::auto(&x_train, &y_train);
//! let json = serde_json::to_string(&model)?;
//! std::fs::write("model.json", json)?;
//!
//! // Load model
//! let json = std::fs::read_to_string("model.json")?;
//! let model: OptimizedPKBoostShannon = serde_json::from_str(&json)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## When to Use PKBoost
//!
//! **✅ Good fit:**
//! - Extreme class imbalance (<5% minority class)
//! - Fraud detection, anomaly detection, rare event prediction
//! - Data that evolves over time (concept drift)
//! - When you want good results without hyperparameter tuning
//!
//! **❌ Consider alternatives for:**
//! - Perfectly balanced datasets (XGBoost may be faster)
//! - Very small datasets (<1,000 samples)
//!
//! ## Author
//!
//! **Pushp Kharat** - [GitHub](https://github.com/Pushp-Kharat1/pkboost)
//!
//! ## License
//!
//! This project is licensed under the GPL-3.0 License.

pub mod adaptive_parallel;
pub mod adversarial;
pub mod auto_params;
pub mod auto_tuner;
pub mod constants;
pub mod fork_parallel;
pub mod histogram_builder;
pub mod huber_loss;
pub mod living_booster;
pub mod living_regressor;
pub mod loss;
pub mod metabolism;
pub mod metrics;
pub mod model;
pub mod multiclass;
pub mod optimized_data;
pub mod partitioned_classifier;
pub mod precision;
pub mod python_bindings;
pub mod regression;
pub mod tree;
pub mod tree_regression;

// Re-exports for convenient access
pub use adversarial::AdversarialEnsemble;
pub use auto_params::{auto_params, AutoHyperParams, DataStats};
pub use constants::*;
pub use histogram_builder::OptimizedHistogramBuilder;
pub use huber_loss::HuberLoss;
pub use living_booster::AdversarialLivingBooster;
pub use living_regressor::{AdaptiveRegressor, SystemState};
pub use loss::{LossType, MSELoss, OptimizedShannonLoss, PoissonLoss};
pub use metabolism::FeatureMetabolism;
pub use metrics::{calculate_pr_auc, calculate_roc_auc, calculate_shannon_entropy};
pub use model::OptimizedPKBoostShannon;
pub use multiclass::MultiClassPKBoost;
pub use optimized_data::CachedHistogram;
pub use optimized_data::TransposedData;
pub use partitioned_classifier::{
    PartitionConfig, PartitionMethod, PartitionedClassifier, PartitionedClassifierBuilder, TaskType,
};
pub use precision::{AdaptiveCompute, PrecisionLevel, ProgressiveBuffer, ProgressivePrecision};
pub use regression::{
    calculate_mad, calculate_mae, calculate_r2, calculate_rmse, detect_outliers,
    MSELoss as RegressionMSELoss, PKBoostRegressor, RegressionLossType,
};
pub use tree::{HistSplitResult, OptimizedTreeShannon, TreeParams};
