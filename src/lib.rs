//! PKBoost: Optimized Gradient Boosting with Shannon Entropy
//! Author: Pushp Kharat 

pub mod histogram_builder;
pub mod loss;
pub mod tree;
pub mod model;
pub mod metrics;
pub mod optimized_data;
pub mod adaptive_parallel;
pub mod auto_params;
pub mod auto_tuner;
pub mod metabolism;
pub mod adversarial;
pub mod living_booster;
pub mod python_bindings;
pub mod regression;
pub mod tree_regression;
pub mod living_regressor;
pub mod constants;
pub mod huber_loss;
pub mod partitioned_classifier;
pub mod multiclass;
pub mod precision;

pub use histogram_builder::OptimizedHistogramBuilder;
pub use loss::OptimizedShannonLoss;
pub use tree::{OptimizedTreeShannon, TreeParams, HistSplitResult};
pub use optimized_data::CachedHistogram;
pub use model::OptimizedPKBoostShannon;
pub use metrics::{calculate_roc_auc, calculate_pr_auc, calculate_shannon_entropy};
pub use optimized_data::TransposedData;
pub use metabolism::FeatureMetabolism;
pub use adversarial::AdversarialEnsemble;
pub use living_booster::AdversarialLivingBooster;
pub use auto_params::{DataStats, auto_params, AutoHyperParams};
pub use regression::{PKBoostRegressor, RegressionLossType, calculate_rmse, calculate_mae, calculate_r2, detect_outliers, calculate_mad};
pub use living_regressor::{AdaptiveRegressor, SystemState};
pub use constants::*;
pub use huber_loss::HuberLoss;
pub use partitioned_classifier::{PartitionedClassifier, PartitionedClassifierBuilder, PartitionConfig, TaskType, PartitionMethod};
pub use multiclass::MultiClassPKBoost;
pub use precision::{PrecisionLevel, ProgressivePrecision, AdaptiveCompute, ProgressiveBuffer};