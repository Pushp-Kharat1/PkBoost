//! PKBoost: Optimized Gradient Boosting with Shannon Entropy
//! Author: Pushp Kharat

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

//What does PKBoost means?
// PKBoost has three main fullforms, which i shift depending on -
//1) Performance-Based Knowledge Booster :- When the model is performing good with no errors and bugs
//2) Pushp_kharat's Booster :- Cause why not, i built this
//3) Pieceofshit Knavish (Scheming; unprincipled; dishonorable.) Booster :- when the fucking thing doesnt works, and i have to sit hours to debug the bloody thing
