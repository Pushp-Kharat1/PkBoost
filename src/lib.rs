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