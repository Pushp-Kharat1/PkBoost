// Constants for PKBoost - extracted magic numbers for maintainability

// Metamorphosis thresholds
pub const METAMORPHOSIS_PERFORMANCE_THRESHOLD: f64 = 0.98;  // 2% degradation allowed
pub const METAMORPHOSIS_ROLLBACK_TOLERANCE: f64 = 1.02;     // 2% degradation before rollback

// Vulnerability detection
pub const VULNERABILITY_ERROR_THRESHOLD: f64 = 0.2;         // 20% error threshold
pub const VULNERABILITY_EMA_ALPHA: f64 = 0.1;               // EMA smoothing factor

// Drift sensitivity thresholds
pub const BASE_DEGRADATION_THRESHOLD: f64 = 0.10;           // 10% base degradation
pub const NOISY_DATA_THRESHOLD_MULTIPLIER: f64 = 1.5;       // 15% for noisy data
pub const NOISE_DETECTION_MULTIPLIER: f64 = 2.0;            // Vuln > 2x baseline = noisy

// Weighted RMSE
pub const RMSE_WEIGHT_RECENT: f64 = 0.5;                    // Most recent weight
pub const RMSE_WEIGHT_MIDDLE: f64 = 0.3;                    // Middle weight
pub const RMSE_WEIGHT_OLDEST: f64 = 0.2;                    // Oldest weight

// Feature metabolism
pub const FEATURE_DEAD_USAGE_THRESHOLD: f64 = 0.01;         // <1% usage = dead
pub const FEATURE_DRIFT_THRESHOLD: f64 = 0.2;               // 20% distribution shift

// Tree building
pub const DEPTH_DECAY_RATE: f64 = 0.1;                      // MI weight decay per depth
pub const MIN_VALIDATION_SIZE: usize = 500;                 // Minimum samples for validation

// Gradient monitoring
pub const GRADIENT_WARNING_THRESHOLD: f64 = 1000.0;         // Warn on large gradients
pub const GRADIENT_CRITICAL_THRESHOLD: f64 = 5000.0;        // Stop on gradient explosion

// Buffer sizing
pub const SMALL_BUFFER_THRESHOLD: usize = 2000;
pub const LARGE_BUFFER_THRESHOLD: usize = 15000;

// Complexity classification
pub const HIGH_COMPLEXITY_THRESHOLD: f64 = 2.5;
pub const MODERATE_COMPLEXITY_THRESHOLD: f64 = 1.2;

// Auto-tuning factors
pub const SIZE_FACTOR_SMALL: f64 = 0.8;
pub const SIZE_FACTOR_LARGE: f64 = 1.2;
pub const COMPLEXITY_FACTOR_HIGH: f64 = 1.3;
pub const COMPLEXITY_FACTOR_LOW: f64 = 0.8;
pub const SEVERITY_FACTOR_VERY_SEVERE: f64 = 1.3;
pub const SEVERITY_FACTOR_SEVERE: f64 = 1.15;
pub const SEVERITY_FACTOR_MILD: f64 = 0.9;

// Learning rate adjustments
pub const BASE_LR_MULTIPLIER: f64 = 1.5;
pub const LR_ADJUSTMENT_HIGH_COMPLEXITY: f64 = 0.9;
pub const LR_ADJUSTMENT_LOW_COMPLEXITY: f64 = 1.1;

// Tree count limits
pub const MIN_TREES_PER_METAMORPHOSIS: usize = 60;
pub const MAX_TREES_PER_METAMORPHOSIS: usize = 120;

// Tree counts by drift type
pub const TREES_SEVERE_DRIFT: usize = 120;
pub const TREES_TEMPORAL_DRIFT: usize = 90;
pub const TREES_VARIANCE_DRIFT: usize = 80;
pub const TREES_LOCALIZED_DRIFT: usize = 40;

// Entropy thresholds
pub const SYSTEMIC_DRIFT_ENTROPY: f64 = 2.5;
pub const LOCALIZED_DRIFT_ENTROPY: f64 = 1.5;

// Combined drift scoring weights
pub const DRIFT_WEIGHT_ENTROPY: f64 = 0.4;
pub const DRIFT_WEIGHT_TEMPORAL: f64 = 0.3;
pub const DRIFT_WEIGHT_VARIANCE: f64 = 0.3;

// Combined drift thresholds
pub const SEVERE_DRIFT_THRESHOLD: f64 = 0.7;
pub const TEMPORAL_DRIFT_THRESHOLD: f64 = 0.5;
pub const VARIANCE_DRIFT_THRESHOLD: f64 = 0.6;

// Division safety
pub const EPSILON: f64 = 1e-10;
