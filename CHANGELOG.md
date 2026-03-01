# Changelog

All notable changes to [PKBoost](https://pypi.org/project/pkboost/) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] - 2026-03-01

### Added

- `predict_contributions(X)` on `PKBoostClassifier` — per-feature Saabas contributions with bias term; output shape `(n_samples, n_features + 1)`, row sums equal raw pre-sigmoid prediction
- Cover tracking per node during `fit_optimized()` for contribution computation
- `node_covers` field in serialized models (`#[serde(default)]` for backwards compat)
- CircleCI integration: PR checks (lint, test, version bump) and release pipeline to AWS CodeArtifact

### Fixed

- Maturin package layout: added `pkboost/__init__.py` so source installs work correctly

## [2.2.2] - 2026-01-24

### Fixed

- Corrected `PKBoostRegressor` class reference in `sklearn_interface.py`

## [2.2.1] - 2026-01-21

### Fixed

- Used `from_shape_vec` for contiguous standard layout transpose to resolve memory layout issues
- Used `as_standard_layout()` for transpose to ensure contiguous memory

## [2.2.0] - 2026-01-21

### Added

- Zero-copy `ArrayView2` migration for true Python-Rust data sharing, eliminating unnecessary copies at the FFI boundary

## [2.1.1] - 2025-12-27

### Added

- Python 3.14 support
- Fork-join parallelism for tree building (experimental), parallelizing histogram building during tree construction on multi-core systems

### Changed

- 4x performance boost via optimized histogram building and memory layout
- Inference speed-up through tighter inner loops

### Fixed

- Serialization/deserialization support and wheel packaging to include the sklearn compatibility layer
- Modernized PyO3 bindings for Python 3.14 compatibility
- Fixed SIMD bug; added f16/bf16/f32/f64 dtype support

## [2.1.0] - 2025-12-16

### Changed

- Bumped version for the serialization and sklearn packaging fixes landing in the next patch
- Added update notification mechanism

## [2.0.2] - 2025-11-05

### Added

- **Poisson Loss for Count Regression** — full support for count-based targets (Y ∈ {0, 1, 2, …}) with log-link function and Newton-Raphson integration
- Unified loss module (`src/loss.rs`) consolidating Poisson, MSE, and Huber losses
- `RegressionLossType::Poisson` variant and `.with_loss()` builder method
- Comprehensive Poisson loss documentation and benchmark test

### Changed

- 6.4% RMSE improvement over MSE on synthetic Poisson data

## [2.0.1] - 2025-11-04

### Added

- Full scikit-learn compatibility with `PKBoostClassifier`, `PKBoostRegressor`, and `PKBoostMultiClassClassifier` wrappers
- Regressor and MultiClass classes exposed in Python bindings

### Fixed

- SIMD bug; added f16/bf16/f32/f64 support

## [2.0.0] - 2025-11-03

### Added

- **Multi-Class Classification** via One-vs-Rest strategy with softmax normalization, per-class auto-tuning, and 92.36% accuracy on the Dry Bean dataset (7 classes)
- **Hierarchical Adaptive Boosting (HAB)** — partition-based ensemble with K-means clustering, 165x faster selective retraining, SimSIMD-accelerated distance calculations, and per-partition drift detection with EMA
- Advanced drift diagnostics: error entropy, temporal patterns, variance monitoring
- Metamorphosis strategies: Conservative, DataAware, FeatureAware
- Prediction uncertainty via ensemble variance and confidence intervals
- Batched prediction (`predict_proba_batch`) for large datasets
- Comprehensive documentation: FEATURES.md, MULTICLASS.md, benchmark results, and API reference

### Changed

- 32–46% core model speedup through loop unrolling in histogram building, conditional entropy skipping at depth > 4, and smart parallelism thresholds
- Parallel specialist training and batched processing in HAB architecture

### Fixed

- Data leakage in synthetic multi-class dataset
- Gradient explosion handling in Living Regressor
- Error handling in HAB metamorphosis
- Removed unused imports and dead code warnings

## [0.1.1] - 2025-10-29

### Changed

- Documentation and packaging improvements

## [0.1.0] - 2025-10-27

### Added

- Initial release of PKBoost — gradient boosting built in Rust for concept drift in imbalanced data
- Shannon entropy-guided tree splitting with Newton-Raphson optimization
- Binary classification with automatic class weighting for extreme imbalance (e.g., 0.2% fraud rates)
- Built-in PR-AUC optimization, early stopping, and histogram-based tree construction
- Parallel processing via Rayon
- Less than 2% performance degradation under drift vs 31.8% (XGBoost) and 42.5% (LightGBM)
- Python 3.8+ support with NumPy integration

[2.2.2]: https://github.com/pushp/PkBoost/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/pushp/PkBoost/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/pushp/PkBoost/compare/v2.1.1...v2.2.0
[2.1.1]: https://github.com/pushp/PkBoost/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/pushp/PkBoost/compare/v2.0.2...v2.1.0
[2.0.2]: https://github.com/pushp/PkBoost/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/pushp/PkBoost/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/pushp/PkBoost/compare/v0.1.1...v2.0.0
[0.1.1]: https://github.com/pushp/PkBoost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/pushp/PkBoost/releases/tag/v0.1.0
