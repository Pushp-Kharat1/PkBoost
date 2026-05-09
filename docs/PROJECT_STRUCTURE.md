# PKBoost Project Structure

This guide explains how the repository is organized so new contributors can quickly find the right files.

## Top-Level Layout

- `src/` - core Rust implementation of PKBoost
- `src/bin/` - runnable Rust binaries; `benchmark_fraud.rs` is the benchmark entrypoint currently declared in `Cargo.toml`
- `docs/` - long-form guides, benchmark reports, and reference material
- `examples/` - Python examples for trying the package quickly
- `pkboost_sklearn/` - scikit-learn compatible Python wrapper
- `tests/` - Python-side tests, currently focused on serialization and sklearn compatibility
- `data/` - packaged sample assets and example artifacts used by project materials
- `raw_data/` - notes about raw datasets
- `benchmark results/` - generated result tables, charts, and changelog files
- `temp/` - ad hoc comparison scripts and experiments

## Rust Source Map

Start in `src/lib.rs`, then branch out by concern:

- `src/model.rs` - main model behavior
- `src/tree.rs` and `src/tree_regression.rs` - tree construction logic
- `src/regression.rs` - regression-specific functionality
- `src/multiclass.rs` - multi-class implementation
- `src/python_bindings.rs` - PyO3 bindings exposed to Python
- `src/metrics.rs` - evaluation metrics such as PR-AUC and ROC-AUC
- `src/auto_tuner.rs` and `src/auto_params.rs` - automatic parameter selection
- `src/adversarial.rs`, `src/living_booster.rs`, and `src/metabolism.rs` - drift monitoring and adaptation logic
- `src/histogram_builder.rs` and `src/optimized_data.rs` - data preparation and histogram internals

## Python-Facing Areas

- `pyproject.toml` configures the Python package build via Maturin
- `pkboost_sklearn/` contains sklearn-style estimator wrappers
- `examples/` shows end-to-end usage patterns
- `tests/` exercises the Python package surface

## Documentation Map

- Start with [README.md](../README.md) for the high-level overview
- Use [README.md](README.md) in this folder as the documentation hub
- Use [PYTHON_BINDINGS.md](PYTHON_BINDINGS.md) for Python install and API guidance
- Use [BENCHMARK_REPRODUCTION.md](BENCHMARK_REPRODUCTION.md) for benchmark steps
- Use [FEATURES.md](FEATURES.md) for a feature inventory

## Good First Places to Contribute

- Broken links or outdated commands in Markdown files
- Missing cross-references between related docs
- Additional examples in `examples/` or `pkboost_sklearn/README.md`
- Clarifications around benchmark setup and repository workflow
