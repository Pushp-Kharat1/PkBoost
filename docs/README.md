# PKBoost Documentation Hub

Use this page as the starting point for the repository's written documentation.

## Start Here

- New users: begin with the project overview in [README.md](../README.md)
- Python users: see [PYTHON_BINDINGS.md](PYTHON_BINDINGS.md)
- Benchmark replication: follow [BENCHMARK_REPRODUCTION.md](BENCHMARK_REPRODUCTION.md)
- Contributors: read [../CONTRIBUTING.md](../CONTRIBUTING.md)

## Guides

### Core usage

- [PYTHON_BINDINGS.md](PYTHON_BINDINGS.md) - Python package installation, APIs, and examples
- [MULTICLASS.md](MULTICLASS.md) - multi-class model behavior and usage notes
- [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) - current examples, utility entrypoints, and notes on archived scripts
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - repository layout and where to start contributing

### Benchmarks and evaluation

- [BENCHMARK_REPRODUCTION.md](BENCHMARK_REPRODUCTION.md) - reproduce the main benchmark flow
- [DRIFT_BENCHMARK_REPORT.md](DRIFT_BENCHMARK_REPORT.md) - drift resilience results and analysis
- [DRYBEAN_DRIFT_RESULTS.md](DRYBEAN_DRIFT_RESULTS.md) - Dry Bean drift benchmark details

### Architecture and research notes

- [FEATURES.md](FEATURES.md) - feature inventory across the library
- [SHANNON_ANALYSIS.md](SHANNON_ANALYSIS.md) - entropy-guided split analysis
- [POISSON_LOSS.md](POISSON_LOSS.md) - Poisson loss support notes
- [HAB_CONCLUSION.md](HAB_CONCLUSION.md) - Hierarchical Adaptive Boosting summary
- [HAB_FINAL_CONCLUSION.md](HAB_FINAL_CONCLUSION.md) - final HAB conclusions

### Release notes and reference material

- [V2_READY.md](V2_READY.md) - v2 release readiness snapshot
- [Math.pdf](Math.pdf) - mathematical derivations
- [../benchmark results/Extras/CHANGELOG_V2.md](../benchmark%20results/Extras/CHANGELOG_V2.md) - v2 changelog
- [../benchmark results/Extras/CHANGELOG_V2.0.2.md](../benchmark%20results/Extras/CHANGELOG_V2.0.2.md) - v2.0.2 changelog

## Repository Pointers

- `src/` contains the Rust implementation and Python bindings entrypoints
- `src/bin/benchmark_fraud.rs` is the currently tracked benchmark binary in the manifest
- `examples/` contains Python examples for quick experimentation
- `pkboost_sklearn/` contains the scikit-learn compatible wrapper
- `tests/` contains Python-side serialization and wrapper tests

## Keeping Docs Current

When you update commands, file paths, or references in docs, confirm that they match the current repository layout before opening a pull request.
