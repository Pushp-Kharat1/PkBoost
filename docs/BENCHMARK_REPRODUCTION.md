# PKBoost Benchmark Reproduction Guide

This guide describes the benchmark workflow that is runnable from the current repository snapshot.

## What Is Included Today

The repository currently includes:

- a Rust benchmark entrypoint at `src/bin/benchmark_fraud.rs`
- Python examples under `examples/`
- sample benchmark CSVs in `data/`
- benchmark result artifacts and reports in `docs/` and `benchmark results/`

The repository does not currently include the older `prepare_data.py`, `run_single_benchmark.py`, `drift_comparison_all.py`, or `scripts/` workflow referenced by some historical materials.

## Prerequisites

### Software

- Rust 1.70+
- Python 3.8+

Check your versions:

```bash
rustc --version
python --version
```

### Python dependencies

For the Python examples:

```bash
pip install maturin numpy pandas scikit-learn lightgbm xgboost
```

## Quick Start With Included Sample Data

Clone the repository:

```bash
git clone https://github.com/Pushp-Kharat1/pkboost.git
cd pkboost
```

Verify the packaged sample files:

```bash
ls data/
```

You should see the credit-card split files:

- `creditcard_train.csv`
- `creditcard_val.csv`
- `creditcard_test.csv`

### Run the Rust benchmark

```bash
cargo run --release --bin benchmark_fraud
```

This is the benchmark binary currently declared in `Cargo.toml`.

### Run the Python benchmark example

Build the local Python package first:

```bash
maturin develop --release
```

Then run:

```bash
python examples/benchmark_fraud.py
```

### Run broader Python API validation

```bash
python examples/comprehensive_example.py
```

This validates the Python-facing APIs with generated data and is useful when checking the bindings after changes.

## Expected Benchmark Direction

On the included credit-card split, the project documentation reports that PKBoost should outperform the comparison models on PR-AUC for this highly imbalanced task.

Reference values used elsewhere in the docs:

| Model | PR-AUC | ROC-AUC | F1 |
|-------|--------|---------|----|
| PKBoost | 0.87-0.88 | about 0.97 | about 0.87 |
| LightGBM | about 0.79 | about 0.92 | about 0.71 |
| XGBoost | about 0.74-0.76 | about 0.91-0.93 | about 0.80 |

Exact values can vary based on environment, compiler settings, and which packaged split or report you compare against.

## Reproducing From Your Own Prepared Data

If you want to benchmark on your own split, prepare CSV files that match the sample layout and place them under `data/`.

Recommended filenames:

- `data/creditcard_train.csv`
- `data/creditcard_val.csv`
- `data/creditcard_test.csv`

Expected format:

- header row present
- target column named `Class`
- numeric feature columns
- one row per observation

Once those files are in place, reuse the same commands:

```bash
cargo run --release --bin benchmark_fraud
python examples/benchmark_fraud.py
```

## Drift Reproduction Notes

The current repository snapshot documents drift behavior primarily through reports rather than a tracked standalone reproduction script.

See:

- [DRIFT_BENCHMARK_REPORT.md](DRIFT_BENCHMARK_REPORT.md)
- [DRYBEAN_DRIFT_RESULTS.md](DRYBEAN_DRIFT_RESULTS.md)

For adaptive API examples on the Python side, use:

```bash
python examples/comprehensive_example.py
```

## Validation Checklist

When verifying benchmark-related documentation changes, it is reasonable to check:

```bash
git status --short
cargo run --release --bin benchmark_fraud
python -m pytest tests/
```

If you only changed docs, at minimum confirm that file paths, binary names, and local Markdown links still resolve.

## Troubleshooting

### `maturin` not found

Install it first:

```bash
pip install maturin
```

### Python example cannot import `pkboost`

Rebuild and install the local extension:

```bash
maturin develop --release
```

### Rust benchmark binary name mismatch

Use the current binary declared in `Cargo.toml`:

```bash
cargo run --release --bin benchmark_fraud
```

### Looking for the old `scripts/` workflow

Those helper scripts are not included in this repository snapshot. Use the current examples and docs instead:

- [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)
- [PYTHON_BINDINGS.md](PYTHON_BINDINGS.md)
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## Related Docs

- [README.md](../README.md)
- [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)
- [PYTHON_BINDINGS.md](PYTHON_BINDINGS.md)
- [DRIFT_BENCHMARK_REPORT.md](DRIFT_BENCHMARK_REPORT.md)
