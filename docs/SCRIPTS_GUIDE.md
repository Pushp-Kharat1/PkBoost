# PKBoost Scripts and Examples Guide

This guide covers the runnable example and utility entrypoints that exist in the current repository snapshot.

## Current Repository Layout

The project currently exposes runnable workflows from these locations:

- `src/bin/benchmark_fraud.rs` - Rust benchmark entrypoint declared in `Cargo.toml`
- `examples/benchmark_fraud.py` - Python benchmark example using the included credit-card split
- `examples/comprehensive_example.py` - broad Python API validation example
- `examples/test_simple.py` - minimal Python smoke test
- `temp/compare_models.py` - experimental comparison script
- `temp/compare_drift.py` - experimental drift comparison script
- `temp/compare_baseline.py` - experimental baseline comparison script

## Rust Benchmark Entry Point

Run the current Rust benchmark with the sample CSV files already checked into `data/`:

```bash
cargo run --release --bin benchmark_fraud
```

This is the main benchmark command referenced by the updated README and contributing docs.

## Python Examples

Install the package locally first if you want to run the Python examples against the repository code:

```bash
pip install maturin
maturin develop --release
```

### `examples/benchmark_fraud.py`

Runs the packaged credit-card benchmark example.

```bash
python examples/benchmark_fraud.py
```

Expected inputs:

- `data/creditcard_train.csv`
- `data/creditcard_val.csv`
- `data/creditcard_test.csv`

### `examples/comprehensive_example.py`

Exercises the Python-facing APIs with generated datasets.

```bash
python examples/comprehensive_example.py
```

Use this when you want a broad sanity check of the bindings and wrapper APIs.

### `examples/test_simple.py`

Runs a minimal classifier setup useful for quick smoke testing.

```bash
python examples/test_simple.py
```

## Experimental Utility Scripts

The `temp/` directory contains ad hoc scripts used for local experimentation.

- `temp/compare_models.py`
- `temp/compare_drift.py`
- `temp/compare_baseline.py`

These scripts are not part of the stable public workflow, so treat them as exploratory utilities rather than documented product entrypoints.

## Historical Note

Some older project materials refer to helper scripts such as `prepare_data.py`, `run_single_benchmark.py`, `drift_comparison_all.py`, and a `scripts/` directory. Those files are not present in this repository snapshot.

If you are updating docs, prefer the current entrypoints under `examples/`, `src/bin/`, and the benchmark reports under `docs/`.

## Related Documentation

- [BENCHMARK_REPRODUCTION.md](BENCHMARK_REPRODUCTION.md) - current benchmark workflow
- [PYTHON_BINDINGS.md](PYTHON_BINDINGS.md) - Python installation and API usage
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - where everything lives in the repo
