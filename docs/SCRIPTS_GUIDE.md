# PKBoost Scripts

Utility scripts for data preparation, benchmarking, and analysis.

## Downloading the package 
```
pip install pkboost
```

## Data Preparation

### `prepare_data.py` - General-Purpose Pipeline

Downloads and preprocesses any Kaggle dataset for PKBoost.

**Usage:**
```bash
python scripts/prepare_data.py <kaggle-slug> <target-column> <positive-class>
```

**Examples:**
```bash
# Credit Card Fraud
python scripts/prepare_data.py mlg-ulb/creditcardfraud Class 1

# Pima Diabetes
python scripts/prepare_data.py uciml/pima-indians-diabetes-database Outcome 1

# Breast Cancer
python scripts/prepare_data.py uciml/breast-cancer-wisconsin-data diagnosis M
```

**What it does:**
1. Downloads from Kaggle
2. Cleans data (handles missing values, drops bad columns)
3. Standardizes numerical features
4. One-hot encodes categorical features
5. Splits 60/20/20 with stratification
6. Saves to `data/train_large.csv`, `val_large.csv`, `test_large.csv`

---

### `create_extreme_imbalance.py` - Synthetic Imbalance

Creates extremely imbalanced datasets for testing.

**Usage:**
```bash
python scripts/create_extreme_imbalance.py
```

**Output:**
- 20K train samples (0.5% positive)
- 5K val samples (0.5% positive)
- 30K test samples (0.5% positive)

Saved to `data/creditcard_train.csv`, etc.

---

### `create_bigger_test.py` - Larger Test Set

Creates a bigger test set for more reliable metrics.

**Usage:**
```bash
python scripts/create_bigger_test.py
```

**Output:**
- 20K train
- 5K val
- 50K test (bigger for statistical significance)

---

### `create_small_test.py` - Fast Testing

Creates small datasets for quick iteration.

**Usage:**
```bash
python scripts/create_small_test.py
```

**Output:**
- 5K train
- 2K val
- 3K test

Perfect for testing code changes without waiting 10+ minutes.

---

## Benchmarking

### `run_single_benchmark.py` - Single Dataset

Runs LightGBM, XGBoost, and PKBoost on prepared data.

**Usage:**
```bash
# After running prepare_data.py
python scripts/run_single_benchmark.py
```

**Output:**
- Training time for each model
- Test metrics (PR-AUC, ROC-AUC, F1, Accuracy)
- Comparison table

---

### `run_all_benchmarks.py` - Multi-Dataset Suite

Automatically downloads, prepares, and benchmarks 6+ datasets.

**Usage:**
```bash
python scripts/run_all_benchmarks.py
```

**Datasets:**
1. Credit Card Fraud
2. Pima Diabetes
3. Breast Cancer Wisconsin
4. Telco Customer Churn
5. IEEE-CIS Fraud
6. NSL-KDD Network Intrusion

**Output:**
- Console: Live progress and results
- `all_benchmarks_results.csv`: Detailed metrics
- Winner count across datasets

**Runtime:** ~1-2 hours

---

## Drift Analysis

### `drift_comparison_all.py` - Drift Resilience Test

Tests how models handle concept drift.

**Usage:**
```bash
python scripts/drift_comparison_all.py
```

**What it does:**
1. Trains LightGBM, XGBoost on clean data
2. Introduces covariate shift (adds noise to features)
3. Tests on drifted data
4. Compares degradation

**Output:**
- Console: Performance before/after drift
- `drift_comparison_complete.png`: Visualization

**Expected results:**
- LightGBM: 42.5% degradation
- XGBoost: 31.8% degradation
- PKBoost: 1.8% degradation (run separately via Rust)

---

## Python Examples

### `example.py` - Basic Usage

Simple demonstration of PKBoostClassifier.

**Usage:**
```bash
python scripts/example.py
```

Generates synthetic data and trains PKBoost.

---

### `example_creditcard.py` - Real Dataset

Full pipeline on Credit Card fraud data.

**Usage:**
```bash
# Requires data/creditcard_train.csv to exist
python scripts/example_creditcard.py
```

Shows:
- Loading CSV data
- Training with auto-tuning
- Evaluation metrics
- Feature importance

---

### `example_creditcard_drift.py` - Adaptive Model

Demonstrates PKBoostAdaptive with drift detection.

**Usage:**
```bash
python scripts/example_creditcard_drift.py
```

Shows:
- Initial training
- Streaming batches
- Vulnerability monitoring
- Automatic metamorphosis
- Performance tracking

---

### `example_drift.py` - Synthetic Drift

Adaptive model with synthetic data.

**Usage:**
```bash
python scripts/example_drift.py
```

Perfect for understanding drift detection without downloading data.

---

## Requirements
```bash
pip install pandas numpy scikit-learn lightgbm xgboost kaggle matplotlib joblib
```

**Kaggle API setup:**
```bash
mkdir -p ~/.kaggle
# Copy your kaggle.json from https://www.kaggle.com/settings
chmod 600 ~/.kaggle/kaggle.json
```

---

## Quick Reference

| Task | Script | Runtime |
|------|--------|---------|
| **Download Credit Card** | `prepare_data.py` | 3-5 min |
| **Create small test** | `create_small_test.py` | 10 sec |
| **Single benchmark** | `run_single_benchmark.py` | 15-20 min |
| **Multi-dataset** | `run_all_benchmarks.py` | 1-2 hours |
| **Drift test** | `drift_comparison_all.py` | 5-10 min |
| **Python example** | `example.py` | 30 sec |

---

## Troubleshooting

**"Kaggle API authentication failed"**
- Set up `~/.kaggle/kaggle.json` with your API token
- Visit dataset page and accept terms

**"File not found: data/creditcard_train.csv"**
- Run `prepare_data.py` first
- Or use `create_extreme_imbalance.py` for quick test data

**"Out of memory"**
- Use `create_small_test.py` for smaller datasets
- Reduce batch size in benchmark scripts

**"LightGBM feature name error"**
- `prepare_data.py` sanitizes names automatically
- Ensure you're using the latest version

---

For more details, see [BENCHMARK_REPRODUCTION.md](BENCHMARK_REPRODUCTION.md)
