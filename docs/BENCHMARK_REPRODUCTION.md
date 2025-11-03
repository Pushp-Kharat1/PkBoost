# PKBoost Benchmark Reproduction Guide

Complete guide for reproducing all benchmarks from the PKBoost paper, including standard performance comparisons and drift resilience tests.

## 📋 Table of Contents

- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Full Reproduction (1-2 hours)](#full-reproduction-1-2-hours)
- [Dataset Preparation](#dataset-preparation)
- [Running Benchmarks](#running-benchmarks)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Software

- **Rust:** 1.70+
```bash
rustc --version  # Should show 1.70+
```
Install: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

- **Python:** 3.8+
```bash
python --version  # Should show 3.8+
```

- **Python packages:**
```bash
pip install pandas numpy scikit-learn lightgbm xgboost kaggle matplotlib joblib
```

### Hardware

- **RAM:** 8GB minimum (16GB recommended for large datasets)
- **Storage:** 2GB for datasets and artifacts
- **CPU:** Multi-core processor (4+ cores for full parallelization)

---

## For Python Quick start please refer to [PKBoostPython.md](PKBoostPython.md)

## Quick Start (5 minutes)

Use included sample data to verify PKBoost works.

### Step 1: Clone Repository
```bash
git clone https://github.com/Pushp-Kharat1/pkboost.git
cd pkboost
```

### Step 2: Verify Sample Data
```bash
ls data/
# Should show: creditcard_train.csv, creditcard_val.csv, creditcard_test.csv
```

These files are included—a subset of the Credit Card dataset.

### Step 3: Run Rust Benchmark
```bash
cargo run --release --bin benchmark
```

**What happens:**
- Compiles PKBoost (~2-3 minutes first time)
- Trains on sample data
- Prints comparison with XGBoost/LightGBM

**Expected output:**
```
=== PKBoost Benchmarking ===
Train: 102,530 samples (0.17% fraud)
Val:   34,177 samples
Test:  34,177 samples

Training PKBoost... 45.2s
  PR-AUC: 0.8782

Training XGBoost... 12.1s
  PR-AUC: 0.7458

Training LightGBM... 9.8s
  PR-AUC: 0.7931

PKBoost improves PR-AUC by +17.9% over XGBoost
```

✅ **Done!** PKBoost is working.

---

## Full Reproduction (1-2 hours)

For complete results matching the paper, download full datasets and run extended benchmarks.

### Dataset Preparation

PKBoost includes a general-purpose data preparation pipeline that works with any Kaggle dataset.

#### Option 1: Credit Card Fraud (Primary Dataset)
```bash
# Download from Kaggle
python prepare_data.py mlg-ulb/creditcardfraud Class 1
```

**What this does:**
1. Downloads Credit Card dataset from Kaggle (143 MB)
2. Preprocesses features (standardization, missing value imputation)
3. Splits 60/20/20 (train/val/test) with stratification
4. Saves to `data/train_large.csv`, `data/val_large.csv`, `data/test_large.csv`

**Output:**
```
--- General-Purpose Data Preparation Pipeline ---
Step 1: Downloading 'mlg-ulb/creditcardfraud' from Kaggle...
Dataset downloaded and fully unzipped.
Step 2: Loading data into pandas...
Step 3: Cleaning data...
Step 4: Defining features and target ('Class')...
Step 5: Building preprocessing pipelines...
Step 6: Performing stratified split (60% train, 20% val, 20% test)...
Step 7: Fitting preprocessor and transforming data...
Step 8: Saving processed data to 'data/' directory...

Final training data shape: (170,884, 31)
```

#### Option 2: Create Smaller Test Datasets

For faster testing, create smaller versions:
```bash
# Creates 20K train, 5K val, 50K test with 0.5% fraud rate
python create_extreme_imbalance.py
```

Or use the bigger test script:
```bash
# Creates bigger test set for more reliable metrics
python create_bigger_test.py
```

#### Option 3: Multiple Datasets at Once

Run benchmarks on 6+ datasets automatically:
```bash
python run_all_benchmarks.py
```

**Datasets included:**
- Credit Card Fraud Detection
- Pima Indians Diabetes
- Breast Cancer Wisconsin
- Telco Customer Churn
- IEEE-CIS Fraud Detection
- NSL-KDD (network intrusion)

**This script:**
1. Downloads each dataset from Kaggle
2. Preprocesses automatically
3. Runs PKBoost, XGBoost, LightGBM
4. Generates comparison table
5. Saves results to `all_benchmarks_results.csv`

**Runtime:** ~1-2 hours depending on internet speed and CPU.

---

## Running Benchmarks

### Standard Performance Comparison (Rust)

Once data is prepared:
```bash
# Uses data/ folder with train/val/test CSV files
cargo run --release --bin benchmark
```

**Runtime:** ~10-15 minutes (Credit Card dataset)

### Standard Performance Comparison (Python)
```bash
# After running prepare_data.py
python run_single_benchmark.py
```

This runs LightGBM, XGBoost, and PKBoost (via Python bindings) on the prepared data.

### Drift Resilience Test

Test how models handle concept drift:
```bash
python drift_comparison_all.py
```

**What this does:**
- Trains models on clean data
- Introduces covariate shift (feature distribution changes)
- Tests on drifted data
- Compares degradation across models
- Generates visualization: `drift_comparison_complete.png`

**Expected output:**
```
=== PHASE 1: NORMAL DATA ===
LightGBM - PR-AUC: 0.7931
XGBoost  - PR-AUC: 0.7458

=== PHASE 2: APPLYING DRIFT ===
LightGBM - PR-AUC: 0.4556 (42.5% degradation)
XGBoost  - PR-AUC: 0.5082 (31.8% degradation)
```

### Adaptive Living Booster Test (Rust)

Test metamorphosis mechanism:
```bash
cargo run --release --bin test_drift
```

Simulates streaming batches with gradual drift and monitors automatic adaptation.

### Adaptive Living Booster (Python)
```bash
python example_creditcard_drift.py
```

Demonstrates real-time drift detection and metamorphosis using Python bindings.

---

## Expected Results

### Credit Card Fraud (0.2% fraud rate)

| Model | PR-AUC | F1-Score | ROC-AUC | Training Time |
|-------|--------|----------|---------|---------------|
| **PKBoost** | **87.8%** | **87.4%** | **97.5%** | 45s |
| XGBoost | 74.5% | 79.8% | 91.7% | 12s |
| LightGBM | 79.3% | 71.3% | 92.1% | 10s |

**Improvements:**
- vs XGBoost: +17.9% PR-AUC
- vs LightGBM: +10.4% PR-AUC

### Drift Resilience (after adding noise to 10 features)

| Model | Baseline | After Drift | Degradation |
|-------|----------|-------------|-------------|
| **PKBoost** | 87.8% | 86.2% | **1.8%** ✅ |
| XGBoost | 74.5% | 50.8% | 31.8% |
| LightGBM | 79.3% | 45.6% | 42.5% |

**Key insight:** PKBoost maintains 98.2% of baseline performance while competitors drop 30-50%.

### Pima Diabetes (35% positive class)

| Model | PR-AUC | F1-Score | ROC-AUC |
|-------|--------|----------|---------|
| **PKBoost** | **98.0%** | **93.7%** | **98.6%** |
| XGBoost | 68.0% | 60.0% | 82.0% |
| LightGBM | 62.9% | 48.8% | 82.4% |

**Note:** Small dataset (n=768) has high variance. Results may vary ±5% across runs.

### Breast Cancer (37% malignant)

| Model | PR-AUC | F1-Score | ROC-AUC |
|-------|--------|----------|---------|
| PKBoost | 97.9% | 93.2% | 98.6% |
| **XGBoost** | **99.2%** | **95.1%** | **99.4%** |
| **LightGBM** | **99.1%** | **96.3%** | **99.2%** |

**Expected:** PKBoost underperforms slightly on balanced data. Use XGBoost for balanced datasets.

### Ionosphere (36% "Bad")

| Model | PR-AUC | F1-Score | ROC-AUC |
|-------|--------|----------|---------|
| **PKBoost** | **98.0%** | **93.7%** | **98.5%** |
| XGBoost | 97.2% | 88.9% | 97.5% |
| LightGBM | 95.4% | 88.9% | 96.0% |

---

## Hardware Specifications

**Reference system** (benchmarks in README):

- **CPU:** Intel i7-10700K (8 cores, 3.8 GHz)
- **RAM:** 32GB DDR4
- **Storage:** NVMe SSD
- **OS:** Ubuntu 22.04
- **Rust:** 1.75.0
- **Python:** 3.11

**Performance variance:**
- **4 cores, 8GB RAM:** 1.5-2× longer training
- **16+ cores, 64GB RAM:** 0.5-0.7× training time
- **Accuracy metrics:** Within ±1-2% regardless of hardware

---

## Troubleshooting

### Issue 1: "File not found: data/creditcard_train.csv"

**Cause:** Dataset not prepared.

**Solution:**
```bash
# Check if raw data exists
ls raw_data/

# If yes, verify it's the right dataset
# If no, download it:
python prepare_data.py mlg-ulb/creditcardfraud Class 1

# Verify output
ls data/
# Should show: train_large.csv, val_large.csv, test_large.csv
```

### Issue 2: Kaggle API Authentication Error

**Error:**
```
Unauthorized: You must accept this dataset's terms of use
```

**Solution:**

1. **Set up Kaggle API credentials:**
```bash
# Create ~/.kaggle/kaggle.json with your API key
# Get it from: https://www.kaggle.com/settings

mkdir -p ~/.kaggle
# Copy your kaggle.json there
chmod 600 ~/.kaggle/kaggle.json
```

2. **Accept dataset terms on Kaggle website:**
   - Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Click "Download" to accept terms

3. **Retry download:**
```bash
python prepare_data.py mlg-ulb/creditcardfraud Class 1
```

### Issue 3: Out of Memory

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

**Option A:** Use smaller dataset
```bash
python create_extreme_imbalance.py  # Creates 20K train instead of 170K
```

**Option B:** Reduce batch size in Python scripts
```python
# In run_single_benchmark.py or similar
# Change batch_size from 10000 to 5000
```

**Option C:** Limit Rust compilation parallelism
```bash
cargo build --release --jobs 1
```

### Issue 4: "ModuleNotFoundError: No module named 'kaggle'"

**Solution:**
```bash
pip install kaggle
```

### Issue 5: LightGBM Feature Name Errors

**Error:**
```
LightGBM: Feature names must be alphanumeric
```

**Already fixed** in `prepare_data.py`:
```python
# Line ~150: Sanitizes feature names
sanitized_feature_names = [re.sub(r'[^A-Za-z0-9_]+', '', name) for name in raw_feature_names]
```

If you still see this, ensure you're using the latest `prepare_data.py`.

### Issue 6: Different Results Than Expected

**Acceptable variance:** ±2-3% on metrics due to:
- Hardware differences
- Random seed variations
- Floating-point precision

**To verify:**
1. Check random seed is 42: `random_state=42` in all splits
2. Verify dataset split ratios: 60/20/20
3. Ensure stratification is enabled
4. Compare trends, not absolute values

### Issue 7: prepare_data.py Takes Forever

**For large datasets** (>1GB):
- Credit Card: ~2-5 minutes
- IEEE-CIS Fraud: ~10-20 minutes
- Home Credit: ~30-60 minutes (feature engineering)

**Progress indicators:**
```
--- Starting Feature Engineering for Home Credit ---
  - Loading application_train.csv...
  - Aggregating bureau data...
  ...
```

If stuck on "Downloading", check internet connection.

---

## Advanced: Custom Datasets

Want to benchmark PKBoost on your own data?

### Step 1: Prepare Your CSV

**Required format:**
- CSV file with header row
- Binary target column (any name)
- Numerical features (categorical will be one-hot encoded automatically)

**Example:**
```csv
feature1,feature2,feature3,my_target
0.123,4.567,8.901,0
-1.234,5.678,9.012,1
```

### Step 2: Run Preparation Pipeline
```bash
python prepare_data.py your-kaggle-slug target_column_name positive_class_label
```

**Example:**
```bash
# For a custom fraud dataset
python prepare_data.py myuser/custom-fraud-data is_fraud 1

# For a medical dataset
python prepare_data.py hospital/disease-prediction diagnosis positive
```

### Step 3: Run Benchmark
```bash
# Rust (uses data/*.csv automatically)
cargo run --release --bin benchmark

# Python
python run_single_benchmark.py
```

### What prepare_data.py Does

1. **Downloads** from Kaggle (or loads local CSV)
2. **Cleans data:**
   - Handles missing values (median/mode imputation)
   - Drops columns with >50% missing
   - Standardizes numerical features
   - One-hot encodes categorical features (≤50 unique values)
3. **Splits:** 60% train, 20% val, 20% test (stratified)
4. **Saves** to `data/train_large.csv`, `val_large.csv`, `test_large.csv`
5. **Saves preprocessor** to `data/preprocessor.pkl` (for inference)

### Supported Dataset Types

✅ **Works with:**
- Binary classification
- Imbalanced classes (PKBoost's strength)
- Mixed numerical/categorical features
- Missing values
- Kaggle datasets or local CSV files

❌ **Not supported:**
- Multi-class classification
- Regression
- Time series (without feature engineering)
- Text/image data (without preprocessing)

---

## Reproducing Paper Results Exactly

For paper submissions or reviews:

### 1. Pin Exact Dependencies
```toml
# In Cargo.toml
[dependencies]
xgboost = "=0.3.0"  # Exact version
lightgbm = "=0.2.0"
```
```bash
# Python
pip install xgboost==1.7.5 lightgbm==4.0.0
```

### 2. Record System Info
```bash
echo "=== System Info ===" > reproduction_info.txt
uname -a >> reproduction_info.txt
rustc --version >> reproduction_info.txt
python --version >> reproduction_info.txt
pip list >> reproduction_info.txt
lscpu | grep "Model name" >> reproduction_info.txt
free -h >> reproduction_info.txt
```

### 3. Run Multiple Times for Statistical Significance
```bash
for i in {1..5}; do
  echo "Run $i"
  cargo run --release --bin benchmark | tee results_run_$i.txt
done
```

Compute mean and standard deviation across runs.

### 4. Use Exact Random Seeds

Verify all scripts use `random_state=42`:
- `prepare_data.py`: Line 105
- `create_extreme_imbalance.py`: Line 12, 18
- All sklearn `train_test_split` calls

---

## File Structure After Preparation
```
pkboost/
├── data/
│   ├── train_large.csv          # 60% of data
│   ├── val_large.csv            # 20% of data
│   ├── test_large.csv           # 20% of data
│   ├── preprocessor.pkl         # For inference
│   ├── creditcard_train.csv     # Sampled versions (optional)
│   ├── creditcard_val.csv
│   └── creditcard_test.csv
├── raw_data/                    # Downloaded datasets (auto-cleaned after)
├── scripts/                     # Python utilities
│   ├── prepare_data.py
│   ├── create_extreme_imbalance.py
│   ├── run_all_benchmarks.py
│   └── drift_comparison_all.py
├── src/                         # Rust source
│   └── bin/
│       ├── benchmark.rs
│       └── test_drift.rs
└── python/                      # Python bindings
    ├── example.py
    ├── example_creditcard.py
    └── example_creditcard_drift.py
```

---

## Questions or Issues?

1. **Check existing issues:** https://github.com/Pushp-Kharat1/pkboost/issues
2. **Open new issue** with:
   - Command you ran
   - Full error message
   - System info (`rustc --version`, `python --version`)
   - Dataset you're using

We respond within 24 hours.

---

## Performance Summary

| Task | Runtime | Notes |
|------|---------|-------|
| **Download dataset** | 2-5 min | Depends on internet speed |
| **Prepare data** | 2-10 min | Credit Card: ~3 min |
| **Rust benchmark** | 10-15 min | Credit Card, full dataset |
| **Python benchmark** | 15-20 min | Includes all 3 libraries |
| **Drift test** | 20-25 min | Includes adaptation cycles |
| **Full multi-dataset** | 1-2 hours | 6+ datasets, comprehensive |

---

**Last updated:** October 28, 2025  
**Maintainer:** Pushp Kharat (kharatpushp16@outlook.com)
