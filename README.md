# PKBoost 

**Gradient boosting that adjusts to concept drift in imbalanced data.**

[![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white&style=for-the-badge)](https://www.rust-lang.org/)
[![PyPI](https://img.shields.io/pypi/v/pkboost?color=blue&logo=pypi&logoColor=white&style=for-the-badge)](https://pypi.org/project/pkboost/)
[![Downloads](https://img.shields.io/pypi/dm/pkboost?color=brightgreen&style=for-the-badge)](https://pypi.org/project/pkboost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Pushp-Kharat1/PKBoost?style=for-the-badge&logo=github)](https://github.com/Pushp-Kharat1/PKBoost/stargazers)



Built from scratch in Rust, PKBoost manages changing data distributions in fraud detection with a fraud rate of 0.2%. It shows less than 2% degradation under drift. In comparison, XGBoost experiences a 31.8% drop and LightGBM a 42.5% drop. PKBoost outperforms XGBoost by 10-18% on the Standard dataset when no drift is applied. It employs information theory with Shannon entropy and Newton Raphson to identify shifts in rare events and trigger an adaptive "metamorphosis" for real-time recovery.

> **"Most boosting libraries overlook concept drift. PKBoost identifies it and evolves to persist."**

**Perfect for:** Streaming fraud detection, real-time medical monitoring, anomaly detection in changing environments, or any scenario where data evolves over time and positive instances are rare.

---

## 🚀 Quick Start

To use it in Python Please refer to: [PKBoostPython.md](PKBoostPython.md)

Clone the repository and build:

```bash
git clone https://github.com/Pushp-Kharat1/pkboost.git
cd pkboost
cargo build --release
```

Run the benchmark:

1. **Use included sample data** (already in `data/`)
```bash
ls data/  # Should show creditcard_train.csv, creditcard_val.csv, etc.
```

2. **Run benchmark**
```bash
cargo run --release --bin benchmark
```

---

## 💻 Basic Usage

To train and predict (see `src/bin/benchmark.rs` for a full example):

```rust
use pkboost::*;
use csv;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load CSV with headers: feature1,feature2,...,Class
    let (x_train, y_train) = load_csv("train.csv")?;
    let (x_val, y_val) = load_csv("val.csv")?;
    let (x_test, y_test) = load_csv("test.csv")?;

    // Auto-configure based on data characteristics
    let mut model = OptimizedPKBoostShannon::auto(&x_train, &y_train);

    // Train with early stopping on validation set
    model.fit(
        &x_train,
        &y_train,
        Some((&x_val, &y_val)),  // Optional validation
        true  // Verbose output
    )?;

    // Predict probabilities (not classes)
    let test_probs = model.predict_proba(&x_test)?;

    // Evaluate
    let pr_auc = calculate_pr_auc(&y_test, &test_probs);
    println!("PR-AUC: {:.4}", pr_auc);

    Ok(())
}

// Helper function (put in your code)
fn load_csv(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let target_col_index = headers.iter().position(|h| h == "Class")
        .ok_or("Class column not found")?;

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for result in reader.records() {
        let record = result?;
        let mut row: Vec<f64> = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i == target_col_index {
                labels.push(value.parse()?);
            } else {
                let parsed_value = if value.is_empty() {
                    f64::NAN
                } else {
                    value.parse()?
                };
                row.push(parsed_value);
            }
        }
        features.push(row);
    }

    Ok((features, labels))
}
```

**Expected CSV format:**
- Header row required
- Target column named "Class" with binary values (0.0 or 1.0)
- All other columns treated as numerical features
- Empty values treated as NaN (median-imputed)
- No categorical support (encode them first)
- For data loading examples, see `src/bin/*.rs` files like `benchmark.rs`. Supports CSV via `csv` crate.

---

## ✨ Key Features

- **Extreme Imbalance Handling:** Automatic class weighting and MI regularization boost recall on rare positives without reducing precision. Binary classification only.
- **Adaptive Hyperparameters:** `auto_tune_principled` profiles your dataset for optimal params—no manual tuning needed.
- **Histogram-Based Trees:** Optimized binning with medians for missing values; supports up to 32 bins per feature for fast splits.
- **Parallelism & Efficiency:** Rayon-based adaptive parallelism detects hardware and scales thresholds dynamically. Efficient batching is used for large datasets.
- **Adaptation Mechanisms:** `AdversarialLivingBooster` monitors vulnerability scores to detect drift and trigger retraining, such as pruning unused features through "metabolism" tracking.
- **Metrics Built-In:** PR-AUC, ROC-AUC, F1@0.5, and threshold optimization are available out-of-the-box.

---

## 📊 Benchmarks

**Testing methodology:** All models use default settings with no hyperparameter tuning. This reflects real-world usage where most practitioners cannot dedicate time to extensive tuning.

PKBoost's auto-tuning provides an edge—it automatically detects imbalance and adjusts parameters. LGBM/XGB can match these results with tuning but require expert knowledge.

**Reproducibility:** All benchmark code is in `src/bin/benchmark.rs`. Data splits: 60% train, 20% val, 20% test. LGBM/XGB used default params from their Rust crates. Full benchmarks (10+ datasets): See `BENCHMARKS.md`.

### Standard Datasets

| Dataset         | Samples  | Imbalance             | Model     | PR-AUC  | F1-AUC  | ROC-AUC |
|------------------|----------|----------------------|-----------|---------|---------|---------|
| **Credit Card**      | 170,884  | 0.2% (extreme)       | **PKBoost**   | **87.8%**   | **87.4%**   | **97.5%**   |
|                  |          |                      | LightGBM  | 79.3%   | 71.3%   | 92.1%   |
|                  |          |                      | XGBoost   | 74.5%   | 79.8%   | 91.7%   |
| *Improvements*     |          |                      | vs LGBM   | **+10.4%**  | **+22.7%**  | **+5.7%**   |
|                  |          |                      | vs XGBoost| **+17.9%**  | **+9.7%**   | **+6.1%**   |
| **Pima Diabetes**     | 460      | 35.0% (balanced)     | **PKBoost**   | **98.0%**   | **93.7%**   | **98.6%**   |
|                  |          |                      | LightGBM  | 62.9%   | 48.8%   | 82.4%   |
|                  |          |                      | XGBoost   | 68.0%   | 60.0%   | 82.0%   |
| *Improvements*     |          |                      | vs LGBM   | **+55.7%**  | **+92.0%**  | **+19.6%**  |
|                  |          |                      | vs XGBoost| **+44.0%**  | **+56.1%**  | **+20.1%**  |
| **Breast Cancer**     | 341      | 37.2% (balanced)     | PKBoost   | 97.9%   | 93.2%   | 98.6%   |
|                  |          |                      | **LightGBM**  | **99.1%**   | **96.3%**   | **99.2%**   |
|                  |          |                      | **XGBoost**   | **99.2%**   | **95.1%**   | **99.4%**   |
| *Improvements*     |          |                      | vs LGBM   | -1.2%   | -3.3%   | -0.7%   |
|                  |          |                      | vs XGBoost| -1.4%   | -2.1%   | -0.8%   |
| **Heart Disease**     | 181      | 45.9% (balanced)     | **PKBoost**   | **87.8%**   | **82.5%**   | **88.5%**   |
| **Ionosphere**        | 210      | 35.7% (balanced)     | **PKBoost**   | **98.0%**   | **93.7%**   | **98.5%**   |
|                  |          |                      | LightGBM  | 95.4%   | 88.9%   | 96.0%   |
|                  |          |                      | XGBoost   | 97.2%   | 88.9%   | 97.5%   |
| *Improvements*     |          |                      | vs LGBM   | **+2.7%**   | **+5.4%**   | **+2.7%**   |
|                  |          |                      | vs XGBoost| **+0.8%**   | **+5.4%**   | **+1.1%**   |
| **Sonar**            | 124      | 46.8% (balanced)     | **PKBoost**   | **91.8%**   | **87.2%**   | **93.6%**   |
| **SpamBase**         | 2,760    | 39.4% (balanced)     | **PKBoost**   | **98.0%**   | **93.3%**   | **98.0%**   |
| **Adult**            | -        | 24.1% (balanced)     | **PKBoost**   | **81.2%**   | **71.9%**   | **92.0%**   |

**Notes:** PR-AUC is prioritized for imbalance; F1@0.5 uses the optimal threshold. Unfilled cells indicate benchmarks in progress. Note on Pima Diabetes: Small datasets (n=460) have high variance due to limited samples. Results may not generalize; re-run with your data for confirmation. Note on Breast Cancer: PKBoost slightly underperforms on nearly balanced datasets (37% minority). This is expected—our optimizations target extreme imbalance. For balanced data, use XGBoost.

### Why PKBoost Wins on Imbalanced Data

**Credit Card Fraud (0.2% minority class):**

- **PKBoost:** 87.8% PR-AUC → Optimal performance maintained.
- **XGBoost:** 74.5% PR-AUC → 15% degradation from balanced baseline.
- **LightGBM:** 79.3% PR-AUC → 10% degradation from balanced baseline.

**Pattern:** As imbalance severity increases (from balanced to 5% to 1% to 0.2%), traditional boosting drops linearly while PKBoost maintains high accuracy.

### Drift Resilience (Credit Card Dataset)

PKBoost features experimental drift detection that monitors model vulnerabilities and can trigger adaptive retraining.

**Benchmark:** After introducing a significant covariate shift (adding noise to 10 features), models were tested on corrupted data:

| Model           | Baseline PR-AUC | After Drift | Degradation |
|------------------|-----------------|-------------|-------------|
| **PKBoost**           | **87.8%**           | **86.2%**      | **1.8%**        |
| LightGBM         | 79.3%           | 45.6%      | 42.5%       |
| XGBoost          | 74.5%           | 50.8%      | 31.8%       |

**PKBoost's robustness comes from:**
- Conservative tree depth, which prevents overfitting to specific distributions
- Quantile-based binning that adapts to feature distributions
- Regularization that reduces sensitivity to noise

**Note:** Adaptive retraining is experimental and didn't trigger in this test. The robustness comes from the base architecture.

---

For more details, see [BENCHMARKS.md](BENCHMARK.md)


## 🎯 When to Use PKBoost

### ✅ Good fit:
- Binary classification (0/1 labels)
- Extreme imbalance (<5% minority class)
- Fraud detection, medical diagnosis, anomaly detection
- Seeking good results without hyperparameter tuning

### ❌ Not suitable for:
- Multi-class classification (not implemented)
- Regression tasks
- Perfectly balanced datasets (use XGBoost, it's faster)
- Datasets with fewer than 1,000 samples (too small for meaningful results)

---

## 🔬 How It Works

**Traditional gradient boosting struggles with extreme imbalance because:**
- Gradient-based splits favor the majority class. More samples lead to stronger gradients.
- Regularization does not consider class rarity.
- Early stopping uses global metrics that overlook minority class performance.

**PKBoost's approach:**
- **Shannon entropy guidance** optimizes splits for information gain on the minority class.
- **Adaptive class weighting** is automatically calculated from data statistics.
- **PR-AUC early stopping** focuses on minority class performance.

**Technical innovation:** Fusing information theory with Newton boosting. Each split maximizes:

```
Gain = GradientGain + λ * InformationGain
```

Where λ is adaptive based on imbalance severity.

### Architecture Flow:

```
[Your Data] → [Auto-Tuner] → [Shannon-Guided Trees] → [Predictions]
                  ↓              ↓                   ↓
            Detects      Entropy + Gradient      PR-AUC
            Imbalance    Split Criterion         Optimized
```

**Core Model:** `OptimizedPKBoostShannon` – Shannon-entropy regularized trees with MI weighting.  
**Data Prep:** `OptimizedHistogramBuilder` – Fast binning, median imputation, parallel transforms.  
**Tuning:** `auto_tune_principled` & `auto_params` – Dataset-aware hyperparameters.  
**Adaptation:** `AdversarialLivingBooster` – Monitors drift through vulnerability scores; triggers retraining, such as feature pruning via metabolism tracking.  
**Parallelism:** `adaptive_parallel` – Hardware-aware Rayon config (cores, RAM detection).  
**Evaluation:** Built-in calculations for PR-AUC, ROC-AUC, and F1.  
**Drift Sims:** Scripts like `test_drift.rs` and `test_static.rs` for baseline comparisons.

See `src/` for full implementation. Binary classification only.

---

## ⚡ Performance

**Training Time (Credit Card, 170K samples):**

- **PKBoost:** ~45s with auto-tuning → 87.8% PR-AUC
- **XGBoost:** ~12s with defaults → 74.5% PR-AUC
- **XGBoost:** ~12s × 50 trials = 10 minutes tuned → ~87% PR-AUC (estimated)

### The Trade-off:
- **PKBoost:** 45 seconds, zero human time
- **XGBoost:** 10+ minutes compute + 2 hours human tuning time

**Choose your bottleneck:** compute time or engineering time.

PKBoost prioritizes accuracy over speed. For production inference, all three have similar prediction latency of around 1ms per sample.

---

## 📋 Requirements

- Rust 1.70+ (2021 edition)
- 8GB+ RAM for large datasets (>100K samples)
- Multi-core CPU recommended (auto-detects and parallelizes)

---

## 🧪 Running Benchmarks & Tests

**Install Rust:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Clone & build:** As above.

**Run:**
```bash
cargo run --release --bin benchmark  # uses data/*.csv
```

**Drift tests:**
```bash
cargo run --bin test_drift
```

Datasets sourced from UCI/ML.

---

## 🛠️ Common Issues

**"error: linker cc not found"**
- Ubuntu/Debian: `sudo apt install build-essential`
- macOS: Install Xcode Command Line Tools

**Out of memory during compilation:**
```bash
cargo build --release --jobs 1  # Limit parallel compilation
```

**Slow training on large datasets:**
- Ensure you're using the `--release` flag
- Check CPU utilization (should be ~800% on 8 cores)

---

## 🤝 Contributing

Open for contributions! Fork & PR: Focus on extensions, optimizations, or new tests. Issues welcome for bugs or dataset requests.

**License:** MIT – Free for commercial use.  
**Contact:** kharatpushp16@outlook.com

---

## 📚 Citation

If you use PKBoost in your research, please cite:

```bibtex
@software{kharat2025pkboost,
  author = {Kharat, Pushp},
  title = {PKBoost: Shannon-Guided Gradient Boosting for Extreme Imbalance},
  year = {2025},
  url = {https://github.com/Pushp-Kharat1/pkboost}
}
```

---

## 📖 Further Reading

[Rust ML Ecosystem](https://www.arewelearningyet.com/)

**Questions?** Open an issue.

---

**Project by Pushp Kharat.** Last updated: October 24, 2025.
