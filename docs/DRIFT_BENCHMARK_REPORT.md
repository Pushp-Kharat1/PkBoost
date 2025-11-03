# PKBoost Drift Benchmark Report  
*Conducted on Credit Card Fraud Dataset – October 30, 2025*

---

## Objective  
Evaluate **PKBoost**, **LightGBM**, and **XGBoost** across **16 realistic drift scenarios** using **PR-AUC** as the primary metric.  
All models trained on the same data split. No hyperparameter tuning beyond defaults and early stopping.

---

## Dataset Summary  
| Split  | Samples | Features | Fraud Rate |
|--------|---------|----------|------------|
| Train  | 170,884 | 30       | ~0.2%      |
| Val    | 56,961  | 30       | —          |
| Test   | 56,962  | 30       | **0.17%** (99 frauds) |

---

## Baseline Performance (No Drift)

| Model      | PR-AUC | ROC-AUC | F1     |
|------------|--------|---------|--------|
| LightGBM   | 0.7931 | 0.9205  | 0.8427 |
| XGBoost    | 0.7625 | 0.9287  | 0.8090 |
| PKBoost    | 0.8740 | 0.9734  | 0.8715 |

> *PKBoost starts with the highest PR-AUC (+0.0809 over LightGBM, +0.1115 over XGBoost).*

---

## Average PR-AUC Across All 16 Scenarios

| Model      | Avg PR-AUC | Avg Degradation |
|------------|------------|-----------------|
| PKBoost    | **0.8509** | **2.82%**       |
| LightGBM   | 0.7031     | 12.10%          |
| XGBoost    | 0.6720     | 12.66%          |

> *PKBoost maintains performance closest to its baseline.*

---

## Scenario-by-Scenario Results (PR-AUC)

| Scenario                          | LightGBM | XGBoost | PKBoost | **Winner** |
|-----------------------------------|----------|---------|---------|------------|
| No Drift (Baseline)               | 0.7931   | 0.7625  | **0.8740** | PKBoost (+0.0809) |
| Mild Covariate (0.2× std)         | 0.7836   | 0.7688  | **0.8705** | PKBoost (+0.0869) |
| Moderate Covariate (0.5× std)     | 0.7700   | 0.7852  | **0.8669** | PKBoost (+0.0817) |
| Severe Covariate (1.0× std)       | 0.7556   | 0.7645  | **0.8520** | PKBoost (+0.0875) |
| Extreme Covariate (2.0× std)      | 0.6998   | 0.7152  | **0.8337** | PKBoost (+0.1185) |
| Sign Flip (Adversarial)           | 0.4814   | 0.5146  | **0.8344** | PKBoost (+0.3198) |
| Gradual Drift                     | 0.7790   | 0.7715  | **0.8674** | PKBoost (+0.0884) |
| Sudden Drift (Half-way)           | 0.7888   | 0.7666  | **0.8639** | PKBoost (+0.0751) |
| Light Noise Injection             | 0.6497   | 0.6687  | **0.8287** | PKBoost (+0.1600) |
| Heavy Noise Injection             | 0.2270   | 0.0717  | **0.7462** | PKBoost (+0.5192) |
| Feature Scaling Drift             | 0.7566   | 0.6665  | **0.8628** | PKBoost (+0.1062) |
| Rotation Drift                    | 0.7864   | 0.7467  | **0.8716** | PKBoost (+0.0852) |
| Outlier Injection (10%)           | 0.7631   | 0.5123  | **0.8687** | PKBoost (+0.1056) |
| Combined Multi-Drift              | 0.7743   | 0.7497  | **0.8503** | PKBoost (+0.0760) |
| Temporal Decay                    | 0.6696   | 0.7085  | **0.8530** | PKBoost (+0.1445) |
| Cyclic/Seasonal Drift             | 0.7721   | 0.7797  | **0.8707** | PKBoost (+0.0910) |

> **PKBoost had the highest PR-AUC in all 16 scenarios.**  
> Margin of victory ranged from **+0.0751** (Sudden Drift) to **+0.5192** (Heavy Noise).

---

## Performance by Drift Type (Average PR-AUC)

| Category           | LightGBM | XGBoost | PKBoost | PKBoost Margin |
|--------------------|----------|---------|---------|----------------|
| Covariate Drift    | 0.7522   | 0.7584  | **0.8558** | **+0.0974** |
| Adversarial        | 0.6223   | 0.5134  | **0.8515** | **+0.2292** |
| Temporal           | 0.7524   | 0.7566  | **0.8638** | **+0.1072** |
| Noise-Based        | 0.4384   | 0.3702  | **0.7875** | **+0.3491** |
| Complex Drifts     | 0.7724   | 0.7210  | **0.8615** | **+0.0891** |

> *PKBoost shows consistent improvement across all categories.*

---

## Most Challenging Scenarios (Lowest Avg PR-AUC)

| Rank | Scenario                | Avg PR-AUC | PKBoost | Best Other | PKBoost Lead |
|------|-------------------------|------------|---------|------------|--------------|
| 1    | Heavy Noise Injection   | 0.3483     | **0.7462** | 0.2270     | **+0.5192**  |
| 2    | Sign Flip (Adversarial) | 0.6101     | **0.8344** | 0.5146     | **+0.3198**  |
| 3    | Light Noise Injection   | 0.7157     | **0.8287** | 0.6687     | **+0.1600**  |
| 4    | Temporal Decay          | 0.7437     | **0.8530** | 0.7085     | **+0.1445**  |
| 5    | Extreme Covariate       | 0.7401     | **0.8337** | 0.7152     | **+0.1185**  |

> *PKBoost's largest gains occur under severe noise and adversarial changes.*

---

## Worst-Case Resilience

| Model      | Worst PR-AUC | Scenario                  |
|------------|--------------|---------------------------|
| PKBoost    | **0.7462**   | Heavy Noise Injection     |
| LightGBM   | 0.2270       | Heavy Noise Injection     |
| XGBoost    | 0.0717       | Heavy Noise Injection     |

> Even in the **most disruptive scenario**, PKBoost retains **PR-AUC > 0.74**, while others drop below **0.23**.

---

## Key Observations

1. **PKBoost never lost a scenario** — highest PR-AUC in all 16 tests.  
2. **Average margin**:  
   - vs LightGBM: **+0.1478**  
   - vs XGBoost: **+0.1789**  
3. **Degradation**:  
   - PKBoost: **2.82%** drop from baseline  
   - LightGBM: **12.10%**  
   - XGBoost: **12.66%**  
4. **Noise & Adversarial resilience**:  
   - Heavy Noise: PKBoost **3.3× better** than LightGBM  
   - Sign Flip: PKBoost **1.6× better** than XGBoost  

> LightGBM and XGBoost are strong models — especially on clean, stable data.  
> But when distribution shifts occur, **PKBoost maintains significantly higher predictive quality**.

---

## Limitations & Fair Notes

- PKBoost uses **adaptive internal mechanisms** (buffer, metamorphosis triggers) not present in standard GBMs.  
- Training time is **longer** than LightGBM/XGBoost (not measured here).  
- All models used **default-like settings** — no exhaustive tuning.  
- Results are **one dataset only** — generalization to other domains untested.

---

## Conclusion

> **PKBoost achieved the highest PR-AUC in every tested drift scenario**, with an average lead of **0.16** and minimal degradation (**2.82%**).  
> LightGBM and XGBoost performed well under mild conditions but degraded sharply under noise, covariate shifts, and adversarial changes.

This is **not a claim of universal superiority** — only a factual report of performance **on this benchmark, under these conditions**.

---

**Files Generated**  
- `drift_detailed_results.csv` – Full per-scenario scores  
- `comprehensive_drift_analysis.png` – Visual summary  
- `baseline_vs_worstcase.png` – Resilience comparison  

*Script: [drift_comparison_all.py](../drift_comparison_all.py)*

---
