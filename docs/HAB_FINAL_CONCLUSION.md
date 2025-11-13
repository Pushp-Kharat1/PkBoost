# HAB/MoE Final Conclusion

## What We Tested

1. **Class-specialized partitioning** → -59% (catastrophic)
2. **Class-weighted ensemble** → -5.7%
3. **Temporal bootstrapping** → -12.5%
4. **Feature bagging** → -5.2%

## Why All Failed

**Root cause**: PKBoost already optimizes for the imbalanced distribution. Ensembling multiple models trained on the same data with different hyperparameters/features doesn't add new information—it just adds noise.

## When HAB/MoE DOES Work

### ✅ Multi-Domain Datasets
```rust
// Example: Combined fraud detection across different industries
let retail_specialist = train_on_retail_fraud();
let banking_specialist = train_on_banking_fraud();
let insurance_specialist = train_on_insurance_fraud();

// Route by domain
let prediction = match transaction.domain {
    Domain::Retail => retail_specialist.predict(x),
    Domain::Banking => banking_specialist.predict(x),
    Domain::Insurance => insurance_specialist.predict(x),
};
```

### ✅ Streaming with Localized Drift
```rust
// Example: Concept drift in specific time windows
let recent_specialist = train_on_last_30_days();
let historical_specialist = train_on_full_history();

// Weighted by recency
let prediction = 0.7 * recent_specialist.predict(x) 
               + 0.3 * historical_specialist.predict(x);
```

### ✅ Multi-Task Learning
```rust
// Example: Predict churn AND lifetime value simultaneously
let churn_specialist = train_for_churn();
let ltv_specialist = train_for_ltv();

// Different objectives, shared features
```

## Recommendation for PKBoost

**For homogeneous tabular fraud/churn detection:**
- Use **single baseline model** (10% better than HAB)
- Focus on **data quality** and **feature engineering**
- Use **adaptive retraining** when drift is detected

**For multi-domain or streaming scenarios:**
- Implement HAB with **domain-specific routing**
- Use **temporal weighting** for recent vs historical data
- Consider **online learning** for continuous adaptation

## Key Insight

> "Ensembles work when specialists see different data distributions. On homogeneous tabular data, a single well-tuned model beats an ensemble of mediocre specialists."

## Dataset Characteristics

**E-Commerce Churn Dataset:**
- 3,941 samples, 10 features
- 17.1% churn rate (moderate imbalance)
- Baseline PR-AUC: 84.93%
- Feature bagging: 80.51% (-5.2%)

**Credit Card Fraud Dataset:**
- 170K samples, 30 features  
- 0.2% fraud rate (extreme imbalance)
- Baseline PR-AUC: 82.97%
- Temporal bootstrap: 72.62% (-12.5%)

## Final Verdict

**HAB is NOT a silver bullet for tabular classification.** It's a specialized technique for multi-domain or streaming scenarios. For standard fraud/churn detection, stick with the baseline single model.
