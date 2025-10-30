import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from pkboost import PKBoostAdaptive
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    df = pd.read_csv(path)
    y = df.pop('Class')
    return df.values, y.values

def evaluate_model(model, X, y, model_type='lgb'):
    if len(np.unique(y)) < 2:
        return 0.0, 0.5, 0.0
    
    if model_type == 'lgb':
        preds = model.predict(X, num_iteration=model.best_iteration)
    elif model_type == 'xgb':
        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat)
    elif model_type == 'pkb':
        preds = model.predict_proba(X)

    pr_auc = average_precision_score(y, preds)
    roc_auc = roc_auc_score(y, preds)
    
    precision, recall, thresholds = precision_recall_curve(y, preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    y_pred = (preds >= optimal_threshold).astype(int)
    f1 = f1_score(y, y_pred, zero_division=0)
    return pr_auc, roc_auc, f1

# ============================================================================
# DRIFT SCENARIO FUNCTIONS
# ============================================================================

def apply_mild_covariate_drift(X):
    """Mild shift - 20% of std"""
    X_drift = X.copy()
    features = [0, 5, 10, 15, 20]
    for feat in features:
        if feat < X_drift.shape[1]:
            shift = 0.2 * np.std(X_drift[:, feat])
            X_drift[:, feat] += shift + np.random.normal(0, 0.05 * np.std(X_drift[:, feat]), len(X_drift))
    return X_drift

def apply_moderate_covariate_drift(X):
    """Moderate shift - 50% of std"""
    X_drift = X.copy()
    features = [0, 5, 10, 15, 20, 25]
    for feat in features:
        if feat < X_drift.shape[1]:
            shift = 0.5 * np.std(X_drift[:, feat])
            X_drift[:, feat] += shift + np.random.normal(0, 0.1 * np.std(X_drift[:, feat]), len(X_drift))
    return X_drift

def apply_severe_covariate_drift(X):
    """Severe shift - 100% of std"""
    X_drift = X.copy()
    features = [0, 5, 10, 15, 20, 25]
    for feat in features:
        if feat < X_drift.shape[1]:
            shift = 1.0 * np.std(X_drift[:, feat])
            X_drift[:, feat] += shift + np.random.normal(0, 0.2 * np.std(X_drift[:, feat]), len(X_drift))
    return X_drift

def apply_extreme_covariate_drift(X):
    """Extreme shift - 200% of std"""
    X_drift = X.copy()
    features = [0, 5, 10, 15, 20, 25, 29]
    for feat in features:
        if feat < X_drift.shape[1]:
            shift = 2.0 * np.std(X_drift[:, feat])
            X_drift[:, feat] += shift + np.random.normal(0, 0.3 * np.std(X_drift[:, feat]), len(X_drift))
    return X_drift

def apply_sign_flip_drift(X):
    """Adversarial - flip signs of key features"""
    X_drift = X.copy()
    features = [5, 10, 15, 20, 25]
    for feat in features:
        if feat < X_drift.shape[1]:
            X_drift[:, feat] = -X_drift[:, feat] + 10.0
    return X_drift

def apply_gradual_drift(X):
    """Gradual increasing drift across samples"""
    X_drift = X.copy()
    n_samples = X_drift.shape[0]
    features = [0, 5, 10, 15, 20]
    for feat in features:
        if feat < X_drift.shape[1]:
            shifts = np.linspace(0, 1.0 * np.std(X_drift[:, feat]), n_samples)
            X_drift[:, feat] += shifts
    return X_drift

def apply_sudden_drift(X):
    """Sudden drift - first half normal, second half drifted"""
    X_drift = X.copy()
    n_samples = X_drift.shape[0]
    split = n_samples // 2
    features = [0, 5, 10, 15, 20, 25]
    for feat in features:
        if feat < X_drift.shape[1]:
            shift = 1.5 * np.std(X_drift[:, feat])
            X_drift[split:, feat] += shift
    return X_drift

def apply_noise_injection(X):
    """Add Gaussian noise to all features"""
    X_drift = X.copy()
    for feat in range(X_drift.shape[1]):
        noise = np.random.normal(0, 0.5 * np.std(X_drift[:, feat]), len(X_drift))
        X_drift[:, feat] += noise
    return X_drift

def apply_heavy_noise(X):
    """Heavy Gaussian noise"""
    X_drift = X.copy()
    for feat in range(X_drift.shape[1]):
        noise = np.random.normal(0, 1.5 * np.std(X_drift[:, feat]), len(X_drift))
        X_drift[:, feat] += noise
    return X_drift

def apply_feature_scaling_drift(X):
    """Scale features by different factors"""
    X_drift = X.copy()
    features = [0, 5, 10, 15, 20, 25]
    scales = [0.5, 1.5, 2.0, 0.3, 2.5, 1.8]
    for feat, scale in zip(features, scales):
        if feat < X_drift.shape[1]:
            X_drift[:, feat] *= scale
    return X_drift

def apply_rotation_drift(X):
    """Rotate feature space (linear combinations)"""
    X_drift = X.copy()
    n_feat = min(10, X_drift.shape[1])
    angle = np.pi / 12
    for i in range(0, n_feat-1, 2):
        if i+1 < X_drift.shape[1]:
            feat1 = X_drift[:, i].copy()
            feat2 = X_drift[:, i+1].copy()
            X_drift[:, i] = feat1 * np.cos(angle) - feat2 * np.sin(angle)
            X_drift[:, i+1] = feat1 * np.sin(angle) + feat2 * np.cos(angle)
    return X_drift

def apply_outlier_injection(X):
    """Inject outliers in 10% of samples"""
    X_drift = X.copy()
    n_outliers = int(0.1 * len(X_drift))
    outlier_idx = np.random.choice(len(X_drift), n_outliers, replace=False)
    features = [0, 5, 10, 15, 20]
    for feat in features:
        if feat < X_drift.shape[1]:
            X_drift[outlier_idx, feat] = np.random.uniform(
                X_drift[:, feat].min() - 5 * np.std(X_drift[:, feat]),
                X_drift[:, feat].max() + 5 * np.std(X_drift[:, feat]),
                n_outliers
            )
    return X_drift

def apply_combined_drift(X):
    """Multiple drift types combined"""
    X_drift = X.copy()
    features = [0, 5, 10, 15, 20, 25]
    for feat in features:
        if feat < X_drift.shape[1]:
            shift = 0.7 * np.std(X_drift[:, feat])
            X_drift[:, feat] += shift
    for feat in range(X_drift.shape[1]):
        noise = np.random.normal(0, 0.3 * np.std(X_drift[:, feat]), len(X_drift))
        X_drift[:, feat] += noise
    for feat in [5, 10, 15]:
        if feat < X_drift.shape[1]:
            X_drift[:, feat] *= 1.5
    return X_drift

def apply_temporal_decay(X):
    """Simulate feature importance decay over time"""
    X_drift = X.copy()
    n_samples = X_drift.shape[0]
    features = [0, 5, 10, 15]
    for feat in features:
        if feat < X_drift.shape[1]:
            decay = np.exp(-np.linspace(0, 2, n_samples))
            X_drift[:, feat] *= decay
    return X_drift

def apply_cyclic_drift(X):
    """Cyclic/seasonal drift pattern"""
    X_drift = X.copy()
    n_samples = X_drift.shape[0]
    features = [0, 5, 10]
    for feat in features:
        if feat < X_drift.shape[1]:
            cycle = np.sin(np.linspace(0, 4*np.pi, n_samples))
            X_drift[:, feat] += cycle * 0.5 * np.std(X_drift[:, feat])
    return X_drift

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE DRIFT TESTING: 16 SCENARIOS")
print("="*80 + "\n")

# Load data
data_path = Path('data')
print("Loading data...")
X_train, y_train = load_data(data_path / 'creditcard_train.csv')
X_val, y_val = load_data(data_path / 'creditcard_val.csv')
X_test, y_test = load_data(data_path / 'creditcard_test.csv')
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

# Convert to contiguous float64
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
y_train = np.ascontiguousarray(y_train, dtype=np.float64)
X_val = np.ascontiguousarray(X_val, dtype=np.float64)
y_val = np.ascontiguousarray(y_val, dtype=np.float64)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)
y_test = np.ascontiguousarray(y_test, dtype=np.float64)

# Train models
print("="*80)
print("TRAINING MODELS")
print("="*80 + "\n")

print("Training LightGBM...")
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
lgb_model = lgb.train(
    {"objective": "binary", "metric": ["auc"], "verbosity": -1},
    train_set, num_boost_round=2000, valid_sets=[val_set],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)
print("LightGBM trained\n")

print("Training XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
xgb_model = xgb.train(
    {"objective": "binary:logistic", "eval_metric": ["auc"], "verbosity": 0},
    dtrain, num_boost_round=2000, evals=[(dval, 'val')],
    early_stopping_rounds=100, verbose_eval=False
)
print("XGBoost trained\n")

print("Training PKBoost...")
pkb_model = PKBoostAdaptive()
pkb_model.fit_initial(X_train, y_train, x_val=X_val, y_val=y_val)
print("PKBoost trained\n")

# Baseline evaluation
print("="*80)
print("BASELINE PERFORMANCE (NO DRIFT)")
print("="*80 + "\n")

lgb_pr, lgb_roc, lgb_f1 = evaluate_model(lgb_model, X_test, y_test, 'lgb')
xgb_pr, xgb_roc, xgb_f1 = evaluate_model(xgb_model, X_test, y_test, 'xgb')
pkb_pr, pkb_roc, pkb_f1 = evaluate_model(pkb_model, X_test, y_test, 'pkb')

print(f"Fraud rate: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.2f}%)\n")
print(f"LightGBM - PR-AUC: {lgb_pr:.4f}, ROC-AUC: {lgb_roc:.4f}, F1: {lgb_f1:.4f}")
print(f"XGBoost  - PR-AUC: {xgb_pr:.4f}, ROC-AUC: {xgb_roc:.4f}, F1: {xgb_f1:.4f}")
print(f"PKBoost  - PR-AUC: {pkb_pr:.4f}, ROC-AUC: {pkb_roc:.4f}, F1: {pkb_f1:.4f}")

baseline_scores = {
    'LightGBM': lgb_pr,
    'XGBoost': xgb_pr,
    'PKBoost': pkb_pr
}

# Define drift scenarios
drift_scenarios = [
    ('No Drift (Baseline)', lambda X: X),
    ('Mild Covariate (0.2x std)', apply_mild_covariate_drift),
    ('Moderate Covariate (0.5x std)', apply_moderate_covariate_drift),
    ('Severe Covariate (1.0x std)', apply_severe_covariate_drift),
    ('Extreme Covariate (2.0x std)', apply_extreme_covariate_drift),
    ('Sign Flip (Adversarial)', apply_sign_flip_drift),
    ('Gradual Drift', apply_gradual_drift),
    ('Sudden Drift (Half-way)', apply_sudden_drift),
    ('Light Noise Injection', apply_noise_injection),
    ('Heavy Noise Injection', apply_heavy_noise),
    ('Feature Scaling Drift', apply_feature_scaling_drift),
    ('Rotation Drift', apply_rotation_drift),
    ('Outlier Injection (10%)', apply_outlier_injection),
    ('Combined Multi-Drift', apply_combined_drift),
    ('Temporal Decay', apply_temporal_decay),
    ('Cyclic/Seasonal Drift', apply_cyclic_drift),
]

# Run all scenarios
print("\n" + "="*80)
print("TESTING DRIFT SCENARIOS")
print("="*80 + "\n")

results = []

for idx, (scenario_name, drift_fn) in enumerate(drift_scenarios, 1):
    print(f"[{idx}/{len(drift_scenarios)}] {scenario_name}")
    print("-" * 80)
    
    try:
        X_drift = drift_fn(X_test)
        
        lgb_pr_drift, _, lgb_f1_drift = evaluate_model(lgb_model, X_drift, y_test, 'lgb')
        xgb_pr_drift, _, xgb_f1_drift = evaluate_model(xgb_model, X_drift, y_test, 'xgb')
        pkb_pr_drift, _, pkb_f1_drift = evaluate_model(pkb_model, X_drift, y_test, 'pkb')
        
        results.append({
            'Scenario': scenario_name,
            'LightGBM_PR': lgb_pr_drift,
            'LightGBM_F1': lgb_f1_drift,
            'XGBoost_PR': xgb_pr_drift,
            'XGBoost_F1': xgb_f1_drift,
            'PKBoost_PR': pkb_pr_drift,
            'PKBoost_F1': pkb_f1_drift,
        })
        
        print(f"  LightGBM: PR-AUC {lgb_pr_drift:.4f}, F1 {lgb_f1_drift:.4f}")
        print(f"  XGBoost:  PR-AUC {xgb_pr_drift:.4f}, F1 {xgb_f1_drift:.4f}")
        print(f"  PKBoost:  PR-AUC {pkb_pr_drift:.4f}, F1 {pkb_f1_drift:.4f}")
        print()
        
    except Exception as e:
        print(f"  Error: {e}\n")
        continue

results_df = pd.DataFrame(results)

# ============================================================================
# ANALYSIS
# ============================================================================

print("="*80)
print("COMPREHENSIVE ANALYSIS")
print("="*80 + "\n")

print("AVERAGE PERFORMANCE ACROSS ALL SCENARIOS:")
print("-" * 80)
avg_lgb = results_df['LightGBM_PR'].mean()
avg_xgb = results_df['XGBoost_PR'].mean()
avg_pkb = results_df['PKBoost_PR'].mean()
print(f"LightGBM: {avg_lgb:.4f}")
print(f"XGBoost:  {avg_xgb:.4f}")
print(f"PKBoost:  {avg_pkb:.4f}")

print(f"\nAVERAGE DEGRADATION FROM BASELINE:")
print("-" * 80)
non_baseline = results_df[results_df['Scenario'] != 'No Drift (Baseline)']
lgb_deg = ((baseline_scores['LightGBM'] - non_baseline['LightGBM_PR'].mean()) / baseline_scores['LightGBM']) * 100
xgb_deg = ((baseline_scores['XGBoost'] - non_baseline['XGBoost_PR'].mean()) / baseline_scores['XGBoost']) * 100
pkb_deg = ((baseline_scores['PKBoost'] - non_baseline['PKBoost_PR'].mean()) / baseline_scores['PKBoost']) * 100

print(f"LightGBM: {lgb_deg:.2f}% degradation")
print(f"XGBoost:  {xgb_deg:.2f}% degradation")
print(f"PKBoost:  {pkb_deg:.2f}% degradation")

print(f"\nSCENARIOS WON (Best PR-AUC):")
print("-" * 80)
winners = {'LightGBM': 0, 'XGBoost': 0, 'PKBoost': 0}
for _, row in results_df.iterrows():
    best_model = max(['LightGBM', 'XGBoost', 'PKBoost'], 
                     key=lambda m: row[f'{m}_PR'])
    winners[best_model] += 1
    print(f"  {row['Scenario']:40s}: {best_model:10s} ({row[best_model+'_PR']:.4f})")

print(f"\nTOTAL WINS:")
print("-" * 80)
for model, count in sorted(winners.items(), key=lambda x: x[1], reverse=True):
    pct = (count / len(results_df)) * 100
    print(f"  {model:12s}: {count}/{len(results_df)} ({pct:.1f}%)")

# Save detailed results
results_df.to_csv('drift_detailed_results.csv', index=False)
print(f"\nDetailed results saved to 'drift_detailed_results.csv'")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80 + "\n")

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

colors = {'LightGBM': '#3498db', 'XGBoost': '#e74c3c', 'PKBoost': '#2ecc71'}

# 1. PR-AUC across all scenarios
ax1 = fig.add_subplot(gs[0, :])
scenarios = results_df['Scenario'].values
x = np.arange(len(scenarios))
width = 0.25

ax1.bar(x - width, results_df['LightGBM_PR'], width, label='LightGBM', 
        color=colors['LightGBM'], alpha=0.85)
ax1.bar(x, results_df['XGBoost_PR'], width, label='XGBoost', 
        color=colors['XGBoost'], alpha=0.85)
ax1.bar(x + width, results_df['PKBoost_PR'], width, label='PKBoost', 
        color=colors['PKBoost'], alpha=0.85)

ax1.axhline(y=0.75, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Reference (0.75)')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('PR-AUC Score', fontsize=12, fontweight='bold')
ax1.set_title('PR-AUC Performance Across All Drift Scenarios', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')

# 2. Average performance
ax2 = fig.add_subplot(gs[1, 0])
models = ['LightGBM', 'XGBoost', 'PKBoost']
avg_scores = [avg_lgb, avg_xgb, avg_pkb]
bars = ax2.bar(models, avg_scores, color=[colors[m] for m in models], 
               edgecolor='black', linewidth=2, alpha=0.85)
for bar in bars:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylabel('Avg PR-AUC', fontsize=11, fontweight='bold')
ax2.set_title('Average Performance\n(All Scenarios)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Degradation comparison
ax3 = fig.add_subplot(gs[1, 1])
degradations = [lgb_deg, xgb_deg, pkb_deg]
bars = ax3.bar(models, degradations, color=[colors[m] for m in models], 
               edgecolor='black', linewidth=2, alpha=0.85)
for bar in bars:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}%', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax3.set_ylabel('Avg Degradation (%)', fontsize=11, fontweight='bold')
ax3.set_title('Performance Degradation\n(vs Baseline)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Wins pie chart
ax4 = fig.add_subplot(gs[1, 2])
winner_values = list(winners.values())
winner_labels = list(winners.keys())
ax4.pie(winner_values, labels=winner_labels, autopct='%1.1f%%',
        colors=[colors[m] for m in winner_labels], startangle=90, 
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('Scenarios Won\n(Best PR-AUC)', fontsize=12, fontweight='bold')

# 5. Line plots for drift severity progression
drift_severity_scenarios = [
    'No Drift (Baseline)',
    'Mild Covariate (0.2x std)',
    'Moderate Covariate (0.5x std)',
    'Severe Covariate (1.0x std)',
    'Extreme Covariate (2.0x std)'
]

severity_data = results_df[results_df['Scenario'].isin(drift_severity_scenarios)]

ax5 = fig.add_subplot(gs[2, :])
x_sev = range(len(drift_severity_scenarios))
for model in models:
    scores = [severity_data[severity_data['Scenario']==s][f'{model}_PR'].values[0] 
              for s in drift_severity_scenarios]
    ax5.plot(x_sev, scores, marker='o', linewidth=3, markersize=10,
             label=model, color=colors[model])

ax5.set_xticks(x_sev)
ax5.set_xticklabels(drift_severity_scenarios, rotation=30, ha='right', fontsize=9)
ax5.set_ylabel('PR-AUC Score', fontsize=11, fontweight='bold')
ax5.set_title('Performance vs Drift Severity (Covariate Drift)', fontsize=13, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

# 6. Degradation heatmap
ax6 = fig.add_subplot(gs[3, :])
heatmap_data = []
for model in models:
    model_degs = []
    for scenario in scenarios:
        baseline = baseline_scores[model]
        current = results_df[results_df['Scenario']==scenario][f'{model}_PR'].values[0]
        deg = ((baseline - current) / baseline) * 100 if scenario != 'No Drift (Baseline)' else 0
        model_degs.append(deg)
    heatmap_data.append(model_degs)

im = ax6.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', vmin=-10, vmax=50)
ax6.set_xticks(range(len(scenarios)))
ax6.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
ax6.set_yticks(range(len(models)))
ax6.set_yticklabels(models, fontsize=11)
ax6.set_title('Degradation Heatmap (% drop from baseline)', fontsize=13, fontweight='bold')

for i in range(len(models)):
    for j in range(len(scenarios)):
        text = ax6.text(j, i, f'{heatmap_data[i][j]:.1f}',
                       ha="center", va="center", color="black", fontsize=7)

cbar = plt.colorbar(im, ax=ax6)
cbar.set_label('Degradation %', fontsize=10)

plt.savefig('comprehensive_drift_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Main visualization saved: 'comprehensive_drift_analysis.png'")

# Additional plot: Baseline vs Worst-case
fig2, ax = plt.subplots(figsize=(14, 8))
worst_scores = {
    'LightGBM': non_baseline['LightGBM_PR'].min(),
    'XGBoost': non_baseline['XGBoost_PR'].min(),
    'PKBoost': non_baseline['PKBoost_PR'].min()
}

x_pos = np.arange(len(models))
width = 0.35

baseline_vals = [baseline_scores[m] for m in models]
worst_vals = [worst_scores[m] for m in models]

ax.bar(x_pos - width/2, baseline_vals, width, label='Baseline', 
       color=[colors[m] for m in models], alpha=0.8, edgecolor='black', linewidth=2)
ax.bar(x_pos + width/2, worst_vals, width, label='Worst-Case Drift', 
       color=[colors[m] for m in models], alpha=0.4, edgecolor='black', linewidth=2)

for i, model in enumerate(models):
    ax.text(i - width/2, baseline_vals[i], f'{baseline_vals[i]:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(i + width/2, worst_vals[i], f'{worst_vals[i]:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('PR-AUC Score', fontsize=12, fontweight='bold')
ax.set_title('Baseline vs Worst-Case Drift Performance', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('baseline_vs_worstcase.png', dpi=300, bbox_inches='tight')
plt.close()

print("Comparison plot saved: 'baseline_vs_worstcase.png'")
