import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('adaptive_regression_metrics.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# RMSE over time
ax = axes[0, 0]
normal = df[df['phase'] == 'normal']
drifted = df[df['phase'] == 'drifted']
ax.plot(normal['observation'], normal['rmse'], 'b-', label='Normal', linewidth=2)
ax.plot(drifted['observation'], drifted['rmse'], 'r-', label='Drifted', linewidth=2)
ax.axvline(x=normal['observation'].max(), color='gray', linestyle='--', alpha=0.5, label='Drift Applied')
ax.set_xlabel('Observations')
ax.set_ylabel('RMSE')
ax.set_title('RMSE Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# R² over time
ax = axes[0, 1]
ax.plot(normal['observation'], normal['r2'], 'b-', label='Normal', linewidth=2)
ax.plot(drifted['observation'], drifted['r2'], 'r-', label='Drifted', linewidth=2)
ax.axvline(x=normal['observation'].max(), color='gray', linestyle='--', alpha=0.5, label='Drift Applied')
ax.set_xlabel('Observations')
ax.set_ylabel('R²')
ax.set_title('R² Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Vulnerability score
ax = axes[1, 0]
ax.plot(df['observation'], df['vuln_score'], 'purple', linewidth=2)
ax.axvline(x=normal['observation'].max(), color='gray', linestyle='--', alpha=0.5, label='Drift Applied')
ax.set_xlabel('Observations')
ax.set_ylabel('Vulnerability Score (EMA)')
ax.set_title('Vulnerability Detection')
ax.legend()
ax.grid(True, alpha=0.3)

# State transitions
ax = axes[1, 1]
state_map = {'Normal': 0, 'Alert { checks_in_alert: 1 }': 1, 'Alert { checks_in_alert: 2 }': 2, 'Metamorphosis': 3}
df['state_num'] = df['state'].map(state_map)
ax.plot(df['observation'], df['state_num'], 'g-', linewidth=2, marker='o', markersize=4)
ax.axvline(x=normal['observation'].max(), color='gray', linestyle='--', alpha=0.5, label='Drift Applied')
ax.set_xlabel('Observations')
ax.set_ylabel('System State')
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['Normal', 'Alert-1', 'Alert-2', 'Metamorphosis'])
ax.set_title('State Machine Transitions')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adaptive_regression_analysis.png', dpi=150, bbox_inches='tight')
print("Plot saved: adaptive_regression_analysis.png")

# Summary statistics
print("\n=== PERFORMANCE SUMMARY ===")
print(f"\nNormal Phase:")
print(f"  RMSE: {normal['rmse'].mean():.4f} ± {normal['rmse'].std():.4f}")
print(f"  R²:   {normal['r2'].mean():.4f} ± {normal['r2'].std():.4f}")

print(f"\nDrifted Phase:")
print(f"  RMSE: {drifted['rmse'].mean():.4f} ± {drifted['rmse'].std():.4f}")
print(f"  R²:   {drifted['r2'].mean():.4f} ± {drifted['r2'].std():.4f}")

degradation = ((drifted['rmse'].mean() - normal['rmse'].mean()) / normal['rmse'].mean()) * 100
print(f"\nDegradation: {degradation:.1f}%")
print(f"Metamorphoses: {df['metamorphosis_count'].max()}")
print(f"Max vulnerability: {df['vuln_score'].max():.4f}")
