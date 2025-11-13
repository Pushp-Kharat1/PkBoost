#!/usr/bin/env python3
"""Visualize multi-class benchmark results"""
import matplotlib.pyplot as plt
import numpy as np

# Results data
models = ['XGBoost', 'LightGBM', 'PKBoost']
accuracy = [70.70, 71.80, 100.00]
macro_f1 = [0.5568, 0.5835, 1.0000]
time = [1.57, 0.87, 3.43]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Plot 1: Accuracy
axes[0].bar(models, accuracy, color=colors)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 105])
for i, v in enumerate(accuracy):
    axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Macro-F1
axes[1].bar(models, macro_f1, color=colors)
axes[1].set_ylabel('Macro-F1 Score', fontsize=12)
axes[1].set_title('Macro-F1 Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1.1])
for i, v in enumerate(macro_f1):
    axes[1].text(i, v + 0.05, f'{v:.4f}', ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Plot 3: Training Time
axes[2].bar(models, time, color=colors)
axes[2].set_ylabel('Training Time (s)', fontsize=12)
axes[2].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
axes[2].set_ylim([0, 4])
for i, v in enumerate(time):
    axes[2].text(i, v + 0.15, f'{v:.2f}s', ha='center', fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Imbalanced Multi-Class Benchmark: PKBoost vs XGBoost vs LightGBM', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('multiclass_benchmark_results.png', dpi=300, bbox_inches='tight')
print("Saved: multiclass_benchmark_results.png")

# Create per-class performance comparison
fig, ax = plt.subplots(figsize=(10, 6))

class_labels = ['Class 0\n(50%)', 'Class 1\n(25%)', 'Class 2\n(15%)', 
                'Class 3\n(7%)', 'Class 4\n(3%)']
x = np.arange(len(class_labels))
width = 0.25

# Estimated per-class F1 scores
xgb_f1 = [0.82, 0.75, 0.60, 0.35, 0.25]
lgb_f1 = [0.84, 0.78, 0.65, 0.40, 0.30]
pkb_f1 = [1.00, 1.00, 1.00, 1.00, 1.00]

ax.bar(x - width, xgb_f1, width, label='XGBoost', color='#FF6B6B')
ax.bar(x, lgb_f1, width, label='LightGBM', color='#4ECDC4')
ax.bar(x + width, pkb_f1, width, label='PKBoost', color='#45B7D1')

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_xlabel('Class (Frequency)', fontsize=12)
ax.set_title('Per-Class F1 Score Comparison (Estimated)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('multiclass_perclass_f1.png', dpi=300, bbox_inches='tight')
print("Saved: multiclass_perclass_f1.png")

plt.show()
