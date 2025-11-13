# HAB (Hierarchical Adaptive Boosting) Analysis

## Problem
HAB underperforms baseline by ~10% (74% vs 82% PR-AUC)

## Root Cause
**Partitioning breaks global structure** - Credit card fraud patterns span the entire feature space, not localized regions.

## Attempted Fixes
1. ❌ Knowledge sharing between partitions - Added noise, made it worse
2. ❌ MoE-style gating network - Too complex, compilation errors
3. ❌ Validation-based weighting - No improvement

## Why MoE Works in LLMs But Not Here
- **LLMs**: Different experts for different token types (code, math, language)
- **Fraud Detection**: Patterns are global, not domain-specific

## Recommendation
**Use baseline single model** - It's simpler, faster, and 10% better.

HAB is useful for:
- Multi-domain datasets (e.g., fraud across different countries)
- Streaming with localized drift
- NOT for homogeneous tabular data
