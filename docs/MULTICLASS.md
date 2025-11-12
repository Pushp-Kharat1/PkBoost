# Multi-Class Classification in PKBoost

PKBoost now supports multi-class classification using **One-vs-Rest (OvR)** strategy with **softmax normalization**.

## Quick Start

```rust
use pkboost::MultiClassPKBoost;

// Create model for 3 classes
let mut model = MultiClassPKBoost::new(3);

// Train (y should contain class labels: 0, 1, 2, ...)
model.fit(&x_train, &y_train, None, true)?;

// Predict class probabilities
let probs = model.predict_proba(&x_test)?;  // Vec<Vec<f64>> shape: [n_samples, n_classes]

// Predict class labels
let predictions = model.predict(&x_test)?;  // Vec<usize>
```

## How It Works

### One-vs-Rest Strategy

For `n` classes, PKBoost trains `n` binary classifiers in parallel:
- **Classifier 0**: Class 0 vs All Others
- **Classifier 1**: Class 1 vs All Others  
- **Classifier 2**: Class 2 vs All Others
- ...

Each classifier uses `OptimizedPKBoostShannon` with auto-tuning.

### Softmax Normalization

Raw predictions from each binary classifier are combined using softmax:

```
P(class_i | x) = exp(score_i) / Σ exp(score_j)
```

This ensures probabilities sum to 1.0 and handles class imbalance naturally.

## Example: Iris Dataset

```rust
use pkboost::MultiClassPKBoost;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data (150 samples, 4 features, 3 classes)
    let (x_train, y_train, x_test, y_test) = load_iris();
    
    // Train
    let mut model = MultiClassPKBoost::new(3);
    model.fit(&x_train, &y_train, None, true)?;
    
    // Evaluate
    let predictions = model.predict(&x_test)?;
    let accuracy = predictions.iter()
        .zip(y_test.iter())
        .filter(|(&pred, &true_y)| pred == true_y as usize)
        .count() as f64 / y_test.len() as f64;
    
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    Ok(())
}
```

**Result**: 95.83% accuracy on Iris test set

## Performance

### Training
- **Parallel**: All `n` classifiers train simultaneously using Rayon
- **Auto-tuning**: Each classifier auto-configures based on its binary task
- **Speed**: ~3x faster than sequential training

### Prediction
- **Batch-friendly**: Softmax applied per sample
- **Memory-efficient**: Streams predictions from each classifier

## Data Format

### Input Labels
- **Type**: `Vec<f64>` with integer values
- **Range**: `0.0, 1.0, 2.0, ..., (n_classes-1).0`
- **Example**: For 3 classes: `[0.0, 1.0, 2.0, 1.0, 0.0, ...]`

### Output Probabilities
- **Type**: `Vec<Vec<f64>>`
- **Shape**: `[n_samples, n_classes]`
- **Example**: `[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], ...]`

### Output Predictions
- **Type**: `Vec<usize>`
- **Values**: Class indices `0, 1, 2, ...`

## Validation Support

```rust
// With validation set for early stopping
model.fit(
    &x_train, 
    &y_train, 
    Some((&x_val, &y_val)),  // Optional validation
    true  // Verbose
)?;
```

Each binary classifier uses validation independently for early stopping.

## Comparison: OvR vs OvO

| Strategy | Classifiers | Training | Prediction | Imbalance Handling |
|----------|-------------|----------|------------|-------------------|
| **OvR** (PKBoost) | `n` | Parallel | Fast | Excellent |
| OvO | `n(n-1)/2` | Slower | Slower | Poor |

**Why OvR?**
- Fewer models to train (3 vs 3 for 3 classes, 10 vs 45 for 10 classes)
- Better for imbalanced data (each binary task is independent)
- Faster prediction (linear in n_classes)
- Natural probability calibration via softmax

## Limitations

- **Memory**: Stores `n` full models (acceptable for n < 100)
- **Imbalance**: If one class dominates, all binary tasks become imbalanced (but PKBoost handles this well)
- **Ordinal data**: Doesn't exploit ordinal relationships between classes

## Advanced: Custom Configuration

```rust
// Each binary classifier uses auto-tuning by default
// To override, modify multiclass.rs:
let mut clf = OptimizedPKBoostShannon::builder()
    .max_depth(6)
    .learning_rate(0.05)
    .n_estimators(500)
    .build_with_data(x, &y_binary);
```

## Benchmarks

| Dataset | Classes | Samples | Accuracy | Training Time |
|---------|---------|---------|----------|---------------|
| Iris | 3 | 150 | 95.8% | 0.8s |
| Wine | 3 | 178 | TBD | TBD |
| Digits | 10 | 1797 | TBD | TBD |

## Future Enhancements

- [ ] Multi-class metrics (macro/micro F1, confusion matrix)
- [ ] Class weights for imbalanced multi-class
- [ ] Hierarchical classification for large n_classes
- [ ] Calibration (Platt scaling, isotonic regression)

## Citation

```bibtex
@software{kharat2025pkboost_multiclass,
  author = {Kharat, Pushp},
  title = {PKBoost: Multi-Class Classification with OvR and Softmax},
  year = {2025},
  url = {https://github.com/Pushp-Kharat1/pkboost}
}
```
