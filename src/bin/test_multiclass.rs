// Test multi-class classification on Iris dataset
use pkboost::MultiClassPKBoost;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Class PKBoost Test (Iris Dataset) ===\n");

    // Iris dataset (150 samples, 4 features, 3 classes)
    let (x_train, y_train, x_test, y_test) = load_iris_data();

    println!("Train: {} samples, Test: {} samples", x_train.len(), x_test.len());
    println!("Classes: 0 (Setosa), 1 (Versicolor), 2 (Virginica)\n");

    // Train multi-class model
    let mut model = MultiClassPKBoost::new(3);
    model.fit(&x_train, &y_train, None, true)?;

    // Predict
    let predictions = model.predict(&x_test)?;
    let probs = model.predict_proba(&x_test)?;

    // Evaluate
    let accuracy = predictions.iter().zip(y_test.iter())
        .filter(|(&pred, &true_y)| pred == true_y as usize)
        .count() as f64 / y_test.len() as f64;

    println!("\n=== Results ===");
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Show first 5 predictions
    println!("\nSample predictions:");
    for i in 0..5.min(predictions.len()) {
        println!("  True: {}, Pred: {}, Probs: [{:.3}, {:.3}, {:.3}]",
                 y_test[i] as usize, predictions[i],
                 probs[i][0], probs[i][1], probs[i][2]);
    }

    Ok(())
}

fn load_iris_data() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    // Simplified Iris dataset (first 120 for train, last 30 for test)
    let data = vec![
        // Setosa (class 0)
        vec![5.1, 3.5, 1.4, 0.2, 0.0], vec![4.9, 3.0, 1.4, 0.2, 0.0], vec![4.7, 3.2, 1.3, 0.2, 0.0],
        vec![4.6, 3.1, 1.5, 0.2, 0.0], vec![5.0, 3.6, 1.4, 0.2, 0.0], vec![5.4, 3.9, 1.7, 0.4, 0.0],
        vec![4.6, 3.4, 1.4, 0.3, 0.0], vec![5.0, 3.4, 1.5, 0.2, 0.0], vec![4.4, 2.9, 1.4, 0.2, 0.0],
        vec![4.9, 3.1, 1.5, 0.1, 0.0], vec![5.4, 3.7, 1.5, 0.2, 0.0], vec![4.8, 3.4, 1.6, 0.2, 0.0],
        vec![4.8, 3.0, 1.4, 0.1, 0.0], vec![4.3, 3.0, 1.1, 0.1, 0.0], vec![5.8, 4.0, 1.2, 0.2, 0.0],
        vec![5.7, 4.4, 1.5, 0.4, 0.0], vec![5.4, 3.9, 1.3, 0.4, 0.0], vec![5.1, 3.5, 1.4, 0.3, 0.0],
        vec![5.7, 3.8, 1.7, 0.3, 0.0], vec![5.1, 3.8, 1.5, 0.3, 0.0], vec![5.4, 3.4, 1.7, 0.2, 0.0],
        vec![5.1, 3.7, 1.5, 0.4, 0.0], vec![4.6, 3.6, 1.0, 0.2, 0.0], vec![5.1, 3.3, 1.7, 0.5, 0.0],
        vec![4.8, 3.4, 1.9, 0.2, 0.0], vec![5.0, 3.0, 1.6, 0.2, 0.0], vec![5.0, 3.4, 1.6, 0.4, 0.0],
        vec![5.2, 3.5, 1.5, 0.2, 0.0], vec![5.2, 3.4, 1.4, 0.2, 0.0], vec![4.7, 3.2, 1.6, 0.2, 0.0],
        vec![4.8, 3.1, 1.6, 0.2, 0.0], vec![5.4, 3.4, 1.5, 0.4, 0.0], vec![5.2, 4.1, 1.5, 0.1, 0.0],
        vec![5.5, 4.2, 1.4, 0.2, 0.0], vec![4.9, 3.1, 1.5, 0.2, 0.0], vec![5.0, 3.2, 1.2, 0.2, 0.0],
        vec![5.5, 3.5, 1.3, 0.2, 0.0], vec![4.9, 3.6, 1.4, 0.1, 0.0], vec![4.4, 3.0, 1.3, 0.2, 0.0],
        vec![5.1, 3.4, 1.5, 0.2, 0.0],
        // Versicolor (class 1)
        vec![7.0, 3.2, 4.7, 1.4, 1.0], vec![6.4, 3.2, 4.5, 1.5, 1.0], vec![6.9, 3.1, 4.9, 1.5, 1.0],
        vec![5.5, 2.3, 4.0, 1.3, 1.0], vec![6.5, 2.8, 4.6, 1.5, 1.0], vec![5.7, 2.8, 4.5, 1.3, 1.0],
        vec![6.3, 3.3, 4.7, 1.6, 1.0], vec![4.9, 2.4, 3.3, 1.0, 1.0], vec![6.6, 2.9, 4.6, 1.3, 1.0],
        vec![5.2, 2.7, 3.9, 1.4, 1.0], vec![5.0, 2.0, 3.5, 1.0, 1.0], vec![5.9, 3.0, 4.2, 1.5, 1.0],
        vec![6.0, 2.2, 4.0, 1.0, 1.0], vec![6.1, 2.9, 4.7, 1.4, 1.0], vec![5.6, 2.9, 3.6, 1.3, 1.0],
        vec![6.7, 3.1, 4.4, 1.4, 1.0], vec![5.6, 3.0, 4.5, 1.5, 1.0], vec![5.8, 2.7, 4.1, 1.0, 1.0],
        vec![6.2, 2.2, 4.5, 1.5, 1.0], vec![5.6, 2.5, 3.9, 1.1, 1.0], vec![5.9, 3.2, 4.8, 1.8, 1.0],
        vec![6.1, 2.8, 4.0, 1.3, 1.0], vec![6.3, 2.5, 4.9, 1.5, 1.0], vec![6.1, 2.8, 4.7, 1.2, 1.0],
        vec![6.4, 2.9, 4.3, 1.3, 1.0], vec![6.6, 3.0, 4.4, 1.4, 1.0], vec![6.8, 2.8, 4.8, 1.4, 1.0],
        vec![6.7, 3.0, 5.0, 1.7, 1.0], vec![6.0, 2.9, 4.5, 1.5, 1.0], vec![5.7, 2.6, 3.5, 1.0, 1.0],
        vec![5.5, 2.4, 3.8, 1.1, 1.0], vec![5.5, 2.4, 3.7, 1.0, 1.0], vec![5.8, 2.7, 3.9, 1.2, 1.0],
        vec![6.0, 2.7, 5.1, 1.6, 1.0], vec![5.4, 3.0, 4.5, 1.5, 1.0], vec![6.0, 3.4, 4.5, 1.6, 1.0],
        vec![6.7, 3.1, 4.7, 1.5, 1.0], vec![6.3, 2.3, 4.4, 1.3, 1.0], vec![5.6, 3.0, 4.1, 1.3, 1.0],
        vec![5.5, 2.5, 4.0, 1.3, 1.0],
        // Virginica (class 2)
        vec![6.3, 3.3, 6.0, 2.5, 2.0], vec![5.8, 2.7, 5.1, 1.9, 2.0], vec![7.1, 3.0, 5.9, 2.1, 2.0],
        vec![6.3, 2.9, 5.6, 1.8, 2.0], vec![6.5, 3.0, 5.8, 2.2, 2.0], vec![7.6, 3.0, 6.6, 2.1, 2.0],
        vec![4.9, 2.5, 4.5, 1.7, 2.0], vec![7.3, 2.9, 6.3, 1.8, 2.0], vec![6.7, 2.5, 5.8, 1.8, 2.0],
        vec![7.2, 3.6, 6.1, 2.5, 2.0], vec![6.5, 3.2, 5.1, 2.0, 2.0], vec![6.4, 2.7, 5.3, 1.9, 2.0],
        vec![6.8, 3.0, 5.5, 2.1, 2.0], vec![5.7, 2.5, 5.0, 2.0, 2.0], vec![5.8, 2.8, 5.1, 2.4, 2.0],
        vec![6.4, 3.2, 5.3, 2.3, 2.0], vec![6.5, 3.0, 5.5, 1.8, 2.0], vec![7.7, 3.8, 6.7, 2.2, 2.0],
        vec![7.7, 2.6, 6.9, 2.3, 2.0], vec![6.0, 2.2, 5.0, 1.5, 2.0], vec![6.9, 3.2, 5.7, 2.3, 2.0],
        vec![5.6, 2.8, 4.9, 2.0, 2.0], vec![7.7, 2.8, 6.7, 2.0, 2.0], vec![6.3, 2.7, 4.9, 1.8, 2.0],
        vec![6.7, 3.3, 5.7, 2.1, 2.0], vec![7.2, 3.2, 6.0, 1.8, 2.0], vec![6.2, 2.8, 4.8, 1.8, 2.0],
        vec![6.1, 3.0, 4.9, 1.8, 2.0], vec![6.4, 2.8, 5.6, 2.1, 2.0], vec![7.2, 3.0, 5.8, 1.6, 2.0],
        vec![7.4, 2.8, 6.1, 1.9, 2.0], vec![7.9, 3.8, 6.4, 2.0, 2.0], vec![6.4, 2.8, 5.6, 2.2, 2.0],
        vec![6.3, 2.8, 5.1, 1.5, 2.0], vec![6.1, 2.6, 5.6, 1.4, 2.0], vec![7.7, 3.0, 6.1, 2.3, 2.0],
        vec![6.3, 3.4, 5.6, 2.4, 2.0], vec![6.4, 3.1, 5.5, 1.8, 2.0], vec![6.0, 3.0, 4.8, 1.8, 2.0],
        vec![6.9, 3.1, 5.4, 2.1, 2.0],
    ];

    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    let mut x_test = Vec::new();
    let mut y_test = Vec::new();

    for (i, row) in data.iter().enumerate() {
        let features = row[..4].to_vec();
        let label = row[4];
        
        if i % 5 == 0 {  // 20% test split
            x_test.push(features);
            y_test.push(label);
        } else {
            x_train.push(features);
            y_train.push(label);
        }
    }

    (x_train, y_train, x_test, y_test)
}
