use mimalloc::MiMalloc;
use pkboost::*;
use std::error::Error;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn load_data(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let target_col_index = headers.iter().position(|h| h == "Class")
        .ok_or("'Class' column not found")?;

    for result in reader.records() {    
        let record = result?;
        let mut feature_row = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i == target_col_index {
                labels.push(value.parse::<f64>()?);
            } else {
                let parsed_value = if value.is_empty() {
                    f64::NAN
                } else {
                    value.parse::<f64>()?
                };
                feature_row.push(parsed_value);
            }
        }
        features.push(feature_row);
    }
    Ok((features, labels))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n=== ADVERSARIAL LIVING BOOSTER TEST ===\n");
    
    let (x_train, y_train) = load_data("data/train_large.csv")?;
    let (x_val, y_val) = load_data("data/val_large.csv")?;
    let (x_test, y_test) = load_data("data/test_large.csv")?;
    
    println!("Creating Adversarial Living Booster...");
    let mut alb = AdversarialLivingBooster::new(&x_train, &y_train);
    
    println!("\nInitial training...");
    alb.fit_initial(&x_train, &y_train, Some((&x_val, &y_val)), true)?;
    
    println!("\n=== SIMULATING STREAMING DATA ===");
    println!("Processing test set in batches...\n");
    
    let batch_size = 1000;
    for i in (0..x_test.len()).step_by(batch_size) {
        let end = (i + batch_size).min(x_test.len());
        let x_batch: Vec<Vec<f64>> = x_test[i..end].to_vec();
        let y_batch: Vec<f64> = y_test[i..end].to_vec();
        
        alb.observe_batch(&x_batch, &y_batch, true)?;
    }
    
    println!("\n=== FINAL STATUS ===");
    println!("System State: {:?}", alb.get_state());
    println!("Vulnerability Score: {:.4}", alb.get_vulnerability_score());
    println!("Metamorphosis Count: {}", alb.get_metamorphosis_count());
    
    Ok(())
}