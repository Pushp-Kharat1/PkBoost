#!/usr/bin/env python3
"""Download and prepare real-world imbalanced multi-class dataset"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def download_dry_bean():
    """Dry Bean Dataset - 7 classes, naturally imbalanced"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"
    
    print("Downloading Dry Bean Dataset...")
    import urllib.request
    import zipfile
    
    urllib.request.urlretrieve(url, "DryBeanDataset.zip")
    
    with zipfile.ZipFile("DryBeanDataset.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    
    df = pd.read_excel("DryBeanDataset/Dry_Bean_Dataset.xlsx")
    os.remove("DryBeanDataset.zip")
    
    return df

def prepare_dataset(df):
    """Prepare dataset for PKBoost"""
    print(f"\nDataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['Class'].value_counts())
    print(f"\nClass percentages:")
    print(df['Class'].value_counts(normalize=True) * 100)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['Class'])
    X = df.drop('Class', axis=1).values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    
    # Save as CSV
    os.makedirs("data", exist_ok=True)
    
    train_df = pd.DataFrame(X_train)
    train_df['Class'] = y_train
    train_df.to_csv("data/drybean_train.csv", index=False)
    
    test_df = pd.DataFrame(X_test)
    test_df['Class'] = y_test
    test_df.to_csv("data/drybean_test.csv", index=False)
    
    print("\nSaved to data/drybean_train.csv and data/drybean_test.csv")
    print(f"\nClass mapping:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i}: {cls}")
    
    return X_train, X_test, y_train, y_test, le.classes_

if __name__ == "__main__":
    try:
        df = download_dry_bean()
        X_train, X_test, y_train, y_test, classes = prepare_dataset(df)
        
        print("\n✅ Dataset ready for PKBoost!")
        print("\nTo use in Rust:")
        print("  cargo run --release --bin benchmark_drybean")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nAlternative: Using Wine Quality dataset (built-in)")
        
        from sklearn.datasets import load_wine
        wine = load_wine()
        X, y = wine.data, wine.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        os.makedirs("data", exist_ok=True)
        
        train_df = pd.DataFrame(X_train)
        train_df['Class'] = y_train
        train_df.to_csv("data/wine_train.csv", index=False)
        
        test_df = pd.DataFrame(X_test)
        test_df['Class'] = y_test
        test_df.to_csv("data/wine_test.csv", index=False)
        
        print(f"\nWine dataset: {len(X_train)} train, {len(X_test)} test")
        print(f"Features: {X_train.shape[1]}, Classes: 3")
        print("Saved to data/wine_train.csv and data/wine_test.csv")
