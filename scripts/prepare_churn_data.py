#!/usr/bin/env python3
"""
Download and prepare E-Commerce Customer Churn dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import zipfile

def download_dataset():
    """Download dataset using Kaggle API"""
    print("Downloading dataset from Kaggle...")
    os.system("kaggle datasets download -d samuelsemaya/e-commerce-customer-churn")
    
    # Extract using Python zipfile
    print("Extracting...")
    with zipfile.ZipFile("e-commerce-customer-churn.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
    print("Download complete!")

def prepare_churn_data():
    """Load and prepare the churn dataset"""
    print("\nLoading E-Commerce Customer Churn dataset...")
    
    # Read the CSV
    df = pd.read_csv("data/data_ecommerce_customer_churn.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Check target distribution
    if 'Churn' in df.columns:
        churn_rate = df['Churn'].mean()
        print(f"\nChurn rate: {churn_rate:.2%}")
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"\nEncoding categorical columns: {categorical_cols.tolist()}")
    for col in categorical_cols:
        if col != 'Churn':  # Don't encode target yet
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Separate features and target
    if 'Churn' in df.columns:
        y = df['Churn'].values
        X = df.drop('Churn', axis=1)
    else:
        # Find likely target column
        print("\nWarning: 'Churn' column not found. Available columns:")
        print(df.columns.tolist())
        return
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split: 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train churn: {y_train.mean():.2%}")
    print(f"Val churn: {y_val.mean():.2%}")
    print(f"Test churn: {y_test.mean():.2%}")
    
    # Save to CSV with 'Class' as target column (PKBoost convention)
    train_df = X_train.copy()
    train_df['Class'] = y_train
    train_df.to_csv("data/churn_train.csv", index=False)
    
    val_df = X_val.copy()
    val_df['Class'] = y_val
    val_df.to_csv("data/churn_val.csv", index=False)
    
    test_df = X_test.copy()
    test_df['Class'] = y_test
    test_df.to_csv("data/churn_test.csv", index=False)
    
    print("\nSaved files:")
    print("  - data/churn_train.csv")
    print("  - data/churn_val.csv")
    print("  - data/churn_test.csv")
    print("\nReady for PKBoost training!")

if __name__ == "__main__":
    # Check if Kaggle API is configured
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("ERROR: Kaggle API not configured!")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
    else:
        download_dataset()
        prepare_churn_data()
