# prepare_data.py (Final, Verbose Version 4.1)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import kaggle
import os
import zipfile
import argparse
import time
import joblib
import re

def engineer_home_credit_features(download_path):
    """
    Loads and engineers features for the Home Credit Default Risk dataset.
    This version includes print statements to show progress.
    """
    print("\n--- Starting Feature Engineering for Home Credit ---")
    
    # Load main application data
    print("  - Loading application_train.csv...")
    df = pd.read_csv(os.path.join(download_path, 'application_train.csv'))
    
    # Load bureau data and create aggregates
    print("  - Loading bureau.csv...")
    bureau = pd.read_csv(os.path.join(download_path, 'bureau.csv'))
    
    print("  - Aggregating bureau data...")
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['mean', 'max'],
        'CREDIT_DAY_OVERDUE': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
    })
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns.ravel()]
    
    print("  - Merging bureau data into main dataframe...")
    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    # Load previous application data and create aggregates
    print("  - Loading previous_application.csv...")
    prev_app = pd.read_csv(os.path.join(download_path, 'previous_application.csv'))
    
    print("  - Aggregating previous application data...")
    prev_app_agg = prev_app.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['mean', 'max'],
        'AMT_CREDIT': ['mean', 'max'],
        'DAYS_DECISION': ['mean', 'min'],
    })
    prev_app_agg.columns = ['PREV_APP_' + '_'.join(col).upper() for col in prev_app_agg.columns.ravel()]
    
    print("  - Merging previous application data into main dataframe...")
    df = df.merge(prev_app_agg, on='SK_ID_CURR', how='left')

    del bureau, bureau_agg, prev_app, prev_app_agg
    print("--- Feature Engineering Complete! ---\n")
    return df

# (The rest of the script is the same as Version 4.0)
def prepare_dataset(dataset_slug: str, target_column: str, positive_class_label: str):
    print("--- General-Purpose Data Preparation Pipeline ---")
    print(f"Step 1: Downloading '{dataset_slug}' from Kaggle...")
    download_path = 'raw_data'
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    clean_slug = dataset_slug
    if dataset_slug.startswith('c/'):
        clean_slug = dataset_slug[2:]
        print(f"  - Detected competition slug. Using clean API slug: '{clean_slug}'")
        kaggle.api.competition_download_files(clean_slug, path=download_path)
    else:
        print("  - Detected standard dataset.")
        kaggle.api.dataset_download_files(clean_slug, path=download_path, unzip=True)
    unzip_complete = False
    while not unzip_complete:
        zip_files = [f for f in os.listdir(download_path) if f.endswith('.zip')]
        if not zip_files:
            unzip_complete = True
            continue
        for item in zip_files:
            zip_path = os.path.join(download_path, item)
            print(f"  - Unzipping {item}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
                os.remove(zip_path)
            except zipfile.BadZipFile:
                print(f"    - Warning: Skipping corrupted or incomplete zip file: {item}")
                os.remove(zip_path)
        time.sleep(1)
    print("Dataset downloaded and fully unzipped.")
    if clean_slug == "home-credit-default-risk":
        df = engineer_home_credit_features(download_path)
    else:
        candidate_files = [os.path.join(root, f) for root, _, files in os.walk(download_path) for f in files]
        train_csv_path = max((f for f in candidate_files if 'train' in os.path.basename(f).lower()), key=os.path.getsize, default=None) or max(candidate_files, key=os.path.getsize)
        print(f"Step 2: Loading data from '{os.path.basename(train_csv_path)}' into pandas...")
        df = pd.read_csv(train_csv_path)
    print("Step 3: Cleaning data...")
    df.columns = [str(col).strip().replace('.', '_').replace('-', '_') for col in df.columns]
    target_column = str(target_column).strip().replace('.', '_').replace('-', '_')
    missing_value_placeholders = ['?', 'NA', 'N/A', '', ' ', -1, '-1']
    for placeholder in missing_value_placeholders:
        df.replace(placeholder, np.nan, inplace=True)
    missing_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < missing_threshold]
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = df[col].str.strip()
    print(f"Step 4: Defining features and target ('{target_column}')...")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {df.columns.tolist()}")
    X = df.drop(target_column, axis=1)
    y = df[target_column].apply(lambda x: 1 if str(x) == str(positive_class_label) else 0)
    del df
    print("Step 5: Building preprocessing pipelines...")
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    max_onehot = 50
    low_card_cats  = [c for c in categorical_features if X[c].nunique() <= max_onehot]
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cats),
    ], remainder='drop')
    print("Step 6: Performing stratified split (60% train, 20% val, 20% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print("Step 7: Fitting preprocessor and transforming data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    raw_feature_names = preprocessor.get_feature_names_out()
    sanitized_feature_names = [re.sub(r'[^A-Za-z0-9_]+', '', name) for name in raw_feature_names]
    print("  - Sanitized feature names to prevent LightGBM errors.")
    output_path = 'data'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"Step 8: Saving processed data to '{output_path}/' directory...")
    pd.DataFrame(X_train_processed, columns=sanitized_feature_names).assign(Class=y_train.values).to_csv(os.path.join(output_path, 'train_large.csv'), index=False)
    pd.DataFrame(X_val_processed, columns=sanitized_feature_names).assign(Class=y_val.values).to_csv(os.path.join(output_path, 'val_large.csv'), index=False)
    pd.DataFrame(X_test_processed, columns=sanitized_feature_names).assign(Class=y_test.values).to_csv(os.path.join(output_path, 'test_large.csv'), index=False)
    joblib.dump(preprocessor, os.path.join(output_path, 'preprocessor.pkl'))
    print("\n--- Preprocessing Complete! ---")
    print(f"Final training data shape: ({X_train_processed.shape[0]}, {X_train_processed.shape[1] + 1})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, preprocess, and split a Kaggle dataset.")
    parser.add_argument("dataset_slug", type=str, help="The slug of the dataset on Kaggle (e.g., 'c/home-credit-default-risk').")
    parser.add_argument("target_column", type=str, help="The name of the column to be used as the target variable.")
    parser.add_argument("positive_class", type=str, help="The label of the positive class in the target column (e.g., '1').")
    args = parser.parse_args()
    prepare_dataset(args.dataset_slug, args.target_column, args.positive_class)