import pandas as pd
import kaggle

# Download credit card dataset
kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path='raw_data', unzip=True)

# Load
df = pd.read_csv('raw_data/creditcard.csv')
print(f"Loaded {len(df)} samples, {df['Class'].sum()} frauds ({df['Class'].mean()*100:.2f}%)")

# Split 60/20/20
from sklearn.model_selection import train_test_split
train, temp = train_test_split(df, test_size=0.4, random_state=42, stratify=df['Class'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['Class'])

# Save
train.to_csv('data/creditcard_train.csv', index=False)
val.to_csv('data/creditcard_val.csv', index=False)
test.to_csv('data/creditcard_test.csv', index=False)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
