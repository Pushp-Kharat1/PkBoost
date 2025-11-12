import pandas as pd

# Load and sample the large files
print("Loading train data...")
train = pd.read_csv('data/train_large.csv')
print(f"Original train size: {len(train)}")

# Sample 10K for train, keeping class balance
train_sample = train.groupby('Class', group_keys=False).apply(lambda x: x.sample(min(len(x), 5000), random_state=42))
print(f"Sampled train size: {len(train_sample)}")

print("Loading val data...")
val = pd.read_csv('data/val_large.csv')
val_sample = val.groupby('Class', group_keys=False).apply(lambda x: x.sample(min(len(x), 2000), random_state=42))
print(f"Sampled val size: {len(val_sample)}")

print("Loading test data...")
test = pd.read_csv('data/test_large.csv')
test_sample = test.groupby('Class', group_keys=False).apply(lambda x: x.sample(min(len(x), 3000), random_state=42))
print(f"Sampled test size: {len(test_sample)}")

# Save
train_sample.to_csv('data/creditcard_train.csv', index=False)
val_sample.to_csv('data/creditcard_val.csv', index=False)
test_sample.to_csv('data/creditcard_test.csv', index=False)

print("\nSmall test files created!")
print(f"Train: {len(train_sample)} samples, {train_sample['Class'].sum()} frauds ({train_sample['Class'].mean()*100:.2f}%)")
print(f"Val: {len(val_sample)} samples, {val_sample['Class'].sum()} frauds ({val_sample['Class'].mean()*100:.2f}%)")
print(f"Test: {len(test_sample)} samples, {test_sample['Class'].sum()} frauds ({test_sample['Class'].mean()*100:.2f}%)")
