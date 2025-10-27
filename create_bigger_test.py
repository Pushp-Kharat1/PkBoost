import pandas as pd

train = pd.read_csv('data/train_large.csv')
val = pd.read_csv('data/val_large.csv')
test = pd.read_csv('data/test_large.csv')

def create_imbalanced(df, fraud_rate=0.005, total_samples=50000):
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]
    
    n_fraud = int(total_samples * fraud_rate)
    n_normal = total_samples - n_fraud
    
    fraud_sample = fraud.sample(min(len(fraud), n_fraud), random_state=42, replace=(len(fraud) < n_fraud))
    normal_sample = normal.sample(min(len(normal), n_normal), random_state=42, replace=(len(normal) < n_normal))
    
    result = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    return result

train_imb = create_imbalanced(train, fraud_rate=0.005, total_samples=20000)
val_imb = create_imbalanced(val, fraud_rate=0.005, total_samples=5000)
test_imb = create_imbalanced(test, fraud_rate=0.005, total_samples=50000)  # Bigger test set

train_imb.to_csv('data/creditcard_train.csv', index=False)
val_imb.to_csv('data/creditcard_val.csv', index=False)
test_imb.to_csv('data/creditcard_test.csv', index=False)

print(f"Created bigger test set: {len(test_imb)} samples, {int(test_imb['Class'].sum())} frauds")
