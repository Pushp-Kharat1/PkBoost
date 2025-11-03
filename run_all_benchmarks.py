import subprocess
import shutil
import os
from pathlib import Path
import time
import json
import pandas as pd

DATASETS = [
    {
        "name": "Credit Card Fraud",
        "slug": "mlg-ulb/creditcardfraud",
        "target": "Class",
        "positive": "1"
    },
    {
        "name": "Pima Indians Diabetes",
        "slug": "uciml/pima-indians-diabetes-database",
        "target": "Outcome",
        "positive": "1"
    },
    {
        "name": "Breast Cancer Wisconsin",
        "slug": "uciml/breast-cancer-wisconsin-data",
        "target": "diagnosis",
        "positive": "M"
    },
    {
        "name": "Telco Customer Churn",
        "slug": "blastchar/telco-customer-churn",
        "target": "Churn",
        "positive": "Yes"
    },
    {
        "name": "IEEE-CIS Fraud Detection",
        "slug": "c/ieee-fraud-detection",
        "target": "isFraud",
        "positive": "1"
    },
    {
        "name": "NSL-KDD",
        "slug": "hassan06/nslkdd",
        "target": "normal",
        "positive": "normal"
    }
]

def clean_directories():
    """Remove all files from raw_data and data directories"""
    print("\nCleaning directories...")
    for dir_name in ['raw_data', 'data']:
        if os.path.exists(dir_name):
            for root, dirs, files in os.walk(dir_name, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except:
                        pass
            print(f"  Cleaned {dir_name}/")
        else:
            os.makedirs(dir_name, exist_ok=True)
            print(f"  Created {dir_name}/")

def run_benchmark(dataset_info, index, total):
    """Run complete benchmark for a single dataset"""
    print("\n" + "="*80)
    print(f"BENCHMARK {index}/{total}: {dataset_info['name']}")
    print("="*80)
    
    # Clean previous data
    clean_directories()
    
    # Prepare dataset
    print(f"\nStep 1: Preparing dataset...")
    try:
        subprocess.run([
            'python', 'prepare_data.py',
            dataset_info['slug'],
            dataset_info['target'],
            dataset_info['positive']
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to prepare dataset: {e}")
        return None
    
    # Run benchmark with live output
    print(f"\nStep 2: Running benchmark...")
    try:
        # Use run() instead of Popen for simpler output handling
        result = subprocess.run(
            ['python', 'run_single_benchmark.py'],
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit
        )
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"Benchmark failed with return code {result.returncode}")
            return None
        
        # Parse results from output
        results = parse_benchmark_results(result.stdout, dataset_info['name'])
        
        # Show immediate comparison for this dataset
        if results:
            print_dataset_comparison(results)
        
        return results
    except Exception as e:
        print(f"Failed to run benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    """Run complete benchmark for a single dataset"""
    print("\n" + "="*80)
    print(f"BENCHMARK {index}/{total}: {dataset_info['name']}")
    print("="*80)
    
    # Clean previous data
    clean_directories()
    
    # Prepare dataset
    print(f"\nStep 1: Preparing dataset...")
    try:
        subprocess.run([
            'python', 'prepare_data.py',
            dataset_info['slug'],
            dataset_info['target'],
            dataset_info['positive']
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to prepare dataset: {e}")
        return None
    
    # Run benchmark with live output
    print(f"\nStep 2: Running benchmark...")
    try:
        # Use run() instead of Popen for simpler output handling
        result = subprocess.run(
            ['python', 'run_single_benchmark.py'],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"Benchmark failed with return code {result.returncode}")
            return None
        
        # Parse results from output
        results = parse_benchmark_results(result.stdout, dataset_info['name'])
        
        # Show immediate comparison for this dataset
        if results:
            print_dataset_comparison(results)
        
        return results
    except Exception as e:
        print(f"Failed to run benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_benchmark_results(output, dataset_name):
    """Extract metrics from benchmark output"""
    results = {'dataset': dataset_name}
    lines = output.split('\n')
    
    current_model = None
    for i, line in enumerate(lines):
        if 'TRAINING LIGHTGBM' in line:
            current_model = 'lightgbm'
        elif 'TRAINING XGBOOST' in line:
            current_model = 'xgboost'
        elif 'TRAINING PKBOOST' in line:
            current_model = 'pkboost'
        elif current_model and 'Test ROC-AUC:' in line:
            # Parse: Test ROC-AUC: 0.98909, PR-AUC: 0.98717, Accuracy: 0.9737, F1: 0.9630
            parts = line.split(',')
            if len(parts) >= 4:
                roc_auc = float(parts[0].split(':')[1].strip())
                pr_auc = float(parts[1].split(':')[1].strip())
                accuracy = float(parts[2].split(':')[1].strip())
                f1 = float(parts[3].split(':')[1].strip())
                results[f'{current_model}_roc_auc'] = roc_auc
                results[f'{current_model}_pr_auc'] = pr_auc
                results[f'{current_model}_accuracy'] = accuracy
                results[f'{current_model}_f1'] = f1
        elif current_model and 'Train time:' in line and 'Inference time:' in line:
            # Parse: Train time: 0.18s, Inference time: 0.0030s
            parts = line.split(',')
            if len(parts) >= 2:
                train_time = float(parts[0].split(':')[1].strip().rstrip('s'))
                inference_time = float(parts[1].split(':')[1].strip().rstrip('s'))
                results[f'{current_model}_train_time'] = train_time
                results[f'{current_model}_inference_time'] = inference_time
        elif current_model == 'pkboost' and 'Test ROC-AUC:' in line and i+2 < len(lines):
            # PKBoost format: Test ROC-AUC:  0.996693
            roc_auc = float(line.split(':')[1].strip())
            results[f'{current_model}_roc_auc'] = roc_auc
            # Look for next lines
            if 'Test PR-AUC:' in lines[i+1]:
                pr_auc = float(lines[i+1].split(':')[1].strip())
                results[f'{current_model}_pr_auc'] = pr_auc
            if 'Training time:' in lines[i+2]:
                train_time = float(lines[i+2].split(':')[1].strip().rstrip('s'))
                results[f'{current_model}_train_time'] = train_time
            if i+3 < len(lines) and 'Inference time:' in lines[i+3]:
                inference_time = float(lines[i+3].split(':')[1].split('(')[0].strip().rstrip('s'))
                results[f'{current_model}_inference_time'] = inference_time
        elif current_model == 'pkboost' and 'Accuracy:' in line and 'Precision' not in line:
            # Parse accuracy from Classification Metrics section
            accuracy = float(line.split(':')[1].strip())
            results[f'{current_model}_accuracy'] = accuracy
        elif current_model == 'pkboost' and 'F1 Score:' in line:
            f1 = float(line.split(':')[1].strip())
            results[f'{current_model}_f1'] = f1
    
    return results

def print_dataset_comparison(result):
    """Print comparison for a single dataset"""
    print("\n" + "="*80)
    print(f"RESULTS FOR: {result['dataset']}")
    print("="*80)
    
    models = ['lightgbm', 'xgboost', 'pkboost']
    
    # ROC-AUC comparison
    print(f"\n{'Model':<15} {'ROC-AUC':>12} {'PR-AUC':>12} {'Accuracy':>12} {'F1':>12}")
    print("-" * 63)
    
    roc_scores = {}
    for model in models:
        roc_key = f'{model}_roc_auc'
        pr_key = f'{model}_pr_auc'
        acc_key = f'{model}_accuracy'
        f1_key = f'{model}_f1'
        
        if roc_key in result:
            roc = result[roc_key]
            pr = result.get(pr_key, 0)
            acc = result.get(acc_key, 0)
            f1 = result.get(f1_key, 0)
            roc_scores[model] = roc
            print(f"{model.upper():<15} {roc:>12.6f} {pr:>12.6f} {acc:>12.6f} {f1:>12.6f}")
    
    # Determine winner
    if roc_scores:
        winner = max(roc_scores, key=roc_scores.get)
        print(f"\n{'WINNER:':<15} {winner.upper()} (ROC-AUC: {roc_scores[winner]:.6f})")
    
    print("="*80)

def print_combined_results(all_results):
    """Print combined comparison across all datasets"""
    print("\n" + "="*80)
    print("COMBINED BENCHMARK RESULTS - ALL DATASETS")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    # Calculate averages
    models = ['lightgbm', 'xgboost', 'pkboost']
    metrics = ['roc_auc', 'pr_auc', 'accuracy', 'f1', 'train_time', 'inference_time']
    
    print("\n" + "="*80)
    print("AVERAGE METRICS ACROSS ALL DATASETS")
    print("="*80)
    print(f"\n{'Metric':<25} {'LightGBM':>12} {'XGBoost':>12} {'PKBoost':>12} {'Best':>12}")
    print("-" * 73)
    
    metric_names = {
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'train_time': 'Train Time (s)',
        'inference_time': 'Inference Time (s)'
    }
    
    for metric in metrics:
        values = {}
        for model in models:
            col = f'{model}_{metric}'
            if col in df.columns:
                values[model] = df[col].mean()
        
        if values:
            if 'time' in metric:
                best = min(values, key=values.get)
            else:
                best = max(values, key=values.get)
            
            row = f"{metric_names[metric]:<25}"
            for model in models:
                if model in values:
                    row += f" {values[model]:>12.6f}"
                else:
                    row += f" {'N/A':>12}"
            row += f" {best.upper():>12}"
            print(row)
    
    # Per-dataset breakdown
    print("\n" + "="*80)
    print("PER-DATASET ROC-AUC COMPARISON")
    print("="*80)
    print(f"\n{'Dataset':<30} {'LightGBM':>12} {'XGBoost':>12} {'PKBoost':>12} {'Best':>12}")
    print("-" * 78)
    
    for _, row in df.iterrows():
        values = {}
        for model in models:
            col = f'{model}_roc_auc'
            if col in row and pd.notna(row[col]):
                values[model] = row[col]
        
        if values:
            best = max(values, key=values.get)
            line = f"{row['dataset']:<30}"
            for model in models:
                if model in values:
                    line += f" {values[model]:>12.6f}"
                else:
                    line += f" {'N/A':>12}"
            line += f" {best.upper():>12}"
            print(line)
    
    # Win count
    print("\n" + "="*80)
    print("MODEL WIN COUNT (Best ROC-AUC per dataset)")
    print("="*80)
    
    win_count = {model: 0 for model in models}
    for _, row in df.iterrows():
        values = {}
        for model in models:
            col = f'{model}_roc_auc'
            if col in row and pd.notna(row[col]):
                values[model] = row[col]
        if values:
            winner = max(values, key=values.get)
            win_count[winner] += 1
    
    for model in models:
        print(f"{model.upper():<15}: {win_count[model]} wins")
    
    # Save to CSV
    df.to_csv('all_benchmarks_results.csv', index=False)
    print("\nDetailed results saved to 'all_benchmarks_results.csv'")

def main():
    print("="*80)
    print("PKBoost Multi-Dataset Benchmark Suite")
    print("="*80)
    print(f"\nTotal datasets to benchmark: {len(DATASETS)}")
    
    all_results = []
    start_time = time.time()
    
    for i, dataset in enumerate(DATASETS, 1):
        result = run_benchmark(dataset, i, len(DATASETS))
        if result:
            all_results.append(result)
        
        if i < len(DATASETS):
            print("\nWaiting 3 seconds before next benchmark...")
            time.sleep(3)
    
    # Combined results
    total_time = time.time() - start_time
    
    if all_results:
        print_combined_results(all_results)
    
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Successful: {len(all_results)}/{len(DATASETS)} benchmarks")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("="*80)

if __name__ == "__main__":
    main()
