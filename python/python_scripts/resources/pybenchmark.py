import pandas as pd
import numpy as np
import time
import subprocess
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys 

class BenchmarkRunner:
    def __init__(self, data_path='data'):
        self.data_path = Path(data_path)
        self.results = {}
        
    def load_data(self):
        """Load the preprocessed data"""
        print("Loading data...")
        self.X_train = pd.read_csv(self.data_path / 'train_large.csv')
        self.y_train = self.X_train.pop('Class')
        
        self.X_val = pd.read_csv(self.data_path / 'val_large.csv')
        self.y_val = self.X_val.pop('Class')
        
        self.X_test = pd.read_csv(self.data_path / 'test_large.csv')
        self.y_test = self.X_test.pop('Class')
        
        # Calculate scale_pos_weight for imbalanced dataset
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        self.scale_pos_weight = n_neg / n_pos
        
        print(f"Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        print(f"Class distribution - Train: {self.y_train.mean():.3f}, Val: {self.y_val.mean():.3f}, Test: {self.y_test.mean():.3f}")
        
    def run_lightgbm(self, use_auto_params=False):
        """Run LightGBM"""
        train_set = lgb.Dataset(self.X_train, label=self.y_train)
        val_set = lgb.Dataset(self.X_val, label=self.y_val, reference=train_set)

        # Out-of-the-box settings
        params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "boosting_type": "gbdt",
            "verbosity": -1
        }

        print("="*80)
        print("TRAINING LIGHTGBM")
        print("="*80)
        print(f"Scale pos weight: {self.scale_pos_weight:.2f}")

        start_time = time.time()
        model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[train_set, val_set],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(25)]
        )
        train_time = time.time() - start_time

        # Inference time
        start_infer = time.time()
        preds_test = model.predict(self.X_test, num_iteration=model.best_iteration)
        inference_time = time.time() - start_infer

        # Metrics
        y_pred = (preds_test >= 0.5).astype(int)
        auc = roc_auc_score(self.y_test, preds_test)
        pr_auc = average_precision_score(self.y_test, preds_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        # Store metrics
        self.results['lightgbm'] = {
            "test_roc_auc": auc,
            "test_pr_auc": pr_auc,
            "accuracy": accuracy,
            "f1": f1,
            "train_time": train_time,
            "inference_time": inference_time,
            "confusion_matrix": cm
        }

        print(f"Test ROC-AUC: {auc:.5f}, PR-AUC: {pr_auc:.5f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Train time: {train_time:.2f}s, Inference time: {inference_time:.4f}s")
        return model

    def run_xgboost(self, use_auto_params=False):
        """Run XGBoost"""
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        # Out-of-the-box settings
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "verbosity": 1
        }

        print("\n" + "="*80)
        print("TRAINING XGBOOST")
        print("="*80)
        print(f"Scale pos weight: {self.scale_pos_weight:.2f}")

        evals = [(dtrain, 'train'), (dval, 'val')]
        
        start_time = time.time()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=25
        )
        train_time = time.time() - start_time

        # Inference time
        start_infer = time.time()
        preds_test = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        inference_time = time.time() - start_infer

        # Metrics
        y_pred = (preds_test >= 0.5).astype(int)
        auc = roc_auc_score(self.y_test, preds_test)
        pr_auc = average_precision_score(self.y_test, preds_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        # Store metrics
        self.results['xgboost'] = {
            "test_roc_auc": auc,
            "test_pr_auc": pr_auc,
            "accuracy": accuracy,
            "f1": f1,
            "train_time": train_time,
            "inference_time": inference_time,
            "confusion_matrix": cm
        }

        print(f"Test ROC-AUC: {auc:.5f}, PR-AUC: {pr_auc:.5f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Train time: {train_time:.2f}s, Inference time: {inference_time:.4f}s")
        return model

    def run_pkboost(self):
        print("\n" + "="*80)
        print("TRAINING PKBOOST")
        print("="*80)

        # Determine correct binary name for platform
        suffix = ".exe" if sys.platform.startswith("win") else ""
        rust_binary = Path(f'target/release/benchmark{suffix}')

        if not rust_binary.exists():
            print("Building Rust binary...")
            subprocess.run(['cargo', 'build', '--release', '--bin', 'benchmark'], check=True)

        start_time = time.time()
        
        # Stream output in real-time
        process = subprocess.Popen(
            [str(rust_binary)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        result_stdout = ""
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # Print live
            result_stdout += line
        
        process.wait()
        total_time = time.time() - start_time
        
        if process.returncode != 0:
            print(f"\nPKBoost exited with code {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, str(rust_binary))

        # Extract metrics
        metrics = {}
        for line in result_stdout.split('\n'):
            if 'Test ROC-AUC:' in line:
                metrics['test_roc_auc'] = float(line.split(':')[1].strip())
            elif 'Test PR-AUC:' in line:
                metrics['test_pr_auc'] = float(line.split(':')[1].strip())
            elif 'Accuracy:' in line:
                metrics['accuracy'] = float(line.split(':')[1].strip())
            elif 'F1 Score:' in line:
                metrics['f1'] = float(line.split(':')[1].strip())
            elif 'Training time:' in line:
                metrics['train_time'] = float(line.split(':')[1].strip().rstrip('s'))
            elif 'Inference time:' in line:
                time_part = line.split(':')[1].split('(')[0].strip().rstrip('s')
                metrics['inference_time'] = float(time_part)

        self.results['pkboost'] = metrics
        return metrics

    def compare_results(self):
        """Compare LightGBM, XGBoost and PKBoost metrics"""
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        available_models = [k for k in ['lightgbm', 'xgboost', 'pkboost'] if k in self.results]
        
        if len(available_models) < 2:
            print("Error: At least two models must be run before comparison")
            return
        
        print("\nMetric Comparison:")
        header = f"{'Metric':<20}"
        for model in available_models:
            header += f" {model.upper():>12}"
        header += f" {'Best':>12}"
        print(header)
        print("-" * (20 + 12 * (len(available_models) + 1)))
        
        metrics_to_compare = [
            ('ROC-AUC', 'test_roc_auc', 'higher'),
            ('PR-AUC', 'test_pr_auc', 'higher'),
            ('Accuracy', 'accuracy', 'higher'),
            ('F1 Score', 'f1', 'higher'),
            ('Train Time (s)', 'train_time', 'lower'),
            ('Inference Time (s)', 'inference_time', 'lower'),
        ]
        
        for metric_name, metric_key, better in metrics_to_compare:
            row = f"{metric_name:<20}"
            values = {}
            
            for model in available_models:
                val = self.results[model].get(metric_key, 0)
                values[model] = val
                row += f" {val:>12.6f}"
            
            # Determine best
            if better == 'higher':
                best_model = max(values, key=values.get)
            else:
                best_model = min(values, key=values.get)
            
            row += f" {best_model.upper():>12}"
            print(row)
        
        # Pairwise comparisons if all three models are available
        if len(available_models) == 3:
            print("\n" + "="*80)
            print("PAIRWISE PERCENTAGE DIFFERENCES")
            print("="*80)
            
            comparisons = [
                ('PKBoost vs LightGBM', 'pkboost', 'lightgbm'),
                ('PKBoost vs XGBoost', 'pkboost', 'xgboost'),
                ('XGBoost vs LightGBM', 'xgboost', 'lightgbm')
            ]
            
            for comp_name, model1, model2 in comparisons:
                print(f"\n{comp_name}:")
                print(f"{'Metric':<20} {'Difference':>15} {'% Change':>12}")
                print("-" * 50)
                
                for metric_name, metric_key, better in metrics_to_compare:
                    val1 = self.results[model1].get(metric_key, 0)
                    val2 = self.results[model2].get(metric_key, 0)
                    diff = val1 - val2
                    pct = (diff / val2 * 100) if val2 != 0 else 0
                    print(f"{metric_name:<20} {diff:>+15.6f} {pct:>+11.2f}%")
        
    def plot_comparison(self):
        """Plot metrics and performance comparison"""
        available_models = [k for k in ['lightgbm', 'xgboost', 'pkboost'] if k in self.results]
        
        if not available_models:
            print("No models have been run yet")
            return
        
        n_models = len(available_models)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Confusion Matrices
        ax = axes[0, 0]
        if 'lightgbm' in self.results:
            cm = self.results['lightgbm']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('LightGBM Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # Metric Comparison
        ax = axes[0, 1]
        metrics = ['test_roc_auc', 'test_pr_auc', 'accuracy', 'f1']
        metric_labels = ['ROC-AUC', 'PR-AUC', 'Accuracy', 'F1']
        
        x = np.arange(len(metrics))
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, model in enumerate(available_models):
            vals = [self.results[model].get(m, 0) for m in metrics]
            offset = (i - n_models/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=model.upper(), alpha=0.8, color=colors[i])
        
        ax.set_ylabel('Score')
        ax.set_title('Metric Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        # Training Time Comparison
        ax = axes[1, 0]
        train_times = [self.results[model]['train_time'] for model in available_models]
        bars = ax.bar(range(len(available_models)), train_times, alpha=0.8, color=colors[:n_models])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.set_xticks(range(len(available_models)))
        ax.set_xticklabels([m.upper() for m in available_models])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s', ha='center', va='bottom')
        
        # Inference Time Comparison
        ax = axes[1, 1]
        infer_times = [self.results[model]['inference_time'] for model in available_models]
        bars = ax.bar(range(len(available_models)), infer_times, alpha=0.8, color=colors[:n_models])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Inference Time Comparison')
        ax.set_xticks(range(len(available_models)))
        ax.set_xticklabels([m.upper() for m in available_models])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
        print("\nPlots saved to 'benchmark_comparison.png'")
        plt.show()


def main():
    benchmark = BenchmarkRunner(data_path='resources/data')
    benchmark.load_data()
    
    # Run all models with out-of-the-box settings
    benchmark.run_lightgbm(use_auto_params=False)
    benchmark.run_xgboost(use_auto_params=False)
    
    try:
        benchmark.run_pkboost()
    except Exception as e:
        print(f"Error running PKBoost: {e}")
        print("Continuing with LightGBM and XGBoost results only...")
    
    benchmark.compare_results()
    benchmark.plot_comparison()


if __name__ == "__main__":
    main()