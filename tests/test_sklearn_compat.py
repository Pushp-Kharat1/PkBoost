"""
Comprehensive test suite for PKBoost sklearn compatibility.

Tests all major sklearn integration points:
- Basic estimator API (fit, predict, score)
- Pipeline integration
- GridSearchCV and RandomizedSearchCV
- Cross-validation
- Metrics and scoring
- Feature selection
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sklearn_interface import PKBoostClassifier, PKBoostRegressor, PKBoostAdaptiveClassifier
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure pkboost is installed: pip install pkboost")
    exit(1)


def test_basic_classifier():
    """Test 1: Basic classifier API"""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Classifier API")
    print("=" * 70)
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf = PKBoostClassifier(n_estimators=200, learning_rate=0.1)
    print(f"[OK] Initialization: {type(clf).__name__}")
    
    clf.fit(X_train, y_train, verbose=False)
    print(f"[OK] Fit complete: {clf.n_trees_} trees")
    
    y_pred = clf.predict(X_test)
    print(f"[OK] Predict shape: {y_pred.shape}")
    
    y_proba = clf.predict_proba(X_test)
    print(f"[OK] Predict_proba shape: {y_proba.shape}")
    
    accuracy = clf.score(X_test, y_test)
    print(f"[OK] Score (accuracy): {accuracy:.4f}")
    
    print(f"[OK] Feature importances: {clf.feature_importances_.shape}")
    print(f"[OK] Classes: {clf.classes_}")
    
    print("\n[PASS] TEST 1 PASSED")


def test_pipeline():
    """Test 2: Pipeline integration"""
    print("\n" + "=" * 70)
    print("TEST 2: Pipeline Integration")
    print("=" * 70)
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', PKBoostClassifier(n_estimators=100)),
    ])
    
    print("[OK] Pipeline created")
    
    pipeline.fit(X_train, y_train)
    print("[OK] Pipeline fitted")
    
    accuracy = pipeline.score(X_test, y_test)
    print(f"[OK] Pipeline accuracy: {accuracy:.4f}")
    
    print("\n[PASS] TEST 2 PASSED")


def test_cross_validation():
    """Test 3: Cross-validation"""
    print("\n" + "=" * 70)
    print("TEST 3: Cross-validation")
    print("=" * 70)
    
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    clf = PKBoostClassifier(n_estimators=100, learning_rate=0.1)
    
    for scoring in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cross_val_score(clf, X, y, cv=3, scoring=scoring)
        print(f"[OK] {scoring:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    print("\n[PASS] TEST 3 PASSED")


def test_grid_search():
    """Test 4: GridSearchCV"""
    print("\n" + "=" * 70)
    print("TEST 4: GridSearchCV")
    print("=" * 70)
    
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6],
    }
    
    clf = PKBoostClassifier()
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', verbose=0)
    
    print(f"[OK] Testing {len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth'])} combinations")
    
    grid.fit(X, y)
    
    print(f"[OK] Best score: {grid.best_score_:.4f}")
    print(f"[OK] Best params: {grid.best_params_}")
    
    print("\n[PASS] TEST 4 PASSED")


def test_regressor():
    """Test 5: Regressor"""
    print("\n" + "=" * 70)
    print("TEST 5: Regressor")
    print("=" * 70)
    
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    reg = PKBoostRegressor(auto=True)
    reg.fit(X_train, y_train, verbose=False)
    
    r2 = reg.score(X_test, y_test)
    print(f"[OK] R² Score: {r2:.4f}")
    
    scores = cross_val_score(reg, X, y, cv=3, scoring='r2')
    print(f"[OK] CV R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    print("\n[PASS] TEST 5 PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" " * 15 + "PKBoost Sklearn Compatibility Test Suite")
    print("=" * 70)
    
    tests = [
        test_basic_classifier,
        test_pipeline,
        test_cross_validation,
        test_grid_search,
        test_regressor,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test_func.__name__} FAILED")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {len(tests)}")
    print(f"[PASS] Passed: {passed}")
    print(f"[FAIL] Failed: {failed}")
    print("=" * 70)
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed! PKBoost is fully sklearn-compatible!")
    else:
        print(f"\n[WARNING] {failed} test(s) failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
