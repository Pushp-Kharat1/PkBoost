"""
PKBoost API Test Script
Tests all exposed Python APIs to verify they are correctly accessible.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

def test_pkboost_classifier():
    """Test PKBoostClassifier API"""
    print("=" * 60)
    print("Testing PKBoostClassifier")
    print("=" * 60)
    
    try:
        from pkboost import PKBoostClassifier
        print("✓ Import PKBoostClassifier: OK")
    except ImportError as e:
        print(f"✗ Import PKBoostClassifier: FAILED - {e}")
        return False
    
    # Generate test data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to float64 contiguous arrays
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    y_test = np.ascontiguousarray(y_test, dtype=np.float64)
    
    # Test constructor
    try:
        clf = PKBoostClassifier()
        print("✓ PKBoostClassifier(): OK")
    except Exception as e:
        print(f"✗ PKBoostClassifier(): FAILED - {e}")
        return False
    
    # Test auto constructor
    try:
        clf_auto = PKBoostClassifier.auto()
        print("✓ PKBoostClassifier.auto(): OK")
    except Exception as e:
        print(f"✗ PKBoostClassifier.auto(): FAILED - {e}")
    
    # Test is_fitted before training (it's a property, not a method)
    try:
        assert clf.is_fitted == False
        print("✓ is_fitted property before training: OK (returns False)")
    except Exception as e:
        print(f"✗ is_fitted before training: FAILED - {e}")
    
    # Test fit
    try:
        clf.fit(X_train, y_train, verbose=False)
        print("✓ fit(): OK")
    except Exception as e:
        print(f"✗ fit(): FAILED - {e}")
        return False
    
    # Test is_fitted after training
    try:
        assert clf.is_fitted == True
        print("✓ is_fitted after training: OK (returns True)")
    except Exception as e:
        print(f"✗ is_fitted after training: FAILED - {e}")
    
    # Test predict_proba
    try:
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        print(f"✓ predict_proba(): OK (shape={proba.shape})")
    except Exception as e:
        print(f"✗ predict_proba(): FAILED - {e}")
    
    # Test predict
    try:
        preds = clf.predict(X_test)
        assert preds.shape == (len(X_test),)
        print(f"✓ predict(): OK (shape={preds.shape})")
    except Exception as e:
        print(f"✗ predict(): FAILED - {e}")
    
    # Test get_feature_importance
    try:
        importance = clf.get_feature_importance()
        assert len(importance) == X_train.shape[1]
        print(f"✓ get_feature_importance(): OK (len={len(importance)})")
    except Exception as e:
        print(f"✗ get_feature_importance(): FAILED - {e}")
    
    # Test get_n_trees
    try:
        n_trees = clf.get_n_trees()
        print(f"✓ get_n_trees(): OK (n_trees={n_trees})")
    except Exception as e:
        print(f"✗ get_n_trees(): FAILED - {e}")
    
    # Test serialization
    try:
        json_str = clf.to_json()
        assert len(json_str) > 0
        print(f"✓ to_json(): OK (len={len(json_str)})")
    except Exception as e:
        print(f"✗ to_json(): FAILED - {e}")
    
    try:
        clf_loaded = PKBoostClassifier.from_json(json_str)
        print("✓ from_json(): OK")
    except Exception as e:
        print(f"✗ from_json(): FAILED - {e}")
    
    try:
        data = clf.to_bytes()
        assert len(data) > 0
        print(f"✓ to_bytes(): OK (len={len(data)})")
    except Exception as e:
        print(f"✗ to_bytes(): FAILED - {e}")
    
    try:
        clf_loaded2 = PKBoostClassifier.from_bytes(data)
        print("✓ from_bytes(): OK")
    except Exception as e:
        print(f"✗ from_bytes(): FAILED - {e}")
    
    # Test save/load (using temp file)
    import tempfile
    import os
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            temp_path = f.name
        clf.save(temp_path)
        print(f"✓ save(): OK")
        clf_loaded3 = PKBoostClassifier.load(temp_path)
        print(f"✓ load(): OK")
        os.unlink(temp_path)
    except Exception as e:
        print(f"✗ save()/load(): FAILED - {e}")
    
    print()
    return True


def test_pkboost_regressor():
    """Test PKBoostRegressor API"""
    print("=" * 60)
    print("Testing PKBoostRegressor")
    print("=" * 60)
    
    try:
        from pkboost import PKBoostRegressor
        print("✓ Import PKBoostRegressor: OK")
    except ImportError as e:
        print(f"✗ Import PKBoostRegressor: FAILED - {e}")
        return False
    
    # Generate test data
    X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to float64 contiguous arrays
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    y_test = np.ascontiguousarray(y_test, dtype=np.float64)
    
    # Test constructor
    try:
        reg = PKBoostRegressor()
        print("✓ PKBoostRegressor(): OK")
    except Exception as e:
        print(f"✗ PKBoostRegressor(): FAILED - {e}")
        return False
    
    # Test auto constructor
    try:
        reg_auto = PKBoostRegressor.auto()
        print("✓ PKBoostRegressor.auto(): OK")
    except Exception as e:
        print(f"✗ PKBoostRegressor.auto(): FAILED - {e}")
    
    # Test fit
    try:
        reg.fit(X_train, y_train, verbose=False)
        print("✓ fit(): OK")
    except Exception as e:
        print(f"✗ fit(): FAILED - {e}")
        return False
    
    # Test is_fitted (property)
    try:
        assert reg.is_fitted == True
        print("✓ is_fitted: OK")
    except Exception as e:
        print(f"✗ is_fitted: FAILED - {e}")
    
    # Test predict
    try:
        preds = reg.predict(X_test)
        assert preds.shape == (len(X_test),)
        print(f"✓ predict(): OK (shape={preds.shape})")
    except Exception as e:
        print(f"✗ predict(): FAILED - {e}")
    
    print()
    return True


def test_pkboost_multiclass():
    """Test PKBoostMultiClass API"""
    print("=" * 60)
    print("Testing PKBoostMultiClass")
    print("=" * 60)
    
    try:
        from pkboost import PKBoostMultiClass
        print("✓ Import PKBoostMultiClass: OK")
    except ImportError as e:
        print(f"✗ Import PKBoostMultiClass: FAILED - {e}")
        return False
    
    # Generate test data (3-class classification)
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                n_informative=10, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to float64 contiguous arrays
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    y_test = np.ascontiguousarray(y_test, dtype=np.float64)
    
    # Test constructor
    try:
        mclf = PKBoostMultiClass(n_classes=3)
        print("✓ PKBoostMultiClass(n_classes=3): OK")
    except Exception as e:
        print(f"✗ PKBoostMultiClass(): FAILED - {e}")
        return False
    
    # Test fit
    try:
        mclf.fit(X_train, y_train, verbose=False)
        print("✓ fit(): OK")
    except Exception as e:
        print(f"✗ fit(): FAILED - {e}")
        return False
    
    # Test is_fitted (property)
    try:
        assert mclf.is_fitted == True
        print("✓ is_fitted: OK")
    except Exception as e:
        print(f"✗ is_fitted: FAILED - {e}")
    
    # Test predict_proba
    try:
        proba = mclf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        print(f"✓ predict_proba(): OK (shape={proba.shape})")
    except Exception as e:
        print(f"✗ predict_proba(): FAILED - {e}")
    
    # Test predict
    try:
        preds = mclf.predict(X_test)
        assert preds.shape == (len(X_test),)
        print(f"✓ predict(): OK (shape={preds.shape})")
    except Exception as e:
        print(f"✗ predict(): FAILED - {e}")
    
    print()
    return True


def test_pkboost_adaptive():
    """Test PKBoostAdaptive API"""
    print("=" * 60)
    print("Testing PKBoostAdaptive")
    print("=" * 60)
    
    try:
        from pkboost import PKBoostAdaptive
        print("✓ Import PKBoostAdaptive: OK")
    except ImportError as e:
        print(f"✗ Import PKBoostAdaptive: FAILED - {e}")
        return False
    
    # Generate test data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to float64 contiguous arrays
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    y_test = np.ascontiguousarray(y_test, dtype=np.float64)
    
    # Test constructor
    try:
        aclf = PKBoostAdaptive()
        print("✓ PKBoostAdaptive(): OK")
    except Exception as e:
        print(f"✗ PKBoostAdaptive(): FAILED - {e}")
        return False
    
    # Test fit_initial
    try:
        aclf.fit_initial(X_train, y_train, verbose=False)
        print("✓ fit_initial(): OK")
    except Exception as e:
        print(f"✗ fit_initial(): FAILED - {e}")
        return False
    
    # Test is_fitted (property)
    try:
        assert aclf.is_fitted == True
        print("✓ is_fitted: OK")
    except Exception as e:
        print(f"✗ is_fitted: FAILED - {e}")
    
    # Test predict_proba
    try:
        proba = aclf.predict_proba(X_test)
        assert proba.shape == (len(X_test),)
        print(f"✓ predict_proba(): OK (shape={proba.shape})")
    except Exception as e:
        print(f"✗ predict_proba(): FAILED - {e}")
    
    # Test predict
    try:
        preds = aclf.predict(X_test)
        assert preds.shape == (len(X_test),)
        print(f"✓ predict(): OK (shape={preds.shape})")
    except Exception as e:
        print(f"✗ predict(): FAILED - {e}")
    
    # Test get_vulnerability_score
    try:
        score = aclf.get_vulnerability_score()
        print(f"✓ get_vulnerability_score(): OK (score={score})")
    except Exception as e:
        print(f"✗ get_vulnerability_score(): FAILED - {e}")
    
    # Test get_state
    try:
        state = aclf.get_state()
        print(f"✓ get_state(): OK (state={state})")
    except Exception as e:
        print(f"✗ get_state(): FAILED - {e}")
    
    # Test get_metamorphosis_count
    try:
        count = aclf.get_metamorphosis_count()
        print(f"✓ get_metamorphosis_count(): OK (count={count})")
    except Exception as e:
        print(f"✗ get_metamorphosis_count(): FAILED - {e}")
    
    # Test observe_batch
    try:
        aclf.observe_batch(X_test[:100], y_test[:100], verbose=False)
        print("✓ observe_batch(): OK")
    except Exception as e:
        print(f"✗ observe_batch(): FAILED - {e}")
    
    print()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PKBoost API Test Suite")
    print("=" * 60 + "\n")
    
    results = {}
    
    results['PKBoostClassifier'] = test_pkboost_classifier()
    results['PKBoostRegressor'] = test_pkboost_regressor()
    results['PKBoostMultiClass'] = test_pkboost_multiclass()
    results['PKBoostAdaptive'] = test_pkboost_adaptive()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print("All API tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    
    exit(0 if all_passed else 1)
