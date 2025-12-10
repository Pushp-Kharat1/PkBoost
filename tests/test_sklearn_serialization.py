"""Test sklearn interface serialization with joblib/pickle."""

import numpy as np
import pytest
import tempfile
import os

try:
    import joblib
except ImportError:
    joblib = None

try:
    import pickle
except ImportError:
    pickle = None

from pkboost_sklearn import PKBoostClassifier


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
    return X, y


@pytest.fixture
def trained_classifier(sample_data):
    """Create a trained sklearn-compatible classifier."""
    X, y = sample_data
    clf = PKBoostClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
    )
    clf.fit(X, y, verbose=False)
    return clf


@pytest.mark.skipif(joblib is None, reason="joblib not installed")
def test_joblib_dump_load(trained_classifier, sample_data):
    """Test serialization with joblib.dump and joblib.load."""
    X, _ = sample_data

    pred_before = trained_classifier.predict_proba(X)[:, 1]

    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        model_path = f.name

    try:
        # Save with joblib
        joblib.dump(trained_classifier, model_path)
        assert os.path.exists(model_path)

        # Load with joblib
        clf_loaded = joblib.load(model_path)

        # Check predictions
        pred_after = clf_loaded.predict_proba(X)[:, 1]
        correlation = np.corrcoef(pred_before, pred_after)[0, 1]
        assert correlation > 0.999, f"Correlation too low: {correlation}"

        # Check attributes are preserved
        assert clf_loaded.n_features_in_ == trained_classifier.n_features_in_
        assert clf_loaded.n_trees_ == trained_classifier.n_trees_
        np.testing.assert_array_equal(clf_loaded.classes_, trained_classifier.classes_)
    finally:
        os.unlink(model_path)


@pytest.mark.skipif(pickle is None, reason="pickle not available")
def test_pickle_dump_load(trained_classifier, sample_data):
    """Test serialization with pickle.dump and pickle.load."""
    X, _ = sample_data

    pred_before = trained_classifier.predict_proba(X)[:, 1]

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        model_path = f.name

    try:
        # Save with pickle
        with open(model_path, 'wb') as f:
            pickle.dump(trained_classifier, f)

        # Load with pickle
        with open(model_path, 'rb') as f:
            clf_loaded = pickle.load(f)

        # Check predictions
        pred_after = clf_loaded.predict_proba(X)[:, 1]
        correlation = np.corrcoef(pred_before, pred_after)[0, 1]
        assert correlation > 0.999, f"Correlation too low: {correlation}"
    finally:
        os.unlink(model_path)


def test_save_load_model(trained_classifier, sample_data):
    """Test save_model and load_model methods."""
    X, _ = sample_data

    pred_before = trained_classifier.predict_proba(X)[:, 1]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        model_path = f.name

    try:
        # Save model
        trained_classifier.save_model(model_path)
        assert os.path.exists(model_path)

        # Load model
        clf_loaded = PKBoostClassifier.load_model(model_path)

        # Check predictions
        pred_after = clf_loaded.predict_proba(X)[:, 1]
        correlation = np.corrcoef(pred_before, pred_after)[0, 1]
        assert correlation > 0.999, f"Correlation too low: {correlation}"

        # Check attributes
        assert clf_loaded.n_trees_ == trained_classifier.n_trees_
    finally:
        os.unlink(model_path)


def test_sklearn_clone(trained_classifier):
    """Test that sklearn clone works."""
    from sklearn.base import clone

    # Clone the unfitted params
    clf_unfitted = PKBoostClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
    )
    clf_cloned = clone(clf_unfitted)

    assert clf_cloned.n_estimators == clf_unfitted.n_estimators
    assert clf_cloned.max_depth == clf_unfitted.max_depth
    assert clf_cloned.learning_rate == clf_unfitted.learning_rate


def test_sklearn_pipeline(sample_data):
    """Test that PKBoostClassifier works in sklearn Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = sample_data

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', PKBoostClassifier(n_estimators=10, max_depth=2)),
    ])

    pipe.fit(X, y)
    pred = pipe.predict(X)

    assert len(pred) == len(y)
    assert set(pred).issubset({0, 1})


@pytest.mark.skipif(joblib is None, reason="joblib not installed")
def test_sklearn_pipeline_joblib(sample_data):
    """Test that sklearn Pipeline with PKBoostClassifier can be serialized."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = sample_data

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', PKBoostClassifier(n_estimators=10, max_depth=2)),
    ])

    pipe.fit(X, y)
    pred_before = pipe.predict_proba(X)[:, 1]

    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        model_path = f.name

    try:
        joblib.dump(pipe, model_path)
        pipe_loaded = joblib.load(model_path)

        pred_after = pipe_loaded.predict_proba(X)[:, 1]
        correlation = np.corrcoef(pred_before, pred_after)[0, 1]
        assert correlation > 0.999, f"Correlation too low: {correlation}"
    finally:
        os.unlink(model_path)


def test_feature_importances_preserved(trained_classifier, sample_data):
    """Test that feature importances are preserved after pickling."""
    X, _ = sample_data

    fi_before = trained_classifier.feature_importances_.copy()

    # Pickle round-trip
    pickled = pickle.dumps(trained_classifier)
    clf_loaded = pickle.loads(pickled)

    fi_after = clf_loaded.feature_importances_

    np.testing.assert_array_equal(fi_before, fi_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
