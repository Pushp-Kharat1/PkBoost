"""Test model serialization functionality."""

import numpy as np
import pkboost
import tempfile
import os
import pytest


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] + 0.5 * X[:, 2] > 0).astype(float)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create a trained model."""
    X, y = sample_data
    clf = pkboost.PKBoostClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
    )
    clf.fit(X, y, verbose=False)
    return clf


def test_to_bytes_from_bytes(trained_model, sample_data):
    """Test serialization to/from bytes."""
    X, _ = sample_data

    # Get predictions before
    pred_before = trained_model.predict_proba(X)

    # Serialize and deserialize
    model_bytes = trained_model.to_bytes()
    assert isinstance(model_bytes, bytes)
    assert len(model_bytes) > 0

    clf_loaded = pkboost.PKBoostClassifier.from_bytes(model_bytes)
    pred_after = clf_loaded.predict_proba(X)

    # Check predictions are highly correlated
    correlation = np.corrcoef(pred_before, pred_after)[0, 1]
    assert correlation > 0.999, f"Correlation too low: {correlation}"

    # Check tree count is preserved
    assert trained_model.get_n_trees() == clf_loaded.get_n_trees()


def test_to_json_from_json(trained_model, sample_data):
    """Test serialization to/from JSON string."""
    X, _ = sample_data

    pred_before = trained_model.predict_proba(X)

    # Serialize and deserialize
    model_json = trained_model.to_json()
    assert isinstance(model_json, str)
    assert len(model_json) > 0

    clf_loaded = pkboost.PKBoostClassifier.from_json(model_json)
    pred_after = clf_loaded.predict_proba(X)

    correlation = np.corrcoef(pred_before, pred_after)[0, 1]
    assert correlation > 0.999, f"Correlation too low: {correlation}"


def test_save_load(trained_model, sample_data):
    """Test saving and loading from file."""
    X, _ = sample_data

    pred_before = trained_model.predict_proba(X)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        model_path = f.name

    try:
        trained_model.save(model_path)
        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0

        clf_loaded = pkboost.PKBoostClassifier.load(model_path)
        pred_after = clf_loaded.predict_proba(X)

        correlation = np.corrcoef(pred_before, pred_after)[0, 1]
        assert correlation > 0.999, f"Correlation too low: {correlation}"
    finally:
        os.unlink(model_path)


def test_feature_importance_preserved(trained_model, sample_data):
    """Test that feature importance is preserved after serialization."""
    X, _ = sample_data

    fi_before = trained_model.get_feature_importance()

    model_json = trained_model.to_json()
    clf_loaded = pkboost.PKBoostClassifier.from_json(model_json)

    fi_after = clf_loaded.get_feature_importance()

    np.testing.assert_array_equal(fi_before, fi_after)


def test_loaded_model_can_predict_new_data(trained_model, sample_data):
    """Test that loaded model can predict on new data."""
    X, _ = sample_data

    # Create new test data
    np.random.seed(123)
    X_new = np.random.randn(100, 10)

    model_bytes = trained_model.to_bytes()
    clf_loaded = pkboost.PKBoostClassifier.from_bytes(model_bytes)

    # Should not raise
    pred = clf_loaded.predict_proba(X_new)
    assert len(pred) == 100
    assert all(0 <= p <= 1 for p in pred)


def test_unfitted_model_cannot_serialize():
    """Test that unfitted model raises error on serialization."""
    clf = pkboost.PKBoostClassifier()

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.to_bytes()

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.to_json()

    with pytest.raises(RuntimeError, match="not fitted"):
        clf.save("/tmp/test.json")


def test_invalid_json_raises_error():
    """Test that invalid JSON raises proper error."""
    with pytest.raises(ValueError, match="Deserialization failed"):
        pkboost.PKBoostClassifier.from_json("not valid json")


def test_invalid_file_raises_error():
    """Test that invalid file path raises proper error."""
    with pytest.raises(ValueError, match="Failed to read"):
        pkboost.PKBoostClassifier.load("/nonexistent/path/model.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
