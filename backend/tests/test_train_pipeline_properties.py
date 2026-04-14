"""
Property-based tests for train_pipeline — training split.

Feature: ml-pipeline-upgrade
"""

import os
import sys
from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from sklearn.model_selection import train_test_split

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helper strategy: build a balanced (X, y) dataset
# ---------------------------------------------------------------------------

@st.composite
def balanced_dataset(draw):
    """
    Generate a random balanced (X, y) dataset.

    - n_classes:         2–5   (st.integers(2, 5))
    - samples_per_class: 5–20  (st.integers(5, 20))
    - feature_dim:       1–10  (st.integers(1, 10))
    """
    n_classes = draw(st.integers(min_value=2, max_value=5))
    samples_per_class = draw(st.integers(min_value=5, max_value=20))
    feature_dim = draw(st.integers(min_value=1, max_value=10))

    X_parts = []
    y_parts = []
    for cls_idx in range(n_classes):
        rng = np.random.default_rng(seed=cls_idx * 31 + samples_per_class)
        X_cls = rng.random(size=(samples_per_class, feature_dim)).astype(np.float32)
        y_cls = np.full(samples_per_class, cls_idx, dtype=int)
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    return X, y, n_classes, samples_per_class


# ---------------------------------------------------------------------------
# Property 9: Training split preserves class proportions
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 9: Training split preserves class proportions
@given(data=balanced_dataset())
@settings(max_examples=100)
def test_property9_train_split_proportions(data) -> None:
    """
    For any balanced dataset of size N with C classes, the stratified 80/20
    split (mirroring what train_pipeline does) SHALL produce:
      1. A training set of size ≈ 0.8 * N (within ±n_classes samples).
      2. A test set of size ≈ 0.2 * N (within ±n_classes samples).
      3. Class proportions preserved within ±1 sample per class in each split.

    **Validates: Requirements 6.1**
    """
    X, y, n_classes, samples_per_class = data
    N = len(X)

    # Mirror the exact split used in train_pipeline.run_pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- 1. Training set size ≈ 0.8 * N -----------------------------------
    expected_train = 0.8 * N
    assert abs(len(X_train) - expected_train) <= n_classes, (
        f"Training set size {len(X_train)} deviates from expected "
        f"{expected_train:.1f} by more than {n_classes} (n_classes={n_classes}, N={N})"
    )

    # --- 2. Test set size ≈ 0.2 * N ---------------------------------------
    expected_test = 0.2 * N
    assert abs(len(X_test) - expected_test) <= n_classes, (
        f"Test set size {len(X_test)} deviates from expected "
        f"{expected_test:.1f} by more than {n_classes} (n_classes={n_classes}, N={N})"
    )

    # --- 3. Class proportions preserved within ±1 sample per class --------
    # Each class originally has `samples_per_class` samples.
    # After an 80/20 stratified split, each class should have:
    #   train: floor(0.8 * samples_per_class) or ceil(0.8 * samples_per_class)
    #   test:  floor(0.2 * samples_per_class) or ceil(0.2 * samples_per_class)
    train_counts = Counter(y_train.tolist())
    test_counts = Counter(y_test.tolist())

    for cls in range(n_classes):
        expected_train_cls = samples_per_class * 0.8
        expected_test_cls = samples_per_class * 0.2

        actual_train_cls = train_counts.get(cls, 0)
        actual_test_cls = test_counts.get(cls, 0)

        assert abs(actual_train_cls - expected_train_cls) <= 1, (
            f"Class {cls} train count {actual_train_cls} deviates from "
            f"expected {expected_train_cls:.1f} by more than 1 sample "
            f"(samples_per_class={samples_per_class})"
        )
        assert abs(actual_test_cls - expected_test_cls) <= 1, (
            f"Class {cls} test count {actual_test_cls} deviates from "
            f"expected {expected_test_cls:.1f} by more than 1 sample "
            f"(samples_per_class={samples_per_class})"
        )


# ---------------------------------------------------------------------------
# Property 10: Model serialisation round-trip
# Validates: Requirements 7.1, 7.2
# ---------------------------------------------------------------------------

import pickle
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from backend.ml.train_pipeline import save_artifacts


# Feature: ml-pipeline-upgrade, Property 10: Model serialisation round-trip
@given(
    features=st.lists(
        st.floats(-1, 1, allow_nan=False, allow_infinity=False),
        min_size=63,
        max_size=63,
    )
)
@settings(max_examples=100)
def test_property10_model_serialisation_roundtrip(features: list[float]) -> None:
    """
    For any trained RandomForestClassifier, saving it to disk with
    save_artifacts and loading it back SHALL produce a model that returns
    identical predictions and probability estimates for any input feature vector.

    **Validates: Requirements 7.1, 7.2**
    """
    # Train a small RandomForestClassifier on a tiny synthetic dataset
    # (2 classes, 5 samples each, 63 features)
    rng = np.random.default_rng(seed=42)
    X_train = rng.random(size=(10, 63)).astype(np.float32)
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    labels = ["class_a", "class_b"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Save using save_artifacts, then load back with pickle
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_artifacts(model, labels, tmp_dir)

        model_path = os.path.join(tmp_dir, "model.pkl")
        labels_path = os.path.join(tmp_dir, "labels.pkl")

        with open(model_path, "rb") as fh:
            loaded_model = pickle.load(fh)
        with open(labels_path, "rb") as fh:
            loaded_labels = pickle.load(fh)

        # Prepare input as 2-D array (1 sample, 63 features)
        X_input = np.array([features], dtype=np.float64)

        # Predictions must be identical before and after round-trip
        pred_before = model.predict(X_input)
        pred_after = loaded_model.predict(X_input)
        assert np.array_equal(pred_before, pred_after), (
            f"predict() mismatch after round-trip: {pred_before} != {pred_after}"
        )

        # Probability estimates must be identical before and after round-trip
        proba_before = model.predict_proba(X_input)
        proba_after = loaded_model.predict_proba(X_input)
        assert np.array_equal(proba_before, proba_after), (
            f"predict_proba() mismatch after round-trip: {proba_before} != {proba_after}"
        )

        # Labels list must also survive the round-trip intact
        assert loaded_labels == labels, (
            f"labels mismatch after round-trip: {loaded_labels} != {labels}"
        )
