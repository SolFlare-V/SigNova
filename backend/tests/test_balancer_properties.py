"""
Property-based tests for balancer.

Feature: ml-pipeline-upgrade
"""

import os
import sys

import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ml.balancer import balance_dataset


# ---------------------------------------------------------------------------
# Helper strategy: build an imbalanced (X, y) dataset
# ---------------------------------------------------------------------------

@st.composite
def imbalanced_dataset(draw):
    """
    Generate a random imbalanced (X, y) dataset.

    - n_classes: 2–5
    - samples per class: 1–20 (varying to create imbalance)
    - feature dimensions: 1–10
    """
    n_classes = draw(st.integers(min_value=2, max_value=5))
    feature_dim = draw(st.integers(min_value=1, max_value=10))

    # Draw a distinct sample count for each class (may differ → imbalance)
    samples_per_class = draw(
        st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=n_classes,
            max_size=n_classes,
        )
    )

    X_parts = []
    y_parts = []
    for cls_idx, n_samples in enumerate(samples_per_class):
        # Use a deterministic seed per class so features are reproducible
        rng = np.random.default_rng(seed=cls_idx * 100 + n_samples)
        X_cls = rng.random(size=(n_samples, feature_dim)).astype(np.float32)
        y_cls = np.full(n_samples, cls_idx, dtype=int)
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Also expose the target_per_class we'll request (median of class counts)
    target = int(np.median(samples_per_class))

    return X, y, n_classes, feature_dim, target, samples_per_class


# ---------------------------------------------------------------------------
# Property 8: Balanced dataset has equal class counts
# Validates: Requirements 5.1, 5.2, 5.3
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 8: Balanced dataset has equal class counts
@given(data=imbalanced_dataset())
@settings(max_examples=100)
def test_property8_balance_equal_counts(data) -> None:
    """
    For any imbalanced feature dataset (X, y) with C classes and a given
    target_per_class, balance_dataset returns arrays where:
      1. Every class appears exactly target_per_class times.
      2. Each feature vector retains its original label (label correspondence).
      3. Output arrays have the correct shapes.

    **Validates: Requirements 5.1, 5.2, 5.3**
    """
    X, y, n_classes, feature_dim, target_per_class, _ = data

    X_bal, y_bal = balance_dataset(X, y, target_per_class=target_per_class)

    # --- 1. Shape checks ---------------------------------------------------
    expected_total = n_classes * target_per_class
    assert X_bal.shape == (expected_total, feature_dim), (
        f"Expected X_balanced shape ({expected_total}, {feature_dim}), "
        f"got {X_bal.shape}"
    )
    assert y_bal.shape == (expected_total,), (
        f"Expected y_balanced shape ({expected_total},), got {y_bal.shape}"
    )

    # --- 2. Equal class counts ---------------------------------------------
    classes = np.unique(y)
    for cls in classes:
        count = int(np.sum(y_bal == cls))
        assert count == target_per_class, (
            f"Class {cls}: expected {target_per_class} samples, got {count}"
        )

    # --- 3. Label correspondence preserved ---------------------------------
    # Build a lookup: original feature vector → set of valid labels
    # (a feature vector may appear multiple times via oversampling, but each
    # occurrence must carry the same label it had in the original dataset)
    original_label: dict[int, int] = {}
    for row_idx in range(X.shape[0]):
        # Use the row index as the key; we verify by matching feature vectors
        pass  # built below

    # For every row in the balanced output, find a matching row in the
    # original dataset and confirm the label matches.
    for i in range(X_bal.shape[0]):
        feat = X_bal[i]
        label = int(y_bal[i])

        # Find all original rows that are close to this feature vector
        diffs = np.abs(X - feat).sum(axis=1)
        matching_original_indices = np.where(diffs < 1e-6)[0]

        assert len(matching_original_indices) > 0, (
            f"Balanced row {i} has no matching row in the original dataset — "
            "balancer introduced a new feature vector."
        )

        # Every matching original row must have the same label
        for orig_idx in matching_original_indices:
            assert int(y[orig_idx]) == label, (
                f"Balanced row {i}: feature matches original row {orig_idx} "
                f"(label {y[orig_idx]}) but balanced label is {label}."
            )
