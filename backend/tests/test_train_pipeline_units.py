"""
Unit tests for train_pipeline.py.

Covers:
  - test_rf_hyperparameters         (Req 6.2) — n_estimators=200, max_depth=20, random_state=42
  - test_accuracy_warning_below_85  (Req 6.4) — warning printed when accuracy < 0.85
  - test_cli_no_args_exits_nonzero  (Req 10.4) — sys.exit(1) when no dataset args
  - test_cli_progress_messages      (Req 10.3) — stdout progress messages at each step
"""

import os
import sys
import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ml.train_pipeline import evaluate, parse_args, run_pipeline, train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_dataset(n_classes: int = 3, samples_per_class: int = 10, n_features: int = 63):
    """Return a small synthetic (X_train, y_train) pair."""
    rng = np.random.default_rng(seed=0)
    X = rng.random(size=(n_classes * samples_per_class, n_features)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), samples_per_class).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Req 6.2 — RF hyperparameters: n_estimators, max_depth, random_state
# ---------------------------------------------------------------------------

def test_rf_hyperparameters():
    """
    Calling train() with n_estimators=200 and max_depth=20 SHALL return a
    RandomForestClassifier with those exact values plus random_state=42.

    Validates: Requirements 6.2, 6.5
    """
    X_train, y_train = _make_tiny_dataset()

    model = train(X_train, y_train, n_estimators=200, max_depth=20)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 200, f"Expected n_estimators=200, got {model.n_estimators}"
    assert model.max_depth == 20, f"Expected max_depth=20, got {model.max_depth}"
    assert model.random_state == 42, f"Expected random_state=42, got {model.random_state}"


# ---------------------------------------------------------------------------
# Req 6.4 — Warning printed when accuracy < 0.85
# ---------------------------------------------------------------------------

def test_accuracy_warning_below_85(capsys):
    """
    evaluate() SHALL print a warning to stdout when test-set accuracy is
    below 0.85.

    Validates: Requirements 6.4
    """
    # Build a model that always predicts class 0
    mock_model = MagicMock(spec=RandomForestClassifier)
    n_test = 20
    # All predictions are class 0; only 1 out of 20 samples is actually class 0
    # → accuracy = 1/20 = 0.05, well below 0.85
    mock_model.predict.return_value = np.zeros(n_test, dtype=np.int32)

    X_test = np.zeros((n_test, 63), dtype=np.float32)
    y_test = np.array([0] + [1] * (n_test - 1), dtype=np.int32)  # accuracy ≈ 0.05
    label_names = ["A", "B"]

    accuracy = evaluate(mock_model, X_test, y_test, label_names)

    captured = capsys.readouterr()
    assert accuracy < 0.85, f"Expected accuracy < 0.85, got {accuracy}"
    assert "WARNING" in captured.out, (
        "Expected a WARNING message in stdout when accuracy < 0.85"
    )
    assert "85%" in captured.out or "0.85" in captured.out, (
        "WARNING message should reference the 85% threshold"
    )


# ---------------------------------------------------------------------------
# Req 10.4 — CLI exits non-zero when no dataset args are provided
# ---------------------------------------------------------------------------

def test_cli_no_args_exits_nonzero():
    """
    parse_args() SHALL call sys.exit(1) when no dataset path arguments
    (--asl-dir, --mnist-csv, --isl-dir) are provided.

    Validates: Requirements 10.4
    """
    with pytest.raises(SystemExit) as exc_info:
        parse_args([])  # empty args list — no dataset paths

    assert exc_info.value.code != 0, (
        f"Expected non-zero exit code, got {exc_info.value.code}"
    )


# ---------------------------------------------------------------------------
# Req 10.3 — Progress messages printed at each pipeline step
# ---------------------------------------------------------------------------

def test_cli_progress_messages(tmp_path, capsys):
    """
    run_pipeline() SHALL print progress messages at the start and end of
    each pipeline step (load, extract, balance, split, train, evaluate, save).

    We mock the heavy I/O steps (load_data, extract_features) so the test
    runs quickly without real dataset files.

    Validates: Requirements 10.3
    """
    n_classes = 3
    samples_per_class = 10
    X, y = _make_tiny_dataset(n_classes=n_classes, samples_per_class=samples_per_class)
    labels = ["A", "B", "C"]

    # Synthetic "samples" list — just enough for load_data to return something
    fake_samples = [(np.zeros((4, 4, 3), dtype=np.float32), lbl)
                    for lbl in labels for _ in range(samples_per_class)]

    args = Namespace(
        asl_dir="/fake/asl",
        mnist_csv=None,
        isl_dir=None,
        output_dir=str(tmp_path),
        n_estimators=10,
        max_depth=5,
        no_augment=True,
    )

    with patch("backend.ml.train_pipeline.load_data", return_value=fake_samples), \
         patch("backend.ml.train_pipeline.extract_features", return_value=(X, y, labels)):
        run_pipeline(args)

    captured = capsys.readouterr()
    stdout = captured.out

    # Steps 3–6 are executed directly by run_pipeline (steps 1 & 2 are inside
    # the mocked load_data / extract_features functions and won't print).
    for step in range(3, 7):
        assert f"Step {step}/6" in stdout, (
            f"Expected '[pipeline] Step {step}/6' progress message in stdout"
        )

    # Splitting step is announced separately (not a numbered step)
    assert "Splitting" in stdout or "split" in stdout.lower(), (
        "Expected a splitting progress message in stdout"
    )

    # Final completion message
    assert "Pipeline complete" in stdout, (
        "Expected a 'Pipeline complete' completion message in stdout"
    )


# ---------------------------------------------------------------------------
# Req 8.3 / 7.1 — Label consistency check raises ValueError on mismatch
# ---------------------------------------------------------------------------

def test_label_consistency_check_raises_on_mismatch(tmp_path):
    """
    run_pipeline() SHALL raise a ValueError when model.classes_ does not
    match list(range(len(labels))), and SHALL NOT save any artifacts.

    Validates: Requirements 8.3, 7.1
    """
    n_classes = 3
    samples_per_class = 10
    X, y = _make_tiny_dataset(n_classes=n_classes, samples_per_class=samples_per_class)
    labels = ["A", "B", "C"]

    fake_samples = [(np.zeros((4, 4, 3), dtype=np.float32), lbl)
                    for lbl in labels for _ in range(samples_per_class)]

    args = Namespace(
        asl_dir="/fake/asl",
        mnist_csv=None,
        isl_dir=None,
        output_dir=str(tmp_path),
        n_estimators=10,
        max_depth=5,
        no_augment=True,
    )

    # Build a mock model whose classes_ do NOT match [0, 1, 2]
    bad_model = MagicMock(spec=RandomForestClassifier)
    bad_model.classes_ = np.array([1, 2, 3])  # shifted — misaligned

    with patch("backend.ml.train_pipeline.load_data", return_value=fake_samples), \
         patch("backend.ml.train_pipeline.extract_features", return_value=(X, y, labels)), \
         patch("backend.ml.train_pipeline.train", return_value=bad_model):
        with pytest.raises(ValueError, match="Label consistency check failed"):
            run_pipeline(args)

    # No model.pkl should have been written
    assert not os.path.exists(os.path.join(str(tmp_path), "model.pkl")), (
        "model.pkl should NOT be saved when label consistency check fails"
    )


def test_label_consistency_check_passes_on_aligned_model(tmp_path):
    """
    run_pipeline() SHALL NOT raise when model.classes_ matches
    list(range(len(labels))).

    Validates: Requirements 8.3, 7.1
    """
    n_classes = 3
    samples_per_class = 10
    X, y = _make_tiny_dataset(n_classes=n_classes, samples_per_class=samples_per_class)
    labels = ["A", "B", "C"]

    fake_samples = [(np.zeros((4, 4, 3), dtype=np.float32), lbl)
                    for lbl in labels for _ in range(samples_per_class)]

    args = Namespace(
        asl_dir="/fake/asl",
        mnist_csv=None,
        isl_dir=None,
        output_dir=str(tmp_path),
        n_estimators=10,
        max_depth=5,
        no_augment=True,
    )

    with patch("backend.ml.train_pipeline.load_data", return_value=fake_samples), \
         patch("backend.ml.train_pipeline.extract_features", return_value=(X, y, labels)):
        # Should complete without raising
        run_pipeline(args)

    assert os.path.exists(os.path.join(str(tmp_path), "model.pkl")), (
        "model.pkl should be saved when label consistency check passes"
    )
