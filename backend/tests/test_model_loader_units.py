"""
Unit tests for model_loader.py.

Covers:
  - test_model_loader_logs_success_failure  (Req 8.1)
  - test_model_loader_loads_both_files      (Req 8.1)
"""

import os
import pickle
import tempfile

import pytest

from backend.app.services.model_loader import ModelLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_loader() -> ModelLoader:
    """Return a ModelLoader with cleared singleton state for test isolation."""
    loader = ModelLoader()
    loader._model = None
    loader._labels = None
    return loader


def _write_pkl(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# Req 8.1 — log success or failure for each file
# ---------------------------------------------------------------------------

def test_model_loader_logs_success_failure(caplog, tmp_path):
    """
    Verify that load_model and load_labels emit the correct log messages
    on success (file exists) and on failure (file missing).

    Validates: Requirements 8.1
    """
    loader = _fresh_loader()

    model_path = str(tmp_path / "model.pkl")
    labels_path = str(tmp_path / "labels.pkl")

    # Write valid pickle files
    _write_pkl(model_path, {"dummy": "model"})
    _write_pkl(labels_path, ["A", "B", "C"])

    # --- success paths ---
    with caplog.at_level("INFO", logger="backend.app.services.model_loader"):
        result_model = loader.load_model(model_path)
        result_labels = loader.load_labels(labels_path)

    assert result_model is True
    assert result_labels is True
    assert any("Model loaded successfully" in r.message for r in caplog.records), (
        "Expected success log for model load"
    )
    assert any("Labels loaded successfully" in r.message for r in caplog.records), (
        "Expected success log for labels load"
    )

    caplog.clear()
    loader._model = None
    loader._labels = None

    # --- failure paths (non-existent files) ---
    missing_model = str(tmp_path / "no_model.pkl")
    missing_labels = str(tmp_path / "no_labels.pkl")

    with caplog.at_level("WARNING", logger="backend.app.services.model_loader"):
        result_model_missing = loader.load_model(missing_model)
        result_labels_missing = loader.load_labels(missing_labels)

    assert result_model_missing is False
    assert result_labels_missing is False
    assert any("not found" in r.message for r in caplog.records), (
        "Expected warning log when model file is missing"
    )
    assert any("not found" in r.message for r in caplog.records), (
        "Expected warning log when labels file is missing"
    )


# ---------------------------------------------------------------------------
# Req 8.1 — both model.pkl and labels.pkl are loaded at startup
# ---------------------------------------------------------------------------

def test_model_loader_loads_both_files(tmp_path):
    """
    Verify that is_loaded is True only after both model.pkl and labels.pkl
    are loaded, and that the model/labels properties return the correct values.

    Validates: Requirements 8.1
    """
    loader = _fresh_loader()

    model_obj = {"type": "RandomForest", "version": 1}
    labels_obj = ["A", "B", "C", "D"]

    model_path = str(tmp_path / "model.pkl")
    labels_path = str(tmp_path / "labels.pkl")

    _write_pkl(model_path, model_obj)
    _write_pkl(labels_path, labels_obj)

    # Before loading anything, is_loaded should be False
    assert loader.is_loaded is False

    # After loading only the model, is_loaded should still be False
    loader.load_model(model_path)
    assert loader.is_loaded is False

    # After loading labels too, is_loaded should be True
    loader.load_labels(labels_path)
    assert loader.is_loaded is True

    # Properties should return the correct deserialized values
    assert loader.model == model_obj
    assert loader.labels == labels_obj
