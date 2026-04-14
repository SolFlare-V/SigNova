"""
Integration test for the end-to-end ML pipeline.

Validates: Requirements 8.2

Covers:
  - Load synthetic dataset via load_asl_dataset
  - extract_features → balance_dataset → stratified split → train → evaluate → save_artifacts
  - Load saved model.pkl and labels.pkl with ModelLoader
  - GestureService.predict returns a valid GesturePrediction
"""

import base64
import os
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import pytest

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Also add backend/ so that `app.*` imports resolve to the same modules used
# by gesture_service.py (which imports `from app.schemas import ...`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.ml.dataset_loader import load_asl_dataset
from backend.ml.train_pipeline import (
    balance,
    evaluate,
    extract_features,
    save_artifacts,
    train,
)
from backend.ml.balancer import balance_dataset
from sklearn.model_selection import train_test_split

from backend.app.services.model_loader import ModelLoader
from backend.app.services.gesture_service import GestureService
from app.schemas import GesturePrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_dataset(root_dir: str, n_classes: int = 5, images_per_class: int = 10) -> None:
    """Create a temp directory with class sub-folders, each containing synthetic PNG images."""
    class_names = ["A", "B", "C", "D", "E"][:n_classes]
    rng = np.random.default_rng(seed=42)

    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(images_per_class):
            # Random 64x64 RGB image
            img_array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            img_path = os.path.join(cls_dir, f"img_{i:03d}.png")
            cv2.imwrite(img_path, img_array)


def _make_valid_landmarks() -> list[float]:
    """Return a valid 63-element landmark feature vector."""
    rng = np.random.default_rng(seed=7)
    return rng.uniform(-1.0, 1.0, size=63).tolist()


def _encode_synthetic_image() -> str:
    """Create a synthetic BGR image and return it as a base64-encoded JPEG string."""
    rng = np.random.default_rng(seed=99)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline():
    """
    Full pipeline smoke test:
      1. Create synthetic dataset (5 classes × 10 images).
      2. Load with load_asl_dataset.
      3. extract_features (mocked — MediaPipe won't detect hands in synthetic images).
      4. balance_dataset → stratified split → train → evaluate → save_artifacts.
      5. Load artifacts with ModelLoader; verify is_loaded.
      6. GestureService.predict with mocked extract_landmarks; verify GesturePrediction.

    Validates: Requirements 8.2
    """
    with tempfile.TemporaryDirectory() as dataset_dir, \
         tempfile.TemporaryDirectory() as model_dir:

        # ------------------------------------------------------------------ #
        # Step 1 — Create synthetic dataset on disk                           #
        # ------------------------------------------------------------------ #
        _create_synthetic_dataset(dataset_dir, n_classes=5, images_per_class=10)

        # ------------------------------------------------------------------ #
        # Step 2 — Load dataset                                               #
        # ------------------------------------------------------------------ #
        samples = load_asl_dataset(dataset_dir, target_size=(224, 224))
        assert len(samples) == 50, f"Expected 50 samples, got {len(samples)}"

        # ------------------------------------------------------------------ #
        # Step 3 — Extract features (mock FeatureExtractor to bypass MediaPipe)
        # ------------------------------------------------------------------ #
        valid_features = _make_valid_landmarks()

        with patch(
            "backend.ml.train_pipeline.FeatureExtractor.extract_features",
            return_value=valid_features,
        ):
            X, y, labels = extract_features(samples, augment=False)

        assert X.shape == (50, 63), f"Expected X shape (50, 63), got {X.shape}"
        assert y.shape == (50,), f"Expected y shape (50,), got {y.shape}"
        assert len(labels) == 5, f"Expected 5 labels, got {len(labels)}"
        assert set(labels) == {"A", "B", "C", "D", "E"}

        # ------------------------------------------------------------------ #
        # Step 4 — Balance                                                    #
        # ------------------------------------------------------------------ #
        X_bal, y_bal = balance_dataset(X, y)
        # Each class should have the same count
        unique, counts = np.unique(y_bal, return_counts=True)
        assert len(set(counts.tolist())) == 1, "All classes should have equal sample counts after balancing"

        # ------------------------------------------------------------------ #
        # Step 5 — Stratified split                                           #
        # ------------------------------------------------------------------ #
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
        )
        assert len(X_train) > 0
        assert len(X_test) > 0

        # ------------------------------------------------------------------ #
        # Step 6 — Train                                                      #
        # ------------------------------------------------------------------ #
        model = train(X_train, y_train, n_estimators=10, max_depth=5)
        assert model is not None

        # ------------------------------------------------------------------ #
        # Step 7 — Evaluate                                                   #
        # ------------------------------------------------------------------ #
        accuracy = evaluate(model, X_test, y_test, labels)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

        # ------------------------------------------------------------------ #
        # Step 8 — Save artifacts                                             #
        # ------------------------------------------------------------------ #
        save_artifacts(model, labels, model_dir)
        model_path = os.path.join(model_dir, "model.pkl")
        labels_path = os.path.join(model_dir, "labels.pkl")
        assert os.path.exists(model_path), "model.pkl should exist after save_artifacts"
        assert os.path.exists(labels_path), "labels.pkl should exist after save_artifacts"

        # ------------------------------------------------------------------ #
        # Step 9 — Load with ModelLoader                                      #
        # ------------------------------------------------------------------ #
        # Reset singleton state so we get a fresh load
        loader = ModelLoader()
        loader._model = None
        loader._labels = None

        model_ok = loader.load_model(model_path)
        labels_ok = loader.load_labels(labels_path)

        assert model_ok, "ModelLoader.load_model should return True"
        assert labels_ok, "ModelLoader.load_labels should return True"
        assert loader.is_loaded, "ModelLoader.is_loaded should be True after loading both files"
        assert loader.labels == labels

        # ------------------------------------------------------------------ #
        # Step 10 — GestureService.predict with mocked extract_landmarks      #
        # ------------------------------------------------------------------ #
        service = GestureService()
        image_b64 = _encode_synthetic_image()

        landmark_array = np.array(_make_valid_landmarks(), dtype=np.float32)

        with patch.object(service, "extract_landmarks", return_value=landmark_array):
            result = service.predict(image_b64, loader.model, loader.labels)

        assert isinstance(result, GesturePrediction), (
            f"Expected GesturePrediction, got {type(result)}"
        )
        assert isinstance(result.gesture, str) and len(result.gesture) > 0, (
            "gesture should be a non-empty string"
        )
        assert isinstance(result.confidence, float), "confidence should be a float"
        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence should be in [0, 1], got {result.confidence}"
        )
        assert result.landmarks_detected is True, (
            "landmarks_detected should be True when landmarks are provided"
        )
        # The gesture is either one of the trained labels or "nothing" (when
        # confidence is below the threshold, the service stores "nothing" in
        # the smoothing buffer per Req 10a.4).
        valid_gestures = set(labels) | {"nothing"}
        assert result.gesture in valid_gestures, (
            f"gesture '{result.gesture}' should be one of {valid_gestures}"
        )
