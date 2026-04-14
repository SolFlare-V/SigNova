"""
Unit tests for GestureService (Task 7.5).

Tests:
  - test_mock_fallback_when_no_model  (Req 8.4)
  - test_no_predict_proba_confidence_zero  (Req 10b.2)
"""

import base64
import os
import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.services.gesture_service import GestureService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_image_b64() -> str:
    """Return a minimal valid base64-encoded 10×10 BGR JPEG."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


def _make_service() -> GestureService:
    """Create a GestureService with MediaPipe disabled."""
    with patch.object(GestureService, "_init_mediapipe", lambda self: None):
        svc = GestureService(window_size=7, confidence_threshold=0.65)
        svc.hand_landmarker = None
    return svc


VALID_LANDMARKS = np.zeros(63, dtype=np.float32)
ASL_LABELS = list(GestureService.GESTURE_LABELS)


# ---------------------------------------------------------------------------
# test_mock_fallback_when_no_model
# Validates: Requirement 8.4
# ---------------------------------------------------------------------------

def test_mock_fallback_when_no_model() -> None:
    """
    WHEN model=None is passed to predict() AND extract_landmarks returns a
    valid 63-element array (hand detected), THE GestureService SHALL fall back
    to mock prediction behaviour and return a valid GesturePrediction that is
    not "No gesture detected", not "unknown", and has landmarks_detected=True.

    Validates: Requirement 8.4
    """
    svc = _make_service()

    with patch.object(svc, "extract_landmarks", return_value=VALID_LANDMARKS):
        result = svc.predict(_make_dummy_image_b64(), model=None, labels=ASL_LABELS)

    # Check duck-type: result must have the GesturePrediction fields
    assert hasattr(result, "gesture") and hasattr(result, "confidence") and hasattr(result, "landmarks_detected"), (
        f"Expected GesturePrediction-like object, got {type(result)}"
    )
    assert result.gesture != "No gesture detected", (
        "Mock fallback should not return 'No gesture detected' when a hand is detected"
    )
    assert result.gesture != "unknown", (
        "Mock fallback should not return 'unknown'"
    )
    assert result.landmarks_detected is True, (
        f"Expected landmarks_detected=True, got {result.landmarks_detected}"
    )
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence {result.confidence} is out of [0.0, 1.0]"
    )


# ---------------------------------------------------------------------------
# test_no_predict_proba_confidence_zero
# Validates: Requirement 10b.2
# ---------------------------------------------------------------------------

def test_no_predict_proba_confidence_zero() -> None:
    """
    WHEN the loaded model does NOT expose a predict_proba method, THE
    GestureService SHALL set the raw confidence to 0.0 and the reported
    confidence in the GesturePrediction SHALL reflect that 0.0 value.

    Validates: Requirement 10b.2
    """
    svc = _make_service()

    # Build a mock model without predict_proba
    model = MagicMock(spec=[])  # spec=[] means no attributes/methods at all
    model.predict = MagicMock(return_value=np.array([0]))

    with patch.object(svc, "extract_landmarks", return_value=VALID_LANDMARKS):
        result = svc.predict(_make_dummy_image_b64(), model=model, labels=ASL_LABELS)

    assert hasattr(result, "gesture") and hasattr(result, "confidence") and hasattr(result, "landmarks_detected"), (
        f"Expected GesturePrediction-like object, got {type(result)}"
    )

    # The buffer entry should have been stored with confidence 0.0
    assert len(svc._buffer) > 0, "Buffer should not be empty after predict"
    stored_gesture, stored_conf = svc._buffer[-1]
    assert stored_conf == 0.0, (
        f"Expected buffer confidence 0.0 when predict_proba is absent, got {stored_conf}"
    )

    # The reported confidence is the mean of matching-frame confidences.
    # With a single frame at 0.0, the reported confidence must also be 0.0.
    assert result.confidence == 0.0, (
        f"Expected reported confidence 0.0 when predict_proba is absent, got {result.confidence}"
    )


# ---------------------------------------------------------------------------
# test_feature_vector_wrong_length_returns_unknown
# Validates: Requirements 3.3, 8.2
# ---------------------------------------------------------------------------

def test_feature_vector_wrong_length_returns_unknown() -> None:
    """
    WHEN extract_landmarks returns an array whose flattened length is NOT 63,
    THE GestureService SHALL log an error and return a GesturePrediction with
    gesture="unknown", confidence=0.0, and landmarks_detected=False.

    Validates: Requirements 3.3, 8.2
    """
    svc = _make_service()

    # Build a mock model (should never be called)
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([0]))

    # Return a landmark array with wrong length (e.g. 60 instead of 63)
    wrong_landmarks = np.zeros(60, dtype=np.float32)

    with patch.object(svc, "extract_landmarks", return_value=wrong_landmarks):
        result = svc.predict(_make_dummy_image_b64(), model=model, labels=ASL_LABELS)

    assert result.gesture == "unknown", (
        f"Expected gesture='unknown' for wrong-length feature vector, got '{result.gesture}'"
    )
    assert result.confidence == 0.0, (
        f"Expected confidence=0.0 for wrong-length feature vector, got {result.confidence}"
    )
    assert result.landmarks_detected is False, (
        f"Expected landmarks_detected=False for wrong-length feature vector, got {result.landmarks_detected}"
    )
    # model.predict should NOT have been called
    model.predict.assert_not_called()
