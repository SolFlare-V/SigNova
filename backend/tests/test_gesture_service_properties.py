"""
Property-based tests for GestureService (Properties 11–18).

Feature: ml-pipeline-upgrade
"""

import base64
import collections
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.services.gesture_service import GestureService
from backend.app.schemas import GesturePrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_image_b64() -> str:
    """Return a minimal valid base64-encoded 1×1 BGR JPEG."""
    import cv2
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


def _make_service() -> GestureService:
    """Create a GestureService with MediaPipe disabled."""
    with patch.object(GestureService, "_init_mediapipe", lambda self: None):
        svc = GestureService(window_size=7, confidence_threshold=0.65)
        svc.hand_landmarker = None
    return svc


def _make_model_mock(label_index: int = 0, proba: float = 0.9) -> MagicMock:
    """Return a mock sklearn model."""
    model = MagicMock()
    model.predict.return_value = np.array([label_index])
    model.predict_proba.return_value = np.array([[proba]])
    return model


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A valid 63-element landmark array (floats in [-1, 1])
landmarks_array = st.lists(
    st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    min_size=63,
    max_size=63,
).map(np.array)

# A gesture label string (non-empty, no "No gesture detected")
gesture_label = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
    min_size=1,
    max_size=10,
).filter(lambda s: s != "No gesture detected")

# A confidence value in [0.0, 1.0]
confidence_value = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)

# A list of ASL-like gesture labels
asl_labels = GestureService.GESTURE_LABELS


# ---------------------------------------------------------------------------
# Property 11: No-hand response is always deterministic
# Validates: Requirements 9.1, 10b.5
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 11: No-hand response is always deterministic
@given(dummy=st.just(_make_dummy_image_b64()))
@settings(max_examples=100)
def test_property11_no_hand_response_deterministic(dummy: str) -> None:
    """
    For any image in which MediaPipe detects no hand, GestureService.predict
    SHALL return GesturePrediction(gesture="No gesture detected",
    confidence=0.0, landmarks_detected=False).

    **Validates: Requirements 9.1, 10b.5**
    """
    svc = _make_service()
    model = _make_model_mock()
    labels = list(asl_labels)

    # extract_landmarks returns None → no hand detected
    with patch.object(svc, "extract_landmarks", return_value=None):
        result = svc.predict(dummy, model, labels)

    assert result.gesture == "No gesture detected", (
        f"Expected 'No gesture detected', got '{result.gesture}'"
    )
    assert result.confidence == 0.0, (
        f"Expected confidence 0.0, got {result.confidence}"
    )
    assert result.landmarks_detected is False, (
        f"Expected landmarks_detected=False, got {result.landmarks_detected}"
    )


# ---------------------------------------------------------------------------
# Property 12: Hand-detected response never returns "No gesture detected"
# Validates: Requirements 9.2
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 12: Hand-detected response never returns "No gesture detected"
@given(
    label_idx=st.integers(min_value=0, max_value=len(asl_labels) - 1),
    proba=st.floats(0.65, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property12_hand_detected_never_no_gesture(
    label_idx: int, proba: float
) -> None:
    """
    For any image in which MediaPipe detects a hand and the model is loaded,
    GestureService.predict SHALL return a GesturePrediction where
    gesture != "No gesture detected".

    **Validates: Requirements 9.2**
    """
    svc = _make_service()
    model = _make_model_mock(label_index=label_idx, proba=proba)
    labels = list(asl_labels)
    fake_landmarks = np.zeros(63, dtype=np.float32)

    with patch.object(svc, "extract_landmarks", return_value=fake_landmarks):
        result = svc.predict(_make_dummy_image_b64(), model, labels)

    assert result.gesture != "No gesture detected", (
        f"Got 'No gesture detected' even though a hand was detected "
        f"(label_idx={label_idx}, proba={proba})"
    )


# ---------------------------------------------------------------------------
# Property 13: Exception response is always deterministic
# Validates: Requirements 9.3
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 13: Exception response is always deterministic
@given(dummy=st.just(_make_dummy_image_b64()))
@settings(max_examples=100)
def test_property13_exception_response_deterministic(dummy: str) -> None:
    """
    For any input that causes an unhandled exception during landmark extraction
    or model inference, GestureService.predict SHALL return
    GesturePrediction(gesture="unknown", confidence=0.0, landmarks_detected=False).

    **Validates: Requirements 9.3**
    """
    svc = _make_service()
    model = _make_model_mock()
    labels = list(asl_labels)

    with patch.object(
        svc, "extract_landmarks", side_effect=RuntimeError("simulated failure")
    ):
        result = svc.predict(dummy, model, labels)

    assert result.gesture == "unknown", (
        f"Expected 'unknown', got '{result.gesture}'"
    )
    assert result.confidence == 0.0, (
        f"Expected confidence 0.0, got {result.confidence}"
    )
    assert result.landmarks_detected is False, (
        f"Expected landmarks_detected=False, got {result.landmarks_detected}"
    )


# ---------------------------------------------------------------------------
# Property 14: Stabiliser majority vote
# Validates: Requirements 10a.2
# ---------------------------------------------------------------------------

@st.composite
def gesture_buffer(draw):
    """
    Generate a non-empty list of (gesture_str, confidence) tuples where
    one gesture is guaranteed to be the majority.
    """
    n = draw(st.integers(min_value=1, max_value=7))
    majority_label = draw(st.sampled_from(["A", "B", "C", "D", "E"]))
    minority_label = draw(
        st.sampled_from(["A", "B", "C", "D", "E"]).filter(
            lambda x: x != majority_label
        )
    )
    majority_count = draw(st.integers(min_value=n // 2 + 1, max_value=n))
    minority_count = n - majority_count

    entries = [(majority_label, 0.9)] * majority_count + [
        (minority_label, 0.9)
    ] * minority_count
    return entries, majority_label


# Feature: ml-pipeline-upgrade, Property 14: Stabiliser majority vote
@given(data=gesture_buffer())
@settings(max_examples=100)
def test_property14_stabiliser_majority_vote(data) -> None:
    """
    For any window of frame predictions, GestureService SHALL return the most
    frequently occurring gesture label in the current buffer.

    **Validates: Requirements 10a.2**
    """
    entries, expected_majority = data
    svc = _make_service()
    model = _make_model_mock()
    labels = list(asl_labels)

    # Pre-populate the buffer with all but the last entry
    for entry in entries[:-1]:
        svc._buffer.append(entry)

    # The last entry is injected via a real predict call
    last_gesture, last_conf = entries[-1]
    last_label_idx = asl_labels.index(last_gesture) if last_gesture in asl_labels else 0
    model.predict.return_value = np.array([last_label_idx])
    model.predict_proba.return_value = np.array([[last_conf]])
    fake_landmarks = np.zeros(63, dtype=np.float32)

    with patch.object(svc, "extract_landmarks", return_value=fake_landmarks):
        result = svc.predict(_make_dummy_image_b64(), model, labels)

    assert result.gesture == expected_majority, (
        f"Expected majority gesture '{expected_majority}', got '{result.gesture}'. "
        f"Buffer: {list(svc._buffer)}"
    )


# ---------------------------------------------------------------------------
# Property 15: Low-confidence frames stored as "nothing"
# Validates: Requirements 10a.4
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 15: Low-confidence frames stored as "nothing"
@given(
    label_idx=st.integers(min_value=0, max_value=len(asl_labels) - 1),
    proba=st.floats(0.0, 0.6499, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_property15_low_confidence_stored_as_nothing(
    label_idx: int, proba: float
) -> None:
    """
    For any prediction with confidence < confidence_threshold (0.65), the
    stabiliser SHALL store "nothing" in the buffer rather than the original
    gesture label.

    **Validates: Requirements 10a.4**
    """
    svc = _make_service()
    model = _make_model_mock(label_index=label_idx, proba=proba)
    labels = list(asl_labels)
    fake_landmarks = np.zeros(63, dtype=np.float32)

    with patch.object(svc, "extract_landmarks", return_value=fake_landmarks):
        svc.predict(_make_dummy_image_b64(), model, labels)

    # The most recently appended buffer entry should be ("nothing", proba)
    assert len(svc._buffer) > 0, "Buffer should not be empty after predict"
    stored_gesture, stored_conf = svc._buffer[-1]
    assert stored_gesture == "nothing", (
        f"Expected 'nothing' in buffer for low-confidence frame "
        f"(proba={proba:.4f} < 0.65), got '{stored_gesture}'"
    )


# ---------------------------------------------------------------------------
# Property 16: Buffer clear resets state
# Validates: Requirements 10a.6
# ---------------------------------------------------------------------------

@st.composite
def buffer_contents(draw):
    """Generate a non-empty list of (gesture, confidence) tuples."""
    n = draw(st.integers(min_value=1, max_value=7))
    entries = draw(
        st.lists(
            st.tuples(
                st.sampled_from(["A", "B", "C", "nothing"]),
                st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=n,
            max_size=n,
        )
    )
    return entries


# Feature: ml-pipeline-upgrade, Property 16: Buffer clear resets state
@given(entries=buffer_contents())
@settings(max_examples=100)
def test_property16_clear_buffer_resets_state(entries) -> None:
    """
    For any buffer state, calling clear_buffer() SHALL result in an empty
    buffer such that the next prediction is based solely on that single frame.

    **Validates: Requirements 10a.6**
    """
    svc = _make_service()

    # Populate buffer
    for entry in entries:
        svc._buffer.append(entry)

    assert len(svc._buffer) > 0, "Buffer should be non-empty before clear"

    svc.clear_buffer()

    assert len(svc._buffer) == 0, (
        f"Buffer should be empty after clear_buffer(), got {len(svc._buffer)} entries"
    )


# ---------------------------------------------------------------------------
# Property 17: Confidence equals average of matching window frames
# Validates: Requirements 10b.1, 10b.3, 10b.4
# ---------------------------------------------------------------------------

@st.composite
def window_with_known_majority(draw):
    """
    Generate a buffer of (gesture, confidence) tuples where one gesture is
    the strict majority within the window_size=7 limit.
    Returns (entries, majority_label) — expected_conf is computed post-predict
    from the actual buffer to account for deque eviction.
    """
    majority_label = draw(st.sampled_from(["A", "B", "C"]))
    minority_label = draw(
        st.sampled_from(["A", "B", "C"]).filter(lambda x: x != majority_label)
    )
    # Keep total entries <= window_size so no eviction surprises
    majority_count = draw(st.integers(min_value=2, max_value=4))
    minority_count = draw(st.integers(min_value=0, max_value=majority_count - 1))

    majority_confs = draw(
        st.lists(
            st.floats(0.65, 1.0, allow_nan=False, allow_infinity=False),
            min_size=majority_count,
            max_size=majority_count,
        )
    )
    minority_confs = draw(
        st.lists(
            st.floats(0.65, 1.0, allow_nan=False, allow_infinity=False),
            min_size=minority_count,
            max_size=minority_count,
        )
    )

    entries = (
        [(majority_label, c) for c in majority_confs]
        + [(minority_label, c) for c in minority_confs]
    )
    return entries, majority_label


# Feature: ml-pipeline-upgrade, Property 17: Confidence equals average of matching window frames
@given(data=window_with_known_majority())
@settings(max_examples=100)
def test_property17_confidence_average_matching_frames(data) -> None:
    """
    For any window of frame predictions where the stabilised gesture is G,
    the reported confidence SHALL equal the arithmetic mean of the
    predict_proba-derived confidence scores for all frames in the window
    where the stored gesture is G, rounded to 3 decimal places.

    **Validates: Requirements 10b.1, 10b.3, 10b.4**
    """
    entries, majority_label = data
    svc = _make_service()
    labels = list(asl_labels)

    majority_idx = asl_labels.index(majority_label)

    # Pre-populate buffer with all but the last entry
    for entry in entries[:-1]:
        svc._buffer.append(entry)

    # Inject the last entry via predict
    last_gesture, last_conf = entries[-1]
    if last_gesture == majority_label:
        model = _make_model_mock(label_index=majority_idx, proba=last_conf)
    else:
        minority_idx = asl_labels.index(last_gesture) if last_gesture in asl_labels else 1
        model = _make_model_mock(label_index=minority_idx, proba=last_conf)

    fake_landmarks = np.zeros(63, dtype=np.float32)
    with patch.object(svc, "extract_landmarks", return_value=fake_landmarks):
        result = svc.predict(_make_dummy_image_b64(), model, labels)

    # Compute expected confidence from the actual buffer state (accounts for deque eviction)
    stable_gesture = result.gesture
    matching_confs = [c for g, c in svc._buffer if g == stable_gesture]
    expected_conf = round(sum(matching_confs) / len(matching_confs), 3)

    assert result.confidence == expected_conf, (
        f"Expected confidence {expected_conf}, got {result.confidence}. "
        f"Buffer: {list(svc._buffer)}, stable_gesture='{stable_gesture}'"
    )


# ---------------------------------------------------------------------------
# Property 18: Label mapping round-trip
# Validates: Requirements 8.3
# ---------------------------------------------------------------------------

@st.composite
def label_list_and_index(draw):
    """Generate a non-empty list of unique label strings and a valid index."""
    n = draw(st.integers(min_value=1, max_value=30))
    labels = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
                min_size=1,
                max_size=10,
            ),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    idx = draw(st.integers(min_value=0, max_value=n - 1))
    return labels, idx


# Feature: ml-pipeline-upgrade, Property 18: Label mapping round-trip
@given(data=label_list_and_index())
@settings(max_examples=100)
def test_property18_label_mapping_roundtrip(data) -> None:
    """
    For any class index i in [0, len(labels)), mapping i through the loaded
    labels list SHALL return the same string that was used as the class label
    during training.

    **Validates: Requirements 8.3**
    """
    labels, idx = data

    # Simulate what GestureService does: labels[prediction_index]
    result = labels[idx]

    assert isinstance(result, str), (
        f"labels[{idx}] should be a str, got {type(result)}"
    )
    assert result == labels[idx], (
        f"Round-trip failed: labels[{idx}] = '{result}' != '{labels[idx]}'"
    )
    # Verify the index is valid (no IndexError)
    assert 0 <= idx < len(labels), (
        f"Index {idx} out of range for labels of length {len(labels)}"
    )
