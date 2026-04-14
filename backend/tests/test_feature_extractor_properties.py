"""
Property-based tests for FeatureExtractor.

Feature: ml-pipeline-upgrade
"""

import math
import sys
import os

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ml.feature_extractor import FeatureExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Landmark:
    """Minimal stand-in for a MediaPipe NormalizedLandmark."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def _make_landmarks(coords: list[float]) -> list[_Landmark]:
    """Convert a flat list of 63 floats into 21 _Landmark objects."""
    assert len(coords) == 63
    return [
        _Landmark(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2])
        for i in range(21)
    ]


# Strategy: 63 finite floats in [-1, 1] (one per coordinate of 21 landmarks)
landmark_coords = st.lists(
    st.floats(-1, 1, allow_nan=False, allow_infinity=False),
    min_size=63,
    max_size=63,
)

# Strategy: positive scalar for scale-invariance test
positive_scalar = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Property 4: Feature extractor output is exactly 63 elements
# Validates: Requirements 3.1, 3.3
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 4: Feature extractor output is exactly 63 elements
@given(coords=landmark_coords)
@settings(max_examples=200)
def test_property4_output_length_is_63(coords: list[float]) -> None:
    """For any 21 valid landmarks, extract_features returns exactly 63 floats."""
    landmarks = _make_landmarks(coords)
    result = FeatureExtractor.extract_features(landmarks)

    assert result is not None, "Expected a list, got None"
    assert len(result) == 63, f"Expected 63 elements, got {len(result)}"
    assert all(isinstance(v, float) for v in result), "All values must be floats"


# ---------------------------------------------------------------------------
# Property 5: Wrist is always at the origin after normalisation
# Validates: Requirements 3a.1
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 5: Wrist is always at the origin after normalisation
@given(coords=landmark_coords)
@settings(max_examples=200)
def test_property5_wrist_at_origin(coords: list[float]) -> None:
    """After normalisation, output[0], output[1], output[2] are all 0.0."""
    landmarks = _make_landmarks(coords)
    result = FeatureExtractor.extract_features(landmarks)

    assert result is not None
    assert math.isclose(result[0], 0.0, abs_tol=1e-9), f"wrist x = {result[0]}"
    assert math.isclose(result[1], 0.0, abs_tol=1e-9), f"wrist y = {result[1]}"
    assert math.isclose(result[2], 0.0, abs_tol=1e-9), f"wrist z = {result[2]}"


# ---------------------------------------------------------------------------
# Property 6: Feature vector is scale-invariant
# Validates: Requirements 3a.2
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 6: Feature vector is scale-invariant
@given(coords=landmark_coords, k=positive_scalar)
@settings(max_examples=200)
def test_property6_scale_invariance(coords: list[float], k: float) -> None:
    """Scaling all landmark coordinates by k produces an identical output vector."""
    landmarks_base = _make_landmarks(coords)
    landmarks_scaled = _make_landmarks([c * k for c in coords])

    result_base = FeatureExtractor.extract_features(landmarks_base)
    result_scaled = FeatureExtractor.extract_features(landmarks_scaled)

    assert result_base is not None
    assert result_scaled is not None
    assert len(result_base) == len(result_scaled) == 63

    for i, (a, b) in enumerate(zip(result_base, result_scaled)):
        assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-9), (
            f"Scale invariance violated at index {i}: base={a}, scaled={b} (k={k})"
        )
