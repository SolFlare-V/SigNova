"""
Property-based tests for augmentor.

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

from backend.ml.augmentor import augment_image


# ---------------------------------------------------------------------------
# Property 7: Augmented images preserve shape and dtype
# Validates: Requirements 4.3, 4.5
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 7: Augmented images preserve shape and dtype
@given(
    height=st.integers(1, 64),
    width=st.integers(1, 64),
    channels=st.integers(1, 3),
)
@settings(max_examples=100)
def test_property7_augment_preserves_shape_dtype(
    height: int, width: int, channels: int
) -> None:
    """
    For any float32 image array, augment_image returns an array with the same
    spatial dimensions (height, width), dtype float32, and all pixel values
    in [0.0, 1.0].

    **Validates: Requirements 4.3, 4.5**
    """
    rng = np.random.default_rng(seed=height * 1000 + width * 10 + channels)

    # augment_image expects shape (H, W, 3); pad single/dual channel to 3
    raw = rng.random(size=(height, width, channels)).astype(np.float32)
    if channels < 3:
        # Repeat last channel to reach 3 channels
        pad = np.repeat(raw[:, :, -1:], 3 - channels, axis=2)
        image = np.concatenate([raw, pad], axis=2).astype(np.float32)
    else:
        image = raw

    result = augment_image(image)

    # Same spatial dimensions
    assert result.shape[0] == height, (
        f"Height changed: expected {height}, got {result.shape[0]}"
    )
    assert result.shape[1] == width, (
        f"Width changed: expected {width}, got {result.shape[1]}"
    )

    # dtype must be float32
    assert result.dtype == np.float32, (
        f"dtype changed: expected float32, got {result.dtype}"
    )

    # All values in [0.0, 1.0]
    assert float(result.min()) >= 0.0, (
        f"Min value {result.min()} is below 0.0"
    )
    assert float(result.max()) <= 1.0, (
        f"Max value {result.max()} is above 1.0"
    )
