"""
Property-based tests for dataset_loader.

Feature: ml-pipeline-upgrade
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st
from PIL import Image

# Ensure backend package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.ml.dataset_loader import load_asl_dataset, preprocess_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_png(path: str, array: np.ndarray) -> None:
    """Save a numpy array as a PNG file using PIL."""
    img = Image.fromarray(array)
    img.save(path)


def _make_rgb_image(h: int, w: int) -> np.ndarray:
    """Create a random uint8 RGB image of shape (h, w, 3)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Property 1: Dataset loader returns correctly typed tuples
# Validates: Requirements 1.5, 2.2, 2.3
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 1: Dataset loader returns correctly typed tuples
@given(
    n_classes=st.integers(1, 4),
    n_images=st.integers(1, 3),
    img_h=st.integers(4, 32),
    img_w=st.integers(4, 32),
)
@settings(max_examples=30)
def test_property1_loader_returns_correctly_typed_tuples(
    n_classes: int, n_images: int, img_h: int, img_w: int
) -> None:
    """
    For any ASL dataset directory with synthetic images, every item returned
    by load_asl_dataset is a tuple of (np.ndarray with dtype float32 and 3
    channels, str).
    """
    with tempfile.TemporaryDirectory() as root_dir:
        # Create class sub-folders with synthetic PNG images
        for cls_idx in range(n_classes):
            cls_name = chr(ord("A") + cls_idx)
            cls_dir = os.path.join(root_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            for img_idx in range(n_images):
                img_array = _make_rgb_image(img_h, img_w)
                img_path = os.path.join(cls_dir, f"img_{img_idx}.png")
                _write_png(img_path, img_array)

        samples = load_asl_dataset(root_dir, target_size=(64, 64))

    assert len(samples) == n_classes * n_images, (
        f"Expected {n_classes * n_images} samples, got {len(samples)}"
    )

    for i, item in enumerate(samples):
        assert isinstance(item, tuple) and len(item) == 2, (
            f"Item {i} is not a 2-tuple: {type(item)}"
        )
        arr, label = item

        assert isinstance(arr, np.ndarray), f"Item {i}: image is not ndarray"
        assert arr.dtype == np.float32, f"Item {i}: dtype is {arr.dtype}, expected float32"
        assert arr.ndim == 3, f"Item {i}: ndim is {arr.ndim}, expected 3"
        assert arr.shape[2] == 3, f"Item {i}: channels is {arr.shape[2]}, expected 3"
        assert isinstance(label, str), f"Item {i}: label is {type(label)}, expected str"
        assert len(label) > 0, f"Item {i}: label is empty string"


# ---------------------------------------------------------------------------
# Property 2: Image preprocessing produces configured output shape
# Validates: Requirements 2.1, 2.2, 2.3
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 2: Image preprocessing produces configured output shape
@given(
    target_h=st.integers(1, 512),
    target_w=st.integers(1, 512),
    src_h=st.integers(1, 64),
    src_w=st.integers(1, 64),
    channels=st.integers(1, 4),
)
@settings(max_examples=100)
def test_property2_preprocess_output_shape_and_range(
    target_h: int, target_w: int, src_h: int, src_w: int, channels: int
) -> None:
    """
    For any input image of any size and channel count, preprocess_image
    returns shape (target_h, target_w, 3) with all values in [0.0, 1.0].
    """
    rng = np.random.default_rng(0)

    if channels == 1:
        # Grayscale (H, W) — no channel dim
        image = rng.integers(0, 256, size=(src_h, src_w), dtype=np.uint8)
    else:
        image = rng.integers(0, 256, size=(src_h, src_w, channels), dtype=np.uint8)

    result = preprocess_image(image, target_size=(target_h, target_w))

    assert result.shape == (target_h, target_w, 3), (
        f"Expected shape ({target_h}, {target_w}, 3), got {result.shape}"
    )
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert float(result.min()) >= 0.0, f"Min value {result.min()} < 0.0"
    assert float(result.max()) <= 1.0, f"Max value {result.max()} > 1.0"


# ---------------------------------------------------------------------------
# Property 3: Loader skips unreadable files without raising
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------

# Feature: ml-pipeline-upgrade, Property 3: Loader skips unreadable files without raising
@given(
    n_valid=st.integers(1, 5),
    n_invalid=st.integers(1, 5),
)
@settings(max_examples=30)
def test_property3_loader_skips_unreadable_files(
    n_valid: int, n_invalid: int
) -> None:
    """
    When a dataset directory contains a mix of valid images and corrupt/invalid
    files, load_asl_dataset returns only valid items and does not raise.
    """
    rng = np.random.default_rng(7)

    with tempfile.TemporaryDirectory() as root_dir:
        cls_dir = os.path.join(root_dir, "A")
        os.makedirs(cls_dir, exist_ok=True)

        # Write valid PNG images
        for i in range(n_valid):
            img_array = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
            img_path = os.path.join(cls_dir, f"valid_{i}.png")
            _write_png(img_path, img_array)

        # Write corrupt files with .png extension (random bytes)
        for i in range(n_invalid):
            corrupt_path = os.path.join(cls_dir, f"corrupt_{i}.png")
            with open(corrupt_path, "wb") as f:
                f.write(rng.bytes(64))

        # Must not raise
        samples = load_asl_dataset(root_dir, target_size=(32, 32))

    # Only valid images should be returned
    assert len(samples) == n_valid, (
        f"Expected {n_valid} valid samples, got {len(samples)}"
    )

    # All returned items must be well-formed
    for arr, label in samples:
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (32, 32, 3)
        assert isinstance(label, str)
