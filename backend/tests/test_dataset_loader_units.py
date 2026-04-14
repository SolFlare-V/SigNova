"""
Unit tests for dataset_loader.py.

Covers:
  - test_isl_uses_same_loader_as_asl  (Req 1.3)
  - test_same_resolution_across_sources  (Req 2.4)
"""

import csv
import inspect
import os
import tempfile

import numpy as np
import pytest

from backend.ml.dataset_loader import (
    _scan_folder,
    load_asl_dataset,
    load_isl_dataset,
    load_mnist_dataset,
)


# ---------------------------------------------------------------------------
# Req 1.3 — ISL uses the same folder-scanning logic as ASL
# ---------------------------------------------------------------------------

def test_isl_uses_same_loader_as_asl():
    """
    Verify that load_isl_dataset and load_asl_dataset both delegate to
    _scan_folder, satisfying Req 1.3 (ISL loaded with same logic as ASL).

    We inspect the source of each function and confirm that '_scan_folder'
    is referenced in both bodies.
    """
    asl_src = inspect.getsource(load_asl_dataset)
    isl_src = inspect.getsource(load_isl_dataset)

    assert "_scan_folder" in asl_src, (
        "load_asl_dataset must call _scan_folder"
    )
    assert "_scan_folder" in isl_src, (
        "load_isl_dataset must call _scan_folder (Req 1.3)"
    )


# ---------------------------------------------------------------------------
# Req 2.4 — consistent target_size across all dataset sources
# ---------------------------------------------------------------------------

def _make_folder_dataset(root: str, label: str = "A", n: int = 2) -> None:
    """Create a minimal folder-based dataset under *root*."""
    import cv2

    class_dir = os.path.join(root, label)
    os.makedirs(class_dir, exist_ok=True)
    for i in range(n):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(class_dir, f"img_{i}.jpg"), img)


def _make_mnist_csv(path: str, n: int = 2) -> None:
    """Create a minimal Sign Language MNIST CSV file at *path*."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # header
        writer.writerow(["label"] + [f"pixel{i}" for i in range(1, 785)])
        for _ in range(n):
            row = [0] + [128] * 784  # label=0 → 'A', all pixels=128
            writer.writerow(row)


def test_same_resolution_across_sources():
    """
    Verify that all three loaders return images with the requested shape
    when the same target_size is supplied (Req 2.4).
    """
    target_size = (64, 64)
    expected_shape = (64, 64, 3)

    with tempfile.TemporaryDirectory() as asl_root, \
         tempfile.TemporaryDirectory() as isl_root, \
         tempfile.TemporaryDirectory() as mnist_dir:

        _make_folder_dataset(asl_root)
        _make_folder_dataset(isl_root)
        mnist_csv = os.path.join(mnist_dir, "mnist.csv")
        _make_mnist_csv(mnist_csv)

        asl_samples = load_asl_dataset(asl_root, target_size=target_size)
        isl_samples = load_isl_dataset(isl_root, target_size=target_size)
        mnist_samples = load_mnist_dataset(mnist_csv, target_size=target_size)

        assert len(asl_samples) > 0, "ASL loader returned no samples"
        assert len(isl_samples) > 0, "ISL loader returned no samples"
        assert len(mnist_samples) > 0, "MNIST loader returned no samples"

        for img, _ in asl_samples:
            assert img.shape == expected_shape, (
                f"ASL image shape {img.shape} != {expected_shape}"
            )

        for img, _ in isl_samples:
            assert img.shape == expected_shape, (
                f"ISL image shape {img.shape} != {expected_shape}"
            )

        for img, _ in mnist_samples:
            assert img.shape == expected_shape, (
                f"MNIST image shape {img.shape} != {expected_shape}"
            )
