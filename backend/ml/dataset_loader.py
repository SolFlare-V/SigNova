"""
Dataset loading module for the SignLang AI ML pipeline.

Provides functions to load and preprocess images from three dataset sources:
  - ASL Alphabet (folder-based, one sub-folder per class)
  - Sign Language MNIST (CSV format, 28×28 grayscale pixel rows)
  - ISL (Indian Sign Language, same folder-based format as ASL)

All loaders return a unified ``list[tuple[np.ndarray, str]]`` where each
array is float32 with shape ``(H, W, 3)`` and pixel values in ``[0.0, 1.0]``.
"""

import csv
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported image file extensions (lower-cased for case-insensitive matching)
_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Mapping from integer label (0–25) to letter string ('A'–'Z')
_MNIST_LABEL_MAP: dict[int, str] = {i: chr(ord("A") + i) for i in range(26)}


def preprocess_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize and normalise a single image for use in the ML pipeline.

    Steps applied in order:
      1. If the image has fewer than 3 colour channels (e.g. grayscale or
         single-channel), convert it to a 3-channel BGR/RGB array.
      2. Resize to ``target_size`` using bilinear interpolation.
      3. Normalise pixel values to ``[0.0, 1.0]`` by dividing by 255.

    Parameters
    ----------
    image:
        Input image as a numpy array.  Accepted shapes:
        ``(H, W)`` (grayscale), ``(H, W, 1)`` (single-channel),
        ``(H, W, 3)`` (BGR/RGB), or ``(H, W, 4)`` (BGRA/RGBA).
    target_size:
        ``(height, width)`` of the output image.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(target_size[0], target_size[1], 3)``
        with all values in ``[0.0, 1.0]``.
    """
    # --- Step 1: ensure 3 channels ---
    if image.ndim == 2:
        # Grayscale (H, W) → (H, W, 3)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        # Single-channel (H, W, 1) → (H, W, 3)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        # BGRA (H, W, 4) → (H, W, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] != 3:
        # Any other channel count (e.g. 2) → convert first channel to 3-channel
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    # If already (H, W, 3), no conversion needed.

    # --- Step 2: resize to target_size (height, width) ---
    # cv2.resize expects (width, height)
    resized = cv2.resize(
        image,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # --- Step 3: normalise to [0.0, 1.0] float32 ---
    return (resized / 255.0).astype(np.float32)


def _scan_folder(
    root_dir: str,
    target_size: tuple[int, int],
) -> list[tuple[np.ndarray, str]]:
    """
    Recursively scan *root_dir* and return preprocessed images with labels.

    The immediate sub-folder name under *root_dir* is used as the label
    string for all images found within that sub-folder (and its descendants).

    Only files with extensions ``.jpg``, ``.jpeg``, or ``.png``
    (case-insensitive) are considered.  Files that cannot be decoded by
    OpenCV are skipped and a warning is logged with the file path.

    Parameters
    ----------
    root_dir:
        Path to the dataset root directory.  Each direct child directory
        is treated as a class label.
    target_size:
        ``(height, width)`` passed to :func:`preprocess_image`.

    Returns
    -------
    list[tuple[np.ndarray, str]]
        Each element is ``(preprocessed_image, label_string)`` where the
        image is a float32 array of shape
        ``(target_size[0], target_size[1], 3)``.
    """
    samples: list[tuple[np.ndarray, str]] = []

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        # Derive label from the immediate sub-folder name relative to root_dir.
        # Files directly inside root_dir (depth 0) are ignored — they have no
        # class sub-folder.
        rel = os.path.relpath(dirpath, root_dir)
        if rel == ".":
            # We are at the root level; no label available — skip files here.
            continue

        # Use only the top-level sub-folder name as the label so that nested
        # directories (e.g. root/A/sub/) still map to label "A".
        label = rel.split(os.sep)[0]

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                continue

            filepath = os.path.join(dirpath, filename)
            img = cv2.imread(filepath)

            if img is None:
                logger.warning(filepath)
                continue

            preprocessed = preprocess_image(img, target_size)
            samples.append((preprocessed, label))

    return samples


def load_asl_dataset(
    root_dir: str,
    target_size: tuple[int, int] = (224, 224),
) -> list[tuple[np.ndarray, str]]:
    """
    Load the ASL Alphabet dataset from a folder-based directory structure.

    Each direct sub-folder of *root_dir* is treated as a class label
    (e.g. ``root_dir/A/``, ``root_dir/B/``, …).  Supported image formats
    are ``.jpg``, ``.jpeg``, and ``.png``.

    Parameters
    ----------
    root_dir:
        Path to the ASL dataset root directory.
    target_size:
        ``(height, width)`` for resizing.  Defaults to ``(224, 224)``.

    Returns
    -------
    list[tuple[np.ndarray, str]]
        Each element is ``(preprocessed_image, label_string)``.
    """
    samples = _scan_folder(root_dir, target_size)
    logger.info("ASL dataset: loaded %d samples from '%s'", len(samples), root_dir)
    return samples


def load_isl_dataset(
    root_dir: str,
    target_size: tuple[int, int] = (224, 224),
) -> list[tuple[np.ndarray, str]]:
    """
    Load the ISL (Indian Sign Language) dataset from a folder-based structure.

    Uses the same :func:`_scan_folder` logic as :func:`load_asl_dataset`.
    Each direct sub-folder of *root_dir* is treated as a class label.

    Parameters
    ----------
    root_dir:
        Path to the ISL dataset root directory.
    target_size:
        ``(height, width)`` for resizing.  Defaults to ``(224, 224)``.

    Returns
    -------
    list[tuple[np.ndarray, str]]
        Each element is ``(preprocessed_image, label_string)``.
    """
    samples = _scan_folder(root_dir, target_size)
    logger.info("ISL dataset: loaded %d samples from '%s'", len(samples), root_dir)
    return samples


def load_mnist_dataset(
    csv_path: str,
    target_size: tuple[int, int] = (224, 224),
) -> list[tuple[np.ndarray, str]]:
    """
    Load the Sign Language MNIST dataset from a CSV file.

    Expected CSV format::

        label,pixel1,pixel2,...,pixel784
        3,107,118,...,255
        ...

    The first row is a header and is skipped.  Each subsequent row contains
    an integer label (0–25, mapping to letters A–Z) followed by 784 pixel
    values representing a 28×28 grayscale image.

    Rows that cannot be parsed (wrong column count, non-integer values, etc.)
    are skipped and a warning is logged.

    Parameters
    ----------
    csv_path:
        Path to the Sign Language MNIST CSV file.
    target_size:
        ``(height, width)`` for resizing.  Defaults to ``(224, 224)``.

    Returns
    -------
    list[tuple[np.ndarray, str]]
        Each element is ``(preprocessed_image, label_string)`` where the
        image has been converted from 28×28 grayscale to 3-channel RGB and
        resized to *target_size*.
    """
    samples: list[tuple[np.ndarray, str]] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)

        # Skip header row
        try:
            next(reader)
        except StopIteration:
            logger.warning("MNIST CSV '%s' is empty.", csv_path)
            return samples

        for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
            try:
                if len(row) != 785:
                    raise ValueError(
                        f"Expected 785 columns, got {len(row)}"
                    )

                int_label = int(row[0])
                pixels = np.array([int(p) for p in row[1:]], dtype=np.uint8)

            except (ValueError, IndexError) as exc:
                logger.warning(
                    "MNIST CSV '%s' row %d skipped: %s", csv_path, row_num, exc
                )
                continue

            # Map integer label to letter string
            label_str = _MNIST_LABEL_MAP.get(int_label, str(int_label))

            # Reshape to (28, 28, 1) grayscale
            gray_image = pixels.reshape(28, 28, 1)

            # preprocess_image handles channel conversion and resize
            preprocessed = preprocess_image(gray_image, target_size)
            samples.append((preprocessed, label_str))

    logger.info("MNIST dataset: loaded %d samples from '%s'", len(samples), csv_path)
    return samples
