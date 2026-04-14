"""
Data augmentation module for the SignLang AI ML training pipeline.

Applies image-level transformations (horizontal flip, small rotation, brightness
adjustment) to training images before landmark extraction. Augmentation is
intentionally mild to preserve gesture structure for MediaPipe detection.

IMPORTANT: This module is for training only. Never call it on validation,
test, or live inference images.
"""

import random
from typing import List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def flip_image(image: np.ndarray) -> np.ndarray:
    """
    Flip an image horizontally (left-right).

    Parameters
    ----------
    image:
        Float32 numpy array of shape (H, W, 3) with values in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Horizontally flipped image with the same shape and dtype.
    """
    return cv2.flip(image, 1)


def rotate_image(
    image: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rotate an image by the given angle around its centre.

    Border pixels are filled by replicating the nearest edge pixel to avoid
    introducing black borders that could confuse MediaPipe.

    Parameters
    ----------
    image:
        Float32 numpy array of shape (H, W, 3) with values in [0.0, 1.0].
    angle:
        Rotation angle in degrees. Positive values rotate counter-clockwise.

    Returns
    -------
    np.ndarray
        Rotated image with the same shape and dtype as the input.
    """
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def adjust_brightness(
    image: np.ndarray,
    factor: float,
) -> np.ndarray:
    """
    Scale pixel values by *factor* and clip the result to [0.0, 1.0].

    Parameters
    ----------
    image:
        Float32 numpy array of shape (H, W, 3) with values in [0.0, 1.0].
    factor:
        Multiplicative brightness factor. Values > 1.0 brighten; < 1.0 darken.

    Returns
    -------
    np.ndarray
        Brightness-adjusted image with the same shape and dtype, clipped to
        [0.0, 1.0].
    """
    adjusted = image * factor
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Core augmentation function
# ---------------------------------------------------------------------------


def augment_image(
    image: np.ndarray,
    flip_prob: float = 0.5,
    rotation_range: Tuple[float, float] = (-10.0, 10.0),
    brightness_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    Apply a single augmentation to a training image.

    Exactly one transform is chosen and applied:
      - With probability *flip_prob*, apply a horizontal flip.
      - Otherwise, apply a random rotation sampled from *rotation_range*.

    Brightness is NOT chained here; use ``augment_dataset`` (or call
    ``adjust_brightness`` directly) to produce a separate brightness variant.
    Keeping geometric and photometric augmentations separate prevents
    compounding distortions that could break MediaPipe landmark detection.

    Parameters
    ----------
    image:
        Float32 numpy array of shape (H, W, 3) with values in [0.0, 1.0].
    flip_prob:
        Probability of choosing a horizontal flip over rotation. Default 0.5.
    rotation_range:
        (min_angle, max_angle) in degrees used when rotation is chosen.
        Default (-10.0, 10.0).
    brightness_range:
        Kept for API compatibility but unused in this function. Brightness
        variants are generated independently in ``augment_dataset``.

    Returns
    -------
    np.ndarray
        Augmented float32 image with the same spatial dimensions as the input.

    Notes
    -----
    This function must only be called on training images. Validation and test
    images must bypass augmentation entirely.
    """
    if random.random() < flip_prob:
        return flip_image(image.copy())
    else:
        angle = random.uniform(rotation_range[0], rotation_range[1])
        return rotate_image(image.copy(), angle)


# ---------------------------------------------------------------------------
# Dataset-level augmentation
# ---------------------------------------------------------------------------


def augment_dataset(
    images: List[np.ndarray],
    labels: List[str],
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Augment a list of training images and return an expanded dataset.

    Strategy (per input image):
      - Always keep the original image.
      - With ~80% probability, generate ONE geometric variant:
          flip OR rotation (chosen randomly, never both).
      - With ~80% probability, generate ONE brightness variant:
          brightness scaling only, no geometric change.

    Geometric and photometric augmentations are kept separate to prevent
    compounding distortions that could degrade MediaPipe landmark detection.
    The dataset grows by at most 3× per image.

    Parameters
    ----------
    images:
        List of float32 numpy arrays, each of shape (H, W, 3) with values in
        [0.0, 1.0].
    labels:
        List of label strings corresponding to each image. Must have the same
        length as *images*.

    Returns
    -------
    augmented_images : List[np.ndarray]
        Original images followed by their augmented variants.
    augmented_labels : List[str]
        Labels aligned with *augmented_images*. Each label matches the class
        of the original image it was derived from.

    Notes
    -----
    - Empty input returns empty lists immediately.
    - Images that are ``None`` or lack the expected shape are skipped with a
      warning printed to stdout.
    - The function prints original vs. augmented counts for quick verification.
    """
    if not images:
        return [], []

    augmented_images: List[np.ndarray] = []
    augmented_labels: List[str] = []

    for idx, (img, label) in enumerate(zip(images, labels)):
        # Validate image
        if img is None:
            print(f"[augmentor] WARNING: image at index {idx} is None — skipping.")
            continue
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            print(
                f"[augmentor] WARNING: image at index {idx} has unexpected shape "
                f"{getattr(img, 'shape', 'unknown')} — skipping."
            )
            continue

        # Keep original
        augmented_images.append(img)
        augmented_labels.append(label)

        # Geometric variant (~80% chance): flip OR rotation, never both
        if random.random() < 0.8:
            if random.random() < 0.5:
                geo = flip_image(img)
            else:
                angle = random.uniform(-10.0, 10.0)
                geo = rotate_image(img, angle)
            augmented_images.append(geo)
            augmented_labels.append(label)

        # Brightness variant (~80% chance): photometric only, no geometry
        if random.random() < 0.8:
            factor = random.uniform(0.8, 1.2)
            bright = adjust_brightness(img, factor)
            augmented_images.append(bright)
            augmented_labels.append(label)

    original_count = len(images)
    augmented_count = len(augmented_images)
    print(
        f"[augmentor] original_count={original_count} → augmented_count={augmented_count} "
        f"(×{augmented_count / original_count:.1f})"
    )

    assert len(augmented_images) == len(augmented_labels), (
        "Label/image count mismatch after augmentation — this is a bug."
    )

    return augmented_images, augmented_labels
