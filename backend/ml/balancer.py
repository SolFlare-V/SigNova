"""
Dataset balancing module for the SignLang AI ML training pipeline.

Equalises sample counts across gesture classes by combining random
undersampling (majority classes) and random oversampling with replacement
(minority classes) to reach a configurable target count per class.

IMPORTANT: This module operates on extracted feature vectors (numpy arrays),
not raw images. It must be called after feature extraction and before the
train/test split.
"""

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    target_per_class: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance a feature dataset so every class has the same number of samples.

    For each class:
    - If the class count exceeds *target_per_class*, randomly undersample
      (without replacement) down to the target.
    - If the class count is below *target_per_class*, randomly oversample
      (with replacement) up to the target.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(N, F)`` — float32 numpy array where N is
        the total number of samples and F is the feature dimension (63 for
        this pipeline).
    y:
        Label array of shape ``(N,)`` containing integer class indices
        corresponding to each row of *X*.
    target_per_class:
        Desired number of samples per class after balancing. When ``None``
        (default), the median class count of the unbalanced dataset is used.

    Returns
    -------
    X_balanced : np.ndarray
        Balanced feature matrix of shape ``(n_classes * target_per_class, F)``.
    y_balanced : np.ndarray
        Balanced label array of shape ``(n_classes * target_per_class,)``.

    Raises
    ------
    ValueError
        If *X* and *y* have different numbers of rows, or if *X* is empty.

    Notes
    -----
    - The output is shuffled so classes are not contiguous.
    - Per-class counts are logged at INFO level before and after balancing.
    - A fixed ``random.seed`` is NOT set here; reproducibility is the
      caller's responsibility (e.g. set ``random_state`` in the pipeline).
    """
    if X.shape[0] == 0:
        logger.warning("[balancer] Empty dataset received — returning as-is.")
        return X, y

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"[balancer] X has {X.shape[0]} rows but y has {y.shape[0]} elements — "
            "they must match."
        )

    # --- Group indices by class -------------------------------------------
    classes = np.unique(y)
    class_indices: dict[int, list[int]] = {
        int(cls): list(np.where(y == cls)[0]) for cls in classes
    }

    # --- Log counts before balancing --------------------------------------
    counts_before = {cls: len(idxs) for cls, idxs in class_indices.items()}
    logger.info("[balancer] Class counts BEFORE balancing: %s", counts_before)
    print(f"[balancer] Class counts BEFORE: {counts_before}")

    # --- Determine target -------------------------------------------------
    if target_per_class is None:
        median_count = int(np.median([len(idxs) for idxs in class_indices.values()]))
        target_per_class = median_count
        logger.info("[balancer] target_per_class defaulted to median = %d", target_per_class)

    # --- Resample each class to target ------------------------------------
    balanced_indices: list[int] = []

    for cls, idxs in class_indices.items():
        n = len(idxs)
        if n >= target_per_class:
            # Undersample: random subset without replacement
            selected = random.sample(idxs, target_per_class)
        else:
            # Oversample: keep all originals, then sample with replacement
            extra_needed = target_per_class - n
            oversampled = random.choices(idxs, k=extra_needed)
            selected = idxs + oversampled

        balanced_indices.extend(selected)

    # --- Shuffle to avoid class-contiguous ordering -----------------------
    random.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    # --- Log counts after balancing ---------------------------------------
    counts_after = {
        int(cls): int(np.sum(y_balanced == cls)) for cls in classes
    }
    logger.info("[balancer] Class counts AFTER balancing: %s", counts_after)
    print(f"[balancer] Class counts AFTER:  {counts_after}")
    print(
        f"[balancer] {X.shape[0]} samples → {X_balanced.shape[0]} samples "
        f"({len(classes)} classes × {target_per_class})"
    )

    return X_balanced, y_balanced
