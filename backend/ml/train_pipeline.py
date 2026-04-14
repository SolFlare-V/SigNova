"""
Training pipeline for the SignLang AI gesture recognition model.

Orchestrates the full end-to-end training workflow:
  load → augment → extract features → balance → split → train → evaluate → save

Usage
-----
    python -m backend.ml.train_pipeline --asl-dir /data/asl
    python -m backend.ml.train_pipeline --mnist-csv /data/sign_mnist_train.csv
    python -m backend.ml.train_pipeline --asl-dir /data/asl --isl-dir /data/isl --no-augment

Run with no dataset arguments to see the usage message.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter
from argparse import Namespace
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from backend.ml.augmentor import augment_image
from backend.ml.balancer import balance_dataset
from backend.ml.dataset_loader import load_asl_dataset, load_isl_dataset, load_mnist_dataset
from backend.ml.feature_extractor import FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> Namespace:
    """
    Parse command-line arguments for the training pipeline.

    Parameters
    ----------
    argv:
        Argument list to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: ``asl_dir``, ``mnist_csv``,
        ``isl_dir``, ``output_dir``, ``n_estimators``, ``max_depth``,
        ``no_augment``.
    """
    parser = argparse.ArgumentParser(
        prog="train_pipeline",
        description="Train a Random Forest gesture classifier from dataset images.",
    )
    parser.add_argument("--asl-dir", type=str, default=None, help="Path to ASL Alphabet dataset root directory.")
    parser.add_argument("--mnist-csv", type=str, default=None, help="Path to Sign Language MNIST CSV file.")
    parser.add_argument("--isl-dir", type=str, default=None, help="Path to ISL dataset root directory.")
    parser.add_argument("--output-dir", type=str, default="backend/ml/models", help="Directory to save model artifacts.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees in the Random Forest.")
    parser.add_argument("--max-depth", type=int, default=20, help="Maximum depth of each tree.")
    parser.add_argument("--no-augment", action="store_true", help="Disable image augmentation during feature extraction.")

    args = parser.parse_args(argv)

    # Require at least one dataset source
    if not any([args.asl_dir, args.mnist_csv, args.isl_dir]):
        parser.print_usage(sys.stderr)
        print(
            "\nerror: at least one dataset argument is required: "
            "--asl-dir, --mnist-csv, or --isl-dir",
            file=sys.stderr,
        )
        sys.exit(1)

    return args


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def load_data(args: Namespace) -> list[tuple[np.ndarray, str]]:
    """
    Load raw images from all configured dataset sources.

    Parameters
    ----------
    args:
        Parsed CLI arguments. Sources are loaded when their path arg is set.

    Returns
    -------
    list[tuple[np.ndarray, str]]
        Combined list of ``(image, label_string)`` tuples from all sources.
    """
    t0 = time.time()
    print("\n[pipeline] Step 1/6 — Loading data...")

    samples: list[tuple[np.ndarray, str]] = []

    if args.asl_dir:
        asl = load_asl_dataset(args.asl_dir)
        print(f"  ASL:   {len(asl):>7,} samples")
        samples.extend(asl)

    if args.mnist_csv:
        mnist = load_mnist_dataset(args.mnist_csv)
        print(f"  MNIST: {len(mnist):>7,} samples")
        samples.extend(mnist)

    if args.isl_dir:
        isl = load_isl_dataset(args.isl_dir)
        print(f"  ISL:   {len(isl):>7,} samples")
        samples.extend(isl)

    elapsed = time.time() - t0
    print(f"[pipeline] Loaded {len(samples):,} total samples in {elapsed:.1f}s")
    return samples


def extract_features(
    samples: list[tuple[np.ndarray, str]],
    augment: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract 63-element landmark feature vectors from raw images.

    For each ``(image, label)`` pair:
      1. Optionally apply ``augment_image`` when *augment* is ``True``.
      2. Run ``FeatureExtractor.extract_features`` via MediaPipe.
      3. Skip the sample when the extractor returns ``None`` (no hand detected).

    Parameters
    ----------
    samples:
        List of ``(image, label_string)`` tuples from :func:`load_data`.
    augment:
        When ``True``, apply a random geometric augmentation to each image
        before landmark extraction.

    Returns
    -------
    X : np.ndarray
        Float32 feature matrix of shape ``(N, 63)``.
    y : np.ndarray
        Integer label array of shape ``(N,)``.
    labels : list[str]
        Ordered list of unique label strings where ``labels[i]`` corresponds
        to class index ``i`` in *y*.
    """
    t0 = time.time()
    print(f"\n[pipeline] Step 2/6 — Extracting features (augment={augment})...")

    # Build a stable label → index mapping from the full sample list
    unique_labels: list[str] = sorted({label for _, label in samples})
    label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}

    X_rows: list[list[float]] = []
    y_vals: list[int] = []
    skipped = 0

    for image, label in samples:
        img = augment_image(image) if augment else image
        features = FeatureExtractor.extract_features(img)
        if features is None:
            skipped += 1
            continue
        X_rows.append(features)
        y_vals.append(label_to_idx[label])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.int32)

    elapsed = time.time() - t0
    print(f"[pipeline] Extracted {len(X):,} feature vectors ({skipped:,} skipped — no hand detected) in {elapsed:.1f}s")
    return X, y, unique_labels


def balance(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance the feature dataset so every class has the same sample count.

    Delegates to :func:`~backend.ml.balancer.balance_dataset` using the
    default median-based target.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(N, 63)``.
    y:
        Integer label array of shape ``(N,)``.

    Returns
    -------
    X_balanced : np.ndarray
    y_balanced : np.ndarray
    """
    t0 = time.time()
    print("\n[pipeline] Step 3/6 — Balancing classes...")
    X_bal, y_bal = balance_dataset(X, y)
    elapsed = time.time() - t0
    print(f"[pipeline] Balanced to {len(X_bal):,} samples in {elapsed:.1f}s")
    return X_bal, y_bal


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    max_depth: int,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the provided feature matrix.

    Parameters
    ----------
    X_train:
        Training feature matrix of shape ``(N, 63)``.
    y_train:
        Training label array of shape ``(N,)``.
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum depth of each tree.

    Returns
    -------
    RandomForestClassifier
        Fitted scikit-learn model.
    """
    t0 = time.time()
    print(f"\n[pipeline] Step 4/6 — Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[pipeline] Training complete in {elapsed:.1f}s")
    return model


def evaluate(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> float:
    """
    Evaluate the trained model on the held-out test set and print a report.

    Prints overall accuracy, a per-class classification report, and a
    confusion matrix.  Prints a warning when accuracy is below 85%.

    Parameters
    ----------
    model:
        Fitted ``RandomForestClassifier``.
    X_test:
        Test feature matrix of shape ``(M, 63)``.
    y_test:
        Test label array of shape ``(M,)``.
    label_names:
        Ordered list of label strings (index matches class integer in *y_test*).

    Returns
    -------
    float
        Overall accuracy on the test set.
    """
    t0 = time.time()
    print("\n[pipeline] Step 5/6 — Evaluating model...")

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    print(f"\n  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Per-class report — use only the classes present in y_test
    present_classes = sorted(np.unique(y_test).tolist())
    present_names = [label_names[i] for i in present_classes]
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=present_names))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    print(cm)

    if accuracy < 0.85:
        print(
            f"\n  WARNING: accuracy {accuracy:.4f} is below the 85% target. "
            "Consider more data, tuning hyperparameters, or additional augmentation."
        )

    elapsed = time.time() - t0
    print(f"\n[pipeline] Evaluation complete in {elapsed:.1f}s")
    return accuracy


def save_artifacts(
    model: RandomForestClassifier,
    labels: list[str],
    output_dir: str,
) -> None:
    """
    Serialise the trained model and label list to disk.

    Creates *output_dir* if it does not exist.  Logs a message when
    existing files are overwritten.

    Parameters
    ----------
    model:
        Fitted ``RandomForestClassifier`` to serialise.
    labels:
        Ordered list of label strings to serialise alongside the model.
    output_dir:
        Directory path where ``model.pkl`` and ``labels.pkl`` are written.
    """
    t0 = time.time()
    print(f"\n[pipeline] Step 6/6 — Saving artifacts to '{output_dir}'...")

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.pkl")
    labels_path = os.path.join(output_dir, "labels.pkl")

    for path, obj, name in [
        (model_path, model, "model.pkl"),
        (labels_path, labels, "labels.pkl"),
    ]:
        if os.path.exists(path):
            logger.info("[pipeline] Overwriting existing file: %s", path)
            print(f"  Overwriting {name}")
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        print(f"  Saved {name} → {path}")

    elapsed = time.time() - t0
    print(f"[pipeline] Artifacts saved in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(args: Namespace) -> None:
    """
    Execute the full training pipeline end-to-end.

    Steps in order:
      1. Load raw images from configured dataset sources.
      2. Extract 63-element feature vectors (with optional augmentation).
      3. Balance classes to the median count.
      4. Stratified 80/20 train/test split (``random_state=42``).
      5. Train ``RandomForestClassifier``.
      6. Evaluate on the test set.
      7. Save ``model.pkl`` and ``labels.pkl``.

    Parameters
    ----------
    args:
        Parsed CLI arguments from :func:`parse_args`.
    """
    t_total = time.time()
    augment = not args.no_augment

    # 1. Load
    samples = load_data(args)
    if not samples:
        print("error: no samples loaded — check dataset paths.", file=sys.stderr)
        sys.exit(1)

    # 2. Extract features
    X, y, labels = extract_features(samples, augment=augment)
    if len(X) == 0:
        print("error: no feature vectors extracted — MediaPipe found no hands.", file=sys.stderr)
        sys.exit(1)
    print(f"Valid samples after feature extraction: {len(X)}")

    # 3. Balance
    X, y = balance(X, y)
    print("Balanced class distribution:", Counter(y.tolist()))

    # 4. Stratified split
    print("\n[pipeline] Splitting dataset (80% train / 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 5. Train
    model = train(X_train, y_train, args.n_estimators, args.max_depth)

    # Label consistency check — ensure model.classes_ aligns with the labels list
    if list(model.classes_) != list(range(len(labels))):
        raise ValueError(
            f"Label consistency check failed: model.classes_ {list(model.classes_)} "
            f"does not match expected indices {list(range(len(labels)))}. "
            "The model was not saved to prevent misaligned predictions."
        )

    # 6. Evaluate
    accuracy = evaluate(model, X_test, y_test, labels)
    if accuracy < 0.85:
        print("WARNING: Accuracy below expected threshold (85%)")

    # Save training metrics summary
    metrics = {
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": float(accuracy),
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[pipeline] Metrics saved → {metrics_path}")

    # 7. Save
    save_artifacts(model, labels, args.output_dir)

    total_elapsed = time.time() - t_total
    print(f"\n[pipeline] Pipeline complete in {total_elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    run_pipeline(parse_args())
