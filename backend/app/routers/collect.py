"""
Data collection endpoint — captures landmark features from webcam frames
and saves them to disk for retraining.
"""
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collect", tags=["Data Collection"])

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml", "data")
LABELS = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# In-memory buffer — flushed to disk on save
_buffer_X: list[list[float]] = []
_buffer_y: list[int] = []


class CollectRequest(BaseModel):
    image_data: str
    label: str          # e.g. "A"


class CollectResponse(BaseModel):
    success: bool
    label: str
    samples_this_label: int
    total_samples: int
    message: str


@router.post("/sample", response_model=CollectResponse)
async def collect_sample(req: CollectRequest):
    """Extract landmarks from a frame and buffer them for training."""
    from app.services.gesture_service import GestureService
    from ml.feature_extractor import FeatureExtractor

    label = req.label.upper().strip()
    if label not in LABELS:
        return CollectResponse(success=False, label=label,
                               samples_this_label=0, total_samples=len(_buffer_y),
                               message=f"Unknown label '{label}'")

    label_idx = LABELS.index(label)

    # Reuse gesture_service decode + extract pipeline
    svc = GestureService.__new__(GestureService)
    svc.__dict__.update(GestureService().__dict__)   # fresh instance

    try:
        image = svc.decode_image(req.image_data)
        landmarks = svc.extract_landmarks(image)
    except Exception as e:
        return CollectResponse(success=False, label=label,
                               samples_this_label=0, total_samples=len(_buffer_y),
                               message=f"Detection failed: {e}")

    if landmarks is None:
        return CollectResponse(success=False, label=label,
                               samples_this_label=0, total_samples=len(_buffer_y),
                               message="No hand detected in frame")

    feats = FeatureExtractor.extract_features_83(landmarks)
    if feats is None:
        return CollectResponse(success=False, label=label,
                               samples_this_label=0, total_samples=len(_buffer_y),
                               message="Feature extraction failed")

    _buffer_X.append(feats)
    _buffer_y.append(label_idx)

    # Also add mild jitter augmentation
    jitter = np.array(feats) + np.random.normal(0, 0.01, len(feats))
    _buffer_X.append(jitter.tolist())
    _buffer_y.append(label_idx)

    count = _buffer_y.count(label_idx)
    return CollectResponse(success=True, label=label,
                           samples_this_label=count // 2,  # divide by 2 (jitter pairs)
                           total_samples=len(_buffer_y) // 2,
                           message=f"Captured sample for '{label}'")


@router.post("/save")
async def save_and_retrain():
    """Merge buffered samples with existing data and retrain the model."""
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    if len(_buffer_X) < 10:
        return {"success": False, "message": "Not enough samples (need at least 10)"}

    os.makedirs(DATA_DIR, exist_ok=True)

    new_X = np.array(_buffer_X, dtype=np.float32)
    new_y = np.array(_buffer_y, dtype=np.int32)

    # Merge with existing data if present
    feat_path  = os.path.join(DATA_DIR, "features.npy")
    label_path = os.path.join(DATA_DIR, "labels.npy")
    if os.path.exists(feat_path) and os.path.exists(label_path):
        old_X = np.load(feat_path).astype(np.float32)
        old_y = np.load(label_path).astype(np.int32)
        # Only keep old samples for classes NOT in the new buffer
        new_classes = set(new_y.tolist())
        keep = np.array([i for i, c in enumerate(old_y) if c not in new_classes])
        if len(keep) > 0:
            merged_X = np.vstack([old_X[keep], new_X])
            merged_y = np.concatenate([old_y[keep], new_y])
        else:
            merged_X, merged_y = new_X, new_y
        logger.info(f"Merged {len(keep)} old samples + {len(new_X)} new samples")
    else:
        merged_X, merged_y = new_X, new_y

    # Save merged dataset
    np.save(feat_path,  merged_X)
    np.save(label_path, merged_y)

    # Retrain
    clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                  random_state=42, n_jobs=-1)
    clf.fit(merged_X, merged_y)

    # Quick eval
    if len(merged_X) >= 10:
        X_tr, X_te, y_tr, y_te = train_test_split(
            merged_X, merged_y, test_size=0.2, random_state=42,
            stratify=merged_y if len(set(merged_y.tolist())) > 1 else None
        )
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
    else:
        clf.fit(merged_X, merged_y)
        acc = 1.0

    # Save model
    model_dir = os.path.join(os.path.dirname(DATA_DIR), "models")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "gesture_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(model_dir, "labels.pkl"), "wb") as f:
        pickle.dump(LABELS, f)

    # Reload the singleton
    from app.services.model_loader import ModelLoader
    ml = ModelLoader()
    ml._model  = clf
    ml._labels = LABELS

    # Clear buffer
    _buffer_X.clear()
    _buffer_y.clear()

    return {
        "success": True,
        "accuracy": round(float(acc), 4),
        "total_samples": len(merged_X),
        "message": f"Model retrained — accuracy {acc*100:.1f}%"
    }


@router.get("/status")
async def collection_status():
    """Return current buffer status."""
    from collections import Counter
    counts = Counter(_buffer_y)
    return {
        "total_buffered": len(_buffer_y) // 2,
        "per_label": {LABELS[k]: v // 2 for k, v in counts.items()},
        "labels": LABELS,
    }


@router.post("/clear")
async def clear_buffer():
    """Clear the in-memory buffer without saving."""
    _buffer_X.clear()
    _buffer_y.clear()
    return {"success": True, "message": "Buffer cleared"}
