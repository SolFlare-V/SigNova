"""
Training Script for Sign Language Gesture Recognition Model

Usage:
    python ml/train_model.py --collect    Collect training data via webcam
    python ml/train_model.py --train      Train model from collected data
    python ml/train_model.py --demo       Generate demo model with synthetic data
"""

import os
import sys
import pickle
import argparse
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

# Fix paths if run from backend dir directly
if os.path.exists("app"):
    sys.path.append(os.path.abspath("."))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

GESTURE_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing'
]


def collect_data():
    """Collect training data using webcam and MediaPipe."""
    try:
        from ml.feature_extractor import FeatureExtractor
        from ml.dataset_utils import DatasetAugmenter
    except ImportError as e:
        print(f"❌ Error importing dependencies: {e}")
        return
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Use DirectShow backend (MSMF is broken on this machine)
    # and set MJPG codec to prevent garbled raw pixel output
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("=" * 50)
    print("Sign Language Data Collection")
    print("=" * 50)
    print(f"Gestures to collect: {', '.join(GESTURE_LABELS[:26])}")
    print("Press 'n' for next gesture, 'q' to quit, SPACE to capture")
    print("=" * 50)
    
    all_data = []
    all_labels = []
    
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python
        
        model_path = os.path.join(MODEL_DIR, 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"❌ Failed to load tasks API: {e}")
        return
    
    for idx, label in enumerate(GESTURE_LABELS[:26]):
        print(f"\n→ Show gesture: '{label}' (Press SPACE to capture, 'n' to skip)")
        samples = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Warning: Failed to grab frame. Exiting current letter.")
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hand_landmarker.detect(mp_image)
            
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    for lm in hand_landmarks:
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            cv2.putText(frame, f"Gesture: {label} | Samples: {samples}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and results.hand_landmarks:
                features = FeatureExtractor.extract_features(results.hand_landmarks[0])
                if len(features) > 0:
                    all_data.append(features)
                    all_labels.append(idx)
                    samples += 1
                    print(f"  ✅ Captured sample {samples} for '{label}'")
                    
                    jittered = DatasetAugmenter.add_jitter(features)
                    all_data.append(jittered)
                    all_labels.append(idx)
                    
            elif key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                _save_data(all_data, all_labels)
                return
    
    cap.release()
    cv2.destroyAllWindows()
    _save_data(all_data, all_labels)


def _save_data(data, labels):
    if len(data) > 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(os.path.join(DATA_DIR, "features.npy"), np.array(data))
        np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(labels))
        print(f"\n✅ Saved {len(data)} samples to {DATA_DIR}")
    else:
        print("\n⚠️ No data collected")


def train_model():
    """Train advanced pipeline (SVM/MLP/RF) from collected data."""
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, accuracy_score
    import warnings
    warnings.filterwarnings('ignore')
    
    features_path = os.path.join(DATA_DIR, "features.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")
    
    if not os.path.exists(features_path):
        print("❌ No training data found. Run with --collect first, or use --demo")
        return
    
    X = np.load(features_path)
    y = np.load(labels_path)
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Classes: {len(set(y))}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🔄 Evaluating base algorithms...")
    # Pipeline: Normalize -> Train
    # Using an SVM with RBF kernel is extremely effective for geometric relations and angles
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42, cache_size=500))
    ])
    
    # GridSearch for Hyperparameter Tuning
    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__gamma': ['scale', 'auto'],
        'clf__kernel': ['rbf', 'poly']
    }
    
    print("🔄 Running GridSearchCV (this may take a moment)...")
    cv = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Best parameters found: {grid_search.best_params_}")
    
    # Evaluation
    print("🔄 Evaluating on test set...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Final Model Accuracy: {accuracy:.4f}")
    target_names = [GESTURE_LABELS[i] for i in sorted(set(y))]
    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")
    
    _save_model(best_model)


def generate_demo_model():
    """Generate a high-dimensional demo model suited for 83 features."""
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    print("🔄 Generating advanced demo model with synthetic geometric data...")
    
    np.random.seed(42)
    n_samples_per_class = 50
    n_features = 83  # NEW feature set size
    n_classes = len(GESTURE_LABELS)
    
    X_all = []
    y_all = []
    
    for class_idx in range(n_classes):
        base_pattern = np.random.randn(n_features) * 2
        for _ in range(n_samples_per_class):
            sample = base_pattern + np.random.randn(n_features) * 0.5
            X_all.append(sample)
            y_all.append(class_idx)
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, kernel='rbf', C=1.0))
    ])
    
    model.fit(X, y)
    
    print(f"✅ Demo model trained on {len(X)} synthetic samples")
    _save_model(model)


def _save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "gesture_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Model Training")
    parser.add_argument("--collect", action="store_true", help="Collect training data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--demo", action="store_true", help="Generate demo model")
    
    args = parser.parse_args()
    
    if args.collect:
        collect_data()
    elif args.train:
        train_model()
    elif args.demo:
        generate_demo_model()
    else:
        print("Usage:")
        print("  python train_model.py --collect   Collect training data via webcam")
        print("  python train_model.py --train     Train model from collected data")
        print("  python train_model.py --demo      Generate demo model with synthetic data")
