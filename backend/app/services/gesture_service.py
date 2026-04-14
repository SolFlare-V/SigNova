"""Gesture recognition service."""
import base64
import collections
import numpy as np
import cv2
import logging
from typing import Optional, Tuple

from app.schemas import GesturePrediction

logger = logging.getLogger(__name__)


class GestureService:
    """Handles gesture recognition from image frames."""
    
    # ASL alphabet labels
    GESTURE_LABELS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing'
    ]
    
    def __init__(self, window_size: int = 3, confidence_threshold: float = 0.5):
        """Initialize gesture service."""
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.mp_hands = None
        self.hands = None
        self._init_mediapipe()

        # Inline deque-based buffer storing (gesture_str, confidence_float) tuples
        self._buffer: collections.deque = collections.deque(maxlen=window_size)

    def clear_buffer(self) -> None:
        """Clear the prediction buffer."""
        self._buffer.clear()
    
    def _init_mediapipe(self):
        """
        Initialize MediaPipe hand detection.

        Uses the Tasks API (HandLandmarker) which is the only supported path
        on mediapipe >= 0.10 / Python 3.13 — mp.solutions was removed in
        that release.  Thresholds are set to 0.2 so that partially visible
        or poorly-lit hands are still detected.
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python
            import os

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'ml', 'models', 'hand_landmarker.task'
            )

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.1,
                min_hand_presence_confidence=0.1,
                min_tracking_confidence=0.1,
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            self.hands = self.hand_landmarker   # alias for compat checks
            self._mp_module = mp
            logger.info("MediaPipe HandLandmarker (Tasks API) initialized — "
                        "all thresholds=0.2")
        except Exception as e:
            logger.warning(f"MediaPipe not available ({e}). Using mock predictions.")
            self.hand_landmarker = None
            self.hands = None
            self._mp_module = None

    def decode_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            return image
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks using MediaPipe HandLandmarker (Tasks API).

        Minimal preprocessing — just resize if needed and BGR→RGB.
        No histogram equalisation or brightness correction: those change
        landmark positions relative to the training data.
        """
        if self.hand_landmarker is None or self._mp_module is None:
            return None

        try:
            from ml.feature_extractor import FeatureExtractor
            mp = self._mp_module

            # Ensure minimum resolution
            h, w = image.shape[:2]
            if h < 224 or w < 224:
                scale = max(224 / h, 224 / w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)

            # BGR → RGB → mp.Image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Run detection
            results = self.hand_landmarker.detect(mp_image)

            if results.hand_landmarks:
                logger.debug("Hand detected — extracting landmarks")
                hand_landmarks = results.hand_landmarks[0]
                features = FeatureExtractor.extract_features_83(hand_landmarks)
                if features is None:
                    logger.debug("FeatureExtractor returned None for detected hand")
                    return None
                return np.array(features, dtype=np.float32)
            else:
                logger.debug("No hand detected in frame")
                return None

        except Exception as e:
            logger.error(f"Error during MediaPipe detection: {e}")
            return None
    
    def predict(self, image_data: str, model, labels: list[str]) -> GesturePrediction:
        """Run full prediction pipeline on a base64 image."""
        try:
            image = self.decode_image(image_data)
            landmarks = self.extract_landmarks(image)

            if landmarks is None:
                return GesturePrediction(
                    gesture="No gesture detected",
                    confidence=0.0,
                    landmarks_detected=False
                )

            if model is None:
                return self._mock_prediction(True)

            # Reshape for model input: (1, 83)
            features = landmarks.reshape(1, -1)
            if len(features.flatten()) != 83:
                logger.error(f"Feature vector length {len(features.flatten())} != 83; skipping prediction.")
                return GesturePrediction(gesture="unknown", confidence=0.0, landmarks_detected=False)

            # Predict class index and decode to label string
            prediction = model.predict(features)[0]
            label_str = labels[prediction]

            # Obtain raw confidence
            if hasattr(model, 'predict_proba'):
                raw_confidence = float(np.max(model.predict_proba(features)[0]))
            elif hasattr(model, 'decision_function'):
                # SVC fallback — softmax over decision scores
                scores = model.decision_function(features)[0]
                exp_s = np.exp(scores - float(np.max(scores)))
                raw_confidence = float(exp_s.max() / exp_s.sum())
            else:
                raw_confidence = 0.75
                logger.warning("Model has no predict_proba or decision_function")

            # Confidence filter: low-confidence frames stored as "nothing"
            if raw_confidence < self.confidence_threshold:
                self._buffer.append(("nothing", raw_confidence))
            else:
                self._buffer.append((label_str, raw_confidence))

            # Stabilised gesture: majority vote over buffer
            gesture_counts: dict = {}
            for g, _ in self._buffer:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            stable_gesture = max(gesture_counts, key=lambda k: gesture_counts[k])

            # Reported confidence: mean of matching-frame confidences, rounded to 3 dp
            matching_confs = [c for g, c in self._buffer if g == stable_gesture]
            reported_conf = round(sum(matching_confs) / len(matching_confs), 3)

            return GesturePrediction(
                gesture=stable_gesture,
                confidence=reported_conf,
                landmarks_detected=True
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return GesturePrediction(
                gesture="unknown",
                confidence=0.0,
                landmarks_detected=False
            )
    
    def _mock_prediction(self, landmarks_detected: bool) -> GesturePrediction:
        """Generate a mock prediction for demo purposes."""
        import random
        if landmarks_detected:
            gesture = random.choice(self.GESTURE_LABELS[:26])
            confidence = round(random.uniform(0.6, 0.95), 3)
        else:
            gesture = "nothing"
            confidence = 0.0
        
        return GesturePrediction(
            gesture=gesture,
            confidence=confidence,
            landmarks_detected=landmarks_detected
        )
