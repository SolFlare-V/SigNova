"""
Feature extraction module for the SignLang AI gesture recognition pipeline.

This module provides the FeatureExtractor class, which converts 21 MediaPipe
hand landmarks into a 63-element wrist-relative, scale-normalised coordinate
vector. The same normalisation is applied identically during both offline
training (train_pipeline.py) and live inference (gesture_service.py) to
guarantee a consistent feature distribution across both contexts.
"""

import numpy as np


class FeatureExtractor:
    """
    Converts MediaPipe hand landmarks into a 63-element feature vector.

    Normalisation approach:
      1. Subtract the wrist landmark (index 0) from all 21 landmarks so that
         the wrist is always at the origin (0, 0, 0) — translation invariance.
      2. Divide all wrist-relative coordinates by the maximum Euclidean
         distance from the wrist to any other landmark — scale invariance.
      3. If the maximum distance is zero (degenerate hand), return the
         zero-centred coordinates unchanged to avoid divide-by-zero.

    The output is a flat list of 63 floats: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
    """

    @classmethod
    def extract_features(cls, hand_landmarks) -> list[float] | None:
        """
        Extract a 63-element wrist-relative, scale-normalised feature vector.

        Parameters
        ----------
        hand_landmarks:
            Either a MediaPipe ``HandLandmarkerResult.hand_landmarks[0]``
            (a list of landmark objects with ``.x``, ``.y``, ``.z`` attributes)
            or any plain list of objects with those attributes.

        Returns
        -------
        list[float] | None
            A list of exactly 63 floats when a valid 21-landmark hand is
            provided, or ``None`` when ``hand_landmarks`` is ``None`` or
            contains fewer than 21 points.
        """
        if hand_landmarks is None:
            return None

        # Normalise input: accept both a list and objects with a .landmark attr
        landmarks_list: list = (
            hand_landmarks
            if isinstance(hand_landmarks, list)
            else hand_landmarks.landmark
            if hasattr(hand_landmarks, "landmark")
            else hand_landmarks
        )

        if len(landmarks_list) < 21:
            return None

        try:
            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks_list[:21]],
                dtype=float,
            )
        except AttributeError:
            return None

        # Step 1 — translation invariance: subtract wrist (index 0)
        wrist = points[0]
        relative = points - wrist

        # Step 2 — scale invariance: divide by max distance from wrist
        # Use indices 1..20 (exclude wrist itself which is always 0)
        distances = np.linalg.norm(relative[1:], axis=1)
        max_dist: float = float(np.max(distances))

        if max_dist > 0.0:
            normalised = relative / max_dist
        else:
            # Degenerate case: all landmarks coincide with the wrist
            normalised = relative

        return normalised.flatten().tolist()

    @classmethod
    def extract_features_83(cls, hand_landmarks) -> list[float] | None:
        """
        Extract an 83-element feature vector compatible with the pre-trained
        gesture_model.pkl (StandardScaler + SVC trained on 83 features).

        Layout: 63 wrist-relative normalised coords  (same as extract_features)
              + 10 joint angles  (thumb + 4 fingers × 2 joints each)
              + 10 pairwise distances  (fingertip-to-fingertip pairs)
        = 83 total floats.
        """
        base = cls.extract_features(hand_landmarks)
        if base is None:
            return None

        # Rebuild the normalised points array from the flat base vector
        pts = np.array(base, dtype=float).reshape(21, 3)

        # ── 10 joint angles ───────────────────────────────────────────────
        # Finger joint triplets: (base, mid, tip) for each of 5 fingers × 2 joints
        joint_triplets = [
            (1, 2, 3),   # thumb MCP-IP
            (2, 3, 4),   # thumb IP-tip
            (5, 6, 7),   # index MCP-PIP
            (6, 7, 8),   # index PIP-DIP
            (9, 10, 11), # middle MCP-PIP
            (10, 11, 12),# middle PIP-DIP
            (13, 14, 15),# ring MCP-PIP
            (14, 15, 16),# ring PIP-DIP
            (17, 18, 19),# pinky MCP-PIP
            (18, 19, 20),# pinky PIP-DIP
        ]
        angles = []
        for a, b, c in joint_triplets:
            v1 = pts[a] - pts[b]
            v2 = pts[c] - pts[b]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles.append(float(np.arccos(cos_a)))
            else:
                angles.append(0.0)

        # ── 10 pairwise fingertip distances ───────────────────────────────
        fingertips = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
        pairs = [(fingertips[i], fingertips[j])
                 for i in range(len(fingertips))
                 for j in range(i + 1, len(fingertips))]  # 10 pairs
        dists = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in pairs]

        return base + angles + dists


def extract_normalized_features(hand_landmarks) -> list[float] | None:
    """
    Backwards-compatibility wrapper around ``FeatureExtractor.extract_features``.

    Returns a list of exactly 63 floats, or ``None`` when no valid hand is
    detected.
    """
    return FeatureExtractor.extract_features(hand_landmarks)
