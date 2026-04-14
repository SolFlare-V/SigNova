import numpy as np

def extract_normalized_features(hand_landmarks):
    """Extract normalized features from MediaPipe hand landmarks."""
    # Ensure it's list-like (works with mediapipe results)
    landmarks_list = hand_landmarks if isinstance(hand_landmarks, list) else hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks
    
    # In MediaPipe, the wrist is landmark 0
    try:
        base_x = landmarks_list[0].x
        base_y = landmarks_list[0].y
        base_z = landmarks_list[0].z
    except Exception:
        base_x, base_y, base_z = 0, 0, 0
    
    features = []
    for lm in landmarks_list:
        try:
            features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        except Exception:
            features.extend([0, 0, 0])
            
    # Normalize features by max absolute value to make it scale-invariant
    max_val = max([abs(val) for val in features]) if features else 0
    if max_val > 0:
        features = [val / max_val for val in features]
        
    return features
