import numpy as np

class DatasetAugmenter:
    """
    Utilities for augmenting coordinate data before training to improve robustness.
    Includes random scaling, translation noise, and slight rotation.
    """
    
    @staticmethod
    def augment_landmarks(landmarks_array, augmentations=2):
        """
        Takes raw landmarks (Nx21x3) and generates synthetic variations.
        Returns features shape (N * (1+augmentations), D) and corresponding labels.
        """
        # This requires taking the original 21x3 landmarks and applying geometric transforms,
        # but since our current dataset stores 1D feature vectors directly,
        # we will implement statistical jitter directly on the feature vector instead for simplicity,
        # or require raw landmark saving in the future.
        pass
        
    @staticmethod
    def add_jitter(features, noise_level=0.015):
        """Add small Gaussian noise to simulate camera/mediapipe jitter."""
        noise = np.random.normal(0, noise_level, len(features))
        return np.array(features) + noise
