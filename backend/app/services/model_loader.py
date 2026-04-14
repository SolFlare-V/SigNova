"""ML Model loader service - loads model and labels once at startup."""
import os
import pickle
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton service for loading and managing ML models."""

    _instance = None
    _model = None
    _labels: Optional[List[str]] = None
    _model_name = "gesture_rf_model"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str) -> bool:
        """Load the ML model from disk."""
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self._model = pickle.load(f)
                logger.info(f"Model loaded successfully from {model_path}")
                return True
            else:
                logger.warning(f"Model file not found at {model_path}. Running in demo mode.")
                self._model = None
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model = None
            return False

    def load_labels(self, labels_path: str) -> bool:
        """
        Load the label list from a pickled file.

        Parameters
        ----------
        labels_path:
            Path to ``labels.pkl`` — a pickled ``list[str]`` where index ``i``
            corresponds to class integer ``i`` produced by the trained model.

        Returns
        -------
        bool
            ``True`` when labels were loaded successfully, ``False`` otherwise.
        """
        try:
            if os.path.exists(labels_path):
                with open(labels_path, 'rb') as f:
                    self._labels = pickle.load(f)
                logger.info(f"Labels loaded successfully from {labels_path}")
                return True
            else:
                logger.warning(f"Labels file not found at {labels_path}. Running in demo mode.")
                self._labels = None
                return False
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            self._labels = None
            return False

    @property
    def model(self):
        """Get the loaded model."""
        return self._model

    @property
    def labels(self) -> Optional[List[str]]:
        """Get the loaded label list, or ``None`` if not yet loaded."""
        return self._labels

    @property
    def is_loaded(self) -> bool:
        """Return True only when both the model and labels are loaded."""
        return self._model is not None and self._labels is not None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    def get_supported_gestures(self) -> List[str]:
        """
        Get list of supported gesture labels.

        Returns the dynamically loaded label list when available, falling back
        to the static default list for demo / pre-training mode.
        """
        if self._labels is not None:
            return self._labels
        return [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'nothing'
        ]
