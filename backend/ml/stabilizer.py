import collections

class PredictionStabilizer:
    """
    Stabilizes real-time gesture predictions by applying a sliding window smoothing filter.
    Helpful to eliminate 'flickering' between classes in adjacent frames.
    """
    
    def __init__(self, window_size=5, confidence_threshold=0.6):
        """
        :param window_size: Number of frames to keep in the buffer.
        :param confidence_threshold: Minimum confidence required to accept a prediction.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.buffer = collections.deque(maxlen=window_size)
    
    def stabilize(self, prediction: str, confidence: float) -> tuple[str, float]:
        """
        Process a new frame prediction.
        Returns the stabilized prediction and its estimated confidence.
        """
        if confidence < self.confidence_threshold:
            # If low confidence, don't strongly weight it, or treat as noise
            prediction = "nothing"
            
        self.buffer.append((prediction, confidence))
        
        # Calculate the mode (most common prediction) in the current window
        counter = collections.Counter([p for p, c in self.buffer])
        
        # Find the most common gesture
        most_common_gesture, count = counter.most_common(1)[0]
        
        # Calculate smoothed confidence: average confidence of the most common prediction
        relevant_confidences = [c for p, c in self.buffer if p == most_common_gesture]
        smoothed_confidence = sum(relevant_confidences) / len(relevant_confidences) if relevant_confidences else 0.0
        
        # Override if the mode is an actual sign but it's only appeared once (and window is full)
        # to prevent instant snapping if it's just a 1-frame jitter
        if len(self.buffer) == self.window_size and count < (self.window_size // 2):
             return "nothing", 0.0
             
        return most_common_gesture, smoothed_confidence
    
    def clear(self):
        """Clear the buffer, useful when the user stops gesturing."""
        self.buffer.clear()
