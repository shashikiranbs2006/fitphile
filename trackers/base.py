import numpy as np


class PoseLandmark:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


class ExerciseTracker:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.state = "Up"
        self.feedback = "Position yourself in front of the camera"
        self.duration_seconds = 0
        self.start_time = None

    def reset(self):
        self.count = 0
        self.state = "Up"
        self.feedback = "Tracker reset. Ready to start!"
        self.duration_seconds = 0
        self.start_time = None

    def calculate_angle(self, a, b, c):
        """Calculates the 2D angle (in degrees) at joint b given points a, b, c."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def get_more_visible_side(self, landmarks, left_indices, right_indices):
        """
        Returns (a, b, c) as (x, y) tuples for the more visible side,
        plus 'left' or 'right' string.
        Uses normalized coordinates directly (no frame dimensions needed).
        """
        left_visibility = sum(landmarks[idx].visibility for idx in left_indices)
        right_visibility = sum(landmarks[idx].visibility for idx in right_indices)

        chosen = left_indices if left_visibility >= right_visibility else right_indices
        side = 'left' if left_visibility >= right_visibility else 'right'

        points = [(landmarks[idx].x, landmarks[idx].y) for idx in chosen]
        return points[0], points[1], points[2], side

    def process_landmarks(self, landmarks):
        """Override in subclasses. Receives list of landmark objects, updates self.count/state/feedback."""
        raise NotImplementedError("Subclasses must implement process_landmarks")
