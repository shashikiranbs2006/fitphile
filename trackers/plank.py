import time
import numpy as np
from trackers.base import ExerciseTracker, PoseLandmark


class PlankTracker(ExerciseTracker):
    def __init__(self):
        super().__init__("plank")
        self.state = "Rest"
        self.last_announced_milestone = 0

    def reset(self):
        super().reset()
        self.state = "Rest"
        self.last_announced_milestone = 0

    def process_landmarks(self, landmarks):
        try:
            left_indices = [
                PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP,
                PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE
            ]
            right_indices = [
                PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP,
                PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE
            ]

            left_vis = sum(landmarks[i].visibility for i in left_indices)
            right_vis = sum(landmarks[i].visibility for i in right_indices)
            chosen = left_indices if left_vis >= right_vis else right_indices

            shoulder = (landmarks[chosen[0]].x, landmarks[chosen[0]].y)
            hip = (landmarks[chosen[1]].x, landmarks[chosen[1]].y)
            knee = (landmarks[chosen[2]].x, landmarks[chosen[2]].y)
            ankle = (landmarks[chosen[3]].x, landmarks[chosen[3]].y)

            hip_angle = self.calculate_angle(shoulder, hip, knee)
            knee_angle = self.calculate_angle(hip, knee, ankle)

            dx = ankle[0] - shoulder[0]
            dy = ankle[1] - shoulder[1]
            body_tilt = np.degrees(np.arctan2(abs(dy), abs(dx))) if dx != 0 else 90.0

            is_correct_form = (hip_angle > 160) and (knee_angle > 160) and (body_tilt < 35)

            if is_correct_form:
                if self.state != "Active":
                    self.state = "Active"
                    self.start_time = time.time() - self.duration_seconds

                self.duration_seconds = int(time.time() - self.start_time)
                self.feedback = f"Good form! Keep holding. Time: {self.duration_seconds}s"

                # Return milestone info so frontend can trigger browser TTS
                milestones = [10, 20, 30, 45, 60, 90, 120, 180]
                for m in milestones:
                    if self.duration_seconds >= m and self.last_announced_milestone < m:
                        self.last_announced_milestone = m
                        break
            else:
                if self.state == "Active":
                    self.state = "Rest"
                    self.feedback = "Form lost! Straighten your hips/knees and level your body."
                else:
                    self.feedback = "Align shoulders, hips, knees, and ankles horizontally."

        except Exception:
            self.feedback = "Place camera side-on for best plank detection."
