from trackers.base import ExerciseTracker, PoseLandmark


class LungeTracker(ExerciseTracker):
    def __init__(self):
        super().__init__("lunge")
        self.state = "Up"

    def process_landmarks(self, landmarks):
        try:
            left_hip = (landmarks[PoseLandmark.LEFT_HIP].x, landmarks[PoseLandmark.LEFT_HIP].y)
            left_knee = (landmarks[PoseLandmark.LEFT_KNEE].x, landmarks[PoseLandmark.LEFT_KNEE].y)
            left_ankle = (landmarks[PoseLandmark.LEFT_ANKLE].x, landmarks[PoseLandmark.LEFT_ANKLE].y)

            right_hip = (landmarks[PoseLandmark.RIGHT_HIP].x, landmarks[PoseLandmark.RIGHT_HIP].y)
            right_knee = (landmarks[PoseLandmark.RIGHT_KNEE].x, landmarks[PoseLandmark.RIGHT_KNEE].y)
            right_ankle = (landmarks[PoseLandmark.RIGHT_ANKLE].x, landmarks[PoseLandmark.RIGHT_ANKLE].y)

            left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

            if left_knee_angle < 100 or right_knee_angle < 100:
                if self.state != "Down":
                    self.state = "Down"
                    self.feedback = "Good depth! Push up to starting position."
            elif left_knee_angle > 140 and right_knee_angle > 140:
                if self.state == "Down":
                    self.count += 1
                    self.state = "Up"
                    self.feedback = f"Great lunge! Total: {self.count}."
            else:
                if self.state == "Down":
                    self.feedback = "Return to upright standing position."
                else:
                    self.feedback = "Step forward and lower your knee."

        except Exception:
            self.feedback = "Align your full body in the camera frame."
