from trackers.base import ExerciseTracker, PoseLandmark


class PushupTracker(ExerciseTracker):
    def __init__(self):
        super().__init__("pushup")
        self.state = "Up"

    def process_landmarks(self, landmarks):
        try:
            left_indices = [PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST]
            right_indices = [PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST]

            shoulder, elbow, wrist, side = self.get_more_visible_side(landmarks, left_indices, right_indices)
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)

            if elbow_angle < 90:
                if self.state != "Down":
                    self.state = "Down"
                    self.feedback = "Great depth! Now push back up."
            elif elbow_angle > 160:
                if self.state == "Down":
                    self.count += 1
                    self.state = "Up"
                    self.feedback = f"Good rep! Total push-ups: {self.count}."
            else:
                if self.state == "Down":
                    self.feedback = "Push up all the way to complete the rep."
                else:
                    self.feedback = "Lower your chest towards the floor."

        except Exception:
            self.feedback = "Adjust your angle so your full arm is visible."
