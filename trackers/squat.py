import time
from trackers.base import ExerciseTracker, PoseLandmark


class SquatTracker(ExerciseTracker):
    def __init__(self):
        super().__init__("squat")
        self.state = "Up"
        self.last_rep_time = time.time()
        self.motivation_given = False

    def reset(self):
        super().reset()
        self.last_rep_time = time.time()
        self.motivation_given = False

    def process_landmarks(self, landmarks):
        try:
            left_indices = [PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE]
            right_indices = [PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE]

            hip, knee, ankle, side = self.get_more_visible_side(landmarks, left_indices, right_indices)
            knee_angle = self.calculate_angle(hip, knee, ankle)

            if knee_angle < 100:
                if self.state != "Down":
                    self.state = "Down"
                    self.feedback = "Good depth! Push through your heels to stand."
            elif knee_angle > 160:
                if self.state == "Down":
                    self.count += 1
                    self.state = "Up"
                    self.feedback = f"Good squat! Keep it up. Total: {self.count}."
                    self.last_rep_time = time.time()
                    self.motivation_given = False
            else:
                if self.state == "Down":
                    self.feedback = "Stand up completely to finish the rep."
                else:
                    self.feedback = "Lower your hips below your knees."

            if time.time() - self.last_rep_time > 12 and not self.motivation_given:
                self.motivation_given = True
                self.feedback = "Keep going! You are doing great!"

        except Exception:
            self.feedback = "Stand profile-wise so your full body is visible."
