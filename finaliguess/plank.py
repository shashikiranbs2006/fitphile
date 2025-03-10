import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading  

# Initialize MediaPipe Pose and TTS
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
engine = pyttsx3.init()

# Set voice properties
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

# Function to announce plank time
def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=run_speech, daemon=True).start()

# Start video capture
cap = cv2.VideoCapture(0)

# Plank tracking variables
plank_start_time = None  # Timer start when plank is detected
plank_seconds = 0  # Total plank hold duration
plank_state = False  # Whether user is in plank position

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w, _ = frame.shape

        # Get key body part positions
        shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
        hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * h
        knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * h
        ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * h

        # Check if body is in plank position
        if abs(shoulder_y - hip_y) < 40 and abs(hip_y - knee_y) < 40 and abs(knee_y - ankle_y) < 40:
            if not plank_state:  # If plank just started
                plank_start_time = time.time()
                plank_state = True
            
            plank_seconds = int(time.time() - plank_start_time)

        else:
            plank_state = False  # Plank lost
            plank_start_time = None  # Reset timer

        # Display plank time
        cv2.putText(frame, f"Plank Time: {plank_seconds} sec", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Announce milestone times
        if plank_seconds in [10, 20, 30, 60, 90, 120]:  # Announce at key intervals
            speak(f"Great job! You have held the plank for {plank_seconds} seconds!")

    cv2.imshow('Plank Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()