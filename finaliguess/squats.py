import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading  
from flask import Flask, Response, render_template

app = Flask(__name__)

# Initialize MediaPipe Pose and TTS
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
engine = pyttsx3.init()

# Set voice properties
engine.setProperty("rate", 170)  # Adjust speech speed if needed
engine.setProperty("volume", 1.0)  # Max volume

def speak(text):
    """Function to convert text to speech in a separate thread."""
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

# Squat tracking variables
squat_count = 0
in_squat = False
feedback_message = ""
last_squat_time = time.time()
motivation_given = False

# Video capture
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def generate_frames():
    global squat_count, in_squat, feedback_message, last_squat_time, motivation_given
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        h, w, _ = frame.shape

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
            knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
            ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

            left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
            left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
            left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))

            right_knee_angle = calculate_angle(hip, knee, ankle)
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            if right_knee_angle < 90 and left_knee_angle < 90:
                in_squat = True
            elif right_knee_angle > 160 and left_knee_angle > 160 and in_squat:
                squat_count += 1
                in_squat = False
                last_squat_time = time.time()
                motivation_given = False
                feedback_message = "Good squat! Keep going!"
                if squat_count % 10 == 0:
                    speak(f"Congratulations! You have completed {squat_count} squats!")
            else:
                feedback_message = "Go lower for a full squat!"

            if time.time() - last_squat_time > 10 and not motivation_given:
                speak("Come on! Keep pushing! Squats are your power!")
                motivation_given = True

            cv2.putText(frame, feedback_message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Squats: {squat_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
