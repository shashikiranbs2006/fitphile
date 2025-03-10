import cv2
import mediapipe as mp
import numpy as np
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

# Function to announce lunge count
def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=run_speech, daemon=True).start()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Start video capture
cap = cv2.VideoCapture(0)

# Lunge tracking variables
lunge_count = 0
lunge_state = "Up"

# Button positions (x1, y1, x2, y2)
RESET_BTN_POS = (50, 400, 200, 450)  
HOME_BTN_POS = (250, 400, 400, 450)  

def check_button_click(x, y):
    global lunge_count
    if RESET_BTN_POS[0] <= x <= RESET_BTN_POS[2] and RESET_BTN_POS[1] <= y <= RESET_BTN_POS[3]:
        lunge_count = 0  # Reset count
        print("Lunge count reset!")

    elif HOME_BTN_POS[0] <= x <= HOME_BTN_POS[2] and HOME_BTN_POS[1] <= y <= HOME_BTN_POS[3]:
        print("Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Mouse click detection
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        check_button_click(x, y)

cv2.namedWindow('Lunge Counter with Buttons')
cv2.setMouseCallback('Lunge Counter with Buttons', mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    h, w, _ = frame.shape

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark

        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
        left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))

        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        right_knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
        right_ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Detect correct lunge form
        if 80 <= left_knee_angle <= 110 or 80 <= right_knee_angle <= 110:  
            if lunge_state == "Up":
                lunge_state = "Down"

        elif left_knee_angle > 130 and right_knee_angle > 130:  
            if lunge_state == "Down":
                lunge_state = "Up"
                lunge_count += 1  

                if lunge_count % 10 == 0:
                    speak(f"Congratulations! You have completed {lunge_count} lunges!")

    # **Draw Buttons**
    cv2.rectangle(frame, (RESET_BTN_POS[0], RESET_BTN_POS[1]), (RESET_BTN_POS[2], RESET_BTN_POS[3]), (0, 0, 255), -1)
    cv2.putText(frame, "RESET", (RESET_BTN_POS[0] + 30, RESET_BTN_POS[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.rectangle(frame, (HOME_BTN_POS[0], HOME_BTN_POS[1]), (HOME_BTN_POS[2], HOME_BTN_POS[3]), (0, 255, 0), -1)
    cv2.putText(frame, "HOME", (HOME_BTN_POS[0] + 30, HOME_BTN_POS[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display count on screen
    cv2.putText(frame, f"Lunges: {lunge_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Lunge Counter with Buttons', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
