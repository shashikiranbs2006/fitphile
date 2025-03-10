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

def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

pushup_count = 0
pushup_state = "Up"

RESET_BTN_POS = (50, 400, 200, 450)  
HOME_BTN_POS = (250, 400, 400, 450)  

def check_button_click(x, y):
    global pushup_count
    if RESET_BTN_POS[0] <= x <= RESET_BTN_POS[2] and RESET_BTN_POS[1] <= y <= RESET_BTN_POS[3]:
        pushup_count = 0  
        print("Push-up count reset!")
    elif HOME_BTN_POS[0] <= x <= HOME_BTN_POS[2] and HOME_BTN_POS[1] <= y <= HOME_BTN_POS[3]:
        print("Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        check_button_click(x, y)

cv2.namedWindow('Push-up Counter')
cv2.setMouseCallback('Push-up Counter', mouse_callback)

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
        
        shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        elbow = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
        wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h))
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        if elbow_angle < 90 and pushup_state == "Up":
            pushup_state = "Down"
        elif elbow_angle > 160 and pushup_state == "Down":
            pushup_state = "Up"
            pushup_count += 1  
            if pushup_count % 10 == 0:
                speak(f"Congratulations! You have completed {pushup_count} push-ups!")

    cv2.rectangle(frame, (RESET_BTN_POS[0], RESET_BTN_POS[1]), (RESET_BTN_POS[2], RESET_BTN_POS[3]), (0, 0, 255), -1)
    cv2.putText(frame, "RESET", (RESET_BTN_POS[0] + 30, RESET_BTN_POS[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.rectangle(frame, (HOME_BTN_POS[0], HOME_BTN_POS[1]), (HOME_BTN_POS[2], HOME_BTN_POS[3]), (0, 255, 0), -1)
    cv2.putText(frame, "HOME", (HOME_BTN_POS[0] + 30, HOME_BTN_POS[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Push-ups: {pushup_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Push-up Counter', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
