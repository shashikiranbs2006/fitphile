import os
import time
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS

import database
from trackers.pushup import PushupTracker
from trackers.squat import SquatTracker
from trackers.lunge import LungeTracker
from trackers.plank import PlankTracker

app = Flask(__name__)
CORS(app)

database.init_db()

# Global session state — no camera, no OpenCV
global_state = {
    "user_id": 1,
    "username": "Guest",
    "active_tracker": None,
    "start_time": 0.0
}


def get_calorie_estimate(exercise, count, duration):
    if exercise == "pushup":
        return count * 0.45
    elif exercise == "squat":
        return count * 0.32
    elif exercise == "lunge":
        return count * 0.35
    elif exercise == "plank":
        return duration * 0.15
    return 0.0


# --- Routes ---

@app.route('/')
def home():
    users = database.get_users()
    return render_template('index.html', users=users, active_user=global_state["username"])


@app.route('/set_user', methods=['POST'])
def set_user():
    data = request.get_json()
    username = data.get('username')
    user_id = data.get('user_id')

    if user_id:
        users = database.get_users()
        selected_user = next((u for u in users if u['id'] == int(user_id)), None)
        if selected_user:
            global_state["user_id"] = selected_user["id"]
            global_state["username"] = selected_user["username"]
            return jsonify({"success": True, "username": selected_user["username"]})
    elif username:
        new_id = database.add_user(username)
        if new_id:
            global_state["user_id"] = new_id
            global_state["username"] = username
            return jsonify({"success": True, "username": username})
        else:
            return jsonify({"success": False, "error": "Username already exists."})

    return jsonify({"success": False, "error": "Invalid profile selection."})


@app.route('/dashboard')
def dashboard():
    user_stats = database.get_user_stats(global_state["user_id"])
    return render_template('dashboard.html', username=global_state["username"],
                           stats=user_stats, active_user=global_state["username"])


@app.route('/workout/<exercise>')
def workout(exercise):
    finish_workout_logic()

    if exercise == "pushups":
        global_state["active_tracker"] = PushupTracker()
    elif exercise == "squats":
        global_state["active_tracker"] = SquatTracker()
    elif exercise == "lunges":
        global_state["active_tracker"] = LungeTracker()
    elif exercise == "plank":
        global_state["active_tracker"] = PlankTracker()
    else:
        return redirect(url_for('dashboard'))

    global_state["start_time"] = time.time()
    return render_template('workout.html', exercise=exercise,
                           username=global_state["username"],
                           active_user=global_state["username"])


@app.route('/api/landmarks', methods=['POST'])
def receive_landmarks():
    """
    Core endpoint: receives normalized landmark data from MediaPipe JS in the browser.
    Runs Python angle-math logic and returns updated rep count + feedback.
    
    Expected JSON body:
    {
        "landmarks": [
            {"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99},
            ...  (33 landmarks total, MediaPipe Pose format)
        ]
    }
    """
    tracker = global_state["active_tracker"]
    if not tracker:
        return jsonify({"active": False})

    data = request.get_json()
    landmarks_raw = data.get("landmarks", [])

    if not landmarks_raw:
        return jsonify({"active": True, "feedback": "No landmarks received."})

    # Convert raw dicts to simple objects with .x .y .z .visibility attributes
    class LM:
        def __init__(self, d):
            self.x = d.get("x", 0.0)
            self.y = d.get("y", 0.0)
            self.z = d.get("z", 0.0)
            self.visibility = d.get("visibility", 0.0)

    landmarks = [LM(lm) for lm in landmarks_raw]

    # Run the tracker's angle logic (no frame rendering — pure math)
    tracker.process_landmarks(landmarks)

    session_duration = int(time.time() - global_state["start_time"])
    calories = get_calorie_estimate(
        tracker.name, tracker.count,
        tracker.duration_seconds if tracker.name == "plank" else session_duration
    )

    return jsonify({
        "active": True,
        "exercise": tracker.name,
        "count": tracker.count,
        "state": tracker.state,
        "feedback": tracker.feedback,
        "duration": tracker.duration_seconds if tracker.name == "plank" else session_duration,
        "calories": round(calories, 2)
    })


@app.route('/api/stats')
def api_stats():
    """Lightweight poll endpoint for session metadata."""
    tracker = global_state["active_tracker"]
    if not tracker:
        return jsonify({"active": False})

    session_duration = int(time.time() - global_state["start_time"])
    calories = get_calorie_estimate(
        tracker.name, tracker.count,
        tracker.duration_seconds if tracker.name == "plank" else session_duration
    )

    return jsonify({
        "active": True,
        "exercise": tracker.name,
        "count": tracker.count,
        "state": tracker.state,
        "feedback": tracker.feedback,
        "duration": tracker.duration_seconds if tracker.name == "plank" else session_duration,
        "calories": round(calories, 2)
    })


def finish_workout_logic():
    tracker = global_state["active_tracker"]
    if tracker:
        total_time = int(time.time() - global_state["start_time"])
        exercise_duration = tracker.duration_seconds if tracker.name == "plank" else total_time
        calories = get_calorie_estimate(tracker.name, tracker.count, exercise_duration)

        if tracker.count > 0 or exercise_duration > 3:
            database.add_workout(
                user_id=global_state["user_id"],
                exercise_type=tracker.name,
                count=tracker.count,
                duration_seconds=exercise_duration,
                calories=calories
            )

        global_state["active_tracker"] = None


@app.route('/api/finish', methods=['POST'])
def finish_workout():
    finish_workout_logic()
    return jsonify({"success": True})


@app.route('/history')
def history():
    history_logs = database.get_user_history(global_state["user_id"])
    user_stats = database.get_user_stats(global_state["user_id"])
    return render_template('history.html', username=global_state["username"],
                           history=history_logs, stats=user_stats,
                           active_user=global_state["username"])


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
