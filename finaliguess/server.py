import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from flask import Flask, render_template
import subprocess

app = Flask(__name__)

# Route to home page (HTML page)
@app.route('/')
def home():
    return render_template('index.html')

# Function to run exercise scripts
def run_script(script_name):
    subprocess.Popen(["python", script_name], creationflags=subprocess.CREATE_NEW_CONSOLE)

@app.route('/exercise/<name>')
def start_exercise(name):
    exercises = {
        "pushups": "pushups.py",
        "plank": "plank.py",
        "lunges": "lunges.py",
        "squats": "squats.py"
    }
    if name in exercises:
        run_script(exercises[name])
        return f"Starting {name} exercise..."
    else:
        return "Invalid exercise!"

if __name__ == '__main__':
    app.run(debug=True)