import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workout.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database with users and workouts tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create workouts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            exercise_type TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            duration_seconds INTEGER DEFAULT 0,
            calories REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    
    # Insert a default Guest user if no users exist
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO users (username) VALUES ('Guest')")
        conn.commit()
        
    conn.close()

def get_users():
    """Fetches all users from the database."""
    conn = get_db_connection()
    users = conn.execute("SELECT * FROM users ORDER BY username ASC").fetchall()
    conn.close()
    return [dict(user) for user in users]

def add_user(username):
    """Adds a new user and returns their ID. Returns None if user already exists."""
    username = username.strip()
    if not username:
        return None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def add_workout(user_id, exercise_type, count, duration_seconds, calories):
    """Inserts a new workout session record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO workouts (user_id, exercise_type, count, duration_seconds, calories, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, exercise_type, count, duration_seconds, round(calories, 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    workout_id = cursor.lastrowid
    conn.close()
    return workout_id

def get_user_history(user_id):
    """Gets the full workout log for a given user."""
    conn = get_db_connection()
    workouts = conn.execute("""
        SELECT * FROM workouts 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(w) for w in workouts]

def get_user_stats(user_id):
    """Generates summary metrics for a given user."""
    conn = get_db_connection()
    stats = conn.execute("""
        SELECT 
            COUNT(id) as total_workouts,
            SUM(CASE WHEN exercise_type != 'plank' THEN count ELSE 0 END) as total_reps,
            SUM(duration_seconds) as total_duration_seconds,
            SUM(calories) as total_calories
        FROM workouts 
        WHERE user_id = ?
    """, (user_id,)).fetchone()
    conn.close()
    
    res = dict(stats)
    # Handle None defaults for empty history
    res['total_workouts'] = res['total_workouts'] or 0
    res['total_reps'] = res['total_reps'] or 0
    res['total_duration_seconds'] = res['total_duration_seconds'] or 0
    res['total_calories'] = round(res['total_calories'] or 0.0, 2)
    return res
