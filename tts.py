"""
TTS is now handled entirely by the browser's Web Speech API (SpeechSynthesisUtterance).
This stub exists so no import errors occur if any legacy code references it.
pyttsx3 is NOT used — it requires a local audio device and cannot run on cloud servers.
"""

def speak(text):
    """No-op. Voice feedback is handled client-side via browser SpeechSynthesis."""
    pass
