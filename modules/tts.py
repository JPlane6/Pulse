# TTS Testing for my MAC, works using the built-in `say` command.
import subprocess
import threading

VOICE = "Ava"

_speaking = False
_lock = threading.Lock()


def speak(text: str):
    global _speaking

    with _lock:
        if _speaking:
            return
        _speaking = True

    def _run():
        global _speaking
        try:
            subprocess.run(["say", "-v", VOICE, text], check=True)
        finally:
            with _lock:
                _speaking = False

    threading.Thread(target=_run, daemon=True).start()