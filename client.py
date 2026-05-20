import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import tempfile
import os
import time
import subprocess
import json
from datetime import datetime
import scipy.signal as signal

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════

SERVER_URL    = "http://192.168.0.156:5000"
MAX_QUESTIONS = 6

SAMPLERATE    = 48000

# Persistent ALSA device name
# This survives reboots and reconnects
AUDIO_DEVICE  = "plughw:CARD=Device,DEV=0"

PIPER_MODEL   = "/home/ayushs0604/Pulse/en_US-amy-medium.onnx"

LOG_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs")
PATIENT_INFO_PATH = os.path.join(LOG_DIR, "patientINFO.json")
CONFIG_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "config.json")

os.makedirs(LOG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  CONFIG HELPERS
# ═══════════════════════════════════════════════════════════════════

def load_config():
    """
    Load config.json from disk.
    Returns {} if file doesn't exist yet.
    """

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    return {}


def save_config(config):
    """
    Save config.json to disk.
    """

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[config] Saved to {CONFIG_PATH}")


def get_room_number(config):
    """
    Get room number from config.
    """

    room = config.get("room_number", "101")

    print(f"[config] Using room number: {room}")

    return str(room).strip()


def get_patient_name(config):
    """
    Get patient name from config.
    Converts:
        'john doe'
    into:
        'JohnDoe'
    """

    name = config.get("patient_name", "Patient").strip()

    print(f"[config] Using patient name: {name}")

    return "".join(word.capitalize() for word in name.split())


# ═══════════════════════════════════════════════════════════════════
#  TEXT TO SPEECH
# ═══════════════════════════════════════════════════════════════════

def speak(text):
    """
    Convert text to speech using Piper
    and play through the USB speaker.
    """

    print(f"[tts] {text}")

    try:

        # ---------------------------------------------------------
        # Start Piper TTS
        # ---------------------------------------------------------

        piper_process = subprocess.Popen(
            [
                "piper",
                "--model",
                PIPER_MODEL,
                "--output_raw"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # ---------------------------------------------------------
        # Start audio playback
        # ---------------------------------------------------------

        aplay_process = subprocess.Popen(
            [
                "aplay",
                "-D",
                AUDIO_DEVICE,
                "-r", "22050",
                "-f", "S16_LE",
                "-c", "1"
            ],
            stdin=piper_process.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # ---------------------------------------------------------
        # Send text into Piper
        # ---------------------------------------------------------

        piper_process.stdin.write(text.encode("utf-8"))
        piper_process.stdin.close()

        # ---------------------------------------------------------
        # Wait for completion
        # ---------------------------------------------------------

        aplay_return_code = aplay_process.wait()
        piper_return_code = piper_process.wait()

        # ---------------------------------------------------------
        # Error reporting
        # ---------------------------------------------------------

        if piper_return_code != 0:
            piper_error = piper_process.stderr.read().decode()
            print(f"[tts] Piper error:\n{piper_error}")

        if aplay_return_code != 0:
            aplay_error = aplay_process.stderr.read().decode()
            print(f"[tts] Aplay error:\n{aplay_error}")

        if piper_return_code == 0 and aplay_return_code == 0:
            print("[tts] Speech completed successfully.")

    except Exception as e:
        print(f"[tts] Unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO RECORDING
# ═══════════════════════════════════════════════════════════════════

def record(duration):
    """
    Record audio from microphone,
    resample to 16kHz for Whisper,
    return raw WAV bytes.
    """

    print(f"[mic] Recording for {duration}s...")

    try:

        audio = sd.rec(
            int(duration * SAMPLERATE),
            samplerate=SAMPLERATE,
            channels=1,
            dtype='float32',
            device=AUDIO_DEVICE
        )

        sd.wait()

    except Exception as e:
        print(f"[mic] Recording failed: {e}")
        return b""

    # -------------------------------------------------------------
    # Resample to 16kHz for Whisper
    # -------------------------------------------------------------

    audio_resampled = signal.resample(
        audio,
        int(len(audio) * 16000 / SAMPLERATE)
    )

    # -------------------------------------------------------------
    # Save temp WAV
    # -------------------------------------------------------------

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:

        wav.write(f.name, 16000, audio_resampled)

        path = f.name

    with open(path, "rb") as f:
        data = f.read()

    os.remove(path)

    return data


# ═══════════════════════════════════════════════════════════════════
#  SERVER COMMUNICATION
# ═══════════════════════════════════════════════════════════════════

def transcribe(audio_bytes):
    """
    Send audio to server for transcription.
    """

    try:

        r = requests.post(
            f"{SERVER_URL}/transcribe",
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30
        )

        result = r.json()

        print(f"[stt] Heard: '{result['text']}'")

        return result["text"], result["status"]

    except Exception as e:

        print(f"[stt] Failed: {e}")

        return "", "STABLE"


def get_next_question(history):
    """
    Ask AI server for next triage question.
    """

    try:

        r = requests.post(
            f"{SERVER_URL}/next_question",
            json={"history": history},
            timeout=60
        )

        return r.json()["question"]

    except Exception as e:

        print(f"[question] Failed: {e}")

        return "DONE"


def check_urgent(history):
    """
    Final urgent assessment check.
    """

    try:

        r = requests.post(
            f"{SERVER_URL}/flag_urgent",
            json={"history": history},
            timeout=60
        )

        return r.json()["flagged_urgent"]

    except Exception as e:

        print(f"[urgent] Failed: {e}")

        return False


# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════

def save_log(session_dir, session_record):
    """
    Save:
        1. Individual session log
        2. Master patientINFO.json
    """

    os.makedirs(session_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Session-specific log
    # -------------------------------------------------------------

    with open(os.path.join(session_dir, "log.json"), "w") as f:
        json.dump(session_record, f, indent=2)

    print(f"[log] Session saved to {session_dir}/log.json")

    # -------------------------------------------------------------
    # Master log
    # -------------------------------------------------------------

    if os.path.exists(PATIENT_INFO_PATH):

        with open(PATIENT_INFO_PATH, "r") as f:
            all_records = json.load(f)

    else:
        all_records = []

    all_records.append(session_record)

    with open(PATIENT_INFO_PATH, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"[log] Appended to patientINFO.json")


# ═══════════════════════════════════════════════════════════════════
#  MAIN ASSESSMENT FLOW
# ═══════════════════════════════════════════════════════════════════

def main():

    # -------------------------------------------------------------
    # Load config
    # -------------------------------------------------------------

    config = load_config()

    room         = get_room_number(config)
    patient_name = get_patient_name(config)

    # -------------------------------------------------------------
    # Session metadata
    # -------------------------------------------------------------

    date     = datetime.now().strftime("%m/%d/%Y")
    time_str = datetime.now().strftime("%I:%M%p")

    session_name = f"{date.replace('/', '-')}_{time_str}_Room{room}_{patient_name}"

    session_dir = os.path.join(LOG_DIR, session_name)

    print(f"\n[session] Starting: {session_name}")

    # -------------------------------------------------------------
    # Greeting
    # -------------------------------------------------------------

    speak(
        f"Hello {patient_name}! "
        f"I am the nurse assistant robot. "
        f"I will ask you a few quick questions."
    )

    # -------------------------------------------------------------
    # Initial triage setup
    # -------------------------------------------------------------

    priority = {
        "STABLE": 0,
        "MONITOR": 1,
        "URGENT": 2
    }

    history = []

    status = "STABLE"

    # -------------------------------------------------------------
    # Mandatory first question
    # -------------------------------------------------------------

    first_q = "On a scale of 1 to 10, how would you rate your pain right now?"

    speak(first_q)

    time.sleep(1)

    answer, answer_status = transcribe(
        record(duration=12)
    )

    history.append({
        "q": first_q,
        "a": answer,
        "status": answer_status
    })

    if priority[answer_status] > priority[status]:
        status = answer_status

    print(f"[status] After Q1: {status}")

    # -------------------------------------------------------------
    # AI-generated follow-up questions
    # -------------------------------------------------------------

    for i in range(MAX_QUESTIONS - 1):

        next_q = get_next_question(history)

        if next_q.strip().upper() == "DONE":

            print(
                f"[patient] AI decided enough "
                f"info gathered after {i + 1} follow-up(s)."
            )

            break

        speak(next_q)

        time.sleep(1)

        answer, answer_status = transcribe(
            record(duration=12)
        )

        history.append({
            "q": next_q,
            "a": answer,
            "status": answer_status
        })

        if priority[answer_status] > priority[status]:
            status = answer_status

        print(f"[status] After Q{i + 2}: {status}")

    # -------------------------------------------------------------
    # Final urgent check
    # -------------------------------------------------------------

    flagged_urgent = check_urgent(history)

    if flagged_urgent:

        status = "URGENT"

        speak(
            f"Based on your responses {patient_name}, "
            f"I am alerting a nurse immediately."
        )

    else:

        speak(
            f"Thank you {patient_name}. "
            f"Your status has been recorded as {status}."
        )

    print(f"\n[final] Status: {status} | Urgent: {flagged_urgent}")

    # -------------------------------------------------------------
    # Save logs
    # -------------------------------------------------------------

    save_log(
        session_dir,
        {
            "header": {
                "patient_name": patient_name,
                "room": room,
                "date": date,
                "time": time_str
            },
            "triage": {
                "final_status": status,
                "flagged_urgent": flagged_urgent
            },
            "assessment": history
        }
    )


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()