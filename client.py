import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import tempfile
import os
import time
import subprocess
import json
from datetime import datetime

# --- Config ---
SERVER_URL    = "http://192.168.0.156:5000"
ROOM          = "001"
PATIENT_NAME  = "Ayush"
MAX_QUESTIONS = 6
SAMPLERATE    = 16000
AUDIO_DEVICE  = 0
PIPER_MODEL   = "/home/ayushs0604/Pulse/en_US-amy-medium.onnx"

LOG_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs")
PATIENT_INFO_PATH = os.path.join(LOG_DIR, "patientINFO.json")
os.makedirs(LOG_DIR, exist_ok=True)


# --- Text to Speech ---
def speak(text):
    print(f"[tts] {text}")
    subprocess.run(
        f'echo "{text}" | piper --model {PIPER_MODEL} --output_raw | aplay -D plughw:0,0 -r 22050 -f S16_LE -c 1',
        shell=True, check=True
    )


# --- Record Audio ---
def record(duration):
    print(f"[mic] Recording for {duration}s...")
    audio = sd.rec(int(duration * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float32', device=AUDIO_DEVICE)
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLERATE, audio)
        path = f.name
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


# --- Send Audio to Server ---
def transcribe(audio_bytes):
    try:
        r = requests.post(f"{SERVER_URL}/transcribe", data=audio_bytes, headers={"Content-Type": "application/octet-stream"}, timeout=30)
        result = r.json()
        print(f"[stt] Heard: '{result['text']}'")
        return result["text"], result["status"]
    except Exception as e:
        print(f"[stt] Failed: {e}")
        return "", "STABLE"


# --- Get Next Question from Server ---
def get_next_question(history):
    try:
        r = requests.post(f"{SERVER_URL}/next_question", json={"history": history}, timeout=60)
        return r.json()["question"]
    except Exception as e:
        print(f"[question] Failed: {e}")
        return "DONE"


# --- Ask Server if Patient is Urgent ---
def check_urgent(history):
    try:
        r = requests.post(f"{SERVER_URL}/flag_urgent", json={"history": history}, timeout=60)
        return r.json()["flagged_urgent"]
    except Exception as e:
        print(f"[urgent] Failed: {e}")
        return False


# --- Save Session to Log ---
def save_log(session_dir, session_record):
    os.makedirs(session_dir, exist_ok=True)

    with open(os.path.join(session_dir, "log.json"), "w") as f:
        json.dump(session_record, f, indent=2)
    print(f"[log] Session saved to {session_dir}/log.json")

    if os.path.exists(PATIENT_INFO_PATH):
        with open(PATIENT_INFO_PATH, "r") as f:
            all_records = json.load(f)
    else:
        all_records = []

    all_records.append(session_record)

    with open(PATIENT_INFO_PATH, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"[log] Appended to patientINFO.json")


# --- Main ---
def main():
    patient_name = "".join(w.capitalize() for w in PATIENT_NAME.strip().split())
    date         = datetime.now().strftime("%m/%d/%Y")
    time_str     = datetime.now().strftime("%I:%M%p")
    session_name = f"{date.replace('/', '-')}_{time_str}_Room{ROOM}_{patient_name}"
    session_dir  = os.path.join(LOG_DIR, session_name)

    print(f"\n[session] {session_name}")

    # --- Greet Patient ---
    speak(f"Hello {patient_name}! I am the nurse assistant robot. I will ask you a few quick questions.")

    # --- First Question: Pain Scale ---
    priority = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}
    history  = []
    status   = "STABLE"

    first_q = "On a scale of 1 to 10, how would you rate your pain right now?"
    speak(first_q)
    time.sleep(1)
    answer, answer_status = transcribe(record(duration=12))
    history.append({"q": first_q, "a": answer, "status": answer_status})
    if priority[answer_status] > priority[status]:
        status = answer_status
    print(f"[status] {status}")

    # --- Dynamic Follow-up Questions ---
    for _ in range(MAX_QUESTIONS - 1):
        next_q = get_next_question(history)

        if next_q.strip().upper() == "DONE":
            print("[patient] AI finished questioning.")
            break

        speak(next_q)
        time.sleep(1)
        answer, answer_status = transcribe(record(duration=12))
        history.append({"q": next_q, "a": answer, "status": answer_status})

        if priority[answer_status] > priority[status]:
            status = answer_status
        print(f"[status] {status}")

    # --- Final Urgent Check ---
    flagged_urgent = check_urgent(history)
    if flagged_urgent:
        status = "URGENT"
        speak(f"Based on your responses {patient_name}, I am alerting a nurse immediately.")
    else:
        speak(f"Thank you {patient_name}. Your status has been recorded as {status}.")

    print(f"\n[final] Status: {status} | Urgent: {flagged_urgent}")

    # --- Save Log ---
    save_log(session_dir, {
        "header": {
            "patient_name": patient_name,
            "room": ROOM,
            "date": date,
            "time": time_str
        },
        "triage": {
            "final_status": status,
            "flagged_urgent": flagged_urgent
        },
        "assessment": history
    })


if __name__ == "__main__":
    main()