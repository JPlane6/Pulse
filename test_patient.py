import cv2
import json
import os
import base64
import time
import threading
from datetime import datetime
from modules import patient, tts, stt

# --- Config ---
ROOM          = "001"
PATIENT_NAME  = "Ayush"
MAX_QUESTIONS = 6

# --- Setup log folder ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

PATIENT_INFO_PATH = os.path.join(LOG_DIR, "patientINFO.json")

def to_camel_case(name):
    return "".join(word.capitalize() for word in name.strip().split())

# --- Session info ---
patient_name = to_camel_case(PATIENT_NAME)
date         = datetime.now().strftime("%m/%d/%Y")
time_str     = datetime.now().strftime("%I:%M%p")

session_name = f"{date.replace('/', '-')}_{time_str}_Room{ROOM}_{patient_name}"
session_dir  = os.path.join(LOG_DIR, session_name)
os.makedirs(session_dir, exist_ok=True)

print(f"\nSession: {session_name}")
print(f"Logging to: {session_dir}\n")

# --- Camera ---
print("Starting camera...")
cap = cv2.VideoCapture(0)

for _ in range(10):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Camera failed")
    exit()

image_path = os.path.join(session_dir, "frame.jpg")
cv2.imwrite(image_path, frame)
print(f"Frame saved to {image_path}")

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# --- Visual triage in background ---
triage_result = {"status": None}

def run_triage():
    triage_result["status"] = patient.visual_triage(frame)

triage_thread = threading.Thread(target=run_triage)
triage_thread.start()
print("Visual triage running in background...")

# --- Greet while triage runs ---
tts.speak_and_wait(f"Hello {patient_name}! I am the nurse assistant robot. I will ask you a few quick questions.")

# --- Wait for triage ---
triage_thread.join()
visual_status = triage_result["status"] or "STABLE"
print(f"Visual triage result: {visual_status}")

if visual_status == "NONE":
    visual_status = "STABLE"

if visual_status == "MONITOR":
    tts.speak_and_wait(f"I noticed you may be a little uncomfortable {patient_name}. Let me ask you some questions.")

# --- Dynamic questions ---
conversation_history = []
priority      = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}
overall_status = visual_status

# First question always pain scale
first_question = "On a scale of 1 to 10, how would you rate your pain right now?"
tts.speak_and_wait(first_question)
print(f"\nQ1: {first_question}")
time.sleep(1)
print("Listening...")
answer = stt.listen(duration=8)
print(f"You said: '{answer}'")
answer_status = patient.assess_response_local(answer)
conversation_history.append({"q": first_question, "a": answer, "status": answer_status})
if priority[answer_status] > priority[overall_status]:
    overall_status = answer_status

# Dynamic follow-up questions
question_count = 1
while question_count < MAX_QUESTIONS:
    next_q = patient.get_next_question(conversation_history)

    if next_q.strip().upper() == "DONE":
        print("[patient] AI decided enough questions asked.")
        break

    tts.speak_and_wait(next_q)
    print(f"\nQ{question_count + 1}: {next_q}")
    time.sleep(1)
    print("Listening...")
    answer = stt.listen(duration=12)
    print(f"You said: '{answer}'")

    answer_status = patient.assess_response_local(answer)
    conversation_history.append({"q": next_q, "a": answer, "status": answer_status})

    if priority[answer_status] > priority[overall_status]:
        overall_status = answer_status

    print(f"Overall status so far: {overall_status}")
    question_count += 1

# --- AI urgent flag ---
flagged_urgent = patient.flag_urgent_ai(conversation_history, visual_status)
if flagged_urgent:
    overall_status = "URGENT"
    tts.speak_and_wait(f"Based on your responses {patient_name}, I am alerting a nurse immediately.")
else:
    tts.speak_and_wait(f"Thank you {patient_name}. Your status has been recorded as {overall_status}.")

print(f"\nFinal status: {overall_status}")
print(f"AI flagged urgent: {flagged_urgent}")

# --- Build session record ---
session_record = {
    "header": {
        "patient_name": patient_name,
        "room": ROOM,
        "date": date,
        "time": time_str
    },
    "triage": {
        "visual_status": visual_status,
        "final_status": overall_status,
        "flagged_urgent": flagged_urgent
    },
    "assessment": conversation_history,
    "image": image_b64
}

# --- Save individual session log ---
json_path = os.path.join(session_dir, "log.json")
with open(json_path, "w") as f:
    json.dump(session_record, f, indent=2)
print(f"Session log saved to {json_path}")

# --- Append to patientINFO.json ---
if os.path.exists(PATIENT_INFO_PATH):
    with open(PATIENT_INFO_PATH, "r") as f:
        all_records = json.load(f)
else:
    all_records = []

all_records.append(session_record)

with open(PATIENT_INFO_PATH, "w") as f:
    json.dump(all_records, f, indent=2)

print(f"Patient record appended to {PATIENT_INFO_PATH}")