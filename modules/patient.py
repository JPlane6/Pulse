import requests
import base64
import cv2
import time
from modules import tts, stt

OLLAMA_URL   = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"

QUESTIONS = [
    "Hi, I am the nurse assistant robot. On a scale of 1 to 10, how would you rate your pain?",
    "Are you feeling any dizziness or nausea?",
    "Have you taken your medication today?",
    "Is there anything you urgently need from a nurse?",
]

URGENT_KEYWORDS  = ["severe", "chest", "can't breathe", "cannot breathe",
                    "help", "emergency", "10", "worst", "unconscious"]
MONITOR_KEYWORDS = ["dizzy", "nausea", "pain", "uncomfortable",
                    "worse", "7", "8", "9", "medication", "no"]


def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def visual_triage(frame):
    image_b64 = frame_to_base64(frame)
    prompt = (
        "You are assisting a nurse robot doing visual patient triage in a hospital. "
        "Look at this image carefully and classify the visible person into exactly one category:\n\n"
        "URGENT — patient appears unconscious, slumped over, not breathing normally, "
        "showing signs of severe distress, or unresponsive.\n\n"
        "MONITOR — patient appears conscious but something looks off: breathing heavily, "
        "slouching unusually, grimacing, visibly uncomfortable, or showing mild distress.\n\n"
        "STABLE — patient appears calm, upright or resting normally, no visible signs of distress.\n\n"
        "If no person is visible, reply NONE.\n"
        "Reply with only one word: URGENT, MONITOR, STABLE, or NONE."
    )

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }, timeout=60)

        result = response.json()["response"].strip().upper()
        print(f"[patient] Visual triage result: {result}")

        if "URGENT"  in result: return "URGENT"
        if "MONITOR" in result: return "MONITOR"
        if "STABLE"  in result: return "STABLE"
        return "NONE"

    except Exception as e:
        print(f"[patient] Visual triage failed: {e}")
        return "NONE"


def assess_response_local(answer):
    answer = answer.lower()
    if any(w in answer for w in URGENT_KEYWORDS):
        return "URGENT"
    if any(w in answer for w in MONITOR_KEYWORDS):
        return "MONITOR"
    return "STABLE"


def run_checkin(frame, lcd, get_speech_input):
    priority = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}

    # --- Step 1: Visual triage ---
    lcd.clear()
    lcd.write_string("Scanning...".ljust(20))
    visual_status = visual_triage(frame)

    if visual_status == "NONE":
        print("[patient] No patient detected, skipping room.")
        return None

    # --- Step 2: Immediate urgent alert ---
    if visual_status == "URGENT":
        tts.speak("Warning. A patient appears to need immediate attention. Alerting nursing staff.")
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string("!! URGENT !!".ljust(20))
        lcd.cursor_pos = (1, 0)
        lcd.write_string("Alert sent".ljust(20))
        return "URGENT"

    # --- Step 3: Greet patient ---
    if visual_status == "MONITOR":
        tts.speak("I noticed you may be uncomfortable. Let me ask you a few questions.")
    else:
        tts.speak("Hello! I am the nurse assistant robot. I will ask you a few quick questions.")

    overall_status = visual_status

    # --- Step 4: Ask questions ---
    for i, question in enumerate(QUESTIONS):
        tts.speak(question)

        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(f"Q{i+1} of {len(QUESTIONS)}".ljust(20))
        lcd.cursor_pos = (1, 0)
        lcd.write_string("Listening...".ljust(20))

        time.sleep(0.5)
        answer = stt.listen(duration=5)
        print(f"[patient] Q: {question} | A: {answer}")

        answer_status = assess_response_local(answer)

        if priority[answer_status] > priority[overall_status]:
            overall_status = answer_status

        lcd.cursor_pos = (2, 0)
        lcd.write_string(f"Status: {overall_status}".ljust(20))
        time.sleep(1.5)

        if overall_status == "URGENT":
            tts.speak("I will alert a nurse immediately.")
            break

    # --- Step 5: Final status ---
    tts.speak(f"Thank you. Your status has been recorded as {overall_status}.")
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string("Final Status:".ljust(20))
    lcd.cursor_pos = (1, 0)
    lcd.write_string(overall_status.ljust(20))

    return overall_status