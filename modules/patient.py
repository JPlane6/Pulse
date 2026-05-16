import requests
import base64
import cv2
import json
import time

OLLAMA_URL   = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"
TEXT_MODEL   = "phi3:mini"

URGENT_KEYWORDS = ["severe", "chest", "can't breathe", "cannot breathe",
                   "help", "emergency", "10", "worst", "unconscious", "dying"]
MONITOR_KEYWORDS = ["dizzy", "nausea", "pain", "uncomfortable",
                    "worse", "7", "8", "9", "medication", "no", "bad"]


def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def visual_triage(frame):
    image_b64 = frame_to_base64(frame)
    prompt = (
        "You are a clinical triage assistant for a nurse robot in a hospital. "
        "Analyze this image carefully and objectively. Focus on visible physical signs only.\n\n"
        "Classify the visible person into exactly one of these categories:\n\n"
        "URGENT — clear signs of medical emergency: unconscious, unresponsive, slumped, "
        "not breathing normally, severe visible distress, seizure, or signs of trauma.\n\n"
        "MONITOR — conscious but showing mild-moderate distress: grimacing, breathing heavily, "
        "visibly uncomfortable, unusual posture, or appears confused.\n\n"
        "STABLE — alert, calm, upright or resting normally, no visible signs of distress.\n\n"
        "NONE — no person visible in the image.\n\n"
        "Important: Do not classify as URGENT unless there are very clear emergency signs. "
        "A person sitting or lying normally should be STABLE. "
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


def get_next_question(conversation_history):
    history_text = "\n".join(
        [f"Q: {qa['q']}\nA: {qa['a']}" for qa in conversation_history]
    )

    prompt = (
        "You are a nurse robot doing a patient assessment. "
        "Based on the conversation so far, generate the single most important follow-up question. "
        "Rules:\n"
        "- Reply with ONLY the question, nothing else\n"
        "- No explanations, no notes, no qualifications\n"
        "- No 'if the patient...' or 'note:' or anything after the question\n"
        "- Short and clinical\n"
        "- Do not repeat questions already asked\n"
        "- If you have enough info, reply with only: DONE\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        "Next question (or DONE):"
    )

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": TEXT_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=60)

        # take only first line to cut off any rambling
        result = response.json()["response"].strip().split("\n")[0]
        print(f"[patient] Next question: {result}")
        return result

    except Exception as e:
        print(f"[patient] Question generation failed: {e}")
        return "DONE"


def assess_response_local(answer):
    answer = answer.lower()
    if any(w in answer for w in URGENT_KEYWORDS):
        return "URGENT"
    if any(w in answer for w in MONITOR_KEYWORDS):
        return "MONITOR"
    return "STABLE"


def flag_urgent_ai(conversation_history, visual_status):
    history_text = "\n".join(
        [f"Q: {qa['q']}\nA: {qa['a']}" for qa in conversation_history]
    )

    prompt = (
        "You are a clinical triage AI assisting a nurse robot. "
        f"The robot's camera classified the patient visually as: {visual_status}\n\n"
        f"The following conversation was recorded:\n{history_text}\n\n"
        "Based on ALL of this information combined, should this patient be flagged as URGENT "
        "and have a nurse alerted immediately?\n"
        "Consider: severe pain (8-10), difficulty breathing, chest pain, confusion, "
        "unresponsiveness, or any combination of concerning symptoms.\n"
        "Reply with only YES or NO."
    )

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": TEXT_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=60)

        result = response.json()["response"].strip().upper().split("\n")[0]
        print(f"[patient] AI urgent flag: {result}")
        return "YES" in result

    except Exception as e:
        print(f"[patient] Urgent flag failed: {e}")
        return False