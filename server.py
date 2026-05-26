# =========================
# server.py
# =========================

import whisper
import requests
import tempfile
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "phi3:mini"
SERVER_PORT = 5001

URGENT_KEYWORDS = [
    "severe",
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "help",
    "emergency",
    "worst pain",
    "unconscious",
    "dying",
    "pressure in chest"
]

MONITOR_KEYWORDS = [
    "dizzy",
    "nausea",
    "pain",
    "uncomfortable",
    "worse",
    "medication",
    "bad",
    "fever",
    "vomiting",
    "migraine"
]

WELCOME_MESSAGE = (
    "Hi, I’m going to ask you a few quick questions "
    "to better understand what’s going on."
)

# ═══════════════════════════════════════════════════════════════════
# LOAD WHISPER
# ═══════════════════════════════════════════════════════════════════

print("[server] Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("[server] Whisper ready!")

# ═══════════════════════════════════════════════════════════════════
# TRIAGE LOGIC
# ═══════════════════════════════════════════════════════════════════

def assess(text):

    text = text.lower()

    if any(w in text for w in URGENT_KEYWORDS):
        return "URGENT"

    if any(w in text for w in MONITOR_KEYWORDS):
        return "MONITOR"

    return "STABLE"

# ═══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════

@app.route("/ping", methods=["GET"])
def ping():

    return jsonify({"status": "ok"})

# ═══════════════════════════════════════════════════════════════════
# WELCOME MESSAGE
# ═══════════════════════════════════════════════════════════════════

@app.route("/welcome", methods=["GET"])
def welcome():

    return jsonify({
        "message": WELCOME_MESSAGE
    })

# ═══════════════════════════════════════════════════════════════════
# TRANSCRIBE AUDIO
# ═══════════════════════════════════════════════════════════════════

@app.route("/transcribe", methods=["POST"])
def transcribe():

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:

            f.write(request.data)
            path = f.name

        result = whisper_model.transcribe(
            path,
            fp16=False,
            language="en"
        )

        text = result["text"].strip()
        status = assess(text)

        os.remove(path)

        if len(text) == 0:

            return jsonify({
                "text": "",
                "status": "STABLE",
                "heard": False
            })

        print(f"[transcribe] '{text}' → {status}")

        return jsonify({
            "text": text,
            "status": status,
            "heard": True
        })

    except Exception as e:

        print(f"[transcribe] Error: {e}")

        return jsonify({
            "text": "",
            "status": "STABLE",
            "heard": False
        })

# ═══════════════════════════════════════════════════════════════════
# NEXT QUESTION
# ═══════════════════════════════════════════════════════════════════

@app.route("/next_question", methods=["POST"])
def next_question():

    try:

        history = request.get_json()["history"]

        history_text = "\n".join([
            f"Patient: {qa['a']}"
            for qa in history
        ])

        prompt = (
            "You are a calm and caring clinical triage assistant.\n"
            "Speak naturally and briefly.\n"
            "Ask only ONE short follow-up question at a time.\n"
            "Ask the MOST PERTINENT unanswered question.\n"
            "Never repeat a question.\n"
            "Never ask for information already given.\n"
            "Make questions explicit and context-aware.\n"
            "Avoid vague wording.\n"
            "Example: if the patient mentions a headache, ask "
            "'Where in your head does it hurt?' "
            "instead of 'Where does it hurt?'\n"
            "Keep questions crisp and conversational.\n"
            "Use natural spoken English.\n"
            "Do not explain anything.\n"
            "Do not ask multiple questions at once.\n\n"

            "You must gather:\n"
            "- Main symptom\n"
            "- Severity\n"
            "- Duration\n"
            "- Breathing issues\n"
            "- Existing conditions\n"
            "- Medications\n\n"

            "When enough information is collected, reply ONLY with: DONE\n\n"

            f"Conversation:\n{history_text}\n\n"
            "Assistant:"
        )

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 35,
                    "num_ctx": 1024,
                    "repeat_penalty": 1.2,
                    "top_k": 20,
                    "top_p": 0.8,
                    "keep_alive": "-1"
                }
            },
            timeout=60
        )

        response_text = r.json()["response"].strip()

        print(f"[next_question] {response_text}")

        return jsonify({
            "question": response_text
        })

    except Exception as e:

        print(f"[next_question] Error: {e}")

        return jsonify({
            "question": "Could you repeat that?"
        })

# ═══════════════════════════════════════════════════════════════════
# FINAL URGENCY CHECK
# ═══════════════════════════════════════════════════════════════════

@app.route("/flag_urgent", methods=["POST"])
def flag_urgent():

    try:

        history = request.get_json()["history"]

        history_text = "\n".join([
            f"Q: {qa['q']}\nA: {qa['a']}"
            for qa in history
        ])

        prompt = (
            "You are reviewing a medical triage conversation.\n"
            "Determine whether emergency attention may be needed.\n"
            "Reply ONLY with YES or NO.\n\n"

            f"Conversation:\n{history_text}"
        )

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 3,
                    "num_ctx": 1024,
                    "keep_alive": "-1"
                }
            },
            timeout=60
        )

        result = r.json()["response"].strip().upper()

        flagged = "YES" in result

        print(f"[flag_urgent] {result}")

        return jsonify({
            "flagged_urgent": flagged
        })

    except Exception as e:

        print(f"[flag_urgent] Error: {e}")

        return jsonify({
            "flagged_urgent": False
        })

# ═══════════════════════════════════════════════════════════════════
# START SERVER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print(f"[server] Starting on port {SERVER_PORT}")

    app.run(
        host="0.0.0.0",
        port=SERVER_PORT,
        debug=False,
        threaded=True
    )