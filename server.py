import whisper
import requests
import tempfile
import os
import re
from flask import Flask, request, jsonify

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

OLLAMA_URL  = "http://localhost:11434/api/generate"
TEXT_MODEL  = "phi3:mini"
SERVER_PORT = 5001

WELCOME_MESSAGE = (
    "Hi, I'm going to ask you a few quick questions "
    "to better understand what's going on."
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

def extract_pain_score(text):
    match = re.search(r'\b(10|[1-9])\b', text)
    if match:
        return int(match.group(1))

    word_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    for word, val in word_map.items():
        if word in text.lower():
            return val

    return None


def assess(text):
    if not text.strip():
        return "STABLE"

    # Don't trust classification on very short or gibberish responses
    words = text.strip().split()
    if len(words) < 2:
        return "STABLE"

    # Pain score — hard numeric signal, most reliable
    score = extract_pain_score(text.lower())
    if score is not None:
        if score >= 8:
            return "URGENT"
        if score >= 5:
            return "MONITOR"

    # AI classification for everything else
    try:
        prompt = (
            "You are a clinical triage assistant.\n"
            "A patient said: \"{text}\"\n"
            "Classify their condition as exactly one of: URGENT, MONITOR, STABLE\n"
            "URGENT = severe pain, chest pain, can't breathe, unconscious, emergency\n"
            "MONITOR = moderate pain, dizziness, nausea, fever, discomfort\n"
            "STABLE = mild or no symptoms, feeling fine\n"
            "If the response is unclear or doesn't describe symptoms, reply: STABLE\n"
            "Reply with only one word: URGENT, MONITOR, or STABLE"
        ).format(text=text)

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 3,
                    "num_ctx": 512,
                    "keep_alive": "-1"
                }
            },
            timeout=30
        )

        result = r.json()["response"].strip().upper().split("\n")[0]

        if "URGENT" in result:
            return "URGENT"
        if "MONITOR" in result:
            return "MONITOR"
        return "STABLE"

    except Exception as e:
        print(f"[assess] AI classification failed, defaulting STABLE: {e}")
        return "STABLE"


# ═══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


# ═══════════════════════════════════════════════════════════════════
# WELCOME
# ═══════════════════════════════════════════════════════════════════

@app.route("/welcome", methods=["GET"])
def welcome():
    return jsonify({"message": WELCOME_MESSAGE})


# ═══════════════════════════════════════════════════════════════════
# TRANSCRIBE
# ═══════════════════════════════════════════════════════════════════

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(request.data)
            path = f.name

        result = whisper_model.transcribe(path, fp16=False, language="en")
        text   = result["text"].strip()
        status = assess(text)
        os.remove(path)

        if len(text) == 0:
            return jsonify({"text": "", "status": "STABLE", "heard": False})

        print(f"[transcribe] '{text}' → {status}")
        return jsonify({"text": text, "status": status, "heard": True})

    except Exception as e:
        print(f"[transcribe] Error: {e}")
        return jsonify({"text": "", "status": "STABLE", "heard": False})


# ═══════════════════════════════════════════════════════════════════
# NEXT QUESTION
# ═══════════════════════════════════════════════════════════════════

@app.route("/next_question", methods=["POST"])
def next_question():
    try:
        history = request.get_json()["history"]

        history_text = "\n".join([
            f"Nurse: {qa['q']}\nPatient: {qa['a']}"
            for qa in history
        ])

        prompt = (
            "You are a calm triage assistant.\n"
            "Ask ONE short follow-up question.\n"
            "Ask the most pertinent missing question.\n"
            "Do not repeat questions.\n"
            "Do not ask for info already given.\n"
            "Use simple everyday words.\n"
            "Be clear and specific.\n"
            "Do not sound robotic.\n"
            "Do not explain anything.\n"
            "Do not ask multiple questions.\n\n"
            "Make Small Questions that are under 15 words. preferably under 10 words. However the response should not cut off, if u have to go to like 16 or 17 words do so it the question is important and will not be cut off.\n"
            "You need to learn:\n"
            "- Main problem\n"
            "- Pain level\n"
            "- How long it has been happening\n"
            "- Breathing problems\n"
            "- Medical conditions\n"
            "- Medications\n\n"
            "If enough info is collected, reply ONLY with: DONE\n\n"
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
                    "num_predict": 20,
                    "num_ctx": 1024,
                    "repeat_penalty": 1.3,
                    "top_k": 20,
                    "top_p": 0.8,
                    "keep_alive": "-1"
                }
            },
            timeout=60
        )

        raw      = r.json()["response"].strip()
        question = raw.split("\n")[0].strip()

        filler_prefixes = [
            "I'm glad to hear that.", "I'm glad to hear that!",
            "Great!", "Good!", "Good to know.", "Okay,", "Okay.",
            "I see.", "I see,", "Alright,", "Alright.", "Sure,",
            "Thank you.", "Thank you!", "Got it.", "Got it,",
            "Of course.", "Of course!", "Certainly.", "Certainly!",
            "Understood.", "Understood,", "Noted.", "Noted,",
            "I understand.", "I understand,", "That's good.", "That's good!",
            "That's helpful.", "That's helpful!", "Thanks for sharing.",
        ]
        for filler in filler_prefixes:
            if question.lower().startswith(filler.lower()):
                question = question[len(filler):].strip()

        # Hard cut at first question mark
        if "?" in question:
            question = question[:question.index("?") + 1].strip()

        print(f"[next_question] {question}")
        return jsonify({"question": question})

    except Exception as e:
        print(f"[next_question] Error: {e}")
        return jsonify({"question": "DONE"})


# ═══════════════════════════════════════════════════════════════════
# FLAG URGENT
# ═══════════════════════════════════════════════════════════════════

@app.route("/flag_urgent", methods=["POST"])
def flag_urgent():
    try:
        history = request.get_json()["history"]

        history_text = "\n".join([
            f"Nurse: {qa['q']}\nPatient: {qa['a']}"
            for qa in history
        ])

        prompt = (
            "You are reviewing a medical triage conversation.\n"
            "Determine whether emergency attention may be needed.\n"
            "Only flag YES if there are clear and definite emergency signals.\n"
            "Unclear or vague responses should be flagged NO.\n"
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

        result  = r.json()["response"].strip().upper()
        flagged = "YES" in result

        print(f"[flag_urgent] {result}")
        return jsonify({"flagged_urgent": flagged})

    except Exception as e:
        print(f"[flag_urgent] Error: {e}")
        return jsonify({"flagged_urgent": False})


# ═══════════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[server] Starting on port {SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=True)