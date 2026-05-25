import whisper
import requests
import tempfile
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "phi3:mini"

URGENT_KEYWORDS  = ["severe", "chest", "can't breathe", "cannot breathe",
                    "help", "emergency", "10", "worst", "unconscious", "dying"]
MONITOR_KEYWORDS = ["dizzy", "nausea", "pain", "uncomfortable",
                    "worse", "7", "8", "9", "medication", "no", "bad"]

# ═══════════════════════════════════════════════════════════════════
#  LOAD WHISPER
# ═══════════════════════════════════════════════════════════════════

print("[server] Loading Whisper...")
whisper_model = whisper.load_model("base")
print("[server] Whisper ready!")


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def assess(text):
    """Keyword-based triage status from transcribed text."""
    text = text.lower()
    if any(w in text for w in URGENT_KEYWORDS):  return "URGENT"
    if any(w in text for w in MONITOR_KEYWORDS): return "MONITOR"
    return "STABLE"


# ═══════════════════════════════════════════════════════════════════
#  ROUTE: Health check
# ═══════════════════════════════════════════════════════════════════

@app.route("/ping", methods=["GET"])
def ping():
    """
    Simple health check endpoint.
    The Pi calls this before starting any session to confirm
    the server is up and reachable. Returns 200 + {"status": "ok"}.
    """
    print("[ping] Pi connected.")
    return jsonify({"status": "ok"})


# ═══════════════════════════════════════════════════════════════════
#  ROUTE: Transcribe audio
# ═══════════════════════════════════════════════════════════════════

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Receive raw WAV bytes from the Pi, run Whisper STT,
    return transcribed text + triage status keyword assessment.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(request.data)
            path = f.name

        result = whisper_model.transcribe(path, fp16=False, language="en")
        text   = result["text"].strip().lower()
        status = assess(text)
        os.remove(path)

        print(f"[transcribe] '{text}' → {status}")
        return jsonify({"text": text, "status": status})

    except Exception as e:
        print(f"[transcribe] Error: {e}")
        return jsonify({"text": "", "status": "STABLE"})


# ═══════════════════════════════════════════════════════════════════
#  ROUTE: Generate next triage question
# ═══════════════════════════════════════════════════════════════════

@app.route("/next_question", methods=["POST"])
def next_question():
    """
    Receive conversation history from the Pi, ask Phi3:mini for
    the single best follow-up question (or DONE if enough info gathered).
    First line only is returned to guard against model rambling.
    """
    try:
        history = request.get_json()["history"]
        history_text = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}" for qa in history])

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

        r = requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        # First line only — guards against Phi3 rambling after the question
        question = r.json()["response"].strip().split("\n")[0]

        print(f"[next_question] {question}")
        return jsonify({"question": question})

    except Exception as e:
        print(f"[next_question] Error: {e}")
        return jsonify({"question": "DONE"})


# ═══════════════════════════════════════════════════════════════════
#  ROUTE: Final urgent flag
# ═══════════════════════════════════════════════════════════════════

@app.route("/flag_urgent", methods=["POST"])
def flag_urgent():
    """
    Receive full conversation history, ask Phi3:mini to make a
    final YES/NO call on whether the patient should be flagged URGENT.
    More reliable than keyword matching alone for edge cases.
    """
    try:
        history = request.get_json()["history"]
        history_text = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}" for qa in history])

        prompt = (
            "You are a clinical triage AI assisting a nurse robot. "
            f"The following patient assessment conversation was recorded:\n{history_text}\n\n"
            "Based on this conversation, should this patient be flagged as URGENT "
            "and have a nurse alerted immediately?\n"
            "Consider: severe pain (8-10), difficulty breathing, chest pain, confusion, "
            "unresponsiveness, or any combination of concerning symptoms.\n"
            "Reply with only YES or NO."
        )

        r = requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        # First line only — guards against extra text after YES/NO
        result  = r.json()["response"].strip().upper().split("\n")[0]
        flagged = "YES" in result

        print(f"[flag_urgent] {result} → flagged={flagged}")
        return jsonify({"flagged_urgent": flagged})

    except Exception as e:
        print(f"[flag_urgent] Error: {e}")
        return jsonify({"flagged_urgent": False})


# ═══════════════════════════════════════════════════════════════════
#  START
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[server] Starting on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5001, debug=False)