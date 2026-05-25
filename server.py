import whisper
import requests
import tempfile
import os
import json
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
#  CONFIG — Optimized for Mac M2 (16GB RAM)
# ═══════════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "phi3:mini"
SERVER_PORT = 5001

# Triage keywords used for immediate, rule-based priority routing
URGENT_KEYWORDS  = ["severe", "chest", "can't breathe", "cannot breathe",
                    "help", "emergency", "10", "worst", "unconscious", "dying"]
MONITOR_KEYWORDS = ["dizzy", "nausea", "pain", "uncomfortable",
                    "worse", "7", "8", "9", "medication", "no", "bad"]

# ═══════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ═══════════════════════════════════════════════════════════════════

print("[server] Loading Whisper STT model...")
whisper_model = whisper.load_model("base")
print("[server] Whisper STT ready!")


# ═══════════════════════════════════════════════════════════════════
#  TRIAGE ASSESSMENT HELPER
# ═══════════════════════════════════════════════════════════════════

def assess(text):
    """Keyword-based triage status matching from transcribed string."""
    text = text.lower()
    if any(w in text for w in URGENT_KEYWORDS):  return "URGENT"
    if any(w in text for w in MONITOR_KEYWORDS): return "MONITOR"
    return "STABLE"


# ═══════════════════════════════════════════════════════════════════
#  ROUTE: Health check
# ═══════════════════════════════════════════════════════════════════

@app.route("/ping", methods=["GET"])
def ping():
    """Simple health check verification endpoint."""
    print("[ping] Pi connected successfully.")
    return jsonify({"status": "ok"})


# ═══════════════════════════════════════════════════════════════════
#  ROUTE: Transcribe audio
# ═══════════════════════════════════════════════════════════════════

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Receives raw WAV bytes from Pi, processes with Whisper, returns text."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(request.data)
            path = f.name

        result = whisper_model.transcribe(path, fp16=False, language="en")
        text   = result["text"].strip().lower()
        status = assess(text)
        os.remove(path)

        print(f"[transcribe] Heard: '{text}' → Evaluated Status: {status}")
        return jsonify({"text": text, "status": status})

    except Exception as e:
        print(f"[transcribe] Error during processing: {e}")
        return jsonify({"text": "", "status": "STABLE"})


# ═══════════════════════════════════════════════════════════════════
#  ROUTE: Stream Next Question (Ultra Low-Latency)
# ═══════════════════════════════════════════════════════════════════

@app.route("/next_question", methods=["POST"])
def next_question():
    """Streams token strings directly from Ollama to achieve instant conversation."""
    try:
        history = request.get_json()["history"]
        history_text = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}" for qa in history])

        # Short, crisp prompt minimizes Mac evaluation overhead
        prompt = (
            "You are a clinical triage nurse robot. Generate the single most important follow-up question.\n"
            "Rules: Only reply with the question. Short and clinical. No notes. No introductions. "
            "If you have enough info, reply with ONLY: DONE\n\n"
            f"Conversation history:\n{history_text}\n\n"
            "Next question (or DONE):"
        )

        def generate():
            try:
                r = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": TEXT_MODEL, 
                        "prompt": prompt, 
                        "stream": True,
                        "options": {
                            "num_predict": 45,     # Prevents bloated responses
                            "num_ctx": 1024,       # Limits memory context window size
                            "temperature": 0.0,    # Zero temp generates text much faster
                            "keep_alive": "-1"     # Locks model into Mac GPU memory permanently
                        }
                    },
                    stream=True,
                    timeout=60
                )
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        done  = chunk.get("done", False)
                        
                        if token:
                            yield token
                        if done:
                            break
            except Exception as e:
                print(f"[next_question] Streaming loop broken: {e}")
                yield "DONE"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        print(f"[next_question] Setup error: {e}")
        return Response("DONE", mimetype="text/plain")


# ═══════════════════════════════════════════════════════════════════
#  ROUTE: Final urgent flag
# ═══════════════════════════════════════════════════════════════════

@app.route("/flag_urgent", methods=["POST"])
def flag_urgent():
    """Asks Phi3 for a definitive evaluation on whether to alert the desk."""
    try:
        history = request.get_json()["history"]
        history_text = "\n".join([f"Q: {qa['q']}\nA: {qa['a']}" for qa in history])

        prompt = (
            "Review this medical history dialogue. Is emergency attention required? "
            "Look for severe pain (8-10), chest compression, breathing struggle, or critical conditions.\n"
            f"Dialogue:\n{history_text}\n\n"
            "Reply with exactly one word, YES or NO:"
        )

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL, 
                "prompt": prompt, 
                "stream": False,
                "options": {
                    "num_predict": 5,
                    "num_ctx": 1024,
                    "temperature": 0.0,
                    "keep_alive": "-1"
                }
            },
            timeout=60
        )
        result  = r.json()["response"].strip().upper().split("\n")[0]
        flagged = "YES" in result

        print(f"[flag_urgent] LLM Decision: {result} → Flagged={flagged}")
        return jsonify({"flagged_urgent": flagged})

    except Exception as e:
        print(f"[flag_urgent] Error evaluating urgency: {e}")
        return jsonify({"flagged_urgent": False})


if __name__ == "__main__":
    print(f"[server] Starting Flask engine on 0.0.0.0:{SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False)