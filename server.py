import whisper
import requests
import tempfile
import os
import re
import json
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
whisper_model = whisper.load_model("small")
print("[server] Whisper ready!")

# ═══════════════════════════════════════════════════════════════════
# TRIAGE KEYWORDS  (fast path — no Phi3 needed)
# ═══════════════════════════════════════════════════════════════════

URGENT_HARD = [
    "can't breathe", "cannot breathe", "chest pain", "chest tightness",
    "heart attack", "unconscious", "unresponsive", "not breathing",
    "stopped breathing", "seizure", "stroke", "overdose", "bleeding out",
    "can't move", "cannot move", "help me", "dying"
]

MONITOR_HARD = [
    "dizzy", "dizziness", "nausea", "vomiting", "fever", "chills",
    "shortness of breath", "hard to breathe", "difficulty breathing",
    "swelling", "rash", "allergic", "infection", "confusion", "disoriented"
]


def extract_pain_score(text):
    match = re.search(r'\b(10|[1-9])\b', text)
    if match:
        return int(match.group(1))
    word_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    for word, val in word_map.items():
        if re.search(rf'\b{word}\b', text.lower()):
            return val
    return None


def keyword_status(text):
    """
    Fast keyword + pain-score classifier.
    Returns 'URGENT', 'MONITOR', 'STABLE', or None if uncertain.
    None means we need Phi3 to decide.
    """
    if not text.strip() or len(text.strip().split()) < 2:
        return "STABLE"

    lower = text.lower()

    for kw in URGENT_HARD:
        if kw in lower:
            print(f"[keyword] URGENT via: '{kw}'")
            return "URGENT"

    for kw in MONITOR_HARD:
        if kw in lower:
            print(f"[keyword] MONITOR via: '{kw}'")
            return "MONITOR"

    score = extract_pain_score(lower)
    if score is not None:
        if score >= 8:  return "URGENT"
        if score >= 4:  return "MONITOR"
        if score <= 3:  return "STABLE"

    return None   # uncertain — let Phi3 handle it


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
# TRANSCRIBE  (Whisper only — no Phi3)
# ═══════════════════════════════════════════════════════════════════

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(request.data)
            path = f.name

        result = whisper_model.transcribe(
            path,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
        )
        text = result["text"].strip()
        os.remove(path)

        if not text:
            return jsonify({"text": "", "heard": False})

        print(f"[transcribe] '{text}'")
        return jsonify({"text": text, "heard": True})

    except Exception as e:
        print(f"[transcribe] Error: {e}")
        return jsonify({"text": "", "heard": False})


# ═══════════════════════════════════════════════════════════════════
# ASSESS + NEXT QUESTION  (single Phi3 call)
# ═══════════════════════════════════════════════════════════════════

@app.route("/assess_and_next", methods=["POST"])
def assess_and_next():
    """
    Replaces two separate endpoints (/transcribe assess + /next_question).

    Receives:
      {
        "history": [{"q": "...", "a": "...", "status": "..."}, ...],
        "last_answer": "the patient's most recent answer text"
      }

    Returns:
      {
        "status":   "URGENT" | "MONITOR" | "STABLE",
        "question": "next question text"  |  "DONE"
      }

    Fast path: if keywords resolve the status unambiguously, we still
    call Phi3 once for the question but skip the classification prompt.
    If keywords are uncertain, one Phi3 call returns BOTH fields via JSON.
    """
    try:
        body        = request.get_json()
        history     = body.get("history", [])
        last_answer = body.get("last_answer", "").strip()

        # ── Fast-path keyword classification ──────────────────────
        fast_status = keyword_status(last_answer)

        history_text = "\n".join([
            f"Nurse: {qa['q']}\nPatient: {qa['a']}"
            for qa in history
        ])

        if fast_status is not None:
            # Status is settled — only ask Phi3 for the next question
            question = _get_next_question_only(history_text, history)
            print(f"[assess_and_next] fast={fast_status} | q='{question}'")
            return jsonify({"status": fast_status, "question": question})

        # ── Slow path: one Phi3 call for both ─────────────────────
        prompt = (
            "You are a clinical triage assistant for a nurse robot.\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"The patient's latest answer: \"{last_answer}\"\n\n"
            "Your job:\n"
            "1. Classify the patient's latest answer as URGENT, MONITOR, or STABLE.\n"
            "   URGENT = life-threatening: severe chest pain, cannot breathe, unconscious, "
            "uncontrolled bleeding, seizure, stroke, pain 9-10/10.\n"
            "   MONITOR = moderate concern: pain 4-8/10, fever, dizziness, nausea, "
            "shortness of breath, confusion.\n"
            "   STABLE = mild or no symptoms: pain 1-3/10, feeling okay.\n"
            "   When in doubt, choose the less severe option.\n\n"
            "2. Ask the single most important follow-up question not yet covered.\n"
            "   Topics to cover: main problem, pain level, duration, breathing, "
            "medical history, medications.\n"
            "   If you have enough info, write DONE instead of a question.\n"
            "   Keep questions under 15 words. Ask only one question.\n\n"
            "Reply with ONLY valid JSON, no extra text, no markdown:\n"
            "{\"status\": \"URGENT|MONITOR|STABLE\", \"question\": \"your question or DONE\"}"
        )

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 60,
                    "num_ctx": 1024,
                    "repeat_penalty": 1.3,
                    "top_k": 20,
                    "top_p": 0.8,
                    "keep_alive": "-1"
                }
            },
            timeout=60
        )

        raw = r.json()["response"].strip()
        print(f"[assess_and_next] raw='{raw}'")

        # Strip markdown fences if Phi3 adds them
        raw = re.sub(r"```json|```", "", raw).strip()

        # Try to parse JSON
        try:
            parsed   = json.loads(raw)
            status   = parsed.get("status",   "STABLE").upper().strip()
            question = parsed.get("question", "DONE").strip()
        except json.JSONDecodeError:
            # Phi3 didn't return clean JSON — extract with regex
            print(f"[assess_and_next] JSON parse failed, extracting with regex")
            status   = _extract_status_from_text(raw)
            question = _extract_question_from_text(raw)

        # Sanitise status
        if status not in ("URGENT", "MONITOR", "STABLE"):
            status = "STABLE"

        # Sanitise question — strip filler, cut at first ?
        question = _clean_question(question)

        print(f"[assess_and_next] status={status} | q='{question}'")
        return jsonify({"status": status, "question": question})

    except Exception as e:
        print(f"[assess_and_next] Error: {e}")
        return jsonify({"status": "STABLE", "question": "DONE"})


# ═══════════════════════════════════════════════════════════════════
# FLAG URGENT  (final holistic check — unchanged)
# ═══════════════════════════════════════════════════════════════════

@app.route("/flag_urgent", methods=["POST"])
def flag_urgent():
    try:
        history = request.get_json()["history"]

        history_text = "\n".join([
            f"Nurse: {qa['q']}\nPatient: {qa['a']}"
            for qa in history
        ])

        urgent_count  = sum(1 for qa in history if qa.get("status") == "URGENT")
        monitor_count = sum(1 for qa in history if qa.get("status") == "MONITOR")

        if urgent_count == 0 and monitor_count < 3:
            print(f"[flag_urgent] No urgent signals (U:{urgent_count} M:{monitor_count}) — skipping AI")
            return jsonify({"flagged_urgent": False})

        prompt = (
            "You are a senior nurse reviewing a completed triage conversation.\n"
            "Decide whether this patient needs IMMEDIATE emergency attention RIGHT NOW.\n\n"
            "Flag YES only if the conversation clearly describes:\n"
            "- Chest pain or pressure\n"
            "- Inability to breathe or severe shortness of breath\n"
            "- Unconsciousness or unresponsiveness\n"
            "- Uncontrolled bleeding\n"
            "- Seizure or stroke symptoms\n"
            "- Pain rated 9 or 10 out of 10\n"
            "- Any other immediately life-threatening condition\n\n"
            "Flag NO for:\n"
            "- Moderate pain (1-8/10) with no other emergency signs\n"
            "- Nausea, fever, dizziness alone\n"
            "- Vague or unclear responses\n"
            "- Discomfort that is not life-threatening\n\n"
            "When in doubt, reply NO.\n\n"
            f"Conversation:\n{history_text}\n\n"
            "Reply ONLY with YES or NO."
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

        result  = r.json()["response"].strip().upper().split("\n")[0]
        flagged = result.startswith("YES")
        print(f"[flag_urgent] AI says: {result} → flagged={flagged}")
        return jsonify({"flagged_urgent": flagged})

    except Exception as e:
        print(f"[flag_urgent] Error: {e}")
        return jsonify({"flagged_urgent": False})


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _get_next_question_only(history_text, history):
    """Called on the fast path — status already known, just need next Q."""
    prompt = (
        "You are a calm triage assistant.\n"
        "Ask ONE short follow-up question.\n"
        "Ask the most pertinent missing question.\n"
        "Do not repeat questions already asked.\n"
        "Use simple everyday words. Be clear and specific.\n"
        "Keep it under 15 words.\n"
        "Topics to cover: main problem, pain level, duration, "
        "breathing problems, medical conditions, medications.\n"
        "If enough info is collected, reply ONLY with: DONE\n\n"
        f"Conversation:\n{history_text}\n\n"
        "Assistant:"
    )

    try:
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
        raw = r.json()["response"].strip().split("\n")[0].strip()
        return _clean_question(raw)
    except Exception as e:
        print(f"[next_question_only] Error: {e}")
        return "DONE"


FILLER_PREFIXES = [
    "I'm glad to hear that.", "I'm glad to hear that!",
    "Great!", "Good!", "Good to know.", "Okay,", "Okay.",
    "I see.", "I see,", "Alright,", "Alright.", "Sure,",
    "Thank you.", "Thank you!", "Got it.", "Got it,",
    "Of course.", "Of course!", "Certainly.", "Certainly!",
    "Understood.", "Understood,", "Noted.", "Noted,",
    "I understand.", "I understand,", "That's good.", "That's good!",
    "That's helpful.", "That's helpful!", "Thanks for sharing.",
]

def _clean_question(question):
    for filler in FILLER_PREFIXES:
        if question.lower().startswith(filler.lower()):
            question = question[len(filler):].strip()
    if "?" in question:
        question = question[:question.index("?") + 1].strip()
    return question


def _extract_status_from_text(text):
    for s in ("URGENT", "MONITOR", "STABLE"):
        if s in text.upper():
            return s
    return "STABLE"


def _extract_question_from_text(text):
    # Look for anything after "question": in the raw text
    match = re.search(r'"question"\s*:\s*"([^"]+)"', text)
    if match:
        return _clean_question(match.group(1))
    # Fall back: find the first sentence ending in ?
    match = re.search(r'([A-Z][^?]+\?)', text)
    if match:
        return _clean_question(match.group(1).strip())
    return "DONE"


# ═══════════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[server] Starting on port {SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=True)