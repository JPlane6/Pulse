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
whisper_model = whisper.load_model("small")   # upgraded from base → small
print("[server] Whisper ready!")

# ═══════════════════════════════════════════════════════════════════
# TRIAGE LOGIC
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
    # Try digit first (more reliable)
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


def assess(text):
    """
    Three-tier classifier.  Returns "URGENT", "MONITOR", or "STABLE".

    Priority order:
      1. Hard-coded keyword shortcuts (fast, reliable)
      2. Pain score parsing
      3. AI classification via phi3:mini (fallback)
    """
    if not text.strip():
        return "STABLE"

    words = text.strip().split()
    if len(words) < 2:
        return "STABLE"

    lower = text.lower()

    # ── 1. Hard keyword shortcuts ───────────────────────────────────
    for kw in URGENT_HARD:
        if kw in lower:
            print(f"[assess] URGENT via keyword: '{kw}'")
            return "URGENT"

    for kw in MONITOR_HARD:
        if kw in lower:
            print(f"[assess] MONITOR via keyword: '{kw}'")
            return "MONITOR"

    # ── 2. Pain score ───────────────────────────────────────────────
    score = extract_pain_score(lower)
    if score is not None:
        if score >= 8:
            return "URGENT"
        if score >= 4:
            return "MONITOR"
        # 1-3 → STABLE, fall through to AI for confirmation

    # ── 3. AI classification ────────────────────────────────────────
    try:
        prompt = (
            "You are a clinical triage assistant.\n"
            "A patient said: \"{text}\"\n\n"
            "Classify their condition as exactly one of: URGENT, MONITOR, STABLE\n\n"
            "URGENT = life-threatening emergency: severe chest pain, cannot breathe, "
            "unconscious, uncontrolled bleeding, seizure, stroke symptoms.\n"
            "MONITOR = moderate concern needing attention soon: moderate pain (4-7/10), "
            "fever, dizziness, nausea, shortness of breath, confusion.\n"
            "STABLE = mild or no symptoms: low pain (1-3/10), feeling okay, minor discomfort.\n\n"
            "IMPORTANT: Only classify as URGENT if there are very clear emergency signals. "
            "When in doubt between URGENT and MONITOR, choose MONITOR. "
            "When in doubt between MONITOR and STABLE, choose STABLE.\n"
            "If the response is vague, unclear, or doesn't describe symptoms, reply: STABLE\n\n"
            "Reply with ONLY one word: URGENT, MONITOR, or STABLE"
        ).format(text=text)

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 5,
                    "num_ctx": 512,
                    "keep_alive": "-1"
                }
            },
            timeout=30
        )

        result = r.json()["response"].strip().upper().split("\n")[0]
        print(f"[assess] AI says: {result}")

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

        result = whisper_model.transcribe(
            path,
            fp16=False,
            language="en",
            condition_on_previous_text=False,   # prevents hallucination loops
            no_speech_threshold=0.6,            # more aggressive silence rejection
            logprob_threshold=-1.0,             # drop low-confidence words
        )
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
    """
    Final holistic pass over the full conversation.
    Much stricter than per-answer assess() — only fires on clear emergencies.
    """
    try:
        history = request.get_json()["history"]

        history_text = "\n".join([
            f"Nurse: {qa['q']}\nPatient: {qa['a']}"
            for qa in history
        ])

        # Count how many answers were already flagged URGENT
        urgent_count  = sum(1 for qa in history if qa.get("status") == "URGENT")
        monitor_count = sum(1 for qa in history if qa.get("status") == "MONITOR")

        # Quick exit: if nothing was flagged URGENT during Q&A, skip the AI call
        # unless 3+ MONITOR flags — then still check
        if urgent_count == 0 and monitor_count < 3:
            print(f"[flag_urgent] No urgent signals during Q&A (U:{urgent_count} M:{monitor_count}) — skipping AI, returning False")
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
        flagged = result.startswith("YES")   # stricter match than "YES" in result

        print(f"[flag_urgent] AI says: {result} → flagged={flagged}")
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