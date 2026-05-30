import whisper
import requests
import re
import json
import base64
import io
import traceback
import numpy as np
import soundfile as sf

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

OLLAMA_URL  = "http://localhost:11434/api/generate"
TEXT_MODEL  = "phi3:mini"
SERVER_PORT = 5001

# OPTIMIZATION: Faster inference with reduced context and predictions
OLLAMA_OPTS_FAST = {
    "temperature":    0.1,
    "num_predict":    15,      # Reduced from 20 - shorter questions
    "num_ctx":        256,     # Reduced from 512 for speed
    "repeat_penalty": 1.3,
    "top_k":          20,
    "top_p":          0.8,
    "keep_alive":     "-1"
}

OLLAMA_OPTS_URGENT = {
    "temperature": 0.0,
    "num_predict": 3,
    "num_ctx":     256,
    "keep_alive":  "-1"
}

# ═══════════════════════════════════════════════════════════════════
# LOAD WHISPER
# ═══════════════════════════════════════════════════════════════════

# OPTIMIZATION: Use 'tiny' for 3-5x faster transcription than 'base'
# Trade: ~5% accuracy loss, acceptable for medical keywords
print("[server] Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("[server] Whisper ready.")

# ═══════════════════════════════════════════════════════════════════
# TRIAGE KEYWORDS
# ═══════════════════════════════════════════════════════════════════

URGENT_HARD = [
    "can't breathe", "cannot breathe", "chest pain", "chest hurt", "chest tightness",
    "heart attack", "heart pain", "unconscious", "unresponsive", "not breathing",
    "stopped breathing", "seizure", "stroke", "overdose", "bleeding out",
    "can't move", "cannot move", "help me", "dying", "severe pain",
    "crushing pain", "pressure in chest", "pain in chest"
]

MONITOR_HARD = [
    "dizzy", "dizziness", "nausea", "vomiting", "fever", "chills",
    "shortness of breath", "hard to breathe", "difficulty breathing",
    "swelling", "rash", "allergic", "infection", "confusion", "disoriented",
    "weak", "weakness", "lightheaded", "faint"
]

# ═══════════════════════════════════════════════════════════════════
# HELPERS
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
        if re.search(rf'\b{word}\b', text.lower()):
            return val
    return None


def keyword_status(text):
    """
    Fast keyword check. Returns a status string if matched, or None if
    the answer needs AI classification.
    """
    if not text.strip():
        return "STABLE"
    lower = text.lower()
    
    # Check for positive/reassuring responses first
    STABLE_PHRASES = [
        "nothing wrong", "nothing is bothering", "nothing bothering", "no problem", 
        "doing fine", "doing okay", "doing well", "feeling fine", "feeling okay", 
        "feeling good", "feel fine", "feel okay", "i'm fine", "i'm okay", "i'm good", 
        "im fine", "im okay", "im good", "all good", "everything's fine", 
        "everything is fine", "no issues", "not bothering", "no pain", 
        "feeling better", "much better", "i am fine", "i am okay", "i am good"
    ]
    for phrase in STABLE_PHRASES:
        if phrase in lower:
            print(f"[keyword] STABLE via reassuring phrase '{phrase}'")
            return "STABLE"
    
    # Check hard-coded urgent keywords
    for kw in URGENT_HARD:
        if kw in lower:
            print(f"[keyword] URGENT via '{kw}'")
            return "URGENT"
    
    # Check hard-coded monitor keywords
    for kw in MONITOR_HARD:
        if kw in lower:
            print(f"[keyword] MONITOR via '{kw}'")
            return "MONITOR"
    
    # Check qualitative pain descriptors
    if "intense" in lower or "severe" in lower or "excruciating" in lower:
        print(f"[keyword] URGENT via intense pain")
        return "URGENT"
    if "moderate" in lower:
        print(f"[keyword] MONITOR via moderate pain")
        return "MONITOR"
    if "mild" in lower or "slight" in lower or "minor" in lower:
        print(f"[keyword] STABLE via mild pain")
        return "STABLE"
    
    # Fallback: Check numeric pain score (1-10) if patient mentions a number
    # Note: We ask for mild/moderate/intense, but accept numbers as backup
    score = extract_pain_score(lower)
    if score is not None:
        if score >= 8: return "URGENT"
        if score >= 4: return "MONITOR"
        return "STABLE"
    
    return None  # needs AI


# ═══════════════════════════════════════════════════════════════════
# WHISPER
# ═══════════════════════════════════════════════════════════════════

def transcribe_bytes(wav_bytes):
    try:
        buf          = io.BytesIO(wav_bytes)
        audio_np, sr = sf.read(buf, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        # OPTIMIZATION: Tiny model with minimal decoding for maximum speed
        result = whisper_model.transcribe(
            audio_np,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            beam_size=1,           # Greedy decode - fastest
            best_of=1,             # No candidates - fastest
            temperature=0.0        # Deterministic
        )
        text = result["text"].strip()
        print(f"[whisper] '{text}'")
        return text
    except Exception as e:
        print(f"[whisper] Error: {repr(e)}")
        traceback.print_exc()
        return ""


# ═══════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════

FILLER_PREFIXES = [
    "I'm glad to hear that.", "Great!", "Good!", "Okay,", "Okay.",
    "I see.", "Alright.", "Sure,", "Thank you.", "Got it.",
    "Of course.", "Certainly.", "Understood.", "Noted.",
]

DONE_SIGNALS = [
    "done", "no further", "no more", "enough information",
    "no other information", "sufficient information", "that's all",
    "reply done", "i have enough", "no additional", "if all", "all covered",
    "all topics", "topics covered", "output done", "output: done",
]


def clean_question(q):
    if not q:
        return "DONE"
    q = q.strip()

    # Strip filler openers
    for filler in FILLER_PREFIXES:
        if q.lower().startswith(filler.lower()):
            q = q[len(filler):].strip()

    # Catch done signals
    q_lower = q.lower()
    for signal in DONE_SIGNALS:
        if signal in q_lower:
            return "DONE"

    # Hard cut at first question mark
    if "?" in q:
        q = q[:q.index("?") + 1].strip()
    else:
        return "DONE"

    if not q or len(q) > 120:
        return "DONE"

    return q


# ═══════════════════════════════════════════════════════════════════
# OLLAMA WRAPPERS
# ═══════════════════════════════════════════════════════════════════

def phi3(prompt, opts):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": prompt, "stream": False, "options": opts},
            timeout=60
        )
        print("[ollama status]", r.status_code)
        if not r.text.strip():
            print("[ollama] empty response")
            return "DONE"
        print("[ollama raw]", r.text[:300])
        r.raise_for_status()
        try:
            parsed = r.json()
        except Exception:
            print("[ollama] invalid JSON")
            return "DONE"
        if "response" not in parsed:
            print("[ollama] missing response field")
            return "DONE"
        return parsed["response"].strip()
    except Exception as e:
        print(f"[ollama] Error: {repr(e)}")
        traceback.print_exc()
        return "DONE"


def phi3_stream(prompt, opts):
    """
    Yields tokens from Ollama as they arrive.
    Stops as soon as a '?' is seen — we only want one question,
    so there's no point streaming past it.
    """
    accumulated = []
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": prompt, "stream": True, "options": opts},
            stream=True,
            timeout=60
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                token = chunk.get("response", "")
                if token:
                    accumulated.append(token)
                    yield token
                    if "?" in "".join(accumulated):
                        break
                if chunk.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"[ollama stream] Error: {repr(e)}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════

def build_question_prompt(full_history_text, questions_asked):
    """
    Patient-friendly prompt: conversational, follows up naturally on responses
    """
    return (
        "You are Pulse, a caring nurse robot talking to a patient.\n\n"
        "YOUR GOAL: Collect ALL these details through natural conversation:\n"
        "  1. Pain intensity → MUST ask EXACTLY: 'Is it intense, moderate, or mild?'\n"
        "  2. Location → Where exactly does it hurt?\n"
        "  3. Duration → How long have they felt this way?\n"
        "  4. Breathing → Any trouble breathing?\n"
        "  5. Medications/Allergies\n\n"
        "STRICT RULES:\n"
        f"- You've asked {questions_asked}/6 questions so far\n"
        "- NEVER say DONE if you've asked fewer than 3 questions\n"
        "- When asking about pain intensity, use EXACTLY these words: 'Is it intense, moderate, or mild?'\n"
        "- Look at what's MISSING from the conversation above\n"
        "- Ask about ONE missing item per question\n"
        "- Keep questions SHORT (under 10 words)\n"
        "- Only say DONE when ALL 5 items collected OR 6 questions asked\n\n"
        f"Conversation so far:\n{full_history_text}\n\n"
        "Your next question:"
    )


def build_status_prompt(patient_answer):
    return (
        f"A patient just said: '{patient_answer}'\n\n"
        "Classify urgency in ONE WORD:\n\n"
        "URGENT = life-threatening symptoms requiring immediate emergency care\n"
        "Examples: chest pain, can't breathe, severe bleeding, unconscious, stroke, intense/severe pain\n\n"
        "MONITOR = concerning symptoms that need attention but not immediate emergency\n"
        "Examples: moderate pain, fever, dizziness, vomiting, difficulty breathing, swelling\n\n"
        "STABLE = no concerning symptoms, patient is fine, or only mild issues\n"
        "Examples: 'nothing wrong', 'feeling fine', 'doing okay', 'no problems', mild discomfort, minor aches\n\n"
        "IMPORTANT: If patient says they're fine/okay/good or has nothing wrong → STABLE\n\n"
        "Answer (URGENT, MONITOR, or STABLE):"
    )


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/turn_stream", methods=["POST"])
def turn_stream():
    """
    OPTIMIZED: Fast transcription (tiny), streams question immediately,
    status classification runs in parallel while Piper speaks.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"error": "invalid body"}), 400

        audio_b64 = body.get("audio_b64", "")
        history   = body.get("history", [])

        if not audio_b64:
            return jsonify({"error": "missing audio_b64"}), 400

        try:
            wav_bytes = base64.b64decode(audio_b64)
        except Exception:
            return jsonify({"error": "invalid base64"}), 400

        # OPTIMIZATION: Tiny Whisper transcribes in ~0.3-0.5s
        text = transcribe_bytes(wav_bytes)

        if not text:
            def empty_gen():
                yield f"data: {json.dumps({'done': True, 'text': '', 'status': 'STABLE', 'question': 'DONE'})}\n\n"
            return Response(empty_gen(), mimetype="text/event-stream")

        # Build full history including this answer
        history_text = "\n".join([
            f"Nurse: {qa.get('q','')}\nPatient: {qa.get('a','')}"
            for qa in history
        ])
        questions_asked = len(history) + 1
        full_history = history_text + f"\nPatient: {text}" if history_text else f"Patient: {text}"

        # Fast keyword check
        fast_status = keyword_status(text)

        def generate_stream():
            # OPTIMIZATION: Stream question immediately, status classifies in parallel
            print(f"[turn_stream] streaming (asked={questions_asked})...")
            question_tokens = []
            
            for token in phi3_stream(
                build_question_prompt(full_history, questions_asked),
                OLLAMA_OPTS_FAST
            ):
                question_tokens.append(token)
                yield f"data: {json.dumps({'t': token})}\n\n"
                if "?" in "".join(question_tokens):
                    break

            question = clean_question("".join(question_tokens).split("\n")[0].strip())

            # Status: keyword if available, else AI (runs WHILE Piper speaks)
            if fast_status is not None:
                status = fast_status
                print(f"[turn_stream] keyword status={status}")
            else:
                status_raw = phi3(build_status_prompt(text), OLLAMA_OPTS_URGENT)
                status = status_raw.upper().strip()
                if status not in ["URGENT", "MONITOR", "STABLE"]:
                    status = "STABLE"
                print(f"[turn_stream] AI status={status}")

            print(f"[turn_stream] done — status={status} question='{question}'")
            yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"

        return Response(generate_stream(), mimetype="text/event-stream")

    except Exception as e:
        print(f"[turn_stream] Error: {repr(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/flag_urgent", methods=["POST"])
def flag_urgent():
    """
    Final pass after all questions are done. Only calls AI if there's
    enough signal — skips the call entirely for obviously stable sessions.
    """
    try:
        body         = request.get_json(force=True, silent=True) or {}
        history      = body.get("history", [])
        history_text = "\n".join([
            f"Nurse: {qa.get('q','')}\nPatient: {qa.get('a','')}" for qa in history
        ])

        urgent_count  = sum(1 for qa in history if qa.get("status") == "URGENT")
        monitor_count = sum(1 for qa in history if qa.get("status") == "MONITOR")

        # If ANY urgent classification, always check with AI
        if urgent_count == 0 and monitor_count == 0:
            print("[flag_urgent] no concerning signals - skipping AI")
            return jsonify({"flagged_urgent": False})

        prompt = (
            "You are a senior triage nurse reviewing this patient conversation.\n\n"
            "Does this patient need IMMEDIATE EMERGENCY care right now?\n\n"
            "Say YES if:\n"
            "- Severe chest pain or heart attack symptoms\n"
            "- Can't breathe or severe breathing difficulty\n"
            "- Unconscious or unresponsive\n"
            "- Severe bleeding or trauma\n"
            "- Stroke symptoms\n"
            "- Intense/severe pain\n\n"
            "Say NO for all other cases (even if they need monitoring).\n\n"
            f"Conversation:\n{history_text}\n\n"
            "Your decision (YES or NO):"
        )

        raw     = phi3(prompt, OLLAMA_OPTS_URGENT)
        flagged = raw.upper().startswith("YES")
        print(f"[flag_urgent] {raw} -> {flagged}")
        return jsonify({"flagged_urgent": flagged})

    except Exception as e:
        print(f"[flag_urgent] Error: {repr(e)}")
        traceback.print_exc()
        return jsonify({"flagged_urgent": False})


# ═══════════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[server] Starting on port {SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=True)