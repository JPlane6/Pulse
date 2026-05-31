import whisper
import requests
import re
import json
import base64
import io
import traceback
import numpy as np
import soundfile as sf
import logging
import random

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# Disable Werkzeug request logging spam for cleaner demo output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

OLLAMA_URL  = "http://localhost:11434/api/generate"
TEXT_MODEL  = "qwen2.5:3b"  # Better instruction following than phi3:mini
SERVER_PORT = 5001

# Question limits - MIN enforced in code logic
MIN_QUESTIONS = 3
MAX_QUESTIONS = 6

# OPTIMIZATION: Qwen 2.5 3B optimized settings
OLLAMA_OPTS_FAST = {
    "temperature": 0.3,        # slightly higher = more natural variation
    "top_p": 0.7,
    "top_k": 20,
    "repeat_penalty": 1.15,
    "num_predict": 40,         # was 18 — allows 10-15 word natural questions
    "num_ctx": 2048,            # was 192 — more context for better follow-ups
    "keep_alive": "-1",
    "stop": ["\n", "Patient:", "Nurse:"]
}

OLLAMA_OPTS_URGENT = {
    "temperature": 0,
    "top_p": 0.1,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "num_predict": 3,           # one word (URGENT/MONITOR/STABLE) + wiggle room
    "num_ctx": 2048,            # must match OLLAMA_OPTS_FAST — prevents model runner reload between calls
    "keep_alive": "-1",
    "stop": ["\n", "Patient:", "Nurse:"]
}

# ═══════════════════════════════════════════════════════════════════
# LOAD WHISPER
# ═══════════════════════════════════════════════════════════════════

print("[server] Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("[server] Whisper ready.")

# ═══════════════════════════════════════════════════════════════════
# TRIAGE KEYWORDS
# ═══════════════════════════════════════════════════════════════════

URGENT_HARD = [
    "can't breathe", "cannot breathe", "chest pain", "chest hurt", "chest tightness",
    "heart attack", "heart pain", "unconscious", "unresponsive", "not breathing",
    "stopped breathing", "seizure", "stroke", "overdose", "bleeding out",
    "can't move", "cannot move", "help me", "dying", "severe pain",
    "crushing pain", "pressure in chest", "pain in chest",
    "my chest hurts", "tight chest", "chest feels tight", "hurting chest"
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

SEVERITY_ORDER = {
    "STABLE": 0,
    "MONITOR": 1,
    "URGENT": 2
}

QUESTION_FALLBACKS = {
    "pain intensity": "Would you say the pain is intense, moderate, or mild?",
    "pain location": "Can you tell me exactly where it hurts?",
    "symptom duration": "How long have you been feeling this way?",
    "breathing status": "Are you having any trouble breathing at all?",
    "medications and allergies": "Are you currently taking any medications or have any allergies?"
}

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
    if not text.strip():
        return "STABLE"
    lower = text.lower()

    for kw in URGENT_HARD:
        if kw in lower:
            print(f"[keyword] URGENT via '{kw}'")
            return "URGENT"

    for kw in MONITOR_HARD:
        if kw in lower:
            print(f"[keyword] MONITOR via '{kw}'")
            return "MONITOR"

    has_intense = "intense" in lower or "severe" in lower or "excruciating" in lower
    has_critical_context = any(word in lower for word in [
        "chest", "heart", "breath", "unconscious", "stroke", "bleeding", "head"
    ])

    if has_intense and has_critical_context:
        print(f"[keyword] URGENT via intense pain + critical context")
        return "URGENT"
    elif has_intense:
        print(f"[keyword] MONITOR via intense pain (no critical context)")
        return "MONITOR"

    if "moderate" in lower:
        return "MONITOR"
    if "mild" in lower or "slight" in lower or "minor" in lower:
        return "STABLE"

    pain_context = any(word in lower for word in ["pain", "hurt", "hurts", "aching", "ache"])
    score = extract_pain_score(lower) if pain_context else None
    if score is not None:
        if score >= 8 and has_critical_context:
            return "URGENT"
        elif score >= 8:
            return "MONITOR"
        elif score >= 4:
            return "MONITOR"
        else:
            return "STABLE"

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
            return "STABLE"

    return None


def extract_patient_text(history):
    return " ".join(qa.get("a", "") for qa in history).lower()


def get_missing_fields(history):
    """
    Returns a SHUFFLED list of missing fields so question order
    varies each session instead of always being the same 5.
    """
    if isinstance(history, str):
        patient_text = history.lower()
    else:
        patient_text = extract_patient_text(history)

    missing = []

    if not any(word in patient_text for word in ["intense", "moderate", "mild", "severe"]):
        missing.append("pain intensity")

    if not any(word in patient_text for word in [
        "chest", "head", "stomach", "arm", "leg", "back", "neck", "knee", "ankle",
        "shoulder", "throat", "abdomen", "hip", "wrist", "jaw", "foot", "hand"
    ]):
        missing.append("pain location")

    if not any(word in patient_text for word in [
        "minutes", "minute", "hours", "hour", "days", "day", "weeks", "week",
        "months", "started", "ago", "yesterday", "today", "morning", "tonight",
        "evening", "afternoon"
    ]):
        missing.append("symptom duration")

    if not any(word in patient_text for word in [
        "breath", "breathing", "wheezing", "suffocating", "gasping"
    ]):
        missing.append("breathing status")

    if not any(word in patient_text for word in [
        "medication", "medicine", "taking", "allergic", "allergy"
    ]):
        missing.append("medications and allergies")

    # Shuffle so order varies between sessions
    random.shuffle(missing)
    return missing


# ═══════════════════════════════════════════════════════════════════
# WHISPER
# ═══════════════════════════════════════════════════════════════════

def transcribe_bytes(wav_bytes):
    try:
        buf          = io.BytesIO(wav_bytes)
        audio_np, sr = sf.read(buf, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        result = whisper_model.transcribe(
            audio_np,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            without_timestamps=True
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
    "have covered", "we have", "information needed", "collected all"
]


def clean_question(q, questions_asked):
    if not q or not q.strip():
        return "What else is bothering you?"

    q = q.strip()

    for filler in FILLER_PREFIXES:
        if q.lower().startswith(filler.lower()):
            q = q[len(filler):].strip()

    q_lower = q.lower().strip()
    if q_lower in DONE_SIGNALS or q_lower == "done":
        return "DONE"

    if "?" in q:
        q = q[:q.index("?") + 1].strip()
    else:
        question_starters = ["what", "where", "when", "how", "why", "is", "are", "can", "do", "does", "have", "has"]
        if any(q.lower().startswith(word) for word in question_starters):
            q = q + "?"
        else:
            if "." in q:
                q = q[:q.index(".") + 1].strip()
            if len(q) > 3 and len(q) < 100:
                q = q.rstrip(".") + "?"

    q = re.sub(r'[.!]+$', '', q).strip()

    if q == "DONE":
        return "DONE"

    if not q.endswith("?"):
        q = q + "?"

    if len(q) > 120:
        q = q[:120]
        last_space = q.rfind(" ")
        if last_space > 50:
            q = q[:last_space].strip()
        if not q.endswith("?"):
            q = q + "?"

    if len(q) < 3 or q == "?":
        return "What else should I know?"

    return q


def sanitize_question(text):
    text = text.strip()

    if text.upper() == "DONE":
        return "DONE"

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

    lines = text.splitlines()
    text = lines[0].strip()

    if "?" in text:
        text = text[:text.index("?") + 1]

    text = re.sub(r'^(okay|alright|well|so|hmm)[,. ]*', '', text, flags=re.I)

    # Increased from 10 to 18 words — allows natural, full-length questions
    MAX_WORDS = 18
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
        if not text.endswith("?"):
            text += "?"

    if len(text.split()) < 2 and text != "DONE":
        return "What symptoms are you experiencing?"

    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# OLLAMA WRAPPERS
# ═══════════════════════════════════════════════════════════════════

def ollama(prompt, opts):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": prompt, "stream": False, "options": opts},
            timeout=20
        )
        print("[ollama status]", r.status_code)
        if not r.text.strip():
            print("[ollama] empty response")
            return ""
        print("[ollama raw]", r.text[:300])
        r.raise_for_status()
        try:
            parsed = r.json()
        except Exception:
            print("[ollama] invalid JSON")
            return ""
        if "response" not in parsed:
            print("[ollama] missing response field")
            return ""
        return parsed["response"].strip()
    except Exception as e:
        print(f"[ollama] Error: {repr(e)}")
        traceback.print_exc()
        return ""


def ollama_stream(prompt, opts):
    accumulated = []
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": prompt, "stream": True, "options": opts},
            stream=True,
            timeout=40
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
                    full = "".join(accumulated)
                    if "?" in full:
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

def build_question_prompt(history, full_history_text, questions_asked):
    """
    Lets the AI decide what to ask next based on the full conversation.
    No rigid next_topic field — removes the repetitive 5-question loop.
    Softened rules allow natural, full-length questions.
    """
    missing = get_missing_fields(history)

    # Short-circuit: enforce MIN_QUESTIONS before allowing DONE
    if questions_asked < MIN_QUESTIONS:
        done_allowed = False
    elif not missing:
        done_allowed = True
    else:
        done_allowed = False

    # Give the AI awareness of what's still uncovered, but don't lock it in
    missing_hint = (
        f"Still uncovered: {', '.join(missing)}." if missing
        else "All key topics have been addressed."
    )

    mode = "DONE_ALLOWED" if done_allowed else "KEEP_ASKING"

    return f"""
You are Pulse, a nurse triage robot conducting a patient assessment.

Conversation so far:
{full_history_text}

{missing_hint}

Mode: {mode}

YOUR TASK:
Ask the single most important follow-up question based on what the patient said.
Pick the topic that matters most right now given their specific answers.

RULES:
- One question only, ending with a single question mark
- 8 to 18 words — long enough to be clear and natural, short enough to be quick
- Sound like a real nurse — warm but professional
- Do NOT repeat any question already asked
- Do NOT ask multiple questions at once
- Do NOT add explanations, diagnoses, or comments after the question

DONE RULES:
- If mode is KEEP_ASKING: never output DONE
- If mode is DONE_ALLOWED and all important info is collected: output only the word DONE

GOOD EXAMPLES:
Can you describe exactly where the pain is located?
How long have you been experiencing these symptoms?
On a scale of mild, moderate, or intense, how would you rate the pain?
Are you currently taking any medications I should know about?
Is the pain constant, or does it come and go?
Have you had any difficulty breathing since this started?
Did anything specific happen right before you started feeling this way?

Output only the question (or DONE):
""".strip()


def build_status_prompt(patient_answer):
    return f"""
Patient message:
"{patient_answer}"

Classify urgency.

URGENT:
- chest pain
- severe breathing problems
- unconscious
- stroke symptoms
- seizure
- overdose
- severe bleeding

MONITOR:
- intense pain (unless with chest/heart/breathing)
- fever
- dizziness
- vomiting
- swelling
- rash
- weakness
- moderate pain
- mild breathing issues

STABLE:
- feels fine
- no symptoms
- mild discomfort
- improving
- no concerns

STRICT RULES:
- Output ONE WORD ONLY
- No explanation
- No punctuation
- No extra text

Allowed outputs:
URGENT
MONITOR
STABLE

Answer:
""".strip()


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/turn_stream", methods=["POST"])
def turn_stream():
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

        text = transcribe_bytes(wav_bytes)

        if not text:
            def empty_gen():
                yield f"data: {json.dumps({'done': True, 'text': '', 'status': 'STABLE', 'question': 'Could you say that again? I did not quite catch it.'})}\n\n"
            return Response(empty_gen(), mimetype="text/event-stream")

        history_text = "\n".join([
            f"Nurse: {qa.get('q','')}\nPatient: {qa.get('a','')}"
            for qa in history
        ])
        questions_asked = sum(1 for qa in history if qa.get("q"))
        full_history = history_text + f"\nPatient: {text}" if history_text else f"Patient: {text}"

        temp_history = history + [{"a": text}]
        missing = get_missing_fields(temp_history)

        def generate_stream():
            print(f"[turn_stream] streaming (asked={questions_asked})...")

            if questions_asked >= MAX_QUESTIONS:
                print(f"[turn_stream] MAX_QUESTIONS reached, forcing DONE")
                question = "DONE"

                keyword_result = keyword_status(text)
                status_raw = ollama(build_status_prompt(text), OLLAMA_OPTS_URGENT)
                status_raw = status_raw.upper().strip()
                if "URGENT" in status_raw:
                    ai_status = "URGENT"
                elif "MONITOR" in status_raw:
                    ai_status = "MONITOR"
                else:
                    ai_status = "STABLE"

                if keyword_result:
                    status = max(keyword_result, ai_status, key=lambda x: SEVERITY_ORDER[x])
                else:
                    status = ai_status

                yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"
                return

            question_tokens = []

            for token in ollama_stream(
                build_question_prompt(temp_history, full_history, questions_asked),
                OLLAMA_OPTS_FAST
            ):
                question_tokens.append(token)
                yield f"data: {json.dumps({'t': token})}\n\n"

            if not question_tokens:
                print(f"[turn_stream] Empty stream, using fallback")
                next_topic = missing[0] if missing else None
                question = QUESTION_FALLBACKS.get(next_topic, "Is there anything else that's been bothering you?")

                keyword_result = keyword_status(text)
                status_raw = ollama(build_status_prompt(text), OLLAMA_OPTS_URGENT)
                status_raw = status_raw.upper().strip()
                if "URGENT" in status_raw:
                    ai_status = "URGENT"
                elif "MONITOR" in status_raw:
                    ai_status = "MONITOR"
                else:
                    ai_status = "STABLE"

                if keyword_result:
                    status = max(keyword_result, ai_status, key=lambda x: SEVERITY_ORDER[x])
                else:
                    status = ai_status

                yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"
                return

            question = clean_question("".join(question_tokens).split("\n")[0].strip(), questions_asked)
            question = sanitize_question(question)

            if "?" not in question and question != "DONE":
                next_topic = missing[0] if missing else None
                question = QUESTION_FALLBACKS.get(next_topic, "Is there anything else that's been bothering you?")

            # Prevent repeated questions
            asked_questions = {qa.get("q", "").lower().strip() for qa in history}
            if question.lower().strip() in asked_questions:
                print(f"[turn_stream] Repeated question detected, using fallback")
                next_topic = missing[0] if missing else None
                question = QUESTION_FALLBACKS.get(next_topic, "Is there anything else that's been bothering you?")

                if question.lower().strip() in asked_questions:
                    unused = [
                        q for q in QUESTION_FALLBACKS.values()
                        if q.lower().strip() not in asked_questions
                    ]
                    question = unused[0] if unused else "Is there anything else you'd like me to know?"

            # Status classification
            keyword_status_result = keyword_status(text)
            if keyword_status_result is not None:
                print(f"[turn_stream] keyword status={keyword_status_result}")

            status_raw = ollama(build_status_prompt(text), OLLAMA_OPTS_URGENT)
            status_raw = status_raw.upper().strip()
            if "URGENT" in status_raw:
                ai_status = "URGENT"
            elif "MONITOR" in status_raw:
                ai_status = "MONITOR"
            else:
                ai_status = "STABLE"
            print(f"[turn_stream] AI status={ai_status}")

            if keyword_status_result is not None:
                status = max(keyword_status_result, ai_status, key=lambda x: SEVERITY_ORDER[x])
            else:
                status = ai_status

            print(f"[turn_stream] done — status={status} question='{question}'")
            yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"

        return Response(generate_stream(), mimetype="text/event-stream")

    except Exception as e:
        print(f"[turn_stream] Error: {repr(e)}")
        traceback.print_exc()
        def error_gen():
            yield f"data: {json.dumps({'done': True, 'text': '', 'status': 'STABLE', 'question': 'Could you say that again? I did not quite catch it.'})}\n\n"
        return Response(error_gen(), mimetype="text/event-stream")


@app.route("/flag_urgent", methods=["POST"])
def flag_urgent():
    try:
        body         = request.get_json(force=True, silent=True) or {}
        history      = body.get("history", [])
        history_text = "\n".join([
            f"Nurse: {qa.get('q','')}\nPatient: {qa.get('a','')}" for qa in history
        ])

        urgent_count  = sum(1 for qa in history if qa.get("status") == "URGENT")
        monitor_count = sum(1 for qa in history if qa.get("status") == "MONITOR")

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

        raw     = ollama(prompt, OLLAMA_OPTS_URGENT)
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
    print(f"[server] Preloading {TEXT_MODEL}...")
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": "hi", "stream": False},
            timeout=30
        )
        print(f"[server] Model preloaded.")
    except Exception as e:
        print(f"[server] Preload warning: {e}")

    print(f"[server] Starting on port {SERVER_PORT}")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False, threaded=True)