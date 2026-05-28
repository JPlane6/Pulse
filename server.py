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

OLLAMA_OPTS_FAST = {
    "temperature":    0.1,
    "num_predict":    25,    # hard cap — a 12-word question ≈ 15 tokens; 25 kills rambling
    "num_ctx":        512,
    "repeat_penalty": 1.3,
    "top_k":          20,
    "top_p":          0.8,
    "keep_alive":     "-1"
}

OLLAMA_OPTS_URGENT = {
    "temperature": 0.0,
    "num_predict": 3,
    "num_ctx":     512,
    "keep_alive":  "-1"
}

# ═══════════════════════════════════════════════════════════════════
# LOAD WHISPER
# ═══════════════════════════════════════════════════════════════════

print("[server] Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("[server] Whisper ready.")

# ═══════════════════════════════════════════════════════════════════
# TRIAGE KEYWORDS
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
    score = extract_pain_score(lower)
    if score is not None:
        if score >= 8: return "URGENT"
        if score >= 4: return "MONITOR"
        return "STABLE"
    return None


# ═══════════════════════════════════════════════════════════════════
# WHISPER
# ═══════════════════════════════════════════════════════════════════

def transcribe_bytes(wav_bytes):
    try:
        buf      = io.BytesIO(wav_bytes)
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


def clean_question(q):
    if not q:
        return "DONE"
    q = q.strip()

    # Strip filler openers
    for filler in FILLER_PREFIXES:
        if q.lower().startswith(filler.lower()):
            q = q[len(filler):].strip()

    # Hard cut at first question mark — kills all trailing instructions/rambling
    if "?" in q:
        q = q[:q.index("?") + 1].strip()

    # Sanity guards
    if not q:
        return "DONE"
    if q.upper() == "DONE":
        return "DONE"
    if len(q) > 120:   # model escaped the num_predict cap somehow
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
            print("[ollama] empty response"); return "DONE"
        print("[ollama raw]", r.text[:300])
        r.raise_for_status()
        try:
            parsed = r.json()
        except Exception:
            print("[ollama] invalid JSON"); return "DONE"
        if "response" not in parsed:
            print("[ollama] missing response field"); return "DONE"
        return parsed["response"].strip()
    except Exception as e:
        print(f"[ollama] Error: {repr(e)}")
        traceback.print_exc()
        return "DONE"


def phi3_stream(prompt, opts):
    """Yields tokens as they arrive from Ollama."""
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
                if "response" in chunk and chunk["response"]:
                    yield chunk["response"]
                if chunk.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"[ollama stream] Error: {repr(e)}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# QUESTION PROMPTS
#
# Two separate prompt builders so the logic is clear and easy to tune.
#
#   build_question_prompt()  — used on the fast (keyword) path where
#     we already know the triage status and just need the next question.
#
#   build_assess_prompt()    — used on the AI path where we need BOTH
#     a status classification AND a next question in one shot (non-
#     streaming, returns JSON).  The question tokens are then streamed
#     separately via build_question_prompt() so the patient hears them
#     as soon as they arrive.
#
# The prompts are deliberately warm and clinical — the nurse robot
# should feel caring, not robotic, while staying terse enough that
# Phi3:mini won't hallucinate instructions into the output.
# ═══════════════════════════════════════════════════════════════════

def build_question_prompt(history_text):
    """
    Returns a prompt for generating the single next triage question.
    Warm but strict: one short question, ends with ?, nothing else.
    This is the original intent — clinically accurate AND empathetic.
    """
    return (
        "You are a caring nurse robot doing a patient triage check-in.\n"
        "Based on the conversation so far, ask the single most important\n"
        "follow-up question to understand the patient's condition better.\n\n"
        "STRICT RULES — follow all of them:\n"
        "- ONE question only\n"
        "- Under 12 words\n"
        "- Must end with a question mark\n"
        "- No preamble, no explanation, no instructions after the question\n"
        "- Output ONLY the question text, nothing else\n"
        "- If you have enough information already, output only: DONE\n\n"
        "Topics to cover (pick the most relevant one not yet asked):\n"
        "pain location, pain level 1-10, duration, breathing difficulty,\n"
        "current medications, recent medical history, allergies.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        "Next question:"
    )


def build_assess_prompt(history_text, last_answer):
    """
    Returns a prompt that classifies the patient's latest answer AND
    generates the next question in one shot (JSON output).
    Used only on the AI path (non-streaming status classification).
    Warm, accurate, and constrained.
    """
    return (
        "You are a caring clinical triage assistant for a nurse robot.\n"
        "Your job is to assess the patient's wellbeing accurately and compassionately.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Patient's latest answer: \"{last_answer}\"\n\n"
        "Step 1 — Classify the latest answer:\n"
        "  URGENT  = life-threatening: severe chest pain, cannot breathe, unconscious,\n"
        "            uncontrolled bleeding, seizure, stroke, pain 9-10/10.\n"
        "  MONITOR = moderate concern: pain 4-8/10, fever, dizziness, nausea,\n"
        "            shortness of breath, confusion, or multiple mild symptoms.\n"
        "  STABLE  = mild or no symptoms: pain 1-3/10, feeling generally okay.\n"
        "  When in doubt, choose the less severe option.\n\n"
        "Step 2 — Ask the single most important follow-up question not yet asked.\n"
        "  - Under 12 words, ends with ?\n"
        "  - Topics: main complaint, pain level, duration, breathing, medications.\n"
        "  - If you have enough info already, write DONE instead.\n\n"
        "Reply with ONLY valid JSON, no markdown, no extra text:\n"
        "{\"status\": \"URGENT|MONITOR|STABLE\", \"question\": \"your question or DONE\"}"
    )


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/turn", methods=["POST"])
def turn():
    """Non-streaming fallback endpoint (kept for compatibility)."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"text": "", "status": "STABLE", "question": "DONE", "error": "invalid body"}), 400

        audio_b64 = body.get("audio_b64", "")
        history   = body.get("history", [])

        if not audio_b64:
            return jsonify({"text": "", "status": "STABLE", "question": "DONE", "error": "missing audio_b64"}), 400

        try:
            wav_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return jsonify({"text": "", "status": "STABLE", "question": "DONE", "error": "invalid base64"}), 400

        text = transcribe_bytes(wav_bytes)
        if not text:
            return jsonify({"text": "", "status": "STABLE", "question": "DONE"})

        history_text = "\n".join([
            f"Nurse: {qa.get('q','')}\nPatient: {qa.get('a','')}" for qa in history
        ])

        fast_status = keyword_status(text)

        if fast_status is not None:
            question = clean_question(
                phi3(build_question_prompt(history_text), OLLAMA_OPTS_FAST).split("\n")[0]
            )
            print(f"[turn] fast={fast_status} q='{question}'")
            return jsonify({"text": text, "status": fast_status, "question": question})

        raw = phi3(build_assess_prompt(history_text, text), OLLAMA_OPTS_FAST)
        raw = re.sub(r"```json|```", "", raw).strip()
        print("[phi3 parsed raw]", raw)

        try:
            parsed   = json.loads(raw)
            status   = parsed.get("status",   "STABLE").upper().strip()
            question = parsed.get("question", "DONE").strip()
        except Exception:
            print("[phi3] JSON parse failed")
            status   = "STABLE"
            question = "DONE"

        if status not in ["URGENT", "MONITOR", "STABLE"]:
            status = "STABLE"

        print(f"[turn] phi3={status} q='{question}'")
        return jsonify({"text": text, "status": status, "question": clean_question(question)})

    except Exception as e:
        print(f"[turn] Error: {repr(e)}")
        traceback.print_exc()
        return jsonify({"text": "", "status": "STABLE", "question": "DONE", "error": str(e)}), 500


@app.route("/turn_stream", methods=["POST"])
def turn_stream():
    """
    Streaming SSE endpoint.
    Transcribes audio, classifies status, then streams the next question
    token-by-token so Piper TTS on the Pi can start speaking immediately.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        if not isinstance(body, dict):
            return jsonify({"text": "", "status": "STABLE", "question": "DONE", "error": "invalid body"}), 400

        audio_b64 = body.get("audio_b64", "")
        history   = body.get("history", [])

        if not audio_b64:
            return jsonify({"text": "", "status": "STABLE", "question": "DONE", "error": "missing audio_b64"}), 400

        try:
            wav_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return jsonify({"text": "", "status": "STABLE", "question": "DONE", "error": "invalid base64"}), 400

        # Transcribe first — everything else depends on it
        text = transcribe_bytes(wav_bytes)

        if not text:
            def empty_gen():
                yield f"data: {json.dumps({'done': True, 'text': '', 'status': 'STABLE', 'question': 'DONE'})}\n\n"
            return Response(empty_gen(), mimetype="text/event-stream")

        history_text = "\n".join([
            f"Nurse: {qa.get('q','')}\nPatient: {qa.get('a','')}" for qa in history
        ])

        fast_status = keyword_status(text)

        def generate_stream():
            # ── Fast path (keyword matched) ────────────────────────
            if fast_status is not None:
                print(f"[turn_stream] fast path: {fast_status}")

                question_tokens = []
                for token in phi3_stream(build_question_prompt(history_text), OLLAMA_OPTS_FAST):
                    question_tokens.append(token)
                    yield f"data: {json.dumps({'t': token})}\n\n"

                question = clean_question("".join(question_tokens).split("\n")[0].strip())
                yield f"data: {json.dumps({'done': True, 'text': text, 'status': fast_status, 'question': question})}\n\n"

            # ── AI assessment path ─────────────────────────────────
            else:
                print("[turn_stream] AI path — classifying status...")

                # Status classification is fast (num_predict=3), non-streaming
                status_prompt = (
                    "Triage nurse. Classify this patient response.\n\n"
                    f"Patient: '{text}'\n\n"
                    "Reply ONLY with one word: URGENT, MONITOR, or STABLE\n"
                    "URGENT = severe chest pain, can't breathe, unconscious, pain 9-10\n"
                    "MONITOR = moderate pain, dizziness, nausea, fever, pain 4-8\n"
                    "STABLE = mild symptoms"
                )
                status_raw = phi3(status_prompt, OLLAMA_OPTS_URGENT)
                status     = status_raw.upper().strip()
                if status not in ["URGENT", "MONITOR", "STABLE"]:
                    status = "STABLE"

                print(f"[turn_stream] classified as {status} — now streaming question...")

                # Stream the question in real time
                question_tokens = []
                for token in phi3_stream(build_question_prompt(history_text + f"\nPatient: {text}"), OLLAMA_OPTS_FAST):
                    question_tokens.append(token)
                    yield f"data: {json.dumps({'t': token})}\n\n"

                question = clean_question("".join(question_tokens).split("\n")[0].strip())
                print(f"[turn_stream] done — status={status}, question='{question}'")
                yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"

        return Response(generate_stream(), mimetype="text/event-stream")

    except Exception as e:
        print(f"[turn_stream] Error: {repr(e)}")
        traceback.print_exc()
        return jsonify({"text": "", "status": "STABLE", "question": "DONE", "error": str(e)}), 500


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

        if urgent_count == 0 and monitor_count < 3:
            print("[flag_urgent] skipping AI — not enough flags")
            return jsonify({"flagged_urgent": False})

        prompt = (
            "You are a senior nurse reviewing a triage conversation.\n"
            "Reply YES only if the patient needs immediate emergency attention.\n"
            "Otherwise reply NO.\n\n"
            f"Conversation:\n{history_text}\n\n"
            "Reply ONLY YES or NO."
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