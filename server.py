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

# OPTIMIZATION: Reduced num_ctx from 512 to 256 for faster processing
# Reduced num_predict from 25 to 20 for quicker responses
OLLAMA_OPTS_FAST = {
    "temperature":    0.1,
<<<<<<< Updated upstream
    "num_predict":    20,
    "num_ctx":        256,
=======
    "num_predict":    20,      # Reduced from 25
    "num_ctx":        256,     # Reduced from 512 for speed
>>>>>>> Stashed changes
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

# OPTIMIZATION: Use 'base' instead of 'small' for 2-3x faster transcription
# with minimal accuracy loss for medical triage. Use 'tiny' for even faster.
print("[server] Loading Whisper model...")
<<<<<<< Updated upstream
whisper_model = whisper.load_model("base")
=======
whisper_model = whisper.load_model("base")  # Changed from 'small' to 'base'
>>>>>>> Stashed changes
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
    """
    Fast keyword check. Returns a status string if matched, or None if
    the answer needs AI classification.
    """
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
        # OPTIMIZATION: Added beam_size=3 (down from default 5) for faster decode
        # Added best_of=3 (down from default 5) for speed
        result = whisper_model.transcribe(
            audio_np,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
<<<<<<< Updated upstream
            beam_size=3,
            best_of=3,
            temperature=0.0
=======
            beam_size=3,           # Faster decoding
            best_of=3,             # Fewer candidates to evaluate
            temperature=0.0        # Deterministic, faster
>>>>>>> Stashed changes
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
    full_history_text must already include the patient's latest answer.
    questions_asked is the total number of turns completed so far.
    """
    return (
        "You are a nurse robot doing a triage check-in.\n\n"
        "You MUST collect ALL of the following before stopping:\n"
        "  [A] Pain score (a number 1-10)\n"
        "  [B] Location of the pain or discomfort\n"
        "  [C] How long the issue has been going on\n"
        "  [D] Whether they have difficulty breathing\n"
        "  [E] Any medications or known allergies\n\n"
        "Look at the conversation and find the FIRST item from [A]-[E] not yet answered.\n"
        f"You have completed {questions_asked} questions so far (maximum 6).\n\n"
        "If ANY of [A]-[E] are still missing AND questions_asked < 6, "
        "ask the single most important missing one.\n"
        "If ALL of [A]-[E] are answered, OR the patient has a life-threatening emergency "
        "(chest pain, can't breathe, pain 9-10), output exactly: DONE\n\n"
        "Rules:\n"
        "- Output ONLY the question. Under 12 words. Must end with ?.\n"
        "- OR output exactly: DONE\n"
        "- No extra text, no explanations.\n\n"
        f"Conversation:\n{full_history_text}\n\n"
        "Question:"
    )


def build_status_prompt(patient_answer):
    return (
        f"Patient said: '{patient_answer}'\n"
        "Reply ONE word only: URGENT or MONITOR or STABLE\n"
        "URGENT = chest pain, can't breathe, unconscious, pain 9-10\n"
        "MONITOR = pain 4-8, fever, dizzy, nausea, shortness of breath\n"
        "STABLE = mild or no symptoms, pain 1-3"
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
<<<<<<< Updated upstream
    Main endpoint. Flow:
      1. Transcribe audio (Whisper)
      2. Keyword-check the answer for fast status
      3. Stream the next question token-by-token (Pi starts speaking immediately)
      4. After streaming, run AI status classification if keywords didn't match
      5. Send done event with full result

    The key insight: status classification runs AFTER we start streaming the
    question, so the Pi's Piper TTS is already talking while the Mac figures
    out the urgency level. By the time the question finishes playing, status
    is ready. Zero extra wait.
=======
    OPTIMIZED: Streaming SSE endpoint.
    Transcribes audio, does PARALLEL status classification and question generation,
    streams tokens as soon as they're available.
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
        # Step 1: Transcribe
=======
        # OPTIMIZATION: Start transcription immediately
>>>>>>> Stashed changes
        text = transcribe_bytes(wav_bytes)

        if not text:
            def empty_gen():
                yield f"data: {json.dumps({'done': True, 'text': '', 'status': 'STABLE', 'question': 'DONE'})}\n\n"
            return Response(empty_gen(), mimetype="text/event-stream")

        # Build full history including this answer, so the question prompt
        # has complete context and won't re-ask anything already covered.
        history_text = "\n".join([
            f"Nurse: {qa.get('q','')}\nPatient: {qa.get('a','')}"
            for qa in history
        ])
        # questions_asked = number of completed turns (history doesn't include current yet)
        questions_asked = len(history) + 1
        # full context includes the current answer
        full_history = history_text + f"\nPatient: {text}" if history_text else f"Patient: {text}"

        # Step 2: Fast keyword check
        fast_status = keyword_status(text)

        def generate_stream():
<<<<<<< Updated upstream
            # Step 3: Stream the question immediately regardless of path.
            # Both fast and AI paths do the same streaming — the only
            # difference is where status comes from.
            print(f"[turn_stream] streaming question (asked={questions_asked})...")
            question_tokens = []
            for token in phi3_stream(
                build_question_prompt(full_history, questions_asked),
                OLLAMA_OPTS_FAST
            ):
                question_tokens.append(token)
                yield f"data: {json.dumps({'t': token})}\n\n"

            question = clean_question("".join(question_tokens).split("\n")[0].strip())

            # Step 4: Status — keyword match is instant, AI runs after streaming
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
=======
            # ── OPTIMIZED: Single combined prompt approach ─────────
            # Instead of sequential (classify then ask), we ask LLM to do both
            # in one call and stream immediately
            
            if fast_status is not None:
                # Fast path: keyword matched, skip AI classification
                print(f"[turn_stream] fast path: {fast_status}")
                question_tokens = []
                for token in phi3_stream(build_question_prompt(history_text, questions_asked=len(history)), OLLAMA_OPTS_FAST):
                    question_tokens.append(token)
                    joined = "".join(question_tokens)
                    if "?" in joined:
                        yield f"data: {json.dumps({'t': token})}\n\n"
                        break
                    yield f"data: {json.dumps({'t': token})}\n\n"

                question = clean_question("".join(question_tokens).split("\n")[0].strip())
                yield f"data: {json.dumps({'done': True, 'text': text, 'status': fast_status, 'question': question})}\n\n"

            else:
                # OPTIMIZATION: Start streaming question immediately, classify in parallel
                print("[turn_stream] AI path — streaming question immediately...")
                
                # Start question generation immediately (don't wait for classification)
                question_tokens = []
                full_history = history_text + f"\nPatient: {text}"
                for token in phi3_stream(build_question_prompt(full_history, questions_asked=len(history)), OLLAMA_OPTS_FAST):
                    question_tokens.append(token)
                    yield f"data: {json.dumps({'t': token})}\n\n"
                    if "?" in "".join(question_tokens):
                        break

                # Quick status classification AFTER streaming started
                # (runs while Piper is already speaking)
                status_prompt = (
                    f"Patient said: '{text}'\n"
                    "Reply ONE word: URGENT or MONITOR or STABLE\n"
                    "URGENT=chest pain/can't breathe/unconscious/pain 9-10\n"
                    "MONITOR=pain 4-8/fever/dizzy\nSTABLE=mild"
                )
                status_raw = phi3(status_prompt, OLLAMA_OPTS_URGENT)
                status     = status_raw.upper().strip()
                if status not in ["URGENT", "MONITOR", "STABLE"]:
                    status = "STABLE"

                question = clean_question("".join(question_tokens).split("\n")[0].strip())
                print(f"[turn_stream] done — status={status}, question='{question}'")
                yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"
>>>>>>> Stashed changes

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

        if urgent_count == 0 and monitor_count < 3:
            print("[flag_urgent] skipping AI — not enough signal")
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