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

OLLAMA_URL = "http://localhost:11434/api/generate"
TEXT_MODEL = "phi3:mini"
SERVER_PORT = 5001

OLLAMA_OPTS_FAST = {
    "temperature": 0.1,
    "num_predict": 60,
    "num_ctx": 512,
    "repeat_penalty": 1.3,
    "top_k": 20,
    "top_p": 0.8,
    "keep_alive": "-1"
}

OLLAMA_OPTS_URGENT = {
    "temperature": 0.0,
    "num_predict": 3,
    "num_ctx": 512,
    "keep_alive": "-1"
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
    "can't breathe",
    "cannot breathe",
    "chest pain",
    "chest tightness",
    "heart attack",
    "unconscious",
    "unresponsive",
    "not breathing",
    "stopped breathing",
    "seizure",
    "stroke",
    "overdose",
    "bleeding out",
    "can't move",
    "cannot move",
    "help me",
    "dying"
]

MONITOR_HARD = [
    "dizzy",
    "dizziness",
    "nausea",
    "vomiting",
    "fever",
    "chills",
    "shortness of breath",
    "hard to breathe",
    "difficulty breathing",
    "swelling",
    "rash",
    "allergic",
    "infection",
    "confusion",
    "disoriented"
]

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def extract_pain_score(text):
    match = re.search(r'\b(10|[1-9])\b', text)

    if match:
        return int(match.group(1))

    word_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10
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
        if score >= 8:
            return "URGENT"

        if score >= 4:
            return "MONITOR"

        return "STABLE"

    return None


# ═══════════════════════════════════════════════════════════════════
# WHISPER
# ═══════════════════════════════════════════════════════════════════

def transcribe_bytes(wav_bytes):

    try:
        buf = io.BytesIO(wav_bytes)

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
    "I'm glad to hear that.",
    "Great!",
    "Good!",
    "Okay,",
    "Okay.",
    "I see.",
    "Alright.",
    "Sure,",
    "Thank you.",
    "Got it.",
    "Of course.",
    "Certainly.",
    "Understood.",
    "Noted.",
]


def clean_question(q):

    if not q:
        return "DONE"

    q = q.strip()

    for filler in FILLER_PREFIXES:
        if q.lower().startswith(filler.lower()):
            q = q[len(filler):].strip()

    if "?" in q:
        q = q[:q.index("?") + 1]

    q = q.strip()

    if not q:
        return "DONE"

    return q


# ═══════════════════════════════════════════════════════════════════
# OLLAMA SAFE WRAPPER
# ═══════════════════════════════════════════════════════════════════

def phi3(prompt, opts):

    try:

        r = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": opts
            },
            timeout=60
        )

        print("[ollama status]", r.status_code)

        if not r.text.strip():
            print("[ollama] empty response")
            return "DONE"

        print("[ollama raw]", r.text[:500])

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
    Streaming version of phi3 - yields tokens as they arrive from Ollama.
    """
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": True,  # Enable streaming
                "options": opts
            },
            stream=True,
            timeout=60
        )

        r.raise_for_status()

        for line in r.iter_lines():
            if not line:
                continue
            
            try:
                chunk = json.loads(line.decode('utf-8'))
                if 'response' in chunk:
                    token = chunk['response']
                    if token:
                        yield token
                
                # Check if this is the last chunk
                if chunk.get('done', False):
                    break
            
            except json.JSONDecodeError:
                continue

    except Exception as e:
        print(f"[ollama stream] Error: {repr(e)}")
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# QUESTION GENERATION
# ═══════════════════════════════════════════════════════════════════

def get_next_question_only(history_text):

    prompt = (
        "You are a calm triage assistant.\n"
        "Ask ONE short follow-up question not yet asked.\n"
        "Topics: main problem, pain level, duration, breathing, medications.\n"
        "Under 15 words.\n"
        "If enough info reply ONLY: DONE\n\n"
        f"Conversation:\n{history_text}\n\nAssistant:"
    )

    raw = phi3(prompt, OLLAMA_OPTS_FAST)

    raw = raw.split("\n")[0].strip()

    return clean_question(raw)


def assess_and_next(history_text, last_answer):

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

    raw = phi3(prompt, OLLAMA_OPTS_FAST)

    raw = re.sub(r"```json|```", "", raw).strip()

    print("[phi3 parsed raw]", raw)

    try:
        parsed = json.loads(raw)
        status = parsed.get("status", "STABLE").upper().strip()
        question = parsed.get("question", "DONE").strip()

    except Exception:
        print("[phi3] JSON parse failed")
        status = "STABLE"
        question = "DONE"

    if status not in ["URGENT", "MONITOR", "STABLE"]:
        status = "STABLE"

    return status, clean_question(question)


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/turn", methods=["POST"])
def turn():

    try:

        body = request.get_json(force=True, silent=True) or {}

        print("[request body type]", type(body))

        if not isinstance(body, dict):
            return jsonify({
                "text": "",
                "status": "STABLE",
                "question": "DONE",
                "error": "invalid request body"
            }), 400

        audio_b64 = body.get("audio_b64", "")
        history = body.get("history", [])

        if not audio_b64:
            return jsonify({
                "text": "",
                "status": "STABLE",
                "question": "DONE",
                "error": "missing audio_b64"
            }), 400

        try:
            wav_bytes = base64.b64decode(audio_b64)

        except Exception as e:
            print("[b64 decode failed]", repr(e))

            return jsonify({
                "text": "",
                "status": "STABLE",
                "question": "DONE",
                "error": "invalid base64"
            }), 400

        text = transcribe_bytes(wav_bytes)

        if not text:

            return jsonify({
                "text": "",
                "status": "STABLE",
                "question": "DONE"
            })

        history_text = "\n".join([
            f"Nurse: {qa.get('q', '')}\nPatient: {qa.get('a', '')}"
            for qa in history
        ])

        fast_status = keyword_status(text)

        if fast_status is not None:

            question = get_next_question_only(history_text)

            print(f"[turn] fast={fast_status} q='{question}'")

            return jsonify({
                "text": text,
                "status": fast_status,
                "question": question
            })

        status, question = assess_and_next(history_text, text)

        print(f"[turn] phi3={status} q='{question}'")

        return jsonify({
            "text": text,
            "status": status,
            "question": question
        })

    except Exception as e:

        print(f"[turn] Error: {repr(e)}")
        traceback.print_exc()

        return jsonify({
            "text": "",
            "status": "STABLE",
            "question": "DONE",
            "error": str(e)
        }), 500


@app.route("/turn_stream", methods=["POST"])
def turn_stream():
    """
    Streaming SSE version of /turn endpoint.
    Streams tokens as they arrive from Ollama, enabling real-time TTS.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        
        if not isinstance(body, dict):
            return jsonify({
                "text": "",
                "status": "STABLE",
                "question": "DONE",
                "error": "invalid request body"
            }), 400

        audio_b64 = body.get("audio_b64", "")
        history = body.get("history", [])

        if not audio_b64:
            return jsonify({
                "text": "",
                "status": "STABLE",
                "question": "DONE",
                "error": "missing audio_b64"
            }), 400

        try:
            wav_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            print("[b64 decode failed]", repr(e))
            return jsonify({
                "text": "",
                "status": "STABLE",
                "question": "DONE",
                "error": "invalid base64"
            }), 400

        # Transcribe audio
        text = transcribe_bytes(wav_bytes)

        if not text:
            def empty_generator():
                yield f"data: {json.dumps({'done': True, 'text': '', 'status': 'STABLE', 'question': 'DONE'})}\n\n"
            
            return Response(empty_generator(), mimetype='text/event-stream')

        history_text = "\n".join([
            f"Nurse: {qa.get('q', '')}\nPatient: {qa.get('a', '')}"
            for qa in history
        ])

        # Check for keyword-based fast status
        fast_status = keyword_status(text)

        def generate_stream():
            """Generator function for SSE streaming"""
            
            if fast_status is not None:
                # Fast path - stream next question
                print(f"[turn_stream] fast path: {fast_status}")
                
                prompt = (
                    "You are a calm triage assistant.\n"
                    "Ask ONE short follow-up question not yet asked.\n"
                    "Topics: main problem, pain level, duration, breathing, medications.\n"
                    "Under 15 words.\n"
                    "If enough info reply ONLY: DONE\n\n"
                    f"Conversation:\n{history_text}\n\nAssistant:"
                )
                
                question_tokens = []
                for token in phi3_stream(prompt, OLLAMA_OPTS_FAST):
                    question_tokens.append(token)
                    # Send each token as SSE
                    yield f"data: {json.dumps({'t': token})}\n\n"
                
                question = ''.join(question_tokens)
                question = question.split("\n")[0].strip()
                question = clean_question(question)
                
                # Send final done event
                yield f"data: {json.dumps({'done': True, 'text': text, 'status': fast_status, 'question': question})}\n\n"
            
            else:
                # AI assessment path - stream question generation in real-time
                print(f"[turn_stream] AI assessment path - streaming")
                
                # Generate status classification (fast, non-streaming)
                status_prompt = (
                    "You are a triage nurse. Classify this patient response.\n\n"
                    f"Patient: '{text}'\n\n"
                    "Reply ONLY with one word: URGENT, MONITOR, or STABLE\n"
                    "URGENT = severe chest pain, can't breathe, unconscious, pain 9-10\n"
                    "MONITOR = moderate pain, dizziness, nausea, fever, pain 4-8\n"
                    "STABLE = mild symptoms"
                )
                
                status_raw = phi3(status_prompt, OLLAMA_OPTS_URGENT)
                status = status_raw.upper().strip()
                if status not in ["URGENT", "MONITOR", "STABLE"]:
                    status = "STABLE"
                
                print(f"[turn_stream] classified as {status}, now streaming question...")
                
                # Stream the follow-up question in real-time
                question_prompt = (
                    "You are a calm triage assistant.\n"
                    "Ask ONE short follow-up question not yet asked.\n"
                    "Topics: main problem, pain level, duration, breathing, medications.\n"
                    "Under 15 words.\n"
                    "If enough info reply ONLY: DONE\n\n"
                    f"Conversation:\n{history_text}\n"
                    f"Patient: {text}\n\nAssistant:"
                )
                
                question_tokens = []
                for token in phi3_stream(question_prompt, OLLAMA_OPTS_FAST):
                    question_tokens.append(token)
                    # Stream each token immediately to client
                    yield f"data: {json.dumps({'t': token})}\n\n"
                
                question = ''.join(question_tokens)
                question = question.split("\n")[0].strip()
                question = clean_question(question)
                
                print(f"[turn_stream] complete - status={status}, question='{question}'")
                
                # Send final done event with metadata
                yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"
        
        return Response(generate_stream(), mimetype='text/event-stream')

    except Exception as e:
        print(f"[turn_stream] Error: {repr(e)}")
        traceback.print_exc()
        
        return jsonify({
            "text": "",
            "status": "STABLE",
            "question": "DONE",
            "error": str(e)
        }), 500


@app.route("/flag_urgent", methods=["POST"])
def flag_urgent():

    try:

        body = request.get_json(force=True, silent=True) or {}

        history = body.get("history", [])

        history_text = "\n".join([
            f"Nurse: {qa.get('q', '')}\nPatient: {qa.get('a', '')}"
            for qa in history
        ])

        urgent_count = sum(
            1 for qa in history
            if qa.get("status") == "URGENT"
        )

        monitor_count = sum(
            1 for qa in history
            if qa.get("status") == "MONITOR"
        )

        if urgent_count == 0 and monitor_count < 3:

            print("[flag_urgent] skipping AI")

            return jsonify({
                "flagged_urgent": False
            })

        prompt = (
            "You are a senior nurse.\n"
            "Reply YES only for immediate emergencies.\n"
            "Otherwise NO.\n\n"
            f"Conversation:\n{history_text}\n\n"
            "Reply ONLY YES or NO."
        )

        raw = phi3(prompt, OLLAMA_OPTS_URGENT)

        flagged = raw.upper().startswith("YES")

        print(f"[flag_urgent] {raw} -> {flagged}")

        return jsonify({
            "flagged_urgent": flagged
        })

    except Exception as e:

        print(f"[flag_urgent] Error: {repr(e)}")
        traceback.print_exc()

        return jsonify({
            "flagged_urgent": False
        })


# ═══════════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print(f"[server] Starting on port {SERVER_PORT}")

    app.run(
        host="0.0.0.0",
        port=SERVER_PORT,
        debug=False,
        threaded=True
    )
