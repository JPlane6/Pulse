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
    "temperature": 0.05,
    "top_p": 0.5,
    "top_k": 10,
    "repeat_penalty": 1.15,
    "num_predict": 18,
    "num_ctx": 192,      # Reduced from 256 - faster on M2
    "keep_alive": "-1",
    "stop": ["\n", "Patient:", "Nurse:"]  # Prevent rambling
}

OLLAMA_OPTS_URGENT = {
    "temperature": 0,
    "top_p": 0.1,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "num_predict": 2,
    "num_ctx": 128,
    "keep_alive": "-1",
    "stop": ["\n", "Patient:", "Nurse:"]  # Prevent rambling
}

# ═══════════════════════════════════════════════════════════════════
# LOAD WHISPER
# ═══════════════════════════════════════════════════════════════════

# OPTIMIZATION: Use 'base' for 3-5x faster transcription than 'base'
# Trade: ~5% accuracy loss, acceptable for medical keywords
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

# Severity ranking for combining keyword + AI classification
SEVERITY_ORDER = {
    "STABLE": 0,
    "MONITOR": 1,
    "URGENT": 2
}

# Deterministic fallbacks for when model outputs garbage
QUESTION_FALLBACKS = {
    "pain intensity (intense/moderate/mild)": "Is it intense, moderate, or mild?",
    "pain location": "Where does it hurt?",
    "symptom duration": "How long symptoms?",
    "breathing status": "Any trouble breathing?",
    "medications and allergies": "Any medications or allergies?"
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
    """
    Fast keyword check. Returns a status string if matched, or None if
    the answer needs AI classification.
    """
    if not text.strip():
        return "STABLE"
    lower = text.lower()
    
    # CRITICAL: Check urgent/monitor FIRST, then stable phrases
    # Order matters: "I'm okay but earlier I fainted" should be MONITOR not STABLE
    
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
    
    # MEDICAL SAFETY: Intense/severe pain ALONE is MONITOR, not URGENT
    # Only URGENT if paired with critical symptoms
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
        print(f"[keyword] MONITOR via moderate pain")
        return "MONITOR"
    if "mild" in lower or "slight" in lower or "minor" in lower:
        print(f"[keyword] STABLE via mild pain")
        return "STABLE"
    
    # Fallback: Check numeric pain score (1-10) if patient mentions a number
    # Note: We ask for mild/moderate/intense, but accept numbers as backup
    # CRITICAL: Only extract if pain context present (avoid "2 days ago" = pain 2)
    pain_context = any(word in lower for word in [
        "pain", "hurt", "hurts", "aching", "ache"
    ])
    score = extract_pain_score(lower) if pain_context else None
    if score is not None:
        # High pain needs critical context to be URGENT
        if score >= 8 and has_critical_context:
            return "URGENT"
        elif score >= 8:
            return "MONITOR"
        elif score >= 4:
            return "MONITOR"
        else:
            return "STABLE"
    
    # NOW check stable/reassuring phrases (after urgent/monitor)
    # This prevents "I'm okay but I fainted" from being marked STABLE
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
    
    return None  # needs AI


def extract_patient_text(history):
    """
    Extract ONLY patient responses (not nurse questions).
    Critical: prevents false positives from nurse's own words.
    """
    return " ".join(
        qa.get("a", "")
        for qa in history
    ).lower()


def get_missing_fields(history):
    """
    Determines what critical info is still missing from conversation.
    ONLY checks patient responses, not nurse questions.
    """
    # Extract only patient text - BIG BUG FIX
    if isinstance(history, str):
        # If called with patient text directly
        patient_text = history.lower()
    else:
        # If called with history array
        patient_text = extract_patient_text(history)
    
    missing = []
    
    # Check for pain intensity
    if not any(word in patient_text for word in ["intense", "moderate", "mild", "severe"]):
        missing.append("pain intensity (intense/moderate/mild)")
    
    # Check for location
    if not any(word in patient_text for word in [
        "chest", "head", "stomach", "arm", "leg", "back", "neck", "knee", "ankle",
        "shoulder", "throat", "abdomen", "hip", "wrist", "jaw", "foot", "hand"
    ]):
        missing.append("pain location")
    
    # Check for duration - FIXED: removed "when", added specific time words
    if not any(word in patient_text for word in [
        "minutes", "minute", "hours", "hour", "days", "day", "weeks", "week", "months", "started", "ago",
        "yesterday", "today", "morning", "tonight", "evening", "afternoon"
    ]):
        missing.append("symptom duration")
    
    # Check for breathing
    if not any(word in patient_text for word in [
        "breath", "breathing", "wheezing", "suffocating", "gasping"
    ]):
        missing.append("breathing status")
    
    # Check for medications/allergies
    if not any(word in patient_text for word in ["medication", "medicine", "taking", "allergic", "allergy"]):
        missing.append("medications and allergies")
    
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
            temperature=0.0,       # Deterministic
            without_timestamps=True  # Tiny speed gain for M2
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
    """Clean question text, strip DONE signals, force valid output."""
    if not q or not q.strip():
        # Should never happen, but ensure we have something
        return "What else is bothering you?"
    
    q = q.strip()

    # Strip filler openers
    for filler in FILLER_PREFIXES:
        if q.lower().startswith(filler.lower()):
            q = q[len(filler):].strip()

    # Check if entire output is a DONE signal (not partial match)
    q_lower = q.lower().strip()
    if q_lower in DONE_SIGNALS or q_lower == "done":
        print(f"[clean_question] Detected DONE signal: '{q}'")
        return "DONE"
    
    # Look for a question mark - extract the first question
    if "?" in q:
        # Take everything up to and including first question mark
        q = q[:q.index("?") + 1].strip()
    else:
        # No question mark - look for question-like patterns
        # If starts with question words, add a question mark
        question_starters = ["what", "where", "when", "how", "why", "is", "are", "can", "do", "does", "have", "has"]
        if any(q.lower().startswith(word) for word in question_starters):
            # Likely a question, just missing the mark - add it
            q = q + "?"
        else:
            # Not a clear question - might be a statement
            # Take first sentence if exists
            if "." in q:
                q = q[:q.index(".") + 1].strip()
            # Still no question mark? Force one if it looks reasonable
            if len(q) > 3 and len(q) < 100:
                q = q.rstrip(".") + "?"
    
    # Remove any remaining punctuation clusters
    q = re.sub(r'[.!]+$', '', q).strip()
    
    # CRITICAL: Protect DONE from mutation to DONE?
    if q == "DONE":
        return "DONE"
    
    if not q.endswith("?"):
        q = q + "?"
    
    # Enforce length limits - truncate if too long
    if len(q) > 120:
        # Find last complete word before limit
        q = q[:120]
        last_space = q.rfind(" ")
        if last_space > 50:  # Keep at least some content
            q = q[:last_space].strip()
        if not q.endswith("?"):
            q = q + "?"
        print(f"[clean_question] Truncated long output to {len(q)} chars")
    
    # Final check - ensure we have actual content
    if len(q) < 3 or q == "?":
        return "What else should I know?"
    
    return q


def sanitize_question(text):
    """
    Post-generation cleanup to enforce strict output format.
    Critical for preventing drift even with good prompts.
    """
    text = text.strip()
    
    # Catch raw DONE outputs first
    if text.upper() == "DONE":
        return "DONE"

    # CRITICAL: Strip Qwen thinking tags if present (some versions add them)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

    # Take only first line
    lines = text.splitlines()
    text = lines[0].strip()

    # Cut at first question mark
    if "?" in text:
        text = text[:text.index("?") + 1]

    # Remove filler word prefixes
    text = re.sub(r'^(okay|alright|well|so|hmm)[,. ]*', '', text, flags=re.I)

    # Enforce 10 word maximum (8 truncates valid medical questions)
    MAX_WORDS = 10
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
        # Ensure question mark at end
        if not text.endswith("?"):
            text += "?"
    
    # Protect against nonsense (too short)
    if len(text.split()) < 2 and text != "DONE":
        return "What symptoms?"

    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# OLLAMA WRAPPERS
# ═══════════════════════════════════════════════════════════════════

def ollama(prompt, opts):
    """Non-streaming Ollama generation."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": prompt, "stream": False, "options": opts},
            timeout=20  # 3B model should respond quickly; >20s = something wrong
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
    """
    Streaming Ollama generation.
    Yields tokens as they arrive, stops after first complete question.
    """
    accumulated = []
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": TEXT_MODEL, "prompt": prompt, "stream": True, "options": opts},
            stream=True,
            timeout=40  # Longer for first generation / model load
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
                    # Fixed: safer termination check
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
    Strict prompt optimized for Qwen 2.5 3B.
    Short-circuit logic enforces MIN_QUESTIONS.
    """
    # BUG FIX: Only check patient responses for missing fields
    missing = get_missing_fields(history)
    next_topic = missing[0] if missing else "how they're feeling overall"
    
    # Short-circuit logic: enforce MIN_QUESTIONS
    if questions_asked < MIN_QUESTIONS:
        done_allowed = False
    elif not missing:
        done_allowed = True
    else:
        done_allowed = False
    
    mode = "DONE_ALLOWED" if done_allowed else "KEEP_ASKING"

    return f"""
You are Pulse, a nurse triage robot.

Conversation:
{full_history_text}

Next topic:
{next_topic}

Mode:
{mode}

STRICT RULES:
- Ask ONE question ONLY
- Maximum 10 words
- No greetings
- No empathy
- No reassurance
- No explanations
- No comments
- No diagnosis
- No filler words
- No extra sentences
- No multiple questions
- Stay on the requested topic ONLY
- Sound natural
- End with ONE question mark

SPECIAL RULE:
If topic is pain severity, ask EXACTLY:
Is it intense, moderate, or mild?

DONE RULES:
- If mode is KEEP_ASKING:
  - NEVER say DONE
- If mode is DONE_ALLOWED:
  - Say ONLY: DONE
  - ONLY if all important information is collected

VALID OUTPUT EXAMPLES:
Where does it hurt?
How long symptoms?
Any trouble breathing?
What medications take?
Is it intense, moderate, or mild?

INVALID OUTPUT EXAMPLES:
Hello, how are you feeling today?
I'm sorry you're experiencing that.
Can you describe your symptoms more?
Where does it hurt and how long?
Okay. Where does it hurt?

Output:
""".strip()


def build_status_prompt(patient_answer):
    """
    Strict classification prompt for Qwen 2.5 3B.
    """
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
                yield f"data: {json.dumps({'done': True, 'text': '', 'status': 'STABLE', 'question': 'Could you repeat that?'})}\n\n"
            return Response(empty_gen(), mimetype="text/event-stream")

        # Build full history including this answer
        history_text = "\n".join([
            f"Nurse: {qa.get('q','')}\nPatient: {qa.get('a','')}"
            for qa in history
        ])
        # FIX: Count actual questions asked, not history entries
        questions_asked = sum(1 for qa in history if qa.get("q"))
        full_history = history_text + f"\nPatient: {text}" if history_text else f"Patient: {text}"
        
        # CRITICAL: Include current answer when checking missing fields
        # Prevents asking "Where does it hurt?" after patient says "my chest hurts"
        temp_history = history + [{"a": text}]
        missing = get_missing_fields(temp_history)

        def generate_stream():
            # OPTIMIZATION: Stream question immediately, status classifies in parallel
            print(f"[turn_stream] streaming (asked={questions_asked})...")
            
            # Hard max enforcement - CRITICAL: still run full AI classification
            if questions_asked >= MAX_QUESTIONS:
                print(f"[turn_stream] MAX_QUESTIONS reached, forcing DONE")
                question = "DONE"
                
                # Run BOTH keyword and AI classification (don't skip AI!)
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
                
                print(f"[turn_stream] MAX reached - status={status} (keyword={keyword_result}, AI={ai_status})")
                yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"
                return
            
            question_tokens = []
            
            for token in ollama_stream(
                build_question_prompt(temp_history, full_history, questions_asked),
                OLLAMA_OPTS_FAST
            ):
                question_tokens.append(token)
                yield f"data: {json.dumps({'t': token})}\n\n"
                # Removed duplicate break - ollama_stream already handles this
            
            # CRITICAL: Handle empty stream (Ollama stall/freeze)
            if not question_tokens:
                print(f"[turn_stream] Empty stream detected, using fallback")
                next_topic = missing[0] if missing else None
                question = QUESTION_FALLBACKS.get(next_topic, "What symptoms?")
                
                # Run full dual-path classification (don't downgrade safety)
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
            question = sanitize_question(question)  # Post-generation cleanup
            
            # Hard fail invalid outputs - use deterministic fallback
            if "?" not in question and question != "DONE":
                print(f"[turn_stream] Invalid question, using fallback for: {missing[0] if missing else 'general'}")
                next_topic = missing[0] if missing else None
                question = QUESTION_FALLBACKS.get(next_topic, "What symptoms?")
            
            # Prevent repeated questions (check ALL history, not just last)
            asked_questions = {
                qa.get("q", "").lower().strip()
                for qa in history
            }
            if question.lower().strip() in asked_questions:
                print(f"[turn_stream] Repeated question detected, using fallback")
                next_topic = missing[0] if missing else None
                question = QUESTION_FALLBACKS.get(next_topic, "What symptoms?")
                
                # Double-check: if fallback also repeated, use first unused fallback
                if question.lower().strip() in asked_questions:
                    unused = [
                        q for q in QUESTION_FALLBACKS.values()
                        if q.lower().strip() not in asked_questions
                    ]
                    if unused:
                        question = unused[0]
                    else:
                        # All fallbacks used - generic question
                        question = "What else should I know?"

            # Status: Run BOTH keyword and AI, take highest severity
            keyword_status_result = keyword_status(text)
            
            if keyword_status_result is not None:
                print(f"[turn_stream] keyword status={keyword_status_result}")
            
            # Always run AI as backup
            status_raw = ollama(build_status_prompt(text), OLLAMA_OPTS_URGENT)
            # Improved parsing - handles "The answer is MONITOR" etc
            status_raw = status_raw.upper().strip()
            if "URGENT" in status_raw:
                ai_status = "URGENT"
            elif "MONITOR" in status_raw:
                ai_status = "MONITOR"
            else:
                ai_status = "STABLE"
            print(f"[turn_stream] AI status={ai_status} (from: {status_raw})")
            
            # Take highest severity between keyword and AI
            if keyword_status_result is not None:
                status = max(keyword_status_result, ai_status, key=lambda x: SEVERITY_ORDER[x])
                print(f"[turn_stream] Final status={status} (keyword={keyword_status_result}, AI={ai_status})")
            else:
                status = ai_status

            print(f"[turn_stream] done — status={status} question='{question}'")
            yield f"data: {json.dumps({'done': True, 'text': text, 'status': status, 'question': question})}\n\n"

        return Response(generate_stream(), mimetype="text/event-stream")

    except Exception as e:
        print(f"[turn_stream] Error: {repr(e)}")
        traceback.print_exc()
        # DEMO SAFETY: Return graceful fallback instead of hard error
        def error_gen():
            yield f"data: {json.dumps({'done': True, 'text': '', 'status': 'STABLE', 'question': 'Could you repeat that?'})}\\n\\n"
        return Response(error_gen(), mimetype="text/event-stream")


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
    # Preload model weights into memory for faster first request
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