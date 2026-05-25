import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import tempfile
import os
import re
import time
import subprocess
import json
from datetime import datetime
import scipy.signal as signal
import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

# ═══════════════════════════════════════════════════════════════════
#  CONFIG — edit these to tune behaviour without touching logic
# ═══════════════════════════════════════════════════════════════════

SERVER_URL    = "http://192.168.0.157:5001"
MAX_QUESTIONS = 6
SAMPLERATE    = 16000          # Record at 16kHz natively — Silero + Whisper both want this

AUDIO_DEVICE_KEYWORD = "USB"   # Keyword matched against aplay/arecord/PortAudio device names

PIPER_MODEL = "/home/ayushs0604/Pulse/en_US-amy-medium.onnx"

# --- Silero VAD tuning ---
VAD_THRESHOLD         = 0.5    # Speech confidence cutoff (0.0-1.0). Raise if noisy room, lower if clipping speech
VAD_SILENCE_DURATION  = 1.5   # Seconds of silence before recording stops
VAD_MAX_DURATION      = 15    # Hard cap in seconds — robot won't wait longer than this
VAD_FRAME_SAMPLES     = 512   # Silero expects 512 samples at 16kHz — do not change

# --- Confirmation loop tuning ---
CONFIRM_SILENCE_DURATION = 0.8  # Shorter silence buffer for yes/no answers
CONFIRM_MAX_DURATION     = 5    # Hard cap for yes/no listen
YES_KEYWORDS = ["yes", "yeah", "yep", "correct", "right", "sure", "mhm", "yup"]
NO_KEYWORDS  = ["no", "nope", "nah", "wrong", "incorrect", "not", "didn't", "that's not"]
MAX_CONFIRM_RETRIES = 2         # How many times to re-record before giving up and accepting anyway

# --- Streaming TTS tuning ---
STREAM_SENTENCE_ENDINGS = [".", "?", "!"]   # Chars that trigger Piper to speak the buffered chunk
STREAM_MIN_CHUNK_WORDS  = 3                 # Don't send to Piper until at least this many words buffered

LOG_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs")
PATIENT_INFO_PATH = os.path.join(LOG_DIR, "patientINFO.json")
CONFIG_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "config.json")

os.makedirs(LOG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  SILERO VAD — load once at startup
# ═══════════════════════════════════════════════════════════════════

print("[vad] Loading Silero VAD model...")
vad_model = load_silero_vad()
print("[vad] Silero VAD ready.")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO DEVICE AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════

def find_alsa_device():
    """
    Scan aplay -l for a USB audio playback device.
    Returns ALSA string like 'plughw:2,0' or None if not found.
    """
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                match = re.search(r"card (\d+):", line)
                if match:
                    device_str = f"plughw:{match.group(1)},0"
                    print(f"[audio] ALSA playback → {device_str}  ({line.strip()})")
                    return device_str
        print("[audio] WARNING: USB playback device not found in aplay -l")
        return None
    except Exception as e:
        print(f"[audio] ALSA detection failed: {e}")
        return None


def find_portaudio_input_device():
    """
    Find USB mic index in PortAudio (integer index sounddevice needs).
    Returns integer index or None if not found.
    """
    devices = sd.query_devices()
    print("[audio] PortAudio device scan:")
    for i, dev in enumerate(devices):
        tag = "  ← USB MIC" if (AUDIO_DEVICE_KEYWORD.lower() in dev["name"].lower() and dev["max_input_channels"] > 0) else ""
        print(f"  [{i:2d}] in={dev['max_input_channels']} out={dev['max_output_channels']}  {dev['name']}{tag}")

    for i, dev in enumerate(devices):
        if AUDIO_DEVICE_KEYWORD.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            print(f"[audio] PortAudio input → index {i} ({dev['name']})")
            return i

    print("[audio] WARNING: USB mic not found in PortAudio list")
    return None


def check_mic_present():
    """Fast ALSA-level check: is a USB capture device registered at all?"""
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                print(f"[audio] arecord mic confirmed: {line.strip()}")
                return True
        print("[audio] WARNING: USB mic not found in arecord -l")
        return False
    except Exception as e:
        print(f"[audio] Mic check failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  SERVER HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════

def check_server():
    try:
        r = requests.get(f"{SERVER_URL}/ping", timeout=5)
        if r.status_code == 200:
            print(f"[server] Connected to {SERVER_URL}")
            return True
        print(f"[server] Unexpected status {r.status_code}")
        return False
    except Exception as e:
        print(f"[server] Cannot reach {SERVER_URL}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  CONFIG HELPERS
# ═══════════════════════════════════════════════════════════════════

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def get_room_number(config):
    room = config.get("room_number", "101")
    print(f"[config] Room: {room}")
    return str(room).strip()


def get_patient_name(config):
    name = config.get("patient_name", "Patient").strip()
    print(f"[config] Patient: {name}")
    return "".join(word.capitalize() for word in name.split())


# ═══════════════════════════════════════════════════════════════════
#  TEXT TO SPEECH
# ═══════════════════════════════════════════════════════════════════

def speak(text, alsa_device):
    """Speak a full string — blocks until done."""
    print(f"[tts] {text}")
    try:
        piper = subprocess.Popen(
            ["piper", "--model", PIPER_MODEL, "--output_raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        aplay = subprocess.Popen(
            ["aplay", "-D", alsa_device, "-r", "22050", "-f", "S16_LE", "-c", "1"],
            stdin=piper.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        piper.stdin.write(text.encode("utf-8"))
        piper.stdin.close()
        aplay.wait()
        piper.wait()
    except Exception as e:
        print(f"[tts] Error: {e}")


def speak_chunk(chunk, alsa_device):
    """
    Speak a single text chunk — used by streaming to play sentence-by-sentence.
    Same as speak() but silences piper errors to keep streaming output clean.
    """
    if not chunk.strip():
        return
    try:
        piper = subprocess.Popen(
            ["piper", "--model", PIPER_MODEL, "--output_raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        aplay = subprocess.Popen(
            ["aplay", "-D", alsa_device, "-r", "22050", "-f", "S16_LE", "-c", "1"],
            stdin=piper.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        piper.stdin.write(chunk.strip().encode("utf-8"))
        piper.stdin.close()
        aplay.wait()
        piper.wait()
    except Exception as e:
        print(f"[tts] Chunk error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO RECORDING — Silero VAD
# ═══════════════════════════════════════════════════════════════════

def record(pa_device_index, silence_duration=VAD_SILENCE_DURATION, max_duration=VAD_MAX_DURATION):
    """
    Record using Silero VAD — stops automatically when patient goes quiet.
    silence_duration: seconds of silence before stopping
    max_duration: hard cap so robot never hangs
    """
    vad_iterator = VADIterator(
        vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLERATE,
        min_silence_duration_ms=int(silence_duration * 1000),
        speech_pad_ms=100
    )

    silence_frames_needed = int(silence_duration * SAMPLERATE / VAD_FRAME_SAMPLES)
    max_frames            = int(max_duration * SAMPLERATE / VAD_FRAME_SAMPLES)

    frames          = []
    silent_frames   = 0
    started_speaking = False

    print(f"[mic] Listening with Silero VAD (index={pa_device_index})...")

    try:
        with sd.InputStream(
            samplerate=SAMPLERATE,
            channels=1,
            dtype='int16',
            device=pa_device_index,
            blocksize=VAD_FRAME_SAMPLES
        ) as stream:
            for _ in range(max_frames):
                frame, _ = stream.read(VAD_FRAME_SAMPLES)
                frames.append(frame.copy())

                # Convert int16 → float32 tensor for Silero
                tensor = torch.frombuffer(frame.tobytes(), dtype=torch.int16).float() / 32768.0
                confidence = vad_model(tensor, SAMPLERATE).item()

                if confidence > VAD_THRESHOLD:
                    started_speaking = True
                    silent_frames = 0
                elif started_speaking:
                    silent_frames += 1
                    if silent_frames >= silence_frames_needed:
                        print(f"[mic] Silence detected — stopping.")
                        break

    except Exception as e:
        print(f"[mic] Recording failed: {e}")
        return b""

    if not frames:
        return b""

    audio = np.concatenate(frames, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLERATE, audio)
        path = f.name

    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


# ═══════════════════════════════════════════════════════════════════
#  SERVER COMMUNICATION
# ═══════════════════════════════════════════════════════════════════

def transcribe(audio_bytes):
    try:
        r = requests.post(
            f"{SERVER_URL}/transcribe",
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30
        )
        result = r.json()
        print(f"[stt] Heard: '{result['text']}'")
        return result["text"], result["status"]
    except Exception as e:
        print(f"[stt] Failed: {e}")
        return "", "STABLE"


def get_next_question_streaming(history, alsa_device):
    """
    Stream tokens from server as they arrive.
    Accumulates into sentence chunks and feeds each to Piper immediately
    so patient hears the question starting within ~1s of it being generated.
    Returns the full question string for logging.
    """
    try:
        r = requests.post(
            f"{SERVER_URL}/next_question",
            json={"history": history},
            stream=True,
            timeout=60
        )

        full_text   = ""
        buffer      = ""

        for chunk in r.iter_content(chunk_size=1, decode_unicode=True):
            if not chunk:
                continue

            buffer    += chunk
            full_text += chunk

            # Check if we've hit a sentence boundary and have enough words
            if any(buffer.rstrip().endswith(end) for end in STREAM_SENTENCE_ENDINGS):
                word_count = len(buffer.strip().split())
                if word_count >= STREAM_MIN_CHUNK_WORDS:
                    print(f"[stream] Speaking chunk: '{buffer.strip()}'")
                    speak_chunk(buffer, alsa_device)
                    buffer = ""

        # Speak any remaining buffer (e.g. question didn't end with punctuation)
        if buffer.strip():
            speak_chunk(buffer, alsa_device)

        result = full_text.strip().split("\n")[0]  # First line safeguard
        print(f"[next_question] Full question: '{result}'")
        return result

    except Exception as e:
        print(f"[question] Streaming failed: {e}")
        return "DONE"


def check_urgent(history):
    try:
        r = requests.post(f"{SERVER_URL}/flag_urgent", json={"history": history}, timeout=60)
        return r.json()["flagged_urgent"]
    except Exception as e:
        print(f"[urgent] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  CONFIRMATION LOOP
# ═══════════════════════════════════════════════════════════════════

def confirm_answer(answer_text, alsa_device, pa_device_index):
    """
    Read back what the robot heard and ask the patient to confirm.
    Returns the confirmed answer text (re-recorded if patient says no).
    Gives up and accepts the original after MAX_CONFIRM_RETRIES failed attempts.
    """
    current_answer = answer_text

    for attempt in range(MAX_CONFIRM_RETRIES + 1):
        # Read back what was heard
        if not current_answer.strip():
            speak("Sorry, I didn't catch that. Could you please repeat?", alsa_device)
        else:
            speak(f"I heard: {current_answer}. Is that correct?", alsa_device)

        # Listen for yes/no
        audio = record(pa_device_index, silence_duration=CONFIRM_SILENCE_DURATION, max_duration=CONFIRM_MAX_DURATION)
        response, _ = transcribe(audio)
        response_lower = response.lower()

        print(f"[confirm] Patient said: '{response_lower}'")

        # Check for yes
        if any(word in response_lower for word in YES_KEYWORDS):
            print("[confirm] Confirmed.")
            return current_answer

        # Check for no
        if any(word in response_lower for word in NO_KEYWORDS):
            if attempt < MAX_CONFIRM_RETRIES:
                speak("Sorry about that. Please say your answer again.", alsa_device)
                audio = record(pa_device_index)
                new_answer, _ = transcribe(audio)
                print(f"[confirm] Re-recorded: '{new_answer}'")
                current_answer = new_answer
            else:
                # Gave up re-trying — accept what we have
                print("[confirm] Max retries reached — accepting current answer.")
                speak("Thank you, I'll note that down.", alsa_device)
                return current_answer
        else:
            # Ambiguous response — treat as confirmed and move on
            print("[confirm] Ambiguous response — treating as confirmed.")
            return current_answer

    return current_answer


# ═══════════════════════════════════════════════════════════════════
#  LCD HELPERS
# ═══════════════════════════════════════════════════════════════════

def show_error_on_lcd(lcd, line2, line3):
    if lcd is None:
        return
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string("!! SERVICE NEEDED !!".ljust(20)[:20])
    lcd.cursor_pos = (1, 0)
    lcd.write_string("".ljust(20))
    lcd.cursor_pos = (2, 0)
    lcd.write_string(line2.center(20)[:20])
    lcd.cursor_pos = (3, 0)
    lcd.write_string(line3.center(20)[:20])


def show_status_on_lcd(lcd, status, flagged_urgent):
    if lcd is None:
        return
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string("== TRIAGE STATUS ==".ljust(20))
    lcd.cursor_pos = (1, 0)
    lcd.write_string("".ljust(20))
    lcd.cursor_pos = (2, 0)
    if status == "URGENT":
        lcd.write_string(">>  !! URGENT !!  <<".ljust(20)[:20])
    elif status == "MONITOR":
        lcd.write_string(">>    MONITOR     <<".ljust(20)[:20])
    else:
        lcd.write_string(">>    STABLE      <<".ljust(20)[:20])
    lcd.cursor_pos = (3, 0)
    if flagged_urgent:
        lcd.write_string("!! NURSE ALERTED !!".ljust(20)[:20])
    else:
        lcd.write_string("Assessment complete".ljust(20)[:20])


def init_lcd():
    try:
        from RPLCD.i2c import CharLCD
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
        lcd.clear()
        return lcd
    except Exception as e:
        print(f"[lcd] Not connected: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════

def save_log(session_dir, session_record):
    os.makedirs(session_dir, exist_ok=True)

    with open(os.path.join(session_dir, "log.json"), "w") as f:
        json.dump(session_record, f, indent=2)
    print(f"[log] Session saved to {session_dir}/log.json")

    all_records = []
    if os.path.exists(PATIENT_INFO_PATH):
        try:
            with open(PATIENT_INFO_PATH, "r") as f:
                content = f.read().strip()
            if content:
                all_records = json.loads(content)
            else:
                print("[log] patientINFO.json was empty — starting fresh")
        except json.JSONDecodeError as e:
            print(f"[log] patientINFO.json corrupt ({e}) — starting fresh")

    all_records.append(session_record)

    with open(PATIENT_INFO_PATH, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"[log] Appended to patientINFO.json ({len(all_records)} total records)")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():

    # ── 1. ALSA playback device ──────────────────────────────────
    alsa_device = find_alsa_device()
    if alsa_device is None:
        print("[main] FATAL: No USB speaker found — cannot speak.")
        return

    # ── 2. LCD (optional) ────────────────────────────────────────
    lcd = init_lcd()
    if lcd:
        lcd.cursor_pos = (0, 0)
        lcd.write_string("PULSE  BOOTING...".ljust(20))

    # ── 3. Mic check (ALSA level) ────────────────────────────────
    if not check_mic_present():
        print("[main] Mic not detected at ALSA level.")
        show_error_on_lcd(lcd, "MIC NOT FOUND", "Check Hardware")
        speak("I cannot hear you right now. I need to go get serviced. Please contact the nurse station.", alsa_device)
        return

    # ── 4. PortAudio input index ─────────────────────────────────
    pa_input_index = find_portaudio_input_device()
    if pa_input_index is None:
        print("[main] Mic found by ALSA but not by PortAudio.")
        show_error_on_lcd(lcd, "MIC NOT FOUND", "Check Hardware")
        speak("I cannot hear you right now. I need to go get serviced. Please contact the nurse station.", alsa_device)
        return

    # ── 5. Server connectivity check ─────────────────────────────
    if not check_server():
        print("[main] Server unreachable.")
        show_error_on_lcd(lcd, "SERVER OFFLINE", "Check Laptop")
        speak("I cannot connect to my brain right now. Let me visit the doctor!", alsa_device)
        return

    # ── 6. Config ────────────────────────────────────────────────
    config       = load_config()
    room         = get_room_number(config)
    patient_name = get_patient_name(config)

    # ── 7. Session metadata ──────────────────────────────────────
    date         = datetime.now().strftime("%m/%d/%Y")
    time_str     = datetime.now().strftime("%I:%M%p")
    session_name = f"{date.replace('/', '-')}_{time_str}_Room{room}_{patient_name}"
    session_dir  = os.path.join(LOG_DIR, session_name)
    print(f"\n[session] Starting: {session_name}")

    if lcd:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string("PULSE  READY".ljust(20))

    # ── 8. Greeting ──────────────────────────────────────────────
    speak(f"Hello {patient_name}! I am the nurse assistant robot. I will ask you a few quick questions.", alsa_device)

    # ── 9. Triage setup ──────────────────────────────────────────
    priority = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}
    history  = []
    status   = "STABLE"

    # ── 10. Mandatory first question ─────────────────────────────
    first_q = "On a scale of 1 to 10, how would you rate your pain right now?"
    speak(first_q, alsa_device)
    time.sleep(0.5)

    audio = record(pa_device_index=pa_input_index)
    answer, answer_status = transcribe(audio)
    answer = confirm_answer(answer, alsa_device, pa_input_index)  # ← confirmation loop

    history.append({"q": first_q, "a": answer, "status": answer_status})
    if priority[answer_status] > priority[status]:
        status = answer_status
    print(f"[status] After Q1: {status}")

    # ── 11. AI follow-up questions (streamed) ────────────────────
    for i in range(MAX_QUESTIONS - 1):
        # Streams tokens → speaks sentence chunks as they arrive
        next_q = get_next_question_streaming(history, alsa_device)

        if next_q.strip().upper() == "DONE":
            print(f"[patient] AI done after {i + 1} follow-up(s).")
            break

        time.sleep(0.5)

        audio = record(pa_device_index=pa_input_index)
        answer, answer_status = transcribe(audio)
        answer = confirm_answer(answer, alsa_device, pa_input_index)  # ← confirmation loop

        history.append({"q": next_q, "a": answer, "status": answer_status})
        if priority[answer_status] > priority[status]:
            status = answer_status
        print(f"[status] After Q{i + 2}: {status}")

    # ── 12. Final urgent check ───────────────────────────────────
    flagged_urgent = check_urgent(history)

    if flagged_urgent:
        status = "URGENT"
        speak(f"Based on your responses {patient_name}, I am alerting a nurse immediately.", alsa_device)
    else:
        speak(f"Thank you {patient_name}. Your status has been recorded as {status}.", alsa_device)

    print(f"\n[final] Status: {status} | Urgent: {flagged_urgent}")

    # ── 13. Show final status on LCD ─────────────────────────────
    show_status_on_lcd(lcd, status, flagged_urgent)

    # ── 14. Save logs ────────────────────────────────────────────
    save_log(
        session_dir,
        {
            "header": {
                "patient_name": patient_name,
                "room": room,
                "date": date,
                "time": time_str
            },
            "triage": {
                "final_status": status,
                "flagged_urgent": flagged_urgent
            },
            "assessment": history
        }
    )


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()