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

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════

SERVER_URL    = "http://192.168.0.157:5000"
MAX_QUESTIONS = 6
SAMPLERATE    = 48000

# Keyword matched against aplay/arecord -l output AND PortAudio device names.
# Your headset shows as "USB Composite Device" — "USB" catches it.
AUDIO_DEVICE_KEYWORD = "USB"

PIPER_MODEL = "/home/ayushs0604/Pulse/en_US-amy-medium.onnx"

LOG_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs")
PATIENT_INFO_PATH = os.path.join(LOG_DIR, "patientINFO.json")
CONFIG_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "config.json")

os.makedirs(LOG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  AUDIO DEVICE AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════

def find_alsa_device():
    """
    Scan aplay -l for a USB audio playback device.
    Returns ALSA string like 'plughw:2,0' (used by aplay for TTS output).
    Returns None if not found.
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
    Find USB mic index in PortAudio (what sounddevice actually uses).

    sounddevice does NOT accept ALSA 'plughw:X,0' strings — it needs a
    PortAudio integer device index. We scan sd.query_devices() for a device
    whose name contains AUDIO_DEVICE_KEYWORD AND has at least 1 input channel.

    Returns integer index, or None if not found.
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
    """
    Ping the server's /ping endpoint before starting the session.
    Returns True if reachable, False otherwise.
    Times out after 5 seconds so the robot doesn't hang.
    """
    try:
        r = requests.get(f"{SERVER_URL}/ping", timeout=5)
        if r.status_code == 200:
            print(f"[server] Connected to {SERVER_URL}")
            return True
        else:
            print(f"[server] Unexpected status {r.status_code} from {SERVER_URL}")
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
#  TEXT TO SPEECH  (ALSA string — aplay handles it fine)
# ═══════════════════════════════════════════════════════════════════

def speak(text, alsa_device):
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
        aplay_rc = aplay.wait()
        piper_rc = piper.wait()
        if piper_rc != 0:
            print(f"[tts] Piper error: {piper.stderr.read().decode()}")
        if aplay_rc != 0:
            print(f"[tts] Aplay error: {aplay.stderr.read().decode()}")
        if piper_rc == 0 and aplay_rc == 0:
            print("[tts] Done.")
    except Exception as e:
        print(f"[tts] Error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO RECORDING  (PortAudio integer index — sounddevice needs this)
# ═══════════════════════════════════════════════════════════════════

def record(duration, pa_device_index):
    """
    Record `duration` seconds. pa_device_index is an integer from
    find_portaudio_input_device(). Returns 16kHz WAV bytes, or b'' on failure.
    """
    print(f"[mic] Recording {duration}s (PortAudio index={pa_device_index})...")
    try:
        audio = sd.rec(
            int(duration * SAMPLERATE),
            samplerate=SAMPLERATE,
            channels=1,
            dtype='float32',
            device=pa_device_index
        )
        sd.wait()
    except Exception as e:
        print(f"[mic] Recording failed: {e}")
        return b""

    audio_resampled = signal.resample(audio, int(len(audio) * 16000 / SAMPLERATE))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, 16000, audio_resampled)
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


def get_next_question(history):
    try:
        r = requests.post(f"{SERVER_URL}/next_question", json={"history": history}, timeout=60)
        return r.json()["question"]
    except Exception as e:
        print(f"[question] Failed: {e}")
        return "DONE"


def check_urgent(history):
    try:
        r = requests.post(f"{SERVER_URL}/flag_urgent", json={"history": history}, timeout=60)
        return r.json()["flagged_urgent"]
    except Exception as e:
        print(f"[urgent] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  LCD HELPERS
# ═══════════════════════════════════════════════════════════════════

def show_error_on_lcd(lcd, line2, line3):
    """Generic full-screen error display."""
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
    """
    Full-screen triage result. Uses all 4 rows — no distances.

    Row 0: == TRIAGE STATUS ==
    Row 1: (blank spacer)
    Row 2: >>  !! URGENT !!  <<  /  >>    MONITOR    <<  /  >>    STABLE    <<
    Row 3: !! NURSE ALERTED !!   /  Assessment complete
    """
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
    """Try to connect to LCD. Returns lcd object or None."""
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

    # Handle missing or corrupt/empty patientINFO.json gracefully
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

    # ── 1. ALSA playback device (speaker / TTS) ─────────────────
    alsa_device = find_alsa_device()
    if alsa_device is None:
        print("[main] FATAL: No USB speaker found — cannot speak.")
        return

    # ── 2. LCD (optional — robot continues without it) ───────────
    lcd = init_lcd()
    if lcd:
        lcd.cursor_pos = (0, 0)
        lcd.write_string("PULSE  BOOTING...".ljust(20))

    # ── 3. Mic check (ALSA level) ────────────────────────────────
    if not check_mic_present():
        print("[main] Mic not detected at ALSA level.")
        show_error_on_lcd(lcd, "MIC NOT FOUND", "Check Hardware")
        speak(
            "I cannot hear you right now. I need to go get serviced. "
            "Please contact the nurse station.",
            alsa_device
        )
        return

    # ── 4. PortAudio input index (sounddevice level) ─────────────
    pa_input_index = find_portaudio_input_device()
    if pa_input_index is None:
        print("[main] Mic found by ALSA but not by PortAudio.")
        show_error_on_lcd(lcd, "MIC NOT FOUND", "Check Hardware")
        speak(
            "I cannot hear you right now. I need to go get serviced. "
            "Please contact the nurse station.",
            alsa_device
        )
        return

    # ── 5. Server connectivity check ─────────────────────────────
    if not check_server():
        print("[main] Server unreachable.")
        show_error_on_lcd(lcd, "SERVER OFFLINE", "Check Laptop")
        speak(
            "I cannot connect to my brain right now."
            "Let me visit the doctor!",
            alsa_device
        )
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
    speak(
        f"Hello {patient_name}! "
        f"I am the nurse assistant robot. "
        f"I will ask you a few quick questions.",
        alsa_device
    )

    # ── 9. Triage setup ──────────────────────────────────────────
    priority = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}
    history  = []
    status   = "STABLE"

    # ── 10. Mandatory first question ─────────────────────────────
    first_q = "On a scale of 1 to 10, how would you rate your pain right now?"
    speak(first_q, alsa_device)
    time.sleep(1)

    answer, answer_status = transcribe(record(duration=12, pa_device_index=pa_input_index))
    history.append({"q": first_q, "a": answer, "status": answer_status})
    if priority[answer_status] > priority[status]:
        status = answer_status
    print(f"[status] After Q1: {status}")

    # ── 11. AI follow-up questions ───────────────────────────────
    for i in range(MAX_QUESTIONS - 1):
        next_q = get_next_question(history)

        if next_q.strip().upper() == "DONE":
            print(f"[patient] AI done after {i + 1} follow-up(s).")
            break

        speak(next_q, alsa_device)
        time.sleep(1)

        answer, answer_status = transcribe(record(duration=12, pa_device_index=pa_input_index))
        history.append({"q": next_q, "a": answer, "status": answer_status})
        if priority[answer_status] > priority[status]:
            status = answer_status
        print(f"[status] After Q{i + 2}: {status}")

    # ── 12. Final urgent check ───────────────────────────────────
    flagged_urgent = check_urgent(history)

    if flagged_urgent:
        status = "URGENT"
        speak(
            f"Based on your responses {patient_name}, "
            f"I am alerting a nurse immediately.",
            alsa_device
        )
    else:
        speak(
            f"Thank you {patient_name}. "
            f"Your status has been recorded as {status}.",
            alsa_device
        )

    print(f"\n[final] Status: {status} | Urgent: {flagged_urgent}")

    # ── 13. Show final status on LCD (full screen) ───────────────
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