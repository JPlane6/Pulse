import sounddevice as sd
import scipy.io.wavfile as wav
import scipy.signal as signal
import requests
from requests.exceptions import Timeout, ConnectionError
import tempfile
import os
import re
import time
import subprocess
import json
import base64
import threading
from datetime import datetime
import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════
#  PI PERFORMANCE TUNING
# ═══════════════════════════════════════════════════════════════════
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
SERVER_URL           = "http://192.168.0.157:5001"
MAX_QUESTIONS        = 6
SAMPLERATE           = 16000
AUDIO_DEVICE_KEYWORD = "USB"
PIPER_MODEL          = "/home/ayushs0604/Pulse/en_US-hfc_female-medium.onnx"

VAD_THRESHOLD        = 0.70
VAD_SILENCE_DURATION = 1.5
VAD_MAX_DURATION     = 15
VAD_FRAME_SAMPLES    = 512
RMS_GATE             = 150

# How many words to buffer before flushing a chunk to Piper.
# Lowered to 2 so short questions don't get stuck waiting for more words.
STREAM_CHUNK_WORDS   = 2

LED_RED_PIN   = 17
LED_GREEN_PIN = 27
LED_BLUE_PIN  = 22

LOG_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs")
PATIENT_INFO_PATH = os.path.join(LOG_DIR, "patientINFO.json")
CONFIG_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "config.json")

os.makedirs(LOG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  VAD — background load so it doesn't block the greeting
# ═══════════════════════════════════════════════════════════════════
vad_model  = None
_vad_ready = threading.Event()

def _load_vad_bg():
    global vad_model
    print("[vad] Loading Silero VAD in background...")
    from silero_vad import load_silero_vad
    vad_model = load_silero_vad()
    _vad_ready.set()
    print("[vad] Ready.")

threading.Thread(target=_load_vad_bg, daemon=True).start()

# ═══════════════════════════════════════════════════════════════════
#  GPIO
# ═══════════════════════════════════════════════════════════════════
GPIO_AVAILABLE = False
RED_LINE = GREEN_LINE = BLUE_LINE = None

try:
    import gpiod
    target_chip = "gpiochip0"
    if not os.path.exists("/dev/gpiochip0") and os.path.exists("/dev/gpiochip4"):
        target_chip = "gpiochip4"
    LED_CHIP   = gpiod.Chip(target_chip)
    RED_LINE   = LED_CHIP.get_line(LED_RED_PIN)
    GREEN_LINE = LED_CHIP.get_line(LED_GREEN_PIN)
    BLUE_LINE  = LED_CHIP.get_line(LED_BLUE_PIN)
    RED_LINE.request(consumer="PULSE",   type=gpiod.LINE_REQ_DIR_OUT)
    GREEN_LINE.request(consumer="PULSE", type=gpiod.LINE_REQ_DIR_OUT)
    BLUE_LINE.request(consumer="PULSE",  type=gpiod.LINE_REQ_DIR_OUT)
    RED_LINE.set_value(0); GREEN_LINE.set_value(0); BLUE_LINE.set_value(0)
    GPIO_AVAILABLE = True
    print(f"[gpio] Linked via {target_chip}")
except Exception as e:
    print(f"[gpio] Bypassed ({e}). Simulation mode.")


def set_status_led(status_type):
    if not GPIO_AVAILABLE:
        print(f"[led] {status_type}")
        return
    try:
        RED_LINE.set_value(0); GREEN_LINE.set_value(0); BLUE_LINE.set_value(0)
        if status_type == "URGENT":    RED_LINE.set_value(1)
        elif status_type == "MONITOR": BLUE_LINE.set_value(1)
        else:                          GREEN_LINE.set_value(1)
    except Exception as e:
        print(f"[gpio] LED error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO DEVICE CACHE
# ═══════════════════════════════════════════════════════════════════
_native_sr       = None
_pa_device_index = None
_resample_up     = None
_resample_down   = None
_alsa_device     = None


def find_alsa_device():
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                m = re.search(r"card (\d+):", line)
                if m:
                    d = f"plughw:{m.group(1)},0"
                    print(f"[audio] ALSA: {d}")
                    return d
    except Exception:
        pass
    return None


def find_portaudio_input_device():
    for i, dev in enumerate(sd.query_devices()):
        if AUDIO_DEVICE_KEYWORD.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None


def init_audio_cache(alsa_dev, pa_index):
    global _native_sr, _resample_up, _resample_down, _pa_device_index, _alsa_device
    _alsa_device     = alsa_dev
    _pa_device_index = pa_index
    info             = sd.query_devices(pa_index, "input")
    _native_sr       = int(info["default_samplerate"])
    from math import gcd
    g              = gcd(SAMPLERATE, _native_sr)
    _resample_up   = SAMPLERATE  // g
    _resample_down = _native_sr  // g
    print(f"[audio] Native={_native_sr}Hz, resample {_native_sr}→{SAMPLERATE} ({_resample_up}/{_resample_down})")


def check_mic_present():
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                return True
    except Exception:
        pass
    return False


def check_server():
    try:
        r = requests.get(f"{SERVER_URL}/ping", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

def get_room_number(cfg): return str(cfg.get("room_number", "101")).strip()
def get_patient_name(cfg):
    return "".join(w.capitalize() for w in cfg.get("patient_name", "Patient").strip().split())


# ═══════════════════════════════════════════════════════════════════
#  TTS ENGINE
# ═══════════════════════════════════════════════════════════════════

def _open_piper_pipeline():
    piper = subprocess.Popen(
        ["piper", "--model", PIPER_MODEL, "--output_raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    aplay = subprocess.Popen(
        ["aplay", "-D", _alsa_device, "-r", "22050", "-f", "S16_LE", "-c", "1"],
        stdin=piper.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return piper, aplay


def _close_pipeline(piper, aplay):
    try:
        piper.stdin.close()
    except Exception:
        pass
    try:
        aplay.wait(timeout=10)
    except Exception:
        pass
    try:
        piper.wait(timeout=3)
    except Exception:
        pass


def speak_plain(text):
    """Blocking TTS for fixed phrases."""
    if not text.strip():
        return
    print(f"[tts] '{text}'")
    piper, aplay = _open_piper_pipeline()
    try:
        piper.stdin.write((text.strip() + "\n").encode("utf-8"))
    except BrokenPipeError:
        pass
    _close_pipeline(piper, aplay)


def speak_stream(sse_response):
    """
    Consumes SSE token stream from /turn_stream, drives Piper in real time.

    FIX 1: buf is flushed immediately when a '?' is seen — this ensures
            the last word of a short question is never left hanging.
    FIX 2: After the SSE loop exits, any remaining buf is always flushed
            before _close_pipeline is called.
    FIX 3: Tokens that contain DONE signals are dropped silently so
            leaked prompt text never reaches Piper.

    Returns (transcribed_text, status, full_question).
    """
    text     = ""
    status   = "STABLE"
    question = "DONE"

    # Signals from the server prompt that must never be spoken
    MUTE_SIGNALS = [
        "done", "if all", "all covered", "topics covered",
        "output done", "output: done", "no further", "no more"
    ]

    piper, aplay = _open_piper_pipeline()
    buf = ""

    def _flush(chunk):
        chunk = chunk.strip()
        if not chunk:
            return
        # Drop any chunk that looks like a leaked instruction
        chunk_lower = chunk.lower()
        for sig in MUTE_SIGNALS:
            if sig in chunk_lower:
                print(f"[tts-stream] MUTED leaked chunk: '{chunk}'")
                return
        print(f"[tts-stream] speaking: '{chunk}'")
        try:
            piper.stdin.write((chunk + " ").encode("utf-8"))
            piper.stdin.flush()
        except BrokenPipeError:
            pass

    try:
        for raw_line in sse_response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            try:
                evt = json.loads(line[6:])
            except Exception:
                continue

            # ── Final event — read metadata, stop streaming ────────
            if evt.get("done"):
                text     = evt.get("text",     "")
                status   = evt.get("status",   "STABLE")
                question = evt.get("question", "DONE")
                break

            token = evt.get("t", "")
            if not token:
                continue

            buf += token

            # Flush immediately on sentence-ending punctuation
            if buf.rstrip()[-1:] in ".?!":
                _flush(buf)
                buf = ""
                continue

            # Flush when we have enough words buffered
            if token.endswith(" ") and len(buf.split()) >= STREAM_CHUNK_WORDS:
                _flush(buf)
                buf = ""

    except Exception as e:
        print(f"[tts-stream] Stream error: {e}")

    # ── FIX: always flush whatever is left in buf ──────────────────
    if buf.strip():
        _flush(buf)

    _close_pipeline(piper, aplay)
    return text, status, question


# ═══════════════════════════════════════════════════════════════════
#  RECORD
# ═══════════════════════════════════════════════════════════════════

def record(silence_duration=VAD_SILENCE_DURATION, max_duration=VAD_MAX_DURATION):
    if not _vad_ready.is_set():
        print("[mic] Waiting for VAD to finish loading...")
        _vad_ready.wait()

    hw_blocksize        = int(VAD_FRAME_SAMPLES * (_native_sr / SAMPLERATE))
    silence_frames_need = int(silence_duration * _native_sr / hw_blocksize)
    max_frames          = int(max_duration * _native_sr / hw_blocksize)

    frames           = []
    silent_frames    = 0
    started_speaking = False

    print("[mic] Listening...")

    try:
        with sd.InputStream(samplerate=_native_sr, channels=1, dtype="int16",
                            device=_pa_device_index, blocksize=hw_blocksize) as stream:
            for _ in range(max_frames):
                raw, _  = stream.read(hw_blocksize)
                frame   = raw.flatten()
                frames.append(frame.copy())

                rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
                if rms < RMS_GATE:
                    if started_speaking:
                        silent_frames += 1
                        if silent_frames >= silence_frames_need:
                            print("[mic] Silence — done.")
                            break
                    continue

                f32       = frame.astype(np.float32) / 32768.0
                resampled = signal.resample_poly(f32, _resample_up, _resample_down)
                conf      = vad_model(torch.from_numpy(resampled).float(), SAMPLERATE).item()

                if conf > VAD_THRESHOLD:
                    started_speaking = True
                    silent_frames    = 0
                elif started_speaking:
                    silent_frames += 1
                    if silent_frames >= silence_frames_need:
                        print("[mic] End of speech detected.")
                        break

    except Exception as e:
        print(f"[mic] Error: {e}")
        return b""

    if not frames:
        return b""

    audio_raw = np.concatenate(frames, axis=0)
    audio_16k = signal.resample_poly(
        audio_raw.astype(np.float32), _resample_up, _resample_down
    ).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLERATE, audio_16k)
        path = f.name
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


# ═══════════════════════════════════════════════════════════════════
#  SERVER CALLS
# ═══════════════════════════════════════════════════════════════════

def turn_stream(audio_bytes, history):
    try:
        payload = {
            "audio_b64": base64.b64encode(audio_bytes).decode(),
            "history":   history
        }
        r = requests.post(
            f"{SERVER_URL}/turn_stream",
            json=payload,
            stream=True,
            timeout=60
        )
        return r
    except Exception as e:
        print(f"[turn_stream] Connection error: {e}")
        return None


def check_urgent(history):
    try:
        r = requests.post(f"{SERVER_URL}/flag_urgent", json={"history": history}, timeout=60)
        return r.json()["flagged_urgent"]
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
#  LCD
# ═══════════════════════════════════════════════════════════════════

def init_lcd():
    try:
        from RPLCD.i2c import CharLCD
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
        lcd.clear()
        return lcd
    except Exception:
        return None


def show_error_on_lcd(lcd, line2, line3):
    if not lcd: return
    lcd.clear()
    lcd.cursor_pos = (0, 0); lcd.write_string("!! SERVICE NEEDED !!")
    lcd.cursor_pos = (2, 0); lcd.write_string(line2.center(20)[:20])
    lcd.cursor_pos = (3, 0); lcd.write_string(line3.center(20)[:20])


def show_status_on_lcd(lcd, status, flagged):
    if not lcd: return
    lcd.clear()
    lcd.cursor_pos = (0, 0); lcd.write_string("== TRIAGE STATUS ==")
    lcd.cursor_pos = (2, 0)
    if status == "URGENT":    lcd.write_string(">>  !! URGENT !!  <<")
    elif status == "MONITOR": lcd.write_string(">>    MONITOR     <<")
    else:                     lcd.write_string(">>    STABLE      <<")
    lcd.cursor_pos = (3, 0)
    lcd.write_string("!! NURSE ALERTED !!" if flagged else "Assessment complete")


# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════

def save_log(session_dir, record_data):
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, "log.json"), "w") as f:
        json.dump(record_data, f, indent=2)
    all_records = []
    if os.path.exists(PATIENT_INFO_PATH):
        try:
            content = open(PATIENT_INFO_PATH).read().strip()
            if content:
                all_records = json.loads(content)
        except Exception:
            pass
    all_records.append(record_data)
    with open(PATIENT_INFO_PATH, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"[log] {len(all_records)} total records.")


# ═══════════════════════════════════════════════════════════════════
#  STATUS ANNOUNCEMENT
#  Spoken at the very end of every session, before the robot moves on.
# ═══════════════════════════════════════════════════════════════════

STATUS_PHRASES = {
    "URGENT":  "Your condition has been marked as urgent. A nurse is being alerted right now.",
    "MONITOR": "Your condition has been marked as monitor. A nurse will check on you shortly.",
    "STABLE":  "Your condition has been marked as stable. You are doing okay.",
}

def announce_status(patient_name, status, flagged):
    """Speak the triage result out loud."""
    if flagged or status == "URGENT":
        phrase = STATUS_PHRASES["URGENT"]
    elif status == "MONITOR":
        phrase = STATUS_PHRASES["MONITOR"]
    else:
        phrase = STATUS_PHRASES["STABLE"]

    speak_plain(f"{patient_name}, {phrase}")
    speak_plain("Feel better soon.")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Hardware discovery ──────────────────────────────────────────
    alsa_device = find_alsa_device()
    if alsa_device is None:
        print("[main] FATAL: No ALSA output found.")
        return
    global _alsa_device
    _alsa_device = alsa_device

    lcd = init_lcd()
    if lcd: lcd.write_string("PULSE BOOTING...")

    if not check_mic_present():
        show_error_on_lcd(lcd, "MIC NOT FOUND", "Check Hardware")
        speak_plain("I cannot hear you. My microphone is disconnected.")
        return

    pa_input_index = find_portaudio_input_device()
    if pa_input_index is None:
        show_error_on_lcd(lcd, "MIC NOT FOUND", "Check Hardware")
        speak_plain("I cannot find the microphone input.")
        return

    init_audio_cache(alsa_device, pa_input_index)

    if not check_server():
        show_error_on_lcd(lcd, "SERVER OFFLINE", "Check Laptop Host")
        speak_plain("I cannot connect to my brain server.")
        return

    cfg          = load_config()
    room         = get_room_number(cfg)
    patient_name = get_patient_name(cfg)
    date         = datetime.now().strftime("%m/%d/%Y")
    time_str     = datetime.now().strftime("%I:%M%p")
    session_dir  = os.path.join(LOG_DIR, f"{date.replace('/','-')}_{time_str}_Room{room}_{patient_name}")

    if lcd:
        lcd.clear(); lcd.write_string("PULSE READY")

    set_status_led("STABLE")

    priority = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}
    history  = []
    status   = "STABLE"

    # ── First question ──────────────────────────────────────────────
    first_q = f"Hello {patient_name}! I am Pulse. What's bothering you today?"
    speak_plain(first_q)

    audio = record()

    sse = turn_stream(audio, [])
    if sse is None:
        print("[main] Server unreachable on first turn.")
        return

    answer, answer_status, next_q = speak_stream(sse)
    history.append({"q": first_q, "a": answer, "status": answer_status})
    if priority[answer_status] > priority[status]:
        status = answer_status
    set_status_led(status)
    print(f"[status] {status}")

    # ── Follow-up loop ──────────────────────────────────────────────
    for i in range(MAX_QUESTIONS - 1):
        if not next_q or next_q.strip().upper() == "DONE":
            print("[main] AI signalled DONE.")
            break

        print(f"\n--- Q{i+2} of {MAX_QUESTIONS}: '{next_q}' ---")

        audio = record()

        sse = turn_stream(audio, history)
        if sse is None:
            print("[main] Server unreachable — ending session.")
            break

        prev_q = next_q
        answer, answer_status, next_q = speak_stream(sse)

        history.append({"q": prev_q, "a": answer, "status": answer_status})
        if priority[answer_status] > priority[status]:
            status = answer_status
        set_status_led(status)
        print(f"[status] {status}")

        urgent_streak = sum(1 for qa in history[-2:] if qa.get("status") == "URGENT")
        if urgent_streak >= 2:
            print("[main] Two consecutive URGENT — ending early.")
            break

    # ── Final decision ──────────────────────────────────────────────
    print("\n--- Finalizing ---")
    flagged = check_urgent(history)

    if flagged:
        status = "URGENT"

    set_status_led(status)

    # ── Announce the result out loud ────────────────────────────────
    announce_status(patient_name, status, flagged)

    show_status_on_lcd(lcd, status, flagged)
    save_log(session_dir, {
        "header":     {"patient_name": patient_name, "room": room, "date": date, "time": time_str},
        "triage":     {"final_status": status, "flagged_urgent": flagged},
        "assessment": history
    })
    print(f"[main] Session complete. Status={status} Flagged={flagged}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if GPIO_AVAILABLE:
            try:
                RED_LINE.set_value(0); GREEN_LINE.set_value(0); BLUE_LINE.set_value(0)
                RED_LINE.release();    GREEN_LINE.release();    BLUE_LINE.release()
            except Exception:
                pass