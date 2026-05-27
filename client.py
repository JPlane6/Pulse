import sounddevice as sd
import scipy.io.wavfile as wav
import requests
from requests.exceptions import Timeout, ConnectionError
import tempfile
import os
import re
import time
import subprocess
import json
import threading
from datetime import datetime
import scipy.signal as signal
import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

# ═══════════════════════════════════════════════════════════════════
#  CRITICAL PI PERFORMANCE TUNING
# ═══════════════════════════════════════════════════════════════════
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

SERVER_URL           = "http://192.168.0.157:5001"
MAX_QUESTIONS        = 6
SAMPLERATE           = 16000
AUDIO_DEVICE_KEYWORD = "USB"
PIPER_MODEL          = "/home/ayushs0604/Pulse/en_US-hfc_female-medium.onnx"

BARGE_IN_ENABLED     = False
VAD_THRESHOLD        = 0.70
BARGE_IN_IGNORE_SECS = 6.0

LED_RED_PIN          = 17
LED_GREEN_PIN        = 27
LED_BLUE_PIN         = 22

VAD_SILENCE_DURATION = 1.5
VAD_MAX_DURATION     = 15
VAD_FRAME_SAMPLES    = 512

LOG_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs")
PATIENT_INFO_PATH = os.path.join(LOG_DIR, "patientINFO.json")
CONFIG_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "config.json")

os.makedirs(LOG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  VAD
# ═══════════════════════════════════════════════════════════════════

print("[vad] Loading Silero VAD framework...")
vad_model = load_silero_vad()
print("[vad] Silero VAD framework loaded.")

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
    RED_LINE.set_value(0)
    GREEN_LINE.set_value(0)
    BLUE_LINE.set_value(0)
    GPIO_AVAILABLE = True
    print(f"[gpio] Linked via {target_chip}")
except Exception as e:
    print(f"[gpio] Bypassed ({e}). Running in simulation mode.")


# ═══════════════════════════════════════════════════════════════════
#  LED
# ═══════════════════════════════════════════════════════════════════

def set_status_led(status_type):
    if not GPIO_AVAILABLE:
        print(f"[led-simulation] Status updated to color profile: {status_type}")
        return
    try:
        RED_LINE.set_value(0)
        GREEN_LINE.set_value(0)
        BLUE_LINE.set_value(0)
        if status_type == "URGENT":
            RED_LINE.set_value(1)
        elif status_type == "MONITOR":
            BLUE_LINE.set_value(1)
        else:
            GREEN_LINE.set_value(1)
    except Exception as e:
        print(f"[gpio] LED error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════

def run_microphone_calibration(pa_device_index):
    print("[calibration] Running 2-second mic diagnostic...")
    try:
        device_info = sd.query_devices(pa_device_index, "input")
        native_sr   = int(device_info["default_samplerate"])
    except Exception:
        native_sr = 48000
    try:
        recording = sd.rec(int(2.0 * native_sr), samplerate=native_sr, channels=1, dtype="int16", device=pa_device_index)
        sd.wait()
        rms = np.sqrt(np.mean(recording.flatten().astype(np.float32) ** 2))
        print(f"[calibration] RMS: {rms:.2f}")
        if rms < 1.0:
            print("[calibration] FATAL: No signal energy detected.")
            return False
        return True
    except Exception as e:
        print(f"[calibration] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  BARGE-IN
# ═══════════════════════════════════════════════════════════════════

def monitor_barge_in(aplay_process, piper_process, pa_device_index):
    start_time = time.time()
    try:
        device_info = sd.query_devices(pa_device_index, "input")
        native_sr   = int(device_info["default_samplerate"])
    except Exception:
        native_sr = 48000
    hw_blocksize = int(VAD_FRAME_SAMPLES * (native_sr / SAMPLERATE))
    try:
        with sd.InputStream(samplerate=native_sr, channels=1, dtype="int16", device=pa_device_index, blocksize=hw_blocksize) as stream:
            while aplay_process.poll() is None:
                frame, _ = stream.read(hw_blocksize)
                if time.time() - start_time < BARGE_IN_IGNORE_SECS:
                    continue
                frame_fixed    = np.ascontiguousarray(frame, dtype=np.int16)
                f32_frame      = frame_fixed.flatten().astype(np.float32) / 32768.0
                target_samples = int(len(f32_frame) * SAMPLERATE / native_sr)
                resampled      = signal.resample(f32_frame, target_samples)
                tensor         = torch.from_numpy(resampled).float()
                if vad_model(tensor, SAMPLERATE).item() > VAD_THRESHOLD:
                    print("\n[barge-in] Voice detected — killing playback.")
                    aplay_process.terminate()
                    piper_process.terminate()
                    break
    except Exception as e:
        print(f"[barge-in] Error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════════

def find_alsa_device():
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                match = re.search(r"card (\d+):", line)
                if match:
                    device_str = f"plughw:{match.group(1)},0"
                    print(f"[audio] ALSA device: {device_str}")
                    return device_str
    except Exception:
        pass
    return None

def find_portaudio_input_device():
    for i, dev in enumerate(sd.query_devices()):
        if AUDIO_DEVICE_KEYWORD.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None

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
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

def get_room_number(config):
    return str(config.get("room_number", "101")).strip()

def get_patient_name(config):
    name = config.get("patient_name", "Patient").strip()
    return "".join(word.capitalize() for word in name.split())


# ═══════════════════════════════════════════════════════════════════
#  TTS
# ═══════════════════════════════════════════════════════════════════

def speak(text, alsa_device, pa_device_index=None):
    print(f"[tts] Speaking: '{text}'")
    try:
        piper = subprocess.Popen(
            ["piper", "--model", PIPER_MODEL, "--output_raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        aplay = subprocess.Popen(
            ["aplay", "-D", alsa_device, "-r", "22050", "-f", "S16_LE", "-c", "1"],
            stdin=piper.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if BARGE_IN_ENABLED and pa_device_index is not None:
            threading.Thread(
                target=monitor_barge_in, args=(aplay, piper, pa_device_index), daemon=True
            ).start()
        piper.stdin.write(text.encode("utf-8"))
        piper.stdin.close()
        aplay.wait()
        piper.wait()
        time.sleep(0.6)
    except Exception as e:
        print(f"[tts] Error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  RECORD
# ═══════════════════════════════════════════════════════════════════

def record(pa_device_index, silence_duration=VAD_SILENCE_DURATION, max_duration=VAD_MAX_DURATION):
    try:
        native_sr = int(sd.query_devices(pa_device_index, "input")["default_samplerate"])
    except Exception:
        native_sr = 48000

    VADIterator(vad_model, threshold=VAD_THRESHOLD, sampling_rate=SAMPLERATE,
                min_silence_duration_ms=int(silence_duration * 1000), speech_pad_ms=100)

    hw_blocksize        = int(VAD_FRAME_SAMPLES * (native_sr / SAMPLERATE))
    silence_frames_need = int(silence_duration * native_sr / hw_blocksize)
    max_frames          = int(max_duration * native_sr / hw_blocksize)

    frames           = []
    silent_frames    = 0
    started_speaking = False

    print(f"[mic] Listening at {native_sr}Hz...")

    try:
        with sd.InputStream(samplerate=native_sr, channels=1, dtype="int16",
                            device=pa_device_index, blocksize=hw_blocksize) as stream:
            for _ in range(max_frames):
                frame           = np.ascontiguousarray(stream.read(hw_blocksize)[0], dtype=np.int16)
                frames.append(frame.copy())
                f32             = frame.flatten().astype(np.float32) / 32768.0
                resampled       = signal.resample(f32, int(len(f32) * SAMPLERATE / native_sr))
                confidence      = vad_model(torch.from_numpy(resampled).float(), SAMPLERATE).item()
                if confidence > VAD_THRESHOLD:
                    started_speaking = True
                    silent_frames    = 0
                elif started_speaking:
                    silent_frames += 1
                    if silent_frames >= silence_frames_need:
                        print("[mic] End of speech detected.")
                        break
    except Exception as e:
        print(f"[mic] Capture error: {e}")
        return b""

    if not frames:
        return b""

    audio_raw  = np.concatenate(frames, axis=0).flatten()
    audio_16k  = signal.resample(audio_raw, int(len(audio_raw) * SAMPLERATE / native_sr)).astype(np.int16)

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

def transcribe(audio_bytes):
    try:
        r   = requests.post(f"{SERVER_URL}/transcribe", data=audio_bytes,
                            headers={"Content-Type": "application/octet-stream"}, timeout=30)
        res = r.json()
        print(f"[stt] '{res['text']}'")
        return res["text"], res["status"]
    except (Timeout, ConnectionError) as e:
        print(f"[stt] Network error: {e}")
        return "", "STABLE"
    except Exception as e:
        print(f"[stt] Error: {e}")
        return "", "STABLE"

def get_next_question(history):
    try:
        r      = requests.post(f"{SERVER_URL}/next_question", json={"history": history}, timeout=60)
        result = r.json()["question"].strip()
        print(f"[next_question] '{result}'")
        return result
    except Exception as e:
        print(f"[next_question] Error: {e}")
        return "DONE"

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

def show_status_on_lcd(lcd, status, flagged_urgent):
    if not lcd: return
    lcd.clear()
    lcd.cursor_pos = (0, 0); lcd.write_string("== TRIAGE STATUS ==")
    lcd.cursor_pos = (2, 0)
    if status == "URGENT":       lcd.write_string(">>  !! URGENT !!  <<")
    elif status == "MONITOR":    lcd.write_string(">>    MONITOR     <<")
    else:                        lcd.write_string(">>    STABLE      <<")
    lcd.cursor_pos = (3, 0)
    lcd.write_string("!! NURSE ALERTED !!" if flagged_urgent else "Assessment complete")


# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════

def save_log(session_dir, session_record):
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, "log.json"), "w") as f:
        json.dump(session_record, f, indent=2)

    all_records = []
    if os.path.exists(PATIENT_INFO_PATH):
        try:
            with open(PATIENT_INFO_PATH, "r") as f:
                content = f.read().strip()
                if content:
                    all_records = json.loads(content)
        except Exception:
            pass

    all_records.append(session_record)
    with open(PATIENT_INFO_PATH, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"[log] Total records: {len(all_records)}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    alsa_device = find_alsa_device()
    if alsa_device is None:
        print("[main] FATAL: No ALSA output found.")
        return

    lcd = init_lcd()
    if lcd: lcd.write_string("PULSE BOOTING...")

    if not check_mic_present() or (pa_input_index := find_portaudio_input_device()) is None:
        show_error_on_lcd(lcd, "MIC NOT FOUND", "Check Hardware")
        speak("I cannot hear you. My microphone is disconnected.", alsa_device)
        return

    if not run_microphone_calibration(pa_input_index):
        show_error_on_lcd(lcd, "MIC MUTED / ERROR", "Run Mixer Checks")
        speak("My microphone is failing diagnostic checks.", alsa_device)
        return

    if not check_server():
        show_error_on_lcd(lcd, "SERVER OFFLINE", "Check Laptop Host")
        speak("I cannot connect to my brain server.", alsa_device)
        return

    config       = load_config()
    room         = get_room_number(config)
    patient_name = get_patient_name(config)
    date         = datetime.now().strftime("%m/%d/%Y")
    time_str     = datetime.now().strftime("%I:%M%p")
    session_dir  = os.path.join(LOG_DIR, f"{date.replace('/', '-')}_{time_str}_Room{room}_{patient_name}")

    if lcd:
        lcd.clear()
        lcd.write_string("PULSE READY")

    set_status_led("STABLE")

    priority = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}
    history  = []
    status   = "STABLE"

    # First question
    first_q = f"Hello {patient_name}! I am Pulse. What's bothering you today?"
    speak(first_q, alsa_device, pa_input_index)
    audio                 = record(pa_device_index=pa_input_index)
    answer, answer_status = transcribe(audio)
    history.append({"q": first_q, "a": answer, "status": answer_status})
    if priority[answer_status] > priority[status]:
        status = answer_status
    set_status_led(status)
    print(f"[status] {status}")

    # Follow-up loop
    for i in range(MAX_QUESTIONS - 1):
        print(f"\n--- Question {i + 2} of {MAX_QUESTIONS} ---")

        next_q = get_next_question(history)

        if next_q.strip().upper() == "DONE":
            print("[main] AI signalled DONE.")
            break

        speak(next_q, alsa_device, pa_input_index)
        audio                 = record(pa_device_index=pa_input_index)
        answer, answer_status = transcribe(audio)
        history.append({"q": next_q, "a": answer, "status": answer_status})
        if priority[answer_status] > priority[status]:
            status = answer_status
        set_status_led(status)
        print(f"[status] {status}")

        # Only short-circuit on URGENT if a hard keyword or very high pain was detected.
        # Do NOT break just because the per-answer classifier returned URGENT —
        # let the final flag_urgent call make the definitive call.
        # We only hard-break if status has been URGENT for 2+ consecutive answers.
        urgent_streak = sum(1 for qa in history[-2:] if qa.get("status") == "URGENT")
        if urgent_streak >= 2:
            print("[main] Two consecutive URGENT answers — ending questions early.")
            break

    # ── Final decision ─────────────────────────────────────────────
    # flag_urgent is the authoritative call.
    # We do NOT OR it with status == URGENT here — the server already
    # considers Q&A urgency counts internally before calling the AI.
    print("\n--- Finalizing ---")
    flagged_urgent = check_urgent(history)

    if flagged_urgent:
        status = "URGENT"
        set_status_led("URGENT")
        speak(f"Based on your responses {patient_name}, I am alerting a nurse immediately.", alsa_device, pa_input_index)
    else:
        # Status may still be MONITOR — that's fine, no nurse alert needed
        speak(f"Thank you {patient_name}. Your assessment is complete.", alsa_device, pa_input_index)

    speak("Thank you for your time. Feel better soon.", alsa_device, pa_input_index)

    show_status_on_lcd(lcd, status, flagged_urgent)
    save_log(session_dir, {
        "header":     {"patient_name": patient_name, "room": room, "date": date, "time": time_str},
        "triage":     {"final_status": status, "flagged_urgent": flagged_urgent},
        "assessment": history
    })


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