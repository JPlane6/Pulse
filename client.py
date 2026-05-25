import sounddevice as sd
import scipy.io.wavfile as wav
import requests
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
#  TOP-LEVEL FEATURES & HARDWARE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

SERVER_URL       = "http://192.168.0.157:5001"
MAX_QUESTIONS    = 6
SAMPLERATE       = 16000          # Internal rate needed by Silero/Whisper
AUDIO_DEVICE_KEYWORD = "USB"      # Identifies your target USB mic & speaker
PIPER_MODEL      = "/home/ayushs0604/Pulse/en_US-amy-medium.onnx"

# --- Feature 3: Barge-In Interruption Toggle ---
BARGE_IN_ENABLED = True           # Set to False to disable user interrupting the robot

# --- Feature 4: RGB Status LED Pin Layout (GPIO BCM numbering) ---
LED_RED_PIN      = 17             # Lights up for URGENT status
LED_GREEN_PIN    = 27             # Lights up for STABLE status
LED_BLUE_PIN     = 22             # Lights up for MONITOR status

# --- Silero VAD configurations ---
VAD_THRESHOLD         = 0.5    # Speech confidence sensitivity cutoff
VAD_SILENCE_DURATION  = 1.5    # Seconds of silence required to close recording
VAD_MAX_DURATION      = 15     # Safety stop cap to prevent infinite loop listening
VAD_FRAME_SAMPLES     = 512    # Core sample slice evaluation window size

# --- Confirmation logic options ---
CONFIRM_SILENCE_DURATION = 0.8
CONFIRM_MAX_DURATION     = 5
YES_KEYWORDS = ["yes", "yeah", "yep", "correct", "right", "sure", "mhm", "yup"]
NO_KEYWORDS  = ["no", "nope", "nah", "wrong", "incorrect", "not", "didn't", "that's not"]
MAX_CONFIRM_RETRIES = 2

LOG_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "logs")
PATIENT_INFO_PATH = os.path.join(LOG_DIR, "patientINFO.json")
CONFIG_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules", "config.json")

os.makedirs(LOG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  INITIALIZE HARDWARE & VAD ENGINE
# ═══════════════════════════════════════════════════════════════════

print("[vad] Loading Silero VAD framework...")
vad_model = load_silero_vad()
print("[vad] Silero VAD framework loaded.")

# Safe initialization of GPIO pins for Pi 5
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LED_RED_PIN, GPIO.OUT)
    GPIO.setup(LED_GREEN_PIN, GPIO.OUT)
    GPIO.setup(LED_BLUE_PIN, GPIO.OUT)
    GPIO.output(LED_RED_PIN, GPIO.LOW)
    GPIO.output(LED_GREEN_PIN, GPIO.LOW)
    GPIO.output(LED_BLUE_PIN, GPIO.LOW)
    GPIO_AVAILABLE = True
    print("[gpio] LED Pins successfully initialized and verified.")
except Exception as e:
    print(f"[gpio] Optional GPIO libraries or LEDs not detected ({e}). Continuing in simulation mode safely.")


# ═══════════════════════════════════════════════════════════════════
#  FEATURE 4: LED HARDWARE WRAPPER (Won't break if missing!)
# ═══════════════════════════════════════════════════════════════════

def set_status_led(status_type):
    """Sets the RGB LED color matching your requirements. Safely falls back if un-wired."""
    if not GPIO_AVAILABLE:
        print(f"[led-simulation] Status updated to color profile: {status_type}")
        return

    try:
        # Clear all pins first
        GPIO.output(LED_RED_PIN, GPIO.LOW)
        GPIO.output(LED_GREEN_PIN, GPIO.LOW)
        GPIO.output(LED_BLUE_PIN, GPIO.LOW)

        if status_type == "URGENT":
            GPIO.output(LED_RED_PIN, GPIO.HIGH)    # Red for Urgent
        elif status_type == "MONITOR":
            GPIO.output(LED_BLUE_PIN, GPIO.HIGH)   # Blue for Monitor
        else:
            GPIO.output(LED_GREEN_PIN, GPIO.HIGH)  # Green for Stable
    except Exception as e:
        print(f"[gpio] LED hardware adjustment hit an exception: {e}")


# ═══════════════════════════════════════════════════════════════════
#  FEATURE 5: HARDWARE CALIBRATION AND BOOT DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

def run_microphone_calibration(pa_device_index):
    """Listens to environment to guarantee the hardware isn't zeroed out or muted."""
    print("[calibration] Executing 2-second microphone system diagnostic check...")
    try:
        device_info = sd.query_devices(pa_device_index, 'input')
        native_sr = int(device_info['default_samplerate'])
    except Exception:
        native_sr = 48000

    try:
        # Record a silent 2 second window frame
        recording = sd.rec(int(2.0 * native_sr), samplerate=native_sr, channels=1, dtype='int16', device=pa_device_index)
        sd.wait()
        
        # Calculate Root Mean Square energy metrics
        rms = np.sqrt(np.mean(recording.flatten()**2))
        print(f"[calibration] Diagnostic captured absolute baseline signal energy RMS: {rms:.2f}")
        
        if rms < 1.0: # Absolute computational silence implies physical connection/mixer fault
            print("[calibration] FATAL: Audio track contains no signal energy. Microphone is muted or unlinked.")
            return False
        return True
    except Exception as e:
        print(f"[calibration] System failed to collect baseline validation: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  FEATURE 3: BARGE-IN INTERRUPTION HANDLER
# ═══════════════════════════════════════════════════════════════════

def monitor_barge_in(aplay_process, piper_process, pa_device_index):
    """Background monitoring thread that instantly stops speech if patient talks."""
    try:
        device_info = sd.query_devices(pa_device_index, 'input')
        native_sr = int(device_info['default_samplerate'])
    except Exception:
        native_sr = 48000

    hw_blocksize = int(VAD_FRAME_SAMPLES * (native_sr / SAMPLERATE))
    
    try:
        with sd.InputStream(samplerate=native_sr, channels=1, dtype='int16', device=pa_device_index, blocksize=hw_blocksize) as stream:
            while aplay_process.poll() is None:  # While audio is actively playing out
                frame, _ = stream.read(hw_blocksize)
                f32_frame = frame.flatten().astype(np.float32) / 32768.0
                target_num_samples = int(len(f32_frame) * SAMPLERATE / native_sr)
                resampled_f32 = signal.resample(f32_frame, target_num_samples)
                tensor = torch.from_numpy(resampled_f32).float()

                if vad_model(tensor, SAMPLERATE).item() > VAD_THRESHOLD:
                    print("\n[barge-in] User voice interruption detected! Terminating playback processes instantly.")
                    aplay_process.terminate()
                    piper_process.terminate()
                    break
    except Exception as e:
        print(f"[barge-in] Interruption engine hit an exception tracking frames: {e}")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO PROBING & HARDWARE LOOKUP UTILITIES
# ═══════════════════════════════════════════════════════════════════

def find_alsa_device():
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                match = re.search(r"card (\d+):", line)
                if match:
                    device_str = f"plughw:{match.group(1)},0"
                    print(f"[audio] Found ALSA speaker output link: {device_str}")
                    return device_str
        return None
    except Exception:
        return None

def find_portaudio_input_device():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if AUDIO_DEVICE_KEYWORD.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None

def check_mic_present():
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                return True
        return False
    except Exception:
        return False

def check_server():
    try:
        r = requests.get(f"{SERVER_URL}/ping", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f: return json.load(f)
    return {}

def get_room_number(config): return str(config.get("room_number", "101")).strip()
def get_patient_name(config):
    name = config.get("patient_name", "Patient").strip()
    return "".join(word.capitalize() for word in name.split())


# ═══════════════════════════════════════════════════════════════════
#  TEXT TO SPEECH (WITH INTEGRATED BARGE-IN MULTITHREADING)
# ═══════════════════════════════════════════════════════════════════

def speak(text, alsa_device, pa_device_index=None):
    """Standard blocking synthesis upgraded with adaptive barge-in capability handles."""
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
        
        # Deploy Interruption processing thread if explicitly enabled and inputs pass
        if BARGE_IN_ENABLED and (pa_device_index is not None):
            barge_thread = threading.Thread(target=monitor_barge_in, args=(aplay, piper, pa_device_index), daemon=True)
            barge_thread.start()

        piper.stdin.write(text.encode("utf-8"))
        piper.stdin.close()
        aplay.wait()
        piper.wait()
        time.sleep(0.6)  # Safety pad gives physical speakers structural completion time
    except Exception as e:
        print(f"[tts] Execution hit error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO CAPTURE & RESAMPLING ENGINE
# ═══════════════════════════════════════════════════════════════════

def record(pa_device_index, silence_duration=VAD_SILENCE_DURATION, max_duration=VAD_MAX_DURATION):
    try:
        device_info = sd.query_devices(pa_device_index, 'input')
        native_sr = int(device_info['default_samplerate'])
    except Exception:
        native_sr = 48000

    vad_iterator = VADIterator(
        vad_model, threshold=VAD_THRESHOLD, sampling_rate=SAMPLERATE,
        min_silence_duration_ms=int(silence_duration * 1000), speech_pad_ms=100
    )

    hw_blocksize = int(VAD_FRAME_SAMPLES * (native_sr / SAMPLERATE))
    silence_frames_needed = int(silence_duration * native_sr / hw_blocksize)
    max_frames            = int(max_duration * native_sr / hw_blocksize)

    frames = []
    silent_frames = 0
    started_speaking = False

    print(f"[mic] Opening hardware capture stream safely at native {native_sr}Hz...")

    try:
        with sd.InputStream(samplerate=native_sr, channels=1, dtype='int16', device=pa_device_index, blocksize=hw_blocksize) as stream:
            for _ in range(max_frames):
                frame, _ = stream.read(hw_blocksize)
                frames.append(frame.copy())

                f32_frame = frame.flatten().astype(np.float32) / 32768.0
                target_num_samples = int(len(f32_frame) * SAMPLERATE / native_sr)
                resampled_f32 = signal.resample(f32_frame, target_num_samples)
                tensor = torch.from_numpy(resampled_f32).float()

                confidence = vad_model(tensor, SAMPLERATE).item()

                if confidence > VAD_THRESHOLD:
                    started_speaking = True
                    silent_frames = 0
                elif started_speaking:
                    silent_frames += 1
                    if silent_frames >= silence_frames_needed:
                        print(f"[mic] End of response detected via VAD.")
                        break
    except Exception as e:
        print(f"[mic] Capture interface failure: {e}")
        return b""

    if not frames: return b""

    audio_raw = np.concatenate(frames, axis=0).flatten()
    total_samples = int(len(audio_raw) * SAMPLERATE / native_sr)
    audio_16k = signal.resample(audio_raw, total_samples).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLERATE, audio_16k)
        path = f.name

    with open(path, "rb") as f: data = f.read()
    os.remove(path)
    return data


# ═══════════════════════════════════════════════════════════════════
#  STREAM CONNECTOR FOR RUNTIME QUESTION EXECUTION
# ═══════════════════════════════════════════════════════════════════

def get_next_question_streaming(history, alsa_device, pa_device_index=None):
    """Pipes token streams directly to Piper processes word-by-word instantly."""
    try:
        r = requests.post(f"{SERVER_URL}/next_question", json={"history": history}, stream=True, timeout=60)
        full_text = ""
        word_buffer = []
        
        piper = subprocess.Popen(
            ["piper", "--model", PIPER_MODEL, "--output_raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        aplay = subprocess.Popen(
            ["aplay", "-D", alsa_device, "-r", "22050", "-f", "S16_LE", "-c", "1"],
            stdin=piper.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        if BARGE_IN_ENABLED and (pa_device_index is not None):
            barge_thread = threading.Thread(target=monitor_barge_in, args=(aplay, piper, pa_device_index), daemon=True)
            barge_thread.start()

        print("[stream] Initializing token stream readout...")
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk: continue
            full_text += chunk
            word_buffer.append(chunk)

            if len(word_buffer) >= 2:
                phrase = "".join(word_buffer)
                print(f"[stream] Writing chunk: '{phrase}'", end="", flush=True)
                if piper.poll() is None: # Secure pipe checks case of sudden user barge-in hits
                    try:
                        piper.stdin.write(phrase.encode("utf-8"))
                        piper.stdin.flush()
                    except IOError:
                        break
                word_buffer = []

        if word_buffer and piper.poll() is None:
            try:
                piper.stdin.write("".join(word_buffer).encode("utf-8"))
                piper.stdin.flush()
            except IOError: pass

        print("\n[stream] Generation complete. Wrapping up speech lines...")
        try: piper.stdin.close()
        except Exception: pass
        aplay.wait()
        piper.wait()
        time.sleep(0.6)

        result = full_text.strip().split("\n")[0]
        return result
    except Exception as e:
        print(f"[stream] Connection process encountered failure: {e}")
        return "DONE"


# ═══════════════════════════════════════════════════════════════════
#  SERVER UTILITY CALL TRANSFERS
# ═══════════════════════════════════════════════════════════════════

def transcribe(audio_bytes):
    try:
        r = requests.post(
            f"{SERVER_URL}/transcribe",
            data=audio_bytes, headers={"Content-Type": "application/octet-stream"},
            timeout=30
        )
        res = r.json()
        print(f"[stt] Server Text Output: '{res['text']}'")
        return res["text"], res["status"]
    except Exception as e:
        print(f"[stt] Remote transcription dropped: {e}")
        return "", "STABLE"

def check_urgent(history):
    try:
        r = requests.post(f"{SERVER_URL}/flag_urgent", json={"history": history}, timeout=60)
        return r.json()["flagged_urgent"]
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
#  FEATURE 2: LOCAL AMBIGUITY EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════

def confirm_answer(answer_text, alsa_device, pa_device_index):
    """Verifies response validity and isolates conversational ambiguity locally."""
    current_answer = answer_text

    for attempt in range(MAX_CONFIRM_RETRIES + 1):
        # Local Ambiguity Capture: Handle empty records or unintelligible static inputs directly
        if not current_answer.strip() or len(current_answer.strip()) < 2:
            print("[confirm-local] Caught empty or ambiguous string entry. Requesting clarification.")
            speak("I didn't quite catch that. Could you please repeat your response clearly?", alsa_device, pa_device_index)
            audio = record(pa_device_index)
            new_answer, _ = transcribe(audio)
            current_answer = new_answer
            continue

        speak(f"I heard: {current_answer}. Is that correct?", alsa_device, pa_device_index)
        audio = record(pa_device_index, silence_duration=CONFIRM_SILENCE_DURATION, max_duration=CONFIRM_MAX_DURATION)
        response, _ = transcribe(audio)
        response_lower = response.lower()

        print(f"[confirm] Checking user confirmation phrase: '{response_lower}'")

        if any(word in response_lower for word in YES_KEYWORDS):
            return current_answer

        if any(word in response_lower for word in NO_KEYWORDS):
            if attempt < MAX_CONFIRM_RETRIES:
                speak("My apologies. Please state your answer again.", alsa_device, pa_device_index)
                audio = record(pa_device_index)
                new_answer, _ = transcribe(audio)
                current_answer = new_answer
            else:
                speak("Thank you, noting that down.", alsa_device, pa_device_index)
                return current_answer
        else:
            print("[confirm] Ambiguous phrasing sequence — pushing cycle forward.")
            return current_answer

    return current_answer


# ═══════════════════════════════════════════════════════════════════
#  LCD DISPLAY LOGIC MODULES
# ═══════════════════════════════════════════════════════════════════

def show_error_on_lcd(lcd, line2, line3):
    if not lcd: return
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string("!! SERVICE NEEDED !!")
    lcd.cursor_pos = (2, 0)
    lcd.write_string(line2.center(20)[:20])
    lcd.cursor_pos = (3, 0)
    lcd.write_string(line3.center(20)[:20])

def show_status_on_lcd(lcd, status, flagged_urgent):
    if not lcd: return
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string("== TRIAGE STATUS ==")
    lcd.cursor_pos = (2, 0)
    if status == "URGENT": lcd.write_string(">>  !! URGENT !!  <<")
    elif status == "MONITOR": lcd.write_string(">>    MONITOR     <<")
    else: lcd.write_string(">>    STABLE      <<")
    lcd.cursor_pos = (3, 0)
    lcd.write_string("!! NURSE ALERTED !!" if flagged_urgent else "Assessment complete")

def init_lcd():
    try:
        from RPLCD.i2c import CharLCD
        lcd = CharLCD("PCF8574", 0x27, cols=20, rows=4)
        lcd.clear()
        return lcd
    except Exception:
        return None

def save_log(session_dir, session_record):
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, "log.json"), "w") as f: json.dump(session_record, f, indent=2)

    all_records = []
    if os.path.exists(PATIENT_INFO_PATH):
        try:
            with open(PATIENT_INFO_PATH, "r") as f:
                content = f.read().strip()
                if content: all_records = json.loads(content)
        except Exception: pass

    all_records.append(session_record)
    with open(PATIENT_INFO_PATH, "w") as f: json.dump(all_records, f, indent=2)
    print(f"[log] Data logged. Grand total records compiled: {len(all_records)}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN SYSTEM PROGRAM RUNTIME
# ═══════════════════════════════════════════════════════════════════

def main():
    alsa_device = find_alsa_device()
    if alsa_device is None:
        print("[main] FATAL: ALSA playback output infrastructure missing.")
        return

    lcd = init_lcd()
    if lcd: lcd.write_string("PULSE BOOTING...")

    if not check_mic_present() or (pa_input_index := find_portaudio_input_device()) is None:
        show_error_on_lcd(lcd, "MIC NOT FOUND", "Check Hardware")
        speak("I cannot hear you right now. My microphone is disconnected.", alsa_device)
        return

    # Feature 5: Run mic hardware audio calibration checks
    if not run_microphone_calibration(pa_input_index):
        show_error_on_lcd(lcd, "MIC MUTED / ERROR", "Run Mixer Checks")
        speak("My microphone is muted or failing diagnostic checks.", alsa_device)
        return

    if not check_server():
        show_error_on_lcd(lcd, "SERVER OFFLINE", "Check Laptop Host")
        speak("I cannot connect to my brain server right now.", alsa_device)
        return

    config       = load_config()
    room         = get_room_number(config)
    patient_name = get_patient_name(config)

    date     = datetime.now().strftime("%m/%d/%Y")
    time_str = datetime.now().strftime("%I:%M%p")
    session_dir = os.path.join(LOG_DIR, f"{date.replace('/', '-')}_{time_str}_Room{room}_{patient_name}")

    if lcd:
        lcd.clear()
        lcd.write_string("PULSE READY")

    # Feature 4 Setup: Initialize Default State LED (Stable = Green)
    set_status_led("STABLE")

    speak(f"Hello {patient_name}! I am your nurse assistant robot. Let's check in on you.", alsa_device, pa_input_index)

    priority = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}
    history  = []
    status   = "STABLE"

    first_q = "On a scale of 1 to 10, how would you rate your pain right now?"
    speak(first_q, alsa_device, pa_input_index)
    
    audio = record(pa_device_index=pa_input_index)
    answer, answer_status = transcribe(audio)
    answer = confirm_answer(answer, alsa_device, pa_input_index)

    history.append({"q": first_q, "a": answer, "status": answer_status})
    if priority[answer_status] > priority[status]: status = answer_status
    
    # Update physical LED status color dynamically
    set_status_led(status)
    print(f"[status-update] Current Patient Status Evaluation: {status}")

    for i in range(MAX_QUESTIONS - 1):
        print(f"\n--- Follow-up Question Cycle {i+1} of {MAX_QUESTIONS-1} ---")
        next_q = get_next_question_streaming(history, alsa_device, pa_input_index)
        
        if next_q.strip().upper() == "DONE":
            print("[main] LLM invoked 'DONE' signal. Ending session loop early.")
            break

        audio = record(pa_device_index=pa_input_index)
        answer, answer_status = transcribe(audio)
        answer = confirm_answer(answer, alsa_device, pa_input_index)

        history.append({"q": next_q, "a": answer, "status": answer_status})
        if priority[answer_status] > priority[status]: status = answer_status
        
        # Update physical LED status color dynamically
        set_status_led(status)
        print(f"[status-update] Current Patient Status Evaluation: {status}")

    print("\n--- Finalizing Triage Decisions ---")
    flagged_urgent = check_urgent(history)
    if flagged_urgent:
        status = "URGENT"
        set_status_led("URGENT")
        speak(f"Based on your responses {patient_name}, I am alerting a nurse immediately.", alsa_device, pa_input_index)
    else:
        speak(f"Thank you {patient_name}. Your assessment is complete.", alsa_device, pa_input_index)

    show_status_on_lcd(lcd, status, flagged_urgent)
    save_log(session_dir, {
        "header": {"patient_name": patient_name, "room": room, "date": date, "time": time_str},
        "triage": {"final_status": status, "flagged_urgent": flagged_urgent},
        "assessment": history
    })

if __name__ == "__main__":
    try:
        main()
    finally:
        if GPIO_AVAILABLE:
            GPIO.cleanup() # Clean up pin allocations gracefully on termination