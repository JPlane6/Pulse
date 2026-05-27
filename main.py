"""
Pulse Triage System - Client Hardware Node
Handles local VAD (Silero), offline TTS (Piper), hardware GPIO/LCD mapping, 
and communicates with the central Pulse AI server for triage assessment.
"""

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
#  VAD INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

print("[vad] Loading Silero VAD framework...")
vad_model = load_silero_vad()
print("[vad] Silero VAD framework loaded successfully.")

# ═══════════════════════════════════════════════════════════════════
#  GPIO INITIALIZATION
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
    
    # Initialize all LEDs to OFF
    RED_LINE.set_value(0)
    GREEN_LINE.set_value(0)
    BLUE_LINE.set_value(0)
    
    GPIO_AVAILABLE = True
    print(f"[gpio] Hardware LEDs linked via {target_chip}")
except Exception as e:
    print(f"[gpio] Hardware bypassed ({e}). Running in simulation mode.")


# ═══════════════════════════════════════════════════════════════════
#  LED CONTROLS
# ═══════════════════════════════════════════════════════════════════

def set_status_led(status_type):
    """
    Updates the hardware LED based on the current triage status.
    URGENT = Red, MONITOR = Blue, STABLE = Green.
    """
    if not GPIO_AVAILABLE:
        print(f"[led-simulation] Status updated to color profile: {status_type}")
        return
    try:
        # Reset all lights first
        RED_LINE.set_value(0)
        GREEN_LINE.set_value(0)
        BLUE_LINE.set_value(0)
        
        # Illuminate target status
        if status_type == "URGENT":
            RED_LINE.set_value(1)
        elif status_type == "MONITOR":
            BLUE_LINE.set_value(1)
        else:
            GREEN_LINE.set_value(1)
    except Exception as e:
        print(f"[gpio] LED state update failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  HARDWARE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

def run_microphone_calibration(pa_device_index):
    """
    Records a 2-second clip to ensure the microphone is capturing actual audio
    energy (RMS) and isn't muted at the hardware/OS level.
    """
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
        print(f"[calibration] RMS Value: {rms:.2f}")
        
        if rms < 1.0:
            print("[calibration] FATAL: No signal energy detected. Check physical mic mute button.")
            return False
        return True
    except Exception as e:
        print(f"[calibration] Diagnostic failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  BARGE-IN / INTERRUPTION MONITORING
# ═══════════════════════════════════════════════════════════════════

def monitor_barge_in(aplay_process, piper_process, pa_device_index):
    """
    Runs in a background thread while Piper is speaking. If it detects
    human speech exceeding the VAD threshold, it kills the TTS processes.
    """
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
                
                # Ignore audio for the first few seconds to prevent TTS echoing triggering barge-in
                if time.time() - start_time < BARGE_IN_IGNORE_SECS:
                    continue
                    
                frame_fixed    = np.ascontiguousarray(frame, dtype=np.int16)
                f32_frame      = frame_fixed.flatten().astype(np.float32) / 32768.0
                target_samples = int(len(f32_frame) * SAMPLERATE / native_sr)
                resampled      = signal.resample(f32_frame, target_samples)
                tensor         = torch.from_numpy(resampled).float()
                
                # Check for interruption
                if vad_model(tensor, SAMPLERATE).item() > VAD_THRESHOLD:
                    print("\n[barge-in] Voice detected — killing playback.")
                    aplay_process.terminate()
                    piper_process.terminate()
                    break
    except Exception as e:
        print(f"[barge-in] Monitoring error: {e}")


# ═══════════════════════════════════════════════════════════════════
#  AUDIO & SYSTEM UTILITIES
# ═══════════════════════════════════════════════════════════════════

def find_alsa_device():
    """Locates the ALSA hardware string for the USB speaker."""
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                match = re.search(r"card (\d+):", line)
                if match:
                    device_str = f"plughw:{match.group(1)},0"
                    print(f"[audio] ALSA output device mapped to: {device_str}")
                    return device_str
    except Exception:
        pass
    return None

def find_portaudio_input_device():
    """Locates the PortAudio index for the USB microphone."""
    for i, dev in enumerate(sd.query_devices()):
        if AUDIO_DEVICE_KEYWORD.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None

def check_mic_present():
    """Quick OS-level check to ensure the USB microphone is plugged in."""
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if AUDIO_DEVICE_KEYWORD.lower() in line.lower() and line.startswith("card"):
                return True
    except Exception:
        pass
    return False

def check_server():
    """Pings the central AI server to ensure connectivity before starting assessment."""
    try:
        r = requests.get(f"{SERVER_URL}/ping", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def load_config():
    """Loads localized room and patient data."""
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
#  TEXT TO SPEECH (PIPER)
# ═══════════════════════════════════════════════════════════════════

def warmup_piper():
    """
    Forces Piper to load its model into the OS page cache using a silent string.
    Dramatically reduces latency on the first spoken sentence.
    """
    print("[tts] Warming up Piper caching...")
    try:
        piper = subprocess.Popen(
            ["piper", "--model", PIPER_MODEL, "--output_raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        piper.stdin.write(b" ")
        piper.stdin.close()
        subprocess.run(["cat"], stdin=piper.stdout, stdout=subprocess.DEVNULL)
        piper.wait()
        print("[tts] Piper warmup complete.")
    except Exception as e:
        print(f"[tts] Warmup failed (non-fatal): {e}")


def speak(text, alsa_device, pa_device_index=None):
    """
    Pipes text into the local Piper AI, which pipes raw audio directly into ALSA.
    """
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
        
        # Start background interruption listener if enabled
        if BARGE_IN_ENABLED and pa_device_index is not None:
            threading.Thread(
                target=monitor_barge_in, args=(aplay, piper, pa_device_index), daemon=True
            ).start()
            
        piper.stdin.write(text.encode("utf-8"))
        piper.stdin.close()
        aplay.wait()
        piper.wait()
        time.sleep(0.6) # Brief pause after finishing sentence
    except Exception as e:
        print(f"[tts] Error generating speech: {e}")


# ═══════════════════════════════════════════════════════════════════
#  VOICE CAPTURE
# ═══════════════════════════════════════════════════════════════════

def record(pa_device_index, silence_duration=VAD_SILENCE_DURATION, max_duration=VAD_MAX_DURATION):
    """
    Listens to the microphone indefinitely until voice is detected, then records
    until it detects 'silence_duration' seconds of quiet, or hits 'max_duration'.
    Returns the recorded audio bytes as a 16kHz WAV payload.
    """
    try:
        native_sr = int(sd.query_devices(pa_device_index, "input")["default_samplerate"])
    except Exception:
        native_sr = 48000

    # Initialize Silero VAD Iterator to track speech boundaries
    VADIterator(vad_model, threshold=VAD_THRESHOLD, sampling_rate=SAMPLERATE,
                min_silence_duration_ms=int(silence_duration * 1000), speech_pad_ms=100)

    hw_blocksize        = int(VAD_FRAME_SAMPLES * (native_sr / SAMPLERATE))
    silence_frames_need = int(silence_duration * native_sr / hw_blocksize)
    max_frames          = int(max_duration * native_sr / hw_blocksize)

    frames           = []
    silent_frames    = 0
    started_speaking = False

    print(f"[mic] Listening dynamically at {native_sr}Hz...")

    try:
        with sd.InputStream(samplerate=native_sr, channels=1, dtype="int16",
                            device=pa_device_index, blocksize=hw_blocksize) as stream:
            for _ in range(max_frames):
                frame           = np.ascontiguousarray(stream.read(hw_blocksize)[0], dtype=np.int16)
                frames.append(frame.copy())
                
                # Format frame for Silero processing
                f32             = frame.flatten().astype(np.float32) / 32768.0
                resampled       = signal.resample(f32, int(len(f32) * SAMPLERATE / native_sr))
                confidence      = vad_model(torch.from_numpy(resampled).float(), SAMPLERATE).item()
                
                # State Machine: Check if user is talking or has stopped talking
                if confidence > VAD_THRESHOLD:
                    started_speaking = True
                    silent_frames    = 0
                elif started_speaking:
                    silent_frames += 1
                    if silent_frames >= silence_frames_need:
                        print("[mic] End of speech detected naturally.")
                        break
    except Exception as e:
        print(f"[mic] Hardware capture error: {e}")
        return b""

    if not frames:
        return b""

    # Compile the final audio array and force downsample to 16kHz for STT model
    audio_raw  = np.concatenate(frames, axis=0).flatten()
    audio_16k  = signal.resample(audio_raw, int(len(audio_raw) * SAMPLERATE / native_sr)).astype(np.int16)

    # Wrap in WAV headers temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLERATE, audio_16k)
        path = f.name

    # Read binary back out for HTTP transfer
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    
    return data


# ═══════════════════════════════════════════════════════════════════
#  SERVER COMMUNICATION
# ═══════════════════════════════════════════════════════════════════

def transcribe(audio_bytes):
    """Sends WAV bytes to central server. Expects text transcription and baseline status back."""
    try:
        r   = requests.post(f"{SERVER_URL}/transcribe", data=audio_bytes,
                            headers={"Content-Type": "application/octet-stream"}, timeout=30)
        res = r.json()
        print(f"[stt] Server returned: '{res['text']}'")
        return res["text"], res["status"]
    except (Timeout, ConnectionError) as e:
        print(f"[stt] Network/Connection error: {e}")
        return "", "STABLE"
    except Exception as e:
        print(f"[stt] Unexpected error: {e}")
        return "", "STABLE"

def get_next_question(history):
    """Passes the conversation history to the AI to generate the next logical follow-up."""
    try:
        r      = requests.post(f"{SERVER_URL}/next_question", json={"history": history}, timeout=60)
        result = r.json()["question"].strip()
        print(f"[llm] Next question generated: '{result}'")
        return result
    except Exception as e:
        print(f"[llm] Failed to generate next question: {e}")
        return "DONE"

def check_urgent(history):
    """Final comprehensive review of the history to catch subtle medical emergencies."""
    try:
        r = requests.post(f"{SERVER_URL}/flag_urgent", json={"history": history}, timeout=60)
        return r.json()["flagged_urgent"]
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
#  LCD DISPLAY
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
    
    if status == "URGENT":       
        lcd.write_string(">>  !! URGENT !!  <<")
    elif status == "MONITOR":    
        lcd.write_string(">>    MONITOR     <<")
    else:                        
        lcd.write_string(">>    STABLE      <<")
        
    lcd.cursor_pos = (3, 0)
    lcd.write_string("!! NURSE ALERTED !!" if flagged_urgent else "Assessment complete")


# ═══════════════════════════════════════════════════════════════════
#  DATA LOGGING
# ═══════════════════════════════════════════════════════════════════

def save_log(session_dir, session_record):
    """Saves session logs locally and appends them to the master patient file."""
    os.makedirs(session_dir, exist_ok=True)
    
    # Save isolated session file
    with open(os.path.join(session_dir, "log.json"), "w") as f:
        json.dump(session_record, f, indent=2)

    # Append to master patient tracker
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
        
    print(f"[sys] Log saved. Total historical records: {len(all_records)}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN TRIAGE LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    # 1. Hardware Verification
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

    warmup_piper()

    # 2. Local State Initialization
    config       = load_config()
    room         = get_room_number(config)
    patient_name = get_patient_name(config)
    date         = datetime.now().strftime("%m/%d/%Y")
    time_str     = datetime.now().strftime("%I:%M%p")
    session_dir  = os.path.join(LOG_DIR, f"{date.replace('/', '-')}_{time_str}_Room{room}_{patient_name}")

    if lcd:
        lcd.clear()
        lcd.write_string("PULSE READY")

    priority = {"STABLE": 0, "MONITOR": 1, "URGENT": 2}
    history  = []
    status   = "STABLE"
    
    set_status_led(status)

    # 3. Assessment Launch
    first_q = f"Hello {patient_name}! I am Pulse. What's bothering you today?"
    speak(first_q, alsa_device, pa_input_index)
    
    audio                 = record(pa_device_index=pa_input_index)
    answer, answer_status = transcribe(audio)
    
    history.append({"q": first_q, "a": answer, "status": answer_status})
    
    # Escalate baseline status if necessary
    if priority[answer_status] > priority[status]:
        status = answer_status
        set_status_led(status)
        
    print(f"[main] Current patient status: {status}")

    # 4. Iterative Assessment Loop
    for i in range(MAX_QUESTIONS - 1):
        print(f"\n--- Question {i + 2} of {MAX_QUESTIONS} ---")

        next_q = get_next_question(history)

        if next_q.strip().upper() == "DONE":
            print("[main] AI determined sufficient information gathered.")
            break

        speak(next_q, alsa_device, pa_input_index)
        audio                 = record(pa_device_index=pa_input_index)
        answer, answer_status = transcribe(audio)
        
        history.append({"q": next_q, "a": answer, "status": answer_status})
        
        if priority[answer_status] > priority[status]:
            status = answer_status
            set_status_led(status)
            
        print(f"[main] Current patient status: {status}")

        # ── EMERGENCY SHORT CIRCUIT ──
        # If the patient has given 3 consecutive answers that trigger "URGENT",
        # bypass remaining questions and immediately call for help.
        if len(history) >= 3:
            recent_statuses = [qa.get("status") for qa in history[-3:]]
            if all(s == "URGENT" for s in recent_statuses):
                print("[main] TRACE: 3 consecutive URGENT triggers detected. Short-circuiting loop.")
                break

    # 5. Final Determination & Hand-off
    print("\n--- Finalizing Triage ---")
    
    # Force flag True if loop ended via the 3-urgent short circuit logic above
    streak_met = (len(history) >= 3 and all(qa.get("status") == "URGENT" for qa in history[-3:]))
    
    # Final check via AI if short-circuit wasn't triggered
    flagged_urgent = streak_met or check_urgent(history)

    # Ensure hardware reflects final decision
    if flagged_urgent:
        status = "URGENT"
    set_status_led(status)

    # Voice readout matching the final status
    speak(f"Thank you {patient_name}, your status has been recorded as {status}.", alsa_device, pa_input_index)

    if status == "URGENT":
        speak("I am alerting a nurse immediately.", alsa_device, pa_input_index)
    else:
        speak("Feel better soon.", alsa_device, pa_input_index)

    # Hardware readouts and logging
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
        # Guarantee hardware LEDs power down on unexpected crash/exit
        if GPIO_AVAILABLE:
            try:
                RED_LINE.set_value(0); GREEN_LINE.set_value(0); BLUE_LINE.set_value(0)
                RED_LINE.release();    GREEN_LINE.release();    BLUE_LINE.release()
                print("[gpio] Hardware safely released.")
            except Exception:
                pass