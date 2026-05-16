import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import scipy.io.wavfile as wav

# Load model once at import time — 'base' is fast and accurate on M2
model = whisper.load_model("base")
print("[stt] Whisper model loaded")


def listen(duration=5, samplerate=16000):
    """
    Records for `duration` seconds and returns
    transcribed text as a lowercase string.
    Returns empty string if nothing heard.
    """
    print("[stt] Listening...")

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    # Save to temp wav file — Whisper needs a file not raw bytes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        wav.write(temp_path, samplerate, audio)

    try:
        result = model.transcribe(temp_path, fp16=False, language="en")
        text = result["text"].strip().lower()
        print(f"[stt] Heard: '{text}'")
        return text
    except Exception as e:
        print(f"[stt] Transcription failed: {e}")
        return ""
    finally:
        os.remove(temp_path)