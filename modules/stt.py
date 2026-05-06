import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import os

VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(
        f"[stt] Vosk model not found at '{VOSK_MODEL_PATH}'. "
        f"Download from https://alphacephei.com/vosk/models and unzip into project root."
    )

model = Model(VOSK_MODEL_PATH)
print(f"[stt] Vosk model loaded from {VOSK_MODEL_PATH}")


def listen(duration=5, samplerate=16000):
    """
    Records for `duration` seconds and returns
    transcribed text as a lowercase string.
    Returns empty string if nothing heard.
    """
    print("[stt] Listening...")
    recognizer = KaldiRecognizer(model, samplerate)

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='int16'
    )
    sd.wait()

    recognizer.AcceptWaveform(audio.tobytes())
    result = json.loads(recognizer.FinalResult())
    text   = result.get("text", "")

    print(f"[stt] Heard: '{text}'")
    return text.lower()