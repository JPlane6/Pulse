from TTS.api import TTS
import os

# initialize once
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

def speak(text):
    """
    Convert text to speech and play it immediately.
    """
    tts.tts_to_file(text=text, file_path="data/output.wav")  # save to file
    os.system("afplay data/output.wav")  # Mac playback