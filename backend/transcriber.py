import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

# Initialize the recognizer
recorder = sr.Recognizer()

# Load the default model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
default_model = whisper.load_model("medium", device=DEVICE)

def transcribe_audio(file_path, model_name="medium", non_english=False):
    try:
        # Load the selected model
        if model_name != "large" and not non_english:
            model_name = model_name + ".en"
        audio_model = whisper.load_model(model_name, device=DEVICE)

        # Recognize the audio
        with sr.AudioFile(file_path) as source:
            audio = recorder.record(source)
            audio_data = audio.get_raw_data()
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result['text'].strip()

        return text
    except Exception as e:
        raise RuntimeError(f"Error transcribing audio: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
