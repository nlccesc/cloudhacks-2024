from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

app = FastAPI()

# Define a request body model
class TranscriptionRequest(BaseModel):
    model: str = "medium"
    non_english: bool = False
    energy_threshold: int = 1000
    record_timeout: float = 6.0
    phrase_timeout: float = 6.0

# Initialize the recognizer
recorder = sr.Recognizer()

# Load the default model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
default_model = whisper.load_model("medium", device=DEVICE)

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    request: TranscriptionRequest = TranscriptionRequest()
):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Load the selected model
        model_name = request.model
        if model_name != "large" and not request.non_english:
            model_name = model_name + ".en"
        audio_model = whisper.load_model(model_name, device=DEVICE)

        # Recognize the audio
        with sr.AudioFile(temp_file_path) as source:
            audio = recorder.record(source)
            audio_data = audio.get_raw_data()
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result['text'].strip()

        # Clean up the temporary file
        os.remove(temp_file_path)

        return JSONResponse(content={"transcription": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
