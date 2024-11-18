import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
from fastapi import HTTPException  # Ensure HTTPException is imported
from io import BytesIO
import numpy as np

# Initialize model and tokenizer once at module level
model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

async def transcribe_audio(file):
    """
    Transcribes audio from an UploadFile object using Wav2Vec2.

    Parameters:
    - file: UploadFile object from FastAPI
    
    Returns:
    - str: Transcription text.
    """
    try:
        # Read audio bytes
        audio_bytes = await file.read()
        
        # Load audio from bytes using torchaudio
        audio_tensor, sample_rate = torchaudio.load(BytesIO(audio_bytes))

        # Resample if needed
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        # Convert to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        # Flatten and normalize the audio tensor
        audio_input = audio_tensor.squeeze().numpy().flatten()
        audio_input = audio_input / np.max(np.abs(audio_input))  # Normalize to [-1, 1]

        # Tokenize and prepare model input
        input_values = tokenizer(audio_input, return_tensors="pt", padding="longest").input_values

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)

        return transcription[0]

    except Exception as e:
        print("Error during audio transcription:", e)
        raise HTTPException(status_code=500, detail="Error processing audio file.")
