import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa

# Load the model and processor from local files
model_path = "../Whisper"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)

def transcribe_video(audio_path: str) -> str:
    # Load audio
    audio, rate = librosa.load(audio_path, sr=16000)  # load audio with sampling rate of 16kHz

    # Process input audio for the model
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    # Perform transcription
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode logits to get the transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription
