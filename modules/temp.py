import torch
import torchaudio
import cv2  # OpenCV for video processing
import numpy as np
from fastapi import HTTPException
from whisper import load_model
import os

# Load Whisper model once at the module level
model = load_model("base")  # Adjust the model size as needed

def extract_audio_from_video(video_file_path):
    """
    Extracts audio from the given video file and returns it as a tensor.

    Parameters:
    - video_file_path: Path to the video file.

    Returns:
    - audio_tensor: Tensor containing audio data.
    - sample_rate: Sample rate of the audio.
    """
    print(f"Trying to open video file at: {video_file_path}")  # Debugging line
    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Could not open video file: {video_file_path}")

    # Extract audio using FFmpeg
    audio_file_path = "extracted_audio.wav"
    command = f"ffmpeg -i \"{video_file_path}\" -ab 160k -ac 1 -ar 16000 -vn \"{audio_file_path}\""  # Ensure mono and correct sample rate
    os.system(command)

    # Check if audio extraction was successful
    if not os.path.exists(audio_file_path):
        raise HTTPException(status_code=400, detail="Audio extraction failed from video.")

    # Load audio with torchaudio
    audio_tensor, sample_rate = torchaudio.load(audio_file_path)
    return audio_tensor, sample_rate

def transcribe_audio(audio_file_path):
    """
    Transcribes audio file using Whisper.

    Parameters:
    - audio_file_path: Path to the audio file.

    Returns:
    - str: Transcription text.
    """
    try:
        audio = model.transcribe(audio_file_path)
        return audio['text']
    except Exception as e:
        print("Error during audio transcription:", e)
        raise HTTPException(status_code=500, detail="Error processing audio file.")

def transcribe_video(video_file_path):
    """
    Transcribes text from the given video file.

    Parameters:
    - video_file_path: Path to the video file.

    Returns:
    - str: Transcription text.
    """
    try:
        audio_tensor, sample_rate = extract_audio_from_video(video_file_path)
        
        # Save audio tensor to a temporary file for Whisper processing
        audio_file_path = "temp_audio.wav"
        torchaudio.save(audio_file_path, audio_tensor, sample_rate=sample_rate)

        transcription = transcribe_audio(audio_file_path)
        return transcription
    except Exception as e:
        print("Error during video transcription:", e)
        raise HTTPException(status_code=500, detail="Error processing video file.")
