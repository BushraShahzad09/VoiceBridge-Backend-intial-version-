from fastapi import FastAPI, HTTPException, UploadFile, File
from modules.audio_transcription import transcribe_audio
from modules.video_transcription import transcribe_video
from modules.ocr_read import extract_text
from modules.sign_language import detect_sign_language
import os
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="VoiceBridge Accessibility Assistant",
    version="1.0",
    description="API Server for handling VoiceBridge Accessibility functionalities."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/transcribe_audio")
async def transcribe_audio_route(file: UploadFile = File(...)):
    try:
        transcription = await transcribe_audio(file)
        print("Transcription result:", transcription)  # Add this to verify output
        return {"output": transcription}
    except HTTPException as e:
        raise e
    except Exception as e:
        print("Error in transcribe_audio_route:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error: Unable to transcribe audio.")


@app.post("/transcribe_video")
async def handle_video_transcription(file: UploadFile = File(...)):
    
    video_file_path = f"temp_video_{file.filename}"
    with open(video_file_path, "wb") as buffer:
        buffer.write(await file.read())

    transcription = transcribe_video(video_file_path)
    
    
    os.remove(video_file_path)  
    os.remove("temp_audio.wav")  
    return {"output": transcription}

@app.post("/extract_text")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for extracting text from an image.
    
    Parameters:
    - file: UploadFile object containing the image file.

    Returns:
    - dict: Extracted text.
    """
    try:
        # Save the uploaded image file
        image_file_path = f"temp_image_{file.filename}"
        with open(image_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text from the image
        text = extract_text(image_file_path)

        # Optionally, delete the temporary image file
        os.remove(image_file_path)

        return {"extracted_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing image.")


@app.post("/detect_sign_language")
async def detect_sign_language_route(file: UploadFile = File(...)):
    try:
        translation = await detect_sign_language(file)
        return {"output": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sign language video: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
