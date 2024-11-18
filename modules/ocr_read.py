import pytesseract
import cv2  # OpenCV for image processing
from fastapi import HTTPException
from PIL import Image

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if different

def extract_text(image_file_path):
    """
    Extracts text from an image using OCR.

    Parameters:
    - image_file_path: Path to the image file.

    Returns:
    - str: Extracted text.
    """
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_file_path)
        if image is None:
            raise HTTPException(status_code=400, detail=f"Could not open image file: {image_file_path}")

        # Convert the image to RGB (from BGR format used by OpenCV)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use pytesseract to do OCR on the image
        extracted_text = pytesseract.image_to_string(Image.fromarray(image))
        return extracted_text.strip()  # Remove any extra whitespace

    except Exception as e:
        print("Error during OCR extraction:", e)
        raise HTTPException(status_code=500, detail="Error processing image file.")
