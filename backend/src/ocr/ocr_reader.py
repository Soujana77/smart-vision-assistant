import cv2
import pytesseract

# Path to Tesseract installation
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def read_text(frame):
    """
    Extract text from a video frame using OCR.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    return text.strip()