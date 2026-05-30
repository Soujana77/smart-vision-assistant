import cv2
import pytesseract
import pyttsx3

# Path to Tesseract installation
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Load image
image_path = "data/images/text_sample.png"

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract text using OCR
text = pytesseract.image_to_string(gray)

print("\nDetected Text:\n")
print(text)

# Speak detected text
if text.strip():

    engine.say(text)

    engine.runAndWait()

else:

    print("No text detected.")