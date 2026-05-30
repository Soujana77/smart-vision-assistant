import cv2
import pytesseract
import pyttsx3
import time

# Path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Open webcam
cap = cv2.VideoCapture(0)

# OCR cooldown
last_text = ""
last_spoken_time = 0
cooldown = 5

print("OCR Assistant Started...")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract text
    text = pytesseract.image_to_string(gray)

    # Clean text
    text = text.strip()

    # Display text on screen
    if text:

        cv2.putText(
            frame,
            text[:50],
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        current_time = time.time()

        # Speak only if text changes
        if (
            text != last_text
            and current_time - last_spoken_time > cooldown
        ):

            print("\nDetected Text:")
            print(text)

            engine.say(text)
            engine.runAndWait()

            last_text = text
            last_spoken_time = current_time

    cv2.imshow("OCR Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()