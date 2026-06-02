import cv2
import time

from detection.yolo_detector import detect_objects
from ocr.ocr_reader import read_text
from navigation.navigator import analyze_path
from voice.speaker import speak

# Open webcam
cap = cv2.VideoCapture(0)

# Voice cooldown
last_spoken_text = ""
last_spoken_time = 0

cooldown = 5

print("Smart Vision Assistant Started")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ===========================
    # Object Detection
    # ===========================

    detections = detect_objects(frame)

    for detection in detections:

        left, top, right, bottom = detection["box"]

        label = (
            f"{detection['class_name']} "
            f"{detection['score']:.2f} "
            f"{detection['direction']} "
            f"({detection['distance']})"
        )

        # Draw bounding box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Draw label
        cv2.putText(
            frame,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # ===========================
    # Navigation Analysis
    # ===========================

    guidance = analyze_path(detections)

    current_time = time.time()

    cv2.putText(
        frame,
        guidance,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # ===========================
    # OCR
    # ===========================

    text = read_text(frame)

    if text:

        cv2.putText(
            frame,
            text[:50],
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # ===========================
    # Voice Logic
    # ===========================

    current_time = time.time()

    if (
        text
        and text != last_spoken_text
        and current_time - last_spoken_time > cooldown
    ):

        speak(text)

        last_spoken_text = text

        last_spoken_time = current_time

    # ===========================
    # Display
    # ===========================

    cv2.imshow(
        "Smart Vision Assistant",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()