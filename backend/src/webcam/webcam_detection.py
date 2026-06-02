import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time

# Load pretrained model
model = hub.load(
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
)

print("Model loaded successfully!")

# COCO labels
labels = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    44: "bottle",
    47: "cup",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    84: "book",
    86: "vase"
}

# Open webcam
cap = cv2.VideoCapture(0)

# Confidence threshold
threshold = 0.7

# FPS tracking
prev_time = 0

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(frame_rgb)

    # Add batch dimension
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = model(input_tensor)

    # Extract outputs
    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)

    # Get frame dimensions
    height, width, _ = frame.shape

    # Loop through detections
    for i in range(len(scores)):

        if scores[i] > threshold:

            ymin, xmin, ymax, xmax = boxes[i]

            # Convert normalized coordinates to image coordinates
            left = int(xmin * width)
            right = int(xmax * width)
            top = int(ymin * height)
            bottom = int(ymax * height)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                3
            )

            # Get class label
            class_id = classes[i]
            class_name = labels.get(class_id, "Unknown")

            # Create label text
            label = f"{class_name}: {scores[i]:.2f}"

            # Draw label text
            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # Calculate FPS
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Show webcam output
    cv2.imshow("Smart Vision Assistant", frame)

    # Exit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()