import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load model
model = hub.load(
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
)

print("Model loaded successfully!")

# COCO labels
labels = {
    1: "person",
    44: "bottle",
    47: "cup",
    62: "chair",
    64: "potted plant",
    67: "cell phone",
    77: "teddy bear",
    84: "book",
    86: "vase"
}

# Open webcam
cap = cv2.VideoCapture(0)

# Confidence threshold
threshold = 0.7

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    input_tensor = tf.convert_to_tensor(frame_rgb)

    # Add batch dimension
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = model(input_tensor)

    # Extract outputs
    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)

    height, width, _ = frame.shape

    # Loop through detections
    for i in range(len(scores)):

        if scores[i] > threshold:

            ymin, xmin, ymax, xmax = boxes[i]

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
                2
            )

            # Get class name
            class_id = classes[i]
            class_name = labels.get(class_id, "Unknown")

            label = f"{class_name}: {scores[i]:.2f}"

            # Draw label
            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    # Show output frame
    cv2.imshow("Smart Vision Assistant", frame)

    # Exit on pressing q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()