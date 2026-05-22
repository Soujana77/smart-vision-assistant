import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3

# Load pretrained model
model = hub.load(
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
)

print("Model loaded successfully!")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set speech speed
engine.setProperty("rate", 150)

# Remember last spoken object
last_announced = ""

# Load image
image_path = "data/images/test.png"

image = cv2.imread(image_path)

# Check if image loaded correctly
if image is None:
    print("Error: Image not found!")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to tensor
input_tensor = tf.convert_to_tensor(image_rgb)

# Add batch dimension
input_tensor = input_tensor[tf.newaxis, ...]

# Run inference
detections = model(input_tensor)

# Extract outputs
boxes = detections["detection_boxes"][0].numpy()
scores = detections["detection_scores"][0].numpy()
classes = detections["detection_classes"][0].numpy().astype(int)

# Image dimensions
height, width, _ = image_rgb.shape

# Confidence threshold
threshold = 0.5

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

# Draw detections
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
            image_rgb,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Get class label
        class_id = classes[i]
        class_name = labels.get(class_id, "Unknown")

        # Create label text
        label = f"{class_name}: {scores[i]:.2f}"

        # Draw label
        cv2.putText(
            image_rgb,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

        # Speak detected object
        if class_name != "Unknown":

            if class_name != last_announced:

                speech = f"{class_name} detected"

                print(speech)

                engine.say(speech)

                engine.runAndWait()

                last_announced = class_name

# Display image
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()