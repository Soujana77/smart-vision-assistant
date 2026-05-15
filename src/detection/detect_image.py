import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load pretrained model
model = hub.load(
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
)

print("Model loaded successfully!")

# Load image
image_path = "data/images/test.png"

image = cv2.imread(image_path)

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

        # Draw rectangle
        cv2.rectangle(
            image_rgb,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Label text
    class_id = classes[i]

class_name = labels.get(class_id, "Unknown")

label = f"{class_name}: {scores[i]:.2f}"

        # Put label above box
cv2.putText(
            image_rgb,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

# Display image
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()