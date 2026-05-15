import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load pretrained SSD MobileNet model
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

# Print output keys
print(detections.keys())