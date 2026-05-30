import tensorflow as tf
import tensorflow_hub as hub

# Load model once when module starts
print("Loading TensorFlow model...")

model = hub.load(
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
)

print("Model loaded successfully!")

# Labels
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

def detect_objects(frame):

    detections_list = []

    height, width, _ = frame.shape

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = model(input_tensor)

    boxes = detections["detection_boxes"][0].numpy()
    scores = detections["detection_scores"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)

    threshold = 0.7

    for i in range(len(scores)):

        if scores[i] > threshold:

            ymin, xmin, ymax, xmax = boxes[i]

            left = int(xmin * width)
            right = int(xmax * width)
            top = int(ymin * height)
            bottom = int(ymax * height)

            class_name = labels.get(
                classes[i],
                "Unknown"
            )

            # Direction
            object_center = (left + right) // 2

            if object_center < width // 3:
                direction = "on the left"

            elif object_center < (2 * width) // 3:
                direction = "ahead"

            else:
                direction = "on the right"

            # Distance estimation
            box_area = (
                (right - left)
                * (bottom - top)
            )

            if box_area > 150000:
                distance = "very close"

            elif box_area > 70000:
                distance = "close"

            else:
                distance = "far"

            detections_list.append({

                "class_name": class_name,

                "direction": direction,

                "distance": distance,

                "box": (
                    left,
                    top,
                    right,
                    bottom
                ),

                "score": float(scores[i])

            })

    return detections_list