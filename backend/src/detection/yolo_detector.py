from ultralytics import YOLO

# Load YOLO model
print("Loading YOLOv8 model...")

# Use SMALL model instead of NANO for better accuracy
model = YOLO("yolov8s.pt")

print("YOLOv8 loaded successfully!")

def detect_objects(frame):

    detections_list = []

    height, width, _ = frame.shape

    # Run YOLO inference
    results = model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            # Lower threshold to detect more objects
            if confidence < 0.5:
                continue

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Direction detection
            object_center = (x1 + x2) // 2

            if object_center < width // 3:
                direction = "on the left"

            elif object_center < (2 * width) // 3:
                direction = "ahead"

            else:
                direction = "on the right"

            # Distance estimation
            box_area = (x2 - x1) * (y2 - y1)

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
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "score": round(confidence, 2)

            })

    return detections_list