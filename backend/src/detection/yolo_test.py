from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully!")

# Load test image
image_path = "data/images/test.png"

image = cv2.imread(image_path)

# Run detection
results = model(image)

# Draw detections
annotated_frame = results[0].plot()

# Show image
cv2.imshow("YOLO Detection", annotated_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()