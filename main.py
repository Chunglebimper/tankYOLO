import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def get_bounding_box_coords(frame):
    # Extract detection info
    detections = results.pandas().xyxy[0]  # Bounding boxes in xmin, ymin, xmax, ymax format

    # Filter by class name (if needed)
    target_class = 'person'
    filtered = detections[detections['name'] == target_class]

    # Loop over detections
    for index, row in filtered.iterrows():
        x1, y1 = row['xmin'], row['ymin']
        x2, y2 = row['xmax'], row['ymax']
        conf = row['confidence']
        print(f"{target_class}: ({x1}, {y1}), ({x2}, {y2}) | Confidence: {conf:.2f}")


model = YOLO("yolo11s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make prediction (list)
    results = model(frame)

    for result in results:

        annotated_frame = result.plot()
        get_boundng_box_coords(annotated_frame)
        cv2.imshow("Real-time Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()