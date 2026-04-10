import cv2
from ultralytics import YOLO
import supervision as sv

MODEL_PATH = "models/tomato_disease_yolov8_best.pt"
VIDEO_SOURCE = 0

model = YOLO(MODEL_PATH)
class_names = model.names

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, conf=0.5)

    detections = sv.Detections.from_ultralytics(results[0])

    labels = [
        class_names[int(class_id)]
        for class_id in detections.class_id
    ]

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    cv2.imshow("Tomato Disease Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()