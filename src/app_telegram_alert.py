import os
import cv2
import time
import requests
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "models/tomato_disease_yolov8_best.pt"
VIDEO_SOURCE = 0   # webcam
# VIDEO_SOURCE = "http://YOUR_IP_CAMERA/video"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SEND_INTERVAL = 10  # seconds
CONFIDENCE_THRESHOLD = 0.5

# =========================
# INITIALIZE MODEL
# =========================
model = YOLO(MODEL_PATH)
class_names = model.names

# =========================
# SUPERVISION ANNOTATORS
# =========================
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# =========================
# TREATMENTS
# =========================
plant_diseases_treatments = {
    "Early Blight": "Apply copper-based fungicide and remove infected leaves.",
    "Late Blight": "Use chlorothalonil or copper fungicide immediately.",
    "Leaf Mold": "Reduce humidity and use copper fungicide.",
    "Septoria": "Remove infected leaves and apply fungicide.",
    "Leaf Miner": "Remove affected leaves and apply suitable treatment.",
    "Yellow Leaf Curl Virus": "Remove infected plants and control whiteflies.",
    "Healthy": "No treatment needed."
}

# =========================
# TELEGRAM FUNCTIONS
# =========================
def send_image(bot_token, chat_id, image_path):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

    with open(image_path, "rb") as photo:
        response = requests.post(
            url,
            files={"photo": photo},
            data={"chat_id": chat_id}
        )

    return response.json()


def send_treatment_messages(bot_token, chat_id, detected_diseases):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    for disease, treatment in detected_diseases.items():
        msg = f"Disease: {disease}\nTreatment: {treatment}"

        requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": msg
            }
        )

# =========================
# VIDEO CAPTURE
# =========================
cap = cv2.VideoCapture(VIDEO_SOURCE)

last_send_time = time.time()

plt.figure(figsize=(10, 6))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (600, 400))

    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD)

    detections = sv.Detections.from_ultralytics(results[0])

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    labels = []
    detected_diseases = {}

    for class_id in detections.class_id:
        label = class_names[int(class_id)]
        labels.append(label)

        treatment = plant_diseases_treatments.get(
            label,
            "No treatment information available."
        )

        detected_diseases[label] = treatment

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.axis("off")
    plt.draw()
    plt.pause(0.01)
    plt.clf()

    current_time = time.time()

    if current_time - last_send_time >= SEND_INTERVAL:
        output_path = "annotated_image.jpg"

        cv2.imwrite(output_path, annotated_frame)

        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            send_image(
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                output_path
            )

            send_treatment_messages(
                TELEGRAM_TOKEN,
                TELEGRAM_CHAT_ID,
                detected_diseases
            )

        last_send_time = current_time

cap.release()
plt.close()