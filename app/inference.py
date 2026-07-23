from ultralytics import YOLO
from PIL import Image
import io

MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

def predict_image(image_bytes: bytes, conf_threshold: float = 0.5):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "denomination": class_name,
            "confidence": round(confidence, 4),
            "bbox": {
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2)
            }
        })

    return detections