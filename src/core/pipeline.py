from core.detector import get_eye_data
from core.preprocess import preprocess_eye
from core.model import predict_eye_state
import cv2

def process_frame(frame):
    eye_data = get_eye_data(frame)
    if not eye_data:
        return None
    
    results = {}
    for side in ["left", "right"]:
        data = eye_data[side]

        processed = preprocess_eye(data["img"])

        state, conf = predict_eye_state(processed)
        
        results[side] = {
            "state": state,
            "confidence": conf,
            "bbox": data["bbox"]
        }
    return results