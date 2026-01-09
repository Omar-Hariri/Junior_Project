from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "V3" /"best_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_eye_state(img):
    pred = model.predict(img, verbose=0)[0][0]
    # Return 0 = awake/open, 1 = closed/drowsy
    pred_class = 1 if pred > 0.5 else 0
    confidence = pred if pred_class == 1 else 1 - pred
    return pred_class, confidence

