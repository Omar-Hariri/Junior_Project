import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
import winsound
#-------------------------------
model = tf.keras.models.load_model("../models/best_model.keras")
print("Model loaded successfully!")
#-------------------------------
def get_square_bbox(indices, landmarks, img_w, img_h, padding=40):
    """
    Calculates a square bounding box around the eye landmarks.
    This prevents aspect ratio distortion (squashing) when resizing.
    """
    x_min = img_w
    y_min = img_h
    x_max = y_max = 0

    # Find the extreme points of the eye
    for idx in indices:
        x = int(landmarks[idx].x * img_w)
        y = int(landmarks[idx].y * img_h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    # Calculate current width and height
    eye_w = x_max - x_min
    eye_h = y_max - y_min

    # Make the box square by using the larger dimension
    max_side = max(eye_w, eye_h)
    
    # Find the center of the eye
    center_x = x_min + eye_w // 2
    center_y = y_min + eye_h // 2

    # Add padding (to include eyebrows/skin like MRL dataset)
    side_len = max_side + padding

    # Calculate new square coordinates
    new_x_min = center_x - side_len // 2
    new_y_min = center_y - side_len // 2
    new_x_max = new_x_min + side_len
    new_y_max = new_y_min + side_len

    # Clamp coordinates to be within image boundaries
    return (
        max(0, new_x_min),
        max(0, new_y_min),
        min(img_w, new_x_max),
        min(img_h, new_y_max)
    )
#-------------------------------
def predict_eye_state(eye_img, model, return_processed=False):
    try:
        eye_img = cv2.resize(eye_img, (64, 64))
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye_img = eye_img.astype("float32") / 255.0

        processed_eye = eye_img.copy()  # what the model sees

        eye_img = np.expand_dims(eye_img, axis=-1)
        eye_img = np.expand_dims(eye_img, axis=0)

        pred = model.predict(eye_img, verbose=0)

        score = float(pred[0][0])
        predicted_class = 1 if score >= 0.5 else 0
        confidence = score if predicted_class == 1 else 1 - score

        if return_processed:
            return predicted_class, confidence, processed_eye

        return predicted_class, confidence

    except Exception as e:
        print("Prediction Error:", e)
        if return_processed:
            return None, 0.0, None
        return None, 0.0


#-------------------------------
# real time implementation
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe Eye Landmark Indices
RIGHT_EYE = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173]
LEFT_EYE  = [362, 263, 387, 386, 385, 373, 380, 381, 382, 398]

# Model Classes
CLASSES = {0: "awake", 1: "sleepy"}

# ----------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    left_state = None
    right_state = None
    left_eye_vis = None
    right_eye_vis = None

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # --- Process Left and Right Eye ---
            for eye_name, indices in [("Left", LEFT_EYE), ("Right", RIGHT_EYE)]:
                
                # 1. Get Square Bounding Box
                bbox = get_square_bbox(indices, face_landmarks.landmark, w, h, padding=30)
                x1, y1, x2, y2 = bbox
                
                # 2. Crop the eye
                eye_crop = frame[y1:y2, x1:x2]
                
                if eye_crop.size > 0:
                    # 3. Predict
                    pred_class, conf, eye_vis = predict_eye_state(eye_crop, model, return_processed=True)
                    if eye_vis is not None:
                        eye_vis = (eye_vis * 255).astype("uint8")  # back to display range
                        if eye_name == "Left":
                            left_eye_vis = eye_vis
                        else:
                            right_eye_vis = eye_vis
                        if left_eye_vis is not None and right_eye_vis is not None:
                            combined_eyes = np.hstack([ right_eye_vis,left_eye_vis])
                            combined_eyes = cv2.resize(combined_eyes, (256, 128))
                            cv2.imshow("Model Input", combined_eyes)


                    if pred_class is not None:
                        state_label = CLASSES.get(pred_class, "Unknown")
                        
                        # Set color: Green for Open, Red for Closed
                        color = (0, 255, 0) if state_label == "awake" else (0, 0, 255)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label on top of each eye
                        cv2.putText(frame, f"{state_label} {int(conf*100)}%", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, color, 1)
                        if eye_name == "Left":
                            left_state = pred_class
                        else:
                            right_state = pred_class
                        
    overall_status = "UNKNOWN"
    overall_color = (255, 255, 255)

    if left_state is not None and right_state is not None:
        if left_state == 0 and right_state == 0:
            overall_status = "AWAKE"
            overall_color = (0, 255, 0)
        elif left_state == 1 and right_state == 1:
            overall_status = "DROWSY"
            overall_color = (0, 0, 255)
            winsound.Beep(1000, 500) # alert
        else:
            overall_status = "UNCERTAIN"
            overall_color = (0, 255, 255)



    cv2.putText(
        frame,
        f"Status: {overall_status}",
        (10, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        overall_color,
        2,
        cv2.LINE_AA
    )

    # Show Main Frame
    cv2.imshow('Eye State Detection', frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#-------------------------------
