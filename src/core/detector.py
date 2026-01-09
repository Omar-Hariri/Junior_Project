import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_eye_data(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    EYES = {
        "left": [362, 263, 387, 385, 373, 380], 
        "right": [33, 133, 160, 158, 144, 153]
    }
    
    eye_results = {}
    for side, indices in EYES.items():
        coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        x_min = min([c[0] for c in coords]) - 10
        y_min = min([c[1] for c in coords]) - 10
        x_max = max([c[0] for c in coords]) + 10
        y_max = max([c[1] for c in coords]) + 10
        
        x1, y1 = max(0, x_min), max(0, y_min)
        x2, y2 = min(w, x_max), min(h, y_max)
        
        eye_img = frame[y1:y2, x1:x2]
        eye_results[side] = {"img": eye_img, "bbox": (x1, y1, x2, y2)}
        
    return eye_results