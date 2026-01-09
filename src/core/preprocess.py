import cv2
import numpy as np

def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, (64, 64))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = eye / 255.0
    eye = np.expand_dims(eye, axis=(0, -1))
    return eye
