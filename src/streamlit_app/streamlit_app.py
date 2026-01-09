import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Project path
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.pipeline import process_frame


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="DDD",
    layout="centered",
)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

def go(page):
    st.session_state.page = page
    st.rerun()

# ==================================================================
# HOME PAGE
# ==================================================================
if st.session_state.page == "home":

    st.markdown("<h1>Driver Drowsiness Detection</h1>",unsafe_allow_html=True,)
    st.write("")
    st.write("")

    # Two centered columns: Image button (left) and Video button (right)
    col_image, col_video = st.columns(2)
    
    with col_image:
        if st.button("Open Image Mode", key="image_mode", use_container_width=True):
            go("image")
    
    with col_video:
        if st.button("Open Video Mode", key="video_mode", use_container_width=True):
            go("video")

# ==================================================================
# IMAGE MODE
# ==================================================================
elif st.session_state.page == "image":

    st.markdown("<h1>Image prediction</h1>", unsafe_allow_html=True)

    source = st.radio(
        "Choose image source",
        ["Upload Image", "Take Photo"],
        horizontal=True
    )

    image_file = None

    if source == "Upload Image":
        image_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    else:
        image_file = st.camera_input("Capture image")

    if image_file:
        image = Image.open(image_file).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = process_frame(frame_bgr)

        if results is None:
            st.warning("No face detected.")
        else:
            # Draw annotations on a copy so we have original & processed
            annotated = frame_bgr.copy()

            for eye, data in results.items():
                x1, y1, x2, y2 = data["bbox"]
                conf = data.get("confidence", None)

                is_closed = data["state"] == 1
                color = (0, 0, 255) if is_closed else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                state_label = "CLOSED" if is_closed else "OPEN"
                
                # Show prediction ON the image with confidence
                if conf is not None:
                    label = f"{state_label} ({conf:.2f})"
                else:
                    label = f"{state_label} (N/A)"
                
                # Put text on the image
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1
                )

            # Show only the processed image with predictions ON it
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

    if st.button("⬅ Back"):
        go("home")

# ==================================================================
# VIDEO MODE
# ==================================================================
elif st.session_state.page == "video":

    st.markdown("<h1>Real-time Prediction</h1>", unsafe_allow_html=True)

    # Center the button (CSS handles centering)
    if not st.session_state.run_camera:
        if st.button("▶ Start Monitoring", type="primary"):
            st.session_state.run_camera = True
            st.rerun()
    else:
        if st.button("⏹ Stop Monitoring"):
            st.session_state.run_camera = False
            st.rerun()

    frame_area = st.empty()
    
    # Initialize voice alert state
    if "last_drowsy_state" not in st.session_state:
        st.session_state.last_drowsy_state = False
    if "alert_triggered" not in st.session_state:
        st.session_state.alert_triggered = False

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Unable to access camera. Please check that a webcam is connected and not used by another app.")
            st.session_state.run_camera = False
        else:

            frame_count = 0
            drowsy_frames = 0

            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                results = process_frame(frame)

                if results:
                    drowsy = (
                        results["left"]["state"] == 1 and
                        results["right"]["state"] == 1
                    )

                    frame_count += 1
                    if drowsy:
                        drowsy_frames += 1

                    # Show AWAKE/DROWSY label ON the video
                    status_text = "DROWSY" if drowsy else "AWAKE"
                    status_color = (0, 0, 255) if drowsy else (0, 255, 0)
                    
                    # Put status label on top center of frame
                    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    cv2.putText(
                        frame,
                        status_text,
                        (text_x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        status_color,
                        3
                    )
                    for eye, data in results.items():
                        x1, y1, x2, y2 = data["bbox"]
                        conf = data.get("conf", None)

                        color = (0, 0, 255) if data["state"] == 1 else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        if conf is not None:
                            label = f"{eye.upper()} | {conf:.2f}"
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )

                frame_area.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

            cap.release()


    if st.button("⬅ Back"):
        st.session_state.run_camera = False
        go("home")