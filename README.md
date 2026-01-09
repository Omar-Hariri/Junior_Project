# Junior Project – A SYSTEM FOR DETECTING FATIGUE THROUGH FACIAL IMAGES OF DRIVERS

## Overview

This project implements an **AI-powered driver monitoring system** focused on detecting **driver drowsiness** using computer vision and deep learning.
It follows a complete **end-to-end machine learning pipeline**, from data preprocessing and model training to **real-time inference via a REST API and a Streamlit web interface**.



---

## Features

- Face detection and preprocessing pipeline
- CNN-based deep learning model (Keras / TensorFlow)
- Multiple training experiments using Jupyter notebooks
- REST API for real-time inference (FastAPI)
- Streamlit web application for interactive testing
- Modular and extensible project architecture

---

## Project Structure

```text
Junior_Project/
│
├── models/
│   └── v3/
│       └── best_model.keras        # Trained CNN model
│
├── notebooks/
│   ├── Train_v1.ipynb              # Initial experiment
│   ├── Train_v2.ipynb              # Improved model
│   └── Train_v3.ipynb              # Final selected model
│
├── src/
│   ├── api/
│   │   └── api.py                  # FastAPI inference service
│   │
│   ├── core/
│   │   ├── detector.py             # Face detection logic
│   │   ├── preprocess.py           # Image preprocessing
│   │   ├── model.py                # Model loading and prediction
│   │   └── pipeline.py             # End-to-end inference pipeline
│   │
│   ├── pipeline_test.py            # Pipeline testing script
│   │
│   └── streamlit_app/
│       ├── __init__.py
│       └── streamlit_app.py        # Streamlit UI
│
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows launcher script
└── README.md
```

---

## Technology Stack

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- FastAPI
- Uvicorn
- Streamlit
- NumPy, Pandas
- Jupyter Notebook

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Omar-Hariri/Junior_Project.git
cd Junior_Project
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Model Training (Optional)

Model training is performed using the notebooks located in:

```text
notebooks/
```

The final trained model is saved at:

```text
models/v3/best_model.keras
```

If you retrain the model, update the model path in the inference code accordingly.

---

## Running the API (FastAPI)

Start the inference API using:

```bash
uvicorn src.api.api:app --reload
```

- API URL:
  `http://127.0.0.1:8000`
- Swagger Documentation:
  `http://127.0.0.1:8000/docs`

---

## Running the Streamlit App

```bash
streamlit run src/streamlit_app/streamlit_app.py
```

This launches a web-based interface for real-time testing and visualization.

---

## Inference Pipeline

```text
Input Image / Video Frame
        ↓
Face Detection
        ↓
Image Preprocessing
        ↓
CNN Model Inference
        ↓
Prediction (Normal / Yawning)
```

---

## Configuration Notes

- Ensure the model path in `core/model.py` points to:
  ```text
  models/v3/best_model.keras
  ```
- Webcam access is required for real-time inference.
- The project is primarily tested on **Windows** (run_app.bat included).

---


