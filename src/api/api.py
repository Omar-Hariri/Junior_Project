from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.pipeline import process_frame
from core.detector import get_eye_data
from core.preprocess import preprocess_eye

app = FastAPI(title="Driver Drowsiness Detection API")

#---------------------------------------------------------------
# Health Check (Output is status)
#---------------------------------------------------------------
@app.get("/")
def health():
   return {"status": "ok"}

def decode_image(file_bytes: bytes):
   nparr = np.frombuffer(file_bytes, np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   if img is None:
      raise HTTPException(status_code=400, detail="Invalid image format")
   return img
#---------------------------------------------------------------
# Full Pipeline (Output is detected eyes (coordinates) and their states)
#---------------------------------------------------------------
@app.post("/process")
async def full_pipeline(file: UploadFile = File(...)):
   contents = await file.read()
   frame = decode_image(contents)

   results = process_frame(frame)

   if not results:
      return {"detected": False, "message": "No face or eyes detected"}

   return {"detected": True, "results": results}
#---------------------------------------------------------------
# Visualize Input (Output is processed eye image (What the model sees))
#---------------------------------------------------------------

@app.post("/visualize-input/{side}")
async def visualize_input(side: str, file: UploadFile = File(...)):
   if side not in ["left", "right"]:
      raise HTTPException(status_code=400, detail="Side must be 'left' or 'right'")

   contents = await file.read()
   frame = decode_image(contents)

   eye_data = get_eye_data(frame)
   if not eye_data or side not in eye_data:
      raise HTTPException(status_code=404, detail="Eye not detected")

   processed_eye = preprocess_eye(eye_data[side]["img"])
   vis_img = (processed_eye[0, :, :, 0] * 255).astype(np.uint8)

   _, encoded_img = cv2.imencode(".png", vis_img)
   return Response(content=encoded_img.tobytes(), media_type="image/png")
#---------------------------------------------------------------
# Detect Only (Output is only detected eyes and their coordinates)
#---------------------------------------------------------------
@app.post("/detect-only")
async def detect_only(file: UploadFile = File(...)):
   contents = await file.read()
   frame = decode_image(contents)

   eye_data = get_eye_data(frame)
   if not eye_data:
      return {"detected": False}

   return {
      "detected": True,
      "coordinates": {k: v["bbox"] for k, v in eye_data.items()}
   }
#---------------------------------------------------------------
