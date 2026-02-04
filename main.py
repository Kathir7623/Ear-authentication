from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import os
import sys
import io
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from detector import EarDetector
from preprocessor import Preprocessor
from extractor import FeatureExtractor
from matcher import Matcher

app = FastAPI(title="Syndicate Biometric API")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Backend Engine
detector = EarDetector()
preprocessor = Preprocessor()
extractor = FeatureExtractor()
matcher = Matcher()

@app.get("/")
async def health_check():
    return {"status": "online", "system": "Syndicate Ear Biometrics"}

@app.post("/identify")
async def identify_ear(file: UploadFile = File(...)):
    # Load Image
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect
    ears = detector.detect(image)
    if not ears:
        # Fallback to full image if detection fails
        ear_crop = image
    else:
        ear_crop = detector.crop_ear(image, ears[0]['box'])

    # Process and Extract
    processed = preprocessor.process(ear_crop)
    norm_img = preprocessor.normalize_for_model(processed)
    embedding = extractor.extract(norm_img)

    # Match
    name, score = matcher.identify(embedding)
    
    return {
        "identity": name,
        "confidence": float(score),
        "ear_detected": len(ears) > 0
    }

@app.post("/register")
async def register_ear(name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ears = detector.detect(image)
    ear_crop = detector.crop_ear(image, ears[0]['box']) if ears else image
    
    processed = preprocessor.process(ear_crop)
    norm_img = preprocessor.normalize_for_model(processed)
    embedding = extractor.extract(norm_img)

    matcher.add_template(name, embedding)
    
    return {"status": "success", "message": f"User {name} registered successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
