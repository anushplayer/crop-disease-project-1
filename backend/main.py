from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import os

app = FastAPI(title="Crop Disease & Recommendation API")

# Load models (assuming they exist)
disease_model = None
recommendation_model = None

class SoilData(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Process image
    image = Image.open(file.file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Mock prediction (replace with actual model)
    prediction = {"disease": "Healthy", "confidence": 0.95}
    
    return JSONResponse(content=prediction)

@app.post("/recommend-crop")
async def recommend_crop(data: SoilData):
    # Mock recommendation (replace with actual model)
    crops = ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize"]
    recommendation = crops[np.random.randint(0, len(crops))]
    
    return JSONResponse(content={"recommended_crop": recommendation})