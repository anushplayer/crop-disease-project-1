from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import os

# Import kagglehub for dataset download
import kagglehub

app = FastAPI(title="Crop Disease & Recommendation API")

# Download latest version of the crop and soil dataset from Kaggle
try:
    dataset_path = kagglehub.dataset_download("shankarpriya2913/crop-and-soil-dataset")
    print("Path to dataset files:", dataset_path)
except Exception as e:
    print("Failed to download dataset from KaggleHub:", e)

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
    soil_type: str = None  # Optional: e.g., "Sandy", "Clay", "Silty", etc.

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
    # Example: Use soil_type in recommendation logic (mock)
    crops_by_soil = {
        "Sandy": ["Peanut", "Watermelon", "Cotton"],
        "Clay": ["Rice", "Wheat", "Soybean"],
        "Silty": ["Maize", "Sugarcane", "Potato"],
        "Loamy": ["Tomato", "Carrot", "Onion"],
        None: ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize"]
    }
    possible_crops = crops_by_soil.get(data.soil_type, crops_by_soil[None])
    recommendation = possible_crops[np.random.randint(0, len(possible_crops))]
    return JSONResponse(content={"recommended_crop": recommendation, "soil_type": data.soil_type})