import os
import sys
import io
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terrain_classifier import TerrainClassifier, TERRAIN_PROFILES

app = FastAPI(
    title="Autonomous Robot Terrain Telemetry API",
    description="ROS 2 and Edge Telemetry service for vision-based off-road terrain classification and soil mechanics estimation.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = TerrainClassifier()

class Base64ImageRequest(BaseModel):
    image_base64: str

class TelemetrySimulationRequest(BaseModel):
    terrain_class: str

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "vision-based-terrain-classification",
        "engine": "texture-spectral-soil-mechanics-v2",
        "supported_classes": list(TERRAIN_PROFILES.keys())
    }

@app.get("/api/v1/profiles")
def get_terrain_profiles():
    return TERRAIN_PROFILES

@app.post("/api/v1/classify/base64")
def classify_base64(req: Base64ImageRequest):
    try:
        # Strip header if present
        data_str = req.image_base64
        if "base64," in data_str:
            data_str = data_str.split("base64,")[1]
        img_bytes = base64.b64decode(data_str)
        img = Image.open(io.BytesIO(img_bytes))
        return classifier.classify_image(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(e)}")

@app.post("/api/v1/classify/upload")
async def classify_upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        return classifier.classify_image(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@app.post("/api/v1/telemetry/simulate")
def simulate_telemetry(req: TelemetrySimulationRequest):
    target = req.terrain_class.capitalize()
    if target in TERRAIN_PROFILES:
        return TERRAIN_PROFILES[target]
    raise HTTPException(status_code=404, detail=f"Terrain class '{req.terrain_class}' not found.")
