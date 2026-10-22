import pytest
import os
import sys
from fastapi.testclient import TestClient
import numpy as np
import io
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert "supported_classes" in res.json()

def test_profiles():
    res = client.get("/api/v1/profiles")
    assert res.status_code == 200
    assert "Desert" in res.json()
    assert "Plains" in res.json()

def test_simulate():
    res = client.post("/api/v1/telemetry/simulate", json={"terrain_class": "Desert"})
    assert res.status_code == 200
    assert res.json()["nav_action"] == "THROTTLE_AND_MOMENTUM"

def test_classify_upload():
    # Create small test image
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :, 1] = 180
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post(
        "/api/v1/classify/upload",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "predicted_class" in data
    assert "soil_mechanics" in data
