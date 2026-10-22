import pytest
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.terrain_classifier import TerrainClassifier, TERRAIN_PROFILES

def test_profiles_structure():
    assert "Desert" in TERRAIN_PROFILES
    assert "Forest" in TERRAIN_PROFILES
    assert "Mountain" in TERRAIN_PROFILES
    assert "Plains" in TERRAIN_PROFILES
    for name, p in TERRAIN_PROFILES.items():
        assert 0.0 < p["friction_coefficient"] <= 1.0
        assert p["max_safe_speed_kmh"] > 0
        assert "nav_action" in p

def test_feature_extraction():
    classifier = TerrainClassifier()
    # Create simple synthetic green image (vegetation)
    green_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    green_arr[:, :, 1] = 200 # Green
    img = Image.fromarray(green_arr)
    feats = classifier.extract_features(img)
    assert feats["g_mean"] > feats["r_mean"]
    assert "texture_roughness" in feats

def test_classification():
    classifier = TerrainClassifier()
    # Create sandy / desert tone image (yellowish / high red and green, low blue)
    desert_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    desert_arr[:, :, 0] = 220 # R
    desert_arr[:, :, 1] = 180 # G
    desert_arr[:, :, 2] = 50  # B
    img = Image.fromarray(desert_arr)
    result = classifier.classify_image(img)
    assert "predicted_class" in result
    assert "soil_mechanics" in result
    assert "navigation_directive" in result
    assert result["soil_mechanics"]["friction_coefficient_mu"] > 0
