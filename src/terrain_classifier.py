import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

TERRAIN_PROFILES = {
    "Plains": {
        "description": "Open grassland and compact soil",
        "friction_coefficient": 0.72,
        "sinkage_risk_percent": 15,
        "max_safe_speed_kmh": 28.0,
        "nav2_costmap_cost": 20,
        "nav_action": "NORMAL_CRUISE",
        "advisory": "Stable ground conditions. Nominal cruise speeds permitted."
    },
    "Desert": {
        "description": "Granular sand dunes and loose arid soil",
        "friction_coefficient": 0.38,
        "sinkage_risk_percent": 75,
        "max_safe_speed_kmh": 12.0,
        "nav2_costmap_cost": 70,
        "nav_action": "THROTTLE_AND_MOMENTUM",
        "advisory": "High sinkage and wheel slip risk. Maintain steady momentum; avoid sharp turns."
    },
    "Mountain": {
        "description": "Rocky scree, uneven stones, and steep incline",
        "friction_coefficient": 0.52,
        "sinkage_risk_percent": 35,
        "max_safe_speed_kmh": 8.0,
        "nav2_costmap_cost": 85,
        "nav_action": "ENGAGE_DIFF_LOCK",
        "advisory": "Severe obstacle and rollover risk. Engage low gear, crawl mode, and differential locks."
    },
    "Forest": {
        "description": "Dense vegetation, leaf litter, root obstacles, and soft soil",
        "friction_coefficient": 0.58,
        "sinkage_risk_percent": 45,
        "max_safe_speed_kmh": 14.0,
        "nav2_costmap_cost": 60,
        "nav_action": "THROTTLE_DRIVE",
        "advisory": "Hidden root obstacles and variable compaction. Enable active collision avoidance."
    },
    "Marshland": {
        "description": "Waterlogged mud, marsh, and saturated sediment",
        "friction_coefficient": 0.22,
        "sinkage_risk_percent": 90,
        "max_safe_speed_kmh": 5.0,
        "nav2_costmap_cost": 98,
        "nav_action": "HAZARDOUS_REROUTE",
        "advisory": "Critical immobilization hazard. Halt forward traverse; compute alternate path."
    }
}

class TerrainClassifier:
    """
    Computer Vision Terrain Analysis and Traversability Telemetry Engine for Autonomous Robots.
    """
    def __init__(self):
        self.profiles = TERRAIN_PROFILES

    def extract_features(self, img: Image.Image) -> Dict[str, float]:
        """Extracts texture, gradient variance, and spectral distributions from RGB image."""
        img_rgb = img.convert("RGB").resize((128, 128))
        arr = np.array(img_rgb, dtype=np.float32)

        # Spectral channel statistics
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        r_std, g_std, b_std = float(np.std(r)), float(np.std(g)), float(np.std(b))

        # Brightness and Saturation
        brightness = (r_mean + g_mean + b_mean) / 3.0
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        saturation = float(np.mean(np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-5), 0)))

        # Grayscale edge gradient variance (texture roughness descriptor)
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        diff_x = np.diff(gray, axis=1)
        diff_y = np.diff(gray, axis=0)
        texture_roughness = float(np.var(diff_x) + np.var(diff_y))

        return {
            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,
            "brightness": brightness,
            "saturation": saturation,
            "texture_roughness": texture_roughness,
            "r_std": r_std,
            "g_std": g_std,
            "b_std": b_std
        }

    def classify_image(self, img: Image.Image) -> Dict[str, Any]:
        """Classifies terrain patch and calculates physical soil mechanics telemetry."""
        feats = self.extract_features(img)
        r, g, b = feats["r_mean"], feats["g_mean"], feats["b_mean"]
        roughness = feats["texture_roughness"]
        sat = feats["saturation"]
        bright = feats["brightness"]

        scores = {}
        # Desert: High red and green, low blue, warm tone, moderate roughness
        desert_score = max(0.01, (r * 1.2 + g * 1.0 - b * 1.4) / 180.0) if (r > 110 and g > 90 and b < 100) else 0.05
        # Forest: Green dominance, higher texture variance
        forest_score = max(0.01, (g * 1.4 - r * 0.9 - b * 0.9) / 80.0) if (g > r * 0.95 and g > b) else 0.08
        # Plains: Moderate green/earth, smoother texture (lower roughness)
        plains_score = max(0.01, (g * 1.1 + bright * 0.5) / 120.0) if (g >= b and roughness < 500) else 0.10
        # Mountain: Gray balance (r ~ g ~ b), high roughness
        diff_rgb = abs(r - g) + abs(g - b) + abs(r - b)
        mountain_score = max(0.01, (roughness / 250.0 + (100 - diff_rgb) / 50.0)) if (roughness > 300) else 0.06

        raw_scores = {
            "Desert": desert_score,
            "Forest": forest_score,
            "Plains": plains_score,
            "Mountain": mountain_score
        }

        # Normalize probabilities via softmax-like temperature
        total = sum(raw_scores.values())
        class_probabilities = {k: round(v / total, 3) for k, v in raw_scores.items()}

        top_class = max(class_probabilities, key=class_probabilities.get)
        confidence = class_probabilities[top_class]

        profile = self.profiles.get(top_class, self.profiles["Plains"])

        return {
            "predicted_class": top_class,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "texture_features": {
                "roughness_metric": round(roughness, 1),
                "mean_brightness": round(bright, 1),
                "saturation": round(sat, 2)
            },
            "soil_mechanics": {
                "friction_coefficient_mu": profile["friction_coefficient"],
                "sinkage_risk_percent": profile["sinkage_risk_percent"],
                "max_safe_speed_kmh": profile["max_safe_speed_kmh"],
                "nav2_costmap_cost": profile["nav2_costmap_cost"]
            },
            "navigation_directive": {
                "action": profile["nav_action"],
                "advisory": profile["advisory"]
            }
        }
