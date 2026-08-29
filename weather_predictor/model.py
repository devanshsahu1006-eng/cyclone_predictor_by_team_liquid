"""
Cyclone ResNet18 Model Definition & Inference Helper
---------------------------------------------------
Provides model architecture definition, weight loading, and IMD cyclone category logic.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_channels: int = 1) -> nn.Module:
    """Builds a modified ResNet18 for 1-channel grayscale satellite imagery."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def load_model(weights_path: str = "cyclone_resnet18.pth") -> nn.Module:
    """Loads state dict from weights_path into ResNet18 model and sets to eval mode."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at '{weights_path}'")
    model = build_model(num_channels=1)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(DEVICE)


def get_imd_category(wind_speed: float) -> str:
    """Returns India Meteorological Department (IMD) cyclone intensity category."""
    if wind_speed < 17:
        return "Low Pressure Area"
    elif 17 <= wind_speed <= 27:
        return "Depression"
    elif 28 <= wind_speed <= 33:
        return "Deep Depression"
    elif 34 <= wind_speed <= 47:
        return "Cyclonic Storm"
    elif 48 <= wind_speed <= 63:
        return "Severe Cyclonic Storm"
    elif 64 <= wind_speed <= 89:
        return "Very Severe Cyclonic Storm"
    elif 90 <= wind_speed <= 119:
        return "Extremely Severe Cyclonic Storm"
    else:
        return "Super Cyclonic Storm"


def get_severity(category: str) -> str:
    """Returns simplified severity level for visual UI color coding."""
    if category in ["Low Pressure Area", "Depression"]:
        return "LOW"
    elif category in ["Deep Depression", "Cyclonic Storm"]:
        return "MODERATE"
    elif category in ["Severe Cyclonic Storm", "Very Severe Cyclonic Storm"]:
        return "SEVERE"
    else:
        return "EXTREME"


def estimate_pressure_hpa(vmax_kt: float) -> float:
    """Estimates central pressure (hPa) using standard empirical relation P = 1010 - 0.92 * Vmax."""
    pressure = 1010.0 - (0.92 * max(0.0, vmax_kt))
    return round(max(870.0, min(1013.0, pressure)), 1)


def estimate_gale_radius_km(vmax_kt: float) -> int:
    """Estimates gale wind radius (km) based on intensity for visual globe scale."""
    if vmax_kt < 28:
        return 80
    elif vmax_kt < 48:
        return 160
    elif vmax_kt < 64:
        return 240
    elif vmax_kt < 90:
        return 340
    else:
        return 450
