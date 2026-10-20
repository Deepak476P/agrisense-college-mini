"""Dataset definitions and explicit input validation."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SPECS = {
    "crop": {
        "file": "crop_recommendation.csv",
        "target": "label",
        "numeric": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        "categorical": [],
        "task": "classification",
    },
    "fertilizer": {
        "file": "fertilizer_data.csv",
        "target": "Fertilizer_Name",
        "numeric": ["Temperature", "Humidity", "Moisture", "Nitrogen", "Phosphorous", "Potassium"],
        "categorical": ["Soil_Type", "Crop_Type"],
        "task": "classification",
    },
    "yield": {
        "file": "crop_yield_data.csv",
        "target": "Yield_tons_per_ha",
        "numeric": ["Area_ha", "Rainfall_mm", "Temperature_c", "Fertilizer_kg_ha"],
        "categorical": ["Crop", "Season", "State", "Irrigation_Type"],
        "task": "regression",
    },
}


def load_data(kind):
    """Keep Production_tons out of yield features: it contains the answer."""
    spec = SPECS[kind]
    frame = pd.read_csv(ROOT / "data" / spec["file"])
    required = spec["numeric"] + spec["categorical"] + [spec["target"]]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("Missing values found. Inspect and document a cleaning strategy first.")
    if frame.duplicated().any():
        raise ValueError("Duplicate rows found. Review them before creating a train/test split.")
    if not np.isfinite(frame[spec["numeric"]].to_numpy(dtype=float)).all():
        raise ValueError("Numerical data must contain only finite values.")
    return frame


def reference_metadata():
    return json.loads((ROOT / "data/reference_metadata.json").read_text(encoding="utf-8"))


def validate_input(kind, values, bundle):
    """Reject invalid values, but separately flag valid values beyond training ranges."""
    spec = SPECS[kind]
    features = spec["numeric"] + spec["categorical"]
    missing = set(features) - set(values)
    if missing:
        raise ValueError(f"Please provide: {', '.join(sorted(missing))}")
    clean, warnings = {}, []
    for name in spec["numeric"]:
        try:
            number = float(values[name])
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be a number.") from None
        if not np.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        low, high = (0.0, float("inf"))
        if name in {"temperature", "Temperature", "Temperature_c"}:
            low, high = -50.0, 60.0
        elif name in {"humidity", "Humidity", "Moisture"}:
            high = 100.0
        elif name == "ph":
            high = 14.0
        if number < low or number > high or (name == "Area_ha" and number <= 0):
            raise ValueError(f"{name} is outside its allowed range.")
        clean[name] = number
        minimum, maximum = bundle["ranges"][name]
        if number < minimum or number > maximum:
            warnings.append(f"{name}={number:g} is outside the training range {minimum:g}–{maximum:g}.")
    for name in spec["categorical"]:
        value = str(values[name])
        if value not in bundle["categories"][name]:
            raise ValueError(f"Unknown {name}: {value}. Choose one of the available dataset categories.")
        clean[name] = value
    return pd.DataFrame([clean], columns=features), warnings
