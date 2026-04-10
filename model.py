from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("crop_model.pkl")

# I kept multiple possible dataset locations because sometimes
# I run backend from different folders during testing.
DATASET_CANDIDATES = [
    Path("../data/Crop_recommendation.csv"),
    Path("data/Crop_recommendation.csv"),
    Path("Crop_recommendation.csv"),
]


def resolve_dataset_path() -> Path:
    """
    Try common dataset locations and return the first valid one.
    """
    for file_path in DATASET_CANDIDATES:
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        "Crop_recommendation.csv not found. Put the dataset in data/ or ../data/."
    )


def load_or_train_model() -> RandomForestClassifier:
    """
    Load saved model if available.
    Otherwise train once and store it as crop_model.pkl.
    """
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as model_file:
            return pickle.load(model_file)

    dataset_path = resolve_dataset_path()
    print(f"[INFO] Training model using dataset: {dataset_path}")

    data = pd.read_csv(dataset_path)

    feature_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = data[feature_columns]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # I tried keeping the model reasonably stable instead of over-complicating it.
    # RandomForest gave reliable performance for this project.
    trained_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_split=4,
        random_state=42
    )
    trained_model.fit(X_train, y_train)

    accuracy = trained_model.score(X_test, y_test)
    print(f"[INFO] Validation accuracy: {accuracy:.4f}")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(trained_model, model_file)

    return trained_model


model = load_or_train_model()


def get_weather_data() -> Dict[str, Any]:
    """
    Get current weather for Meerut using Open-Meteo.
    If API fails, fallback values are returned so project still works offline.
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=28.9845&longitude=77.7064"
            "&current=temperature_2m,relative_humidity_2m"
            "&daily=rain_sum"
            "&timezone=Asia/Kolkata"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        api_data = response.json()

        return {
            "temperature": float(api_data["current"]["temperature_2m"]),
            "humidity": float(api_data["current"]["relative_humidity_2m"]),
            "rainfall": float(api_data["daily"]["rain_sum"][0]) if api_data["daily"]["rain_sum"] else 0.0,
            "source": "Open-Meteo"
        }

    except Exception:
        # Fallback keeps system running even if internet/API is unavailable.
        return {
            "temperature": 27.0,
            "humidity": 65.0,
            "rainfall": 0.0,
            "source": "Fallback weather"
        }


def analyze_soil(sensor_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Very basic soil interpretation for dashboard explanation.
    """
    moisture = sensor_data.get("moisture", 0)
    ph_value = sensor_data.get("ph", 6.5)

    if moisture < 35:
        moisture_status = "Low"
    elif moisture <= 70:
        moisture_status = "Optimal"
    else:
        moisture_status = "High"

    if ph_value < 6:
        ph_status = "Acidic"
    elif ph_value <= 7.5:
        ph_status = "Neutral"
    else:
        ph_status = "Alkaline"

    return {
        "moisture_status": moisture_status,
        "ph_status": ph_status
    }


def fertilizer_recommendation(sensor_data: Dict[str, Any]) -> str:
    """
    Simple rule-based fertilizer recommendation.
    This is intentionally kept easy to explain during viva.
    """
    suggestions: List[str] = []

    if sensor_data.get("N", 0) < 40:
        suggestions.append("Urea for nitrogen boost")

    if sensor_data.get("P", 0) < 30:
        suggestions.append("DAP for phosphorus boost")

    if sensor_data.get("K", 0) < 30:
        suggestions.append("MOP / Potash for potassium boost")

    if not suggestions:
        suggestions.append("Balanced NPK fertilizer")

    return " | ".join(suggestions)


def irrigation_advice(moisture: int) -> str:
    """
    Small helper for dashboard readability.
    """
    if moisture < 35:
        return "Irrigation needed soon"
    if moisture <= 70:
        return "Soil moisture is in a healthy range"
    return "Reduce watering and improve drainage"


def get_market_price(crop: str) -> Dict[str, Any]:
    """
    Placeholder mandi pricing.
    This is not live market integration yet.
    """
    price_map = {
        "Rice": 1800,
        "Wheat": 2300,
        "Maize": 1600,
        "Millet": 1500,
        "Cotton": 6200,
        "Sugarcane": 340,
        "Muskmelon": 1700,
        "Watermelon": 1500,
        "Mango": 3200,
    }

    return {
        "name": "Meerut / Delhi Mandi",
        "price_per_quintal": price_map.get(crop, 1700),
        "note": "Indicative mandi price placeholder. Replace with live agri-market API later."
    }


def build_tips(sensor_data: Dict[str, Any], predicted_crop: str, soil_status: Dict[str, str]) -> List[str]:
    """
    Human-readable guidance shown in API response.
    """
    tips = [
        f"Predicted best crop: {predicted_crop}.",
        f"Soil moisture is {soil_status['moisture_status'].lower()}.",
        f"Soil pH trend is {soil_status['ph_status'].lower()}."
    ]

    if sensor_data.get("moisture", 0) < 35:
        tips.append("Use light irrigation immediately and recheck after 30 minutes.")

    if sensor_data.get("temperature", 25) > 35:
        tips.append("Temperature is high. Prefer irrigation in early morning or evening.")

    if sensor_data.get("P", 0) < 30 or sensor_data.get("K", 0) < 30:
        tips.append("Nutrient support is recommended before the next sowing cycle.")

    return tips


def get_recommendation(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function used by backend.
    Combines sensor data, weather data, ML prediction, and rule-based suggestions.
    """
    weather_data = get_weather_data()
    soil_status = analyze_soil(sensor_data)

    input_features = pd.DataFrame([
        {
            "N": sensor_data.get("N", 50),
            "P": sensor_data.get("P", 30),
            "K": sensor_data.get("K", 20),
            "temperature": sensor_data.get("temperature", weather_data["temperature"]),
            "humidity": weather_data["humidity"],
            "ph": sensor_data.get("ph", 6.5),
            "rainfall": weather_data["rainfall"]
        }
    ])

    predicted_label = model.predict(input_features)[0]
    crop_name = str(predicted_label).capitalize()

    probabilities = model.predict_proba(input_features)[0]
    confidence_score = float(max(probabilities))

    fertilizer_text = fertilizer_recommendation(sensor_data)
    irrigation_text = irrigation_advice(sensor_data.get("moisture", 0))
    market_info = get_market_price(crop_name)
    tips = build_tips(sensor_data, crop_name, soil_status)

    return {
        "crop": crop_name,
        "fertilizer": fertilizer_text,
        "irrigation": irrigation_text,
        "market": market_info,
        "confidence": (
            "High" if confidence_score >= 0.75
            else "Medium" if confidence_score >= 0.45
            else "Low"
        ),
        "confidence_score": round(confidence_score, 4),
        "weather": weather_data,
        "soil_status": soil_status,
        "tips": tips
    }