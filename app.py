import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import requests
import traceback

# -------------------------------------------------
# Load ECG Model
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_lstm_mitbih_final.h5")

print("Looking for model at:", MODEL_PATH)

ecg_model = None
model_error = None

try:
    print("Loading ECG AI model...")
    ecg_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully")

except Exception as e:
    print("Model loading failed")
    traceback.print_exc()
    model_error = str(e)
    ecg_model = None


# -------------------------------------------------
# ThingSpeak Configuration
# -------------------------------------------------

THINGSPEAK_CHANNEL_ID = "3143087"
THINGSPEAK_READ_API_KEY = "0BUW0FE0IF72M1VH"


# -------------------------------------------------
# FastAPI App
# -------------------------------------------------

app = FastAPI(title="HeartSense AI – Medical Decision Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# Health Check
# -------------------------------------------------

@app.get("/")
def home():
    return {"status": "HeartSense AI Backend Running"}


@app.get("/health")
def health():
    return {
        "backend": "running",
        "model_loaded": ecg_model is not None,
        "model_error": model_error
    }


# -------------------------------------------------
# ThingSpeak Data Readers
# -------------------------------------------------

def read_latest():
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json?api_key={THINGSPEAK_READ_API_KEY}"
    d = requests.get(url, timeout=5).json()

    return {
        "ecg": float(d.get("field4") or 0),
        "spo2": float(d.get("field6") or 0),
        "gsr": float(d.get("field5") or 0),
        "temp": float(d.get("field7") or 0)
    }


# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------

@app.get("/thingspeak-final-risk")
def thingspeak_final_risk():

    if ecg_model is None:
        return {"error": "Model not loaded"}

    data = read_latest()

    ecg = np.array([data["ecg"]] * 180)
    ecg = ecg.reshape(1, 180, 1)

    prediction = ecg_model.predict(ecg, verbose=0)

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "raw_data": data
    }
