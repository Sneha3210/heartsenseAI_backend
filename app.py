import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import requests
import traceback

# -------------------------------------------------
# Model Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KERAS_MODEL = os.path.join(BASE_DIR, "cnn_lstm_mitbih_final.keras")
H5_MODEL = os.path.join(BASE_DIR, "cnn_lstm_mitbih_final.h5")

print("KERAS MODEL:", KERAS_MODEL)
print("H5 MODEL:", H5_MODEL)

ecg_model = None

try:

    # If H5 model already exists → load it
    if os.path.exists(H5_MODEL):
        print("Loading existing H5 model")
        ecg_model = tf.keras.models.load_model(H5_MODEL, compile=False)

    # Otherwise convert from .keras
    else:
        print("Loading KERAS model")
        temp_model = tf.keras.models.load_model(KERAS_MODEL, compile=False)

        print("Converting model to H5")
        temp_model.save(H5_MODEL)

        ecg_model = temp_model

    print("Model loaded successfully")

except Exception as e:
    print("Model loading failed")
    traceback.print_exc()
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
# Health Endpoint
# -------------------------------------------------

@app.get("/")
def home():
    return {"status": "HeartSense AI Backend Running"}


@app.get("/health")
def health():
    return {
        "backend": "running",
        "model_loaded": ecg_model is not None
    }
