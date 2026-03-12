import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
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
model_error = None

try:

    # If converted model exists
    if os.path.exists(H5_MODEL):
        print("Loading existing H5 model")
        ecg_model = tf.keras.models.load_model(H5_MODEL, compile=False)

    else:
        print("Loading KERAS model")

        temp_model = tf.keras.models.load_model(
            KERAS_MODEL,
            compile=False,
            safe_mode=False
        )

        print("Saving converted H5 model")
        temp_model.save(H5_MODEL)

        ecg_model = temp_model

    print("Model loaded successfully")

except Exception as e:
    print("Model loading failed")
    traceback.print_exc()
    model_error = str(e)
    ecg_model = None


# -------------------------------------------------
# FastAPI
# -------------------------------------------------

app = FastAPI(title="HeartSense AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/debug-files")
def debug_files():
    return {
        "files_in_directory": os.listdir(BASE_DIR),
        "keras_exists": os.path.exists(KERAS_MODEL),
        "h5_exists": os.path.exists(H5_MODEL)
    }
