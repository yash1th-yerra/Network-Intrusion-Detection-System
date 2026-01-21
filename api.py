from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import uvicorn
import contextlib

MODEL_PATH = "models/modified_bilstm_attention_model.h5"
LABEL_ENCODER_PATH = './label_classes/le2_classes.npy'

# Global variables
models = {}

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and resources on startup."""
    try:
        print("Loading model...")
        models["model"] = load_model(MODEL_PATH, compile=False)
        models["classes"] = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    models.clear()

app = FastAPI(title="NIDS API", lifespan=lifespan)

class DetectionInput(BaseModel):
    features: List[float] = Field(..., min_length=93, max_length=93, description="List of 93 normalized network features")

@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": "model" in models}

@app.post("/predict")
def predict(input_data: DetectionInput):
    if "model" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess: reshape to (1, 1, 93)
        features = np.array(input_data.features, dtype=np.float32)
        features = features.reshape(1, 1, 93)
        
        # Predict
        prediction_probs = models["model"].predict(features, verbose=0)
        predicted_idx = np.argmax(prediction_probs, axis=1)[0]
        predicted_class = models["classes"][predicted_idx]
        confidence = float(np.max(prediction_probs))
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                str(cls): float(prob) 
                for cls, prob in zip(models["classes"], prediction_probs[0])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
