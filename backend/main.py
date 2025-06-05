# backend.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np
import uvicorn
import os

app = FastAPI(title="FlickWise Hybrid Recommender API")

# CORS setup for local frontend use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the path to the rootfiles directory
ROOTFILES_DIR = os.path.join(os.path.dirname(__file__), "..", "rootfiles")

# Load model and artifacts once
MODEL_PATH = os.path.join(ROOTFILES_DIR, "flickwise_hybrid_model.keras")
ARTIFACTS = {
    "user_scaler": os.path.join(ROOTFILES_DIR, "user_numeric_scaler.pkl"),
    "content_scaler": os.path.join(ROOTFILES_DIR, "content_numeric_scaler.pkl"),
    "user_map": os.path.join(ROOTFILES_DIR, "user_id_mapping.csv"),
    "content_map": os.path.join(ROOTFILES_DIR, "content_id_mapping.csv")
}

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    user_scaler = joblib.load(ARTIFACTS["user_scaler"])
    content_scaler = joblib.load(ARTIFACTS["content_scaler"])
    user_map = pd.read_csv(ARTIFACTS["user_map"])
    content_map = pd.read_csv(ARTIFACTS["content_map"])
except Exception as e:
    raise RuntimeError(f"Failed to load model or artifacts: {e}")

# Input schema
class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 5

# Predict route
@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        user_row = user_map[user_map['user_id'] == request.user_id]
        if user_row.empty:
            raise HTTPException(status_code=404, detail="User ID not found")

        user_index = int(user_row['user_index'].values[0])
        user_vector = np.array([[user_index]] * len(content_map))
        content_vector = np.array(content_map['content_index'])

        predictions = model.predict([user_vector, content_vector], verbose=0)
        top_indices = predictions.flatten().argsort()[-request.top_k:][::-1]
        recommended_content_ids = content_map.iloc[top_indices]['content_id'].tolist()

        return {"recommended_content": recommended_content_ids}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
