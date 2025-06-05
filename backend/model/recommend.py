# backend/model/recommend.py
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib

MODEL_PATH = "artifacts/flickwise_hybrid_model.keras"
USER_ID_MAP_PATH = "artifacts/user_id_mapping.csv"
CONTENT_ID_MAP_PATH = "artifacts/content_id_mapping.csv"
USER_SCALER_PATH = "artifacts/user_numeric_scaler.pkl"
CONTENT_SCALER_PATH = "artifacts/content_numeric_scaler.pkl"

# Load model and scalers
model = tf.keras.models.load_model(MODEL_PATH)
user_id_map = pd.read_csv(USER_ID_MAP_PATH)
content_id_map = pd.read_csv(CONTENT_ID_MAP_PATH)
user_scaler = joblib.load(USER_SCALER_PATH)
content_scaler = joblib.load(CONTENT_SCALER_PATH)

# For demo, mock features for all content (should match training format)
def get_content_features():
    content_features = content_id_map.copy()
    content_features['dummy1'] = 0  # Replace with real features
    content_features['dummy2'] = 0
    content_scaled = content_scaler.transform(content_features.drop(columns=['content_id']))
    return content_features['content_id'].values, content_scaled

def get_user_features(user_id):
    if user_id not in user_id_map['user_id'].values:
        raise ValueError(f"User ID {user_id} not found in training data.")

    user_numeric_id = user_id_map[user_id_map['user_id'] == user_id].index[0]
    user_features = np.zeros((1, user_scaler.n_features_in_))  # Replace with real user features
    user_features_scaled = user_scaler.transform(user_features)
    return user_numeric_id, user_features_scaled

def get_recommendations(user_id, top_k=10):
    user_numeric_id, user_features_scaled = get_user_features(user_id)
    content_ids, content_features_scaled = get_content_features()

    user_id_array = np.array([user_numeric_id] * len(content_ids))
    predictions = model.predict({
        "user_id_input": user_id_array,
        "user_features": np.repeat(user_features_scaled, len(content_ids), axis=0),
        "content_id_input": np.arange(len(content_ids)),
        "content_features": content_features_scaled
    }, verbose=0)

    top_indices = predictions.reshape(-1).argsort()[::-1][:top_k]
    top_recommendations = content_ids[top_indices]
    return top_recommendations.tolist()
