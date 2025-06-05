import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, save_model
from training.train import prepare_data_for_retraining, build_hybrid_model  # assuming modularized

MODEL_PATH = "./artifacts/flickwise_hybrid_model.keras"
USER_ID_MAP_PATH = "./user_id_mapping.csv"
CONTENT_ID_MAP_PATH = "./content_id_mapping.csv"


def retrain_model(csv_path: str):
    """
    Retrain the hybrid model using new CSV data uploaded by the user.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Uploaded CSV file not found.")

    # Step 1: Load new dataset
    new_data = pd.read_csv(csv_path)

    # Step 2: Preprocess and prepare features (you can define this based on train.py logic)
    X_train, X_val, y_train, y_val, user_features, content_features = prepare_data_for_retraining(new_data)

    # Step 3: Load or build model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = build_hybrid_model(user_features, content_features)  # from train.py

    # Step 4: Retrain model
    model.fit(
        x={
            "user_id_input": X_train["user_id_input"],
            "content_id_input": X_train["content_id_input"],
            "user_features": X_train["user_features"],
            "content_features": X_train["content_features"]
        },
        y=y_train,
        validation_data=(
            {
                "user_id_input": X_val["user_id_input"],
                "content_id_input": X_val["content_id_input"],
                "user_features": X_val["user_features"],
                "content_features": X_val["content_features"]
            },
            y_val
        ),
        batch_size=256,
        epochs=10,
        verbose=1
    )

    # Step 5: Save updated model
    save_model(model, MODEL_PATH)
    return True
