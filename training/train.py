# Final production-grade training code based on 9 optimization strategies
# Key Improvements:
# - Data validation and logging
# - ID mapping persistence
# - Robust pipeline structure
# - Feature drift mitigation
# - Modular training interface
# - Enhanced model architecture
# - Distribution-aware data handling
# - Regularization
# - Automatic saving of model and preprocessors

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import AdamW

# --- 1. Config & Logging Setup ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
EMBEDDING_DIM_USER = 32
EMBEDDING_DIM_MOVIE = 32
EMBEDDING_DIM_SMALL = 16  # for plan, churn, genre, original
DENSE_UNITS = [256, 128]
INITIAL_LR = 1e-3
FIRST_DECAY_STEPS = 10000
T_MUL = 2.0
M_MUL = 0.9
ALPHA = 1e-5
EPOCHS = 50
BATCH_SIZE = 256
TEST_SIZE = 0.15

DATA_PATH = "../data/flickwise_synthetic_data.csv"
ARTIFACTS_DIR = "./artifacts"
LOGS_DIR = "./logs"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Use mixed precision for speed and memory
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("mixed_float16")

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def set_random_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seeds set to {seed}")

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created directory: {path}")

def save_pickle(obj, path: str):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Saved: {path}")

# ------------------------------------------------------------
# 1. Data Loading & Initial Inspection
# ------------------------------------------------------------
def load_raw_data(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[
        'subscription_start_date', 'date', 'release_date'
    ])
    print(f"✅ Data loaded. Columns: {list(df.columns)}")
    print(f"Initial data shape: {df.shape}")
    return df

# ------------------------------------------------------------
# 2. User Profiling
# ------------------------------------------------------------
def build_user_profiles(df: pd.DataFrame, num_users: int = 5000) -> pd.DataFrame:
    user_df = df.groupby("user_id").agg({
        "plan_type": "first",
        "churned": "first",
        "daily_watch_time_minutes": "mean",
        "watch_duration_seconds": "mean",
        "watchlist_count": "mean",
        "support_tickets_opened": "mean",
        "last_interaction_days_ago": "mean",
        "subscription_start_date": lambda dates: (
            pd.to_datetime("2025-06-01") - pd.to_datetime(dates.min())
        ).days
    }).reset_index().rename(columns={
        "daily_watch_time_minutes": "avg_watch_time",
        "watch_duration_seconds": "avg_watch_duration",
        "watchlist_count": "avg_watchlist",
        "support_tickets_opened": "avg_tickets",
        "last_interaction_days_ago": "mean_interaction_gap",
        "subscription_start_date": "tenure_days"
    })

    user_profiles = user_df.head(num_users).copy()
    print(f"--- User Profiling ---")
    print(f"User features shape: {user_profiles.shape}")
    return user_profiles

# ------------------------------------------------------------
# 3. Content Profiling
# ------------------------------------------------------------
def build_content_profiles(df: pd.DataFrame, num_contents: int = 900) -> pd.DataFrame:
    content_agg = df.groupby("content_id").agg({
        "genre": "first",
        "duration_minutes": "first",
        "release_date": lambda dates: (
            pd.to_datetime("2025-06-01") - pd.to_datetime(dates.min())
        ),
        "is_original": "first"
    }).reset_index().rename(columns={"release_date": "age_timedelta"})
    content_agg["content_age_days"] = content_agg["age_timedelta"].dt.days
    content_agg = content_agg.drop(columns=["age_timedelta"])
    content_profiles = content_agg.head(num_contents).copy()

    null_counts = content_profiles.isnull().sum()
    if null_counts.sum() > 0:
        print("⚠️ WARNING: Null values found in content_features. Consider imputation.")
        print(f"{null_counts[null_counts > 0]}")

    print(f"--- Content Profiling ---")
    print(f"Content features shape: {content_profiles.shape}")
    return content_profiles

# ------------------------------------------------------------
# 4. Ratings Matrix Preparation
# ------------------------------------------------------------
def build_ratings_matrix(df: pd.DataFrame, users: pd.DataFrame, contents: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["user_id"].isin(users["user_id"]) & df["content_id"].isin(contents["content_id"])].copy()

    if "avg_rating" in filtered.columns:
        filtered = filtered[["user_id", "content_id", "avg_rating", "activity_timestamp"]].rename(columns={"avg_rating": "rating"})
    else:
        filtered = filtered[["user_id", "content_id", "watch_duration_seconds", "activity_timestamp"]].rename(columns={"watch_duration_seconds": "rating"})

    if "activity_timestamp" in filtered.columns:
        filtered = filtered.sort_values("activity_timestamp", ascending=False)
    else:
        print("No 'activity_timestamp' column found. Skipping sort.")

    filtered = filtered.drop_duplicates(subset=["user_id", "content_id"], keep="first")

    if "activity_timestamp" in filtered.columns:
        filtered = filtered.drop(columns=["activity_timestamp"])

    print(f"--- Ratings Matrix Preparation ---")
    print(f"Ratings matrix shape: {filtered.shape}")
    return filtered.reset_index(drop=True)

# ------------------------------------------------------------
# 5. ID Mapping (Users & Movies)
# ------------------------------------------------------------
def create_id_mappings(users: pd.DataFrame, contents: pd.DataFrame) -> (dict, dict):
    user_ids = users["user_id"].unique().tolist()
    movie_ids = contents["content_id"].unique().tolist()

    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    ensure_dir(ARTIFACTS_DIR)
    pd.DataFrame({"user_id": user_ids, "user_idx": list(range(len(user_ids)))}) \
        .to_csv(os.path.join(ARTIFACTS_DIR, "user_id_mapping.csv"), index=False)
    pd.DataFrame({"content_id": movie_ids, "movie_idx": list(range(len(movie_ids)))}) \
        .to_csv(os.path.join(ARTIFACTS_DIR, "content_id_mapping.csv"), index=False)

    print(f"--- ID Mapping ---")
    print(f"✅ ID mapping: {len(user_ids)} users, {len(movie_ids)} movies. Maps saved.")
    return user2idx, movie2idx

# ------------------------------------------------------------
# 6. Feature Scaling & Encoding
# ------------------------------------------------------------
def scale_and_encode_user_features(user_profiles: pd.DataFrame) -> (pd.DataFrame, dict):
    numeric_cols = ["avg_watch_time", "avg_watch_duration", "avg_watchlist",
                    "avg_tickets", "mean_interaction_gap", "tenure_days"]
    scaler = MinMaxScaler()
    user_profiles[numeric_cols] = scaler.fit_transform(user_profiles[numeric_cols])
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "user_numeric_scaler.pkl"))
    print(f"✅ Saved: {os.path.join(ARTIFACTS_DIR, 'user_numeric_scaler.pkl')}")

    print(f"--- Feature Scaling & Encoding (User) ---")
    print("✅ User features scaled and saved.")
    return user_profiles, {"user_numeric_scaler": scaler}


def scale_and_encode_content_features(content_profiles: pd.DataFrame) -> (pd.DataFrame, dict):
    for col in ["duration_minutes", "content_age_days"]:
        if content_profiles[col].isnull().any():
            med = content_profiles[col].median()
            content_profiles[col].fillna(med, inplace=True)
            print(f"Filled NaNs in content_features '{col}' with median: {med}")

    scaler = MinMaxScaler()
    content_profiles[["duration_minutes", "content_age_days"]] = scaler.fit_transform(
        content_profiles[["duration_minutes", "content_age_days"]]
    )
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "content_numeric_scaler.pkl"))
    print(f"✅ Saved: {os.path.join(ARTIFACTS_DIR, 'content_numeric_scaler.pkl')}")

    le_genre = LabelEncoder()
    content_profiles["genre_enc"] = le_genre.fit_transform(content_profiles["genre"])
    joblib.dump(le_genre, os.path.join(ARTIFACTS_DIR, "genre_encoder.pkl"))
    print(f"✅ Saved: {os.path.join(ARTIFACTS_DIR, 'genre_encoder.pkl')}")

    le_orig = LabelEncoder()
    content_profiles["orig_enc"] = le_orig.fit_transform(content_profiles["is_original"])
    joblib.dump(le_orig, os.path.join(ARTIFACTS_DIR, "orig_encoder.pkl"))
    print(f"✅ Saved: {os.path.join(ARTIFACTS_DIR, 'orig_encoder.pkl')}")

    print(f"--- Feature Scaling & Encoding (Content) ---")
    print("✅ Content features scaled and saved.")
    return content_profiles, {
        "content_numeric_scaler": scaler,
        "genre_encoder": le_genre,
        "orig_encoder": le_orig
    }

# ------------------------------------------------------------
# 7. Build Hybrid Engagement Model
# ------------------------------------------------------------
def build_hybrid_model(
    num_users: int,
    num_movies: int,
    num_plans: int,
    num_churn_vals: int,
    num_genres: int,
    num_orig_vals: int
) -> Model:
    # Input Layers
    user_id_input      = Input(shape=(), name="user_id_input", dtype="int32")
    movie_id_input     = Input(shape=(), name="movie_id_input", dtype="int32")
    plan_enc           = Input(shape=(), name="plan_enc", dtype="int32")
    churn_enc          = Input(shape=(), name="churn_enc", dtype="int32")
    genre_enc          = Input(shape=(), name="genre_enc", dtype="int32")
    orig_enc           = Input(shape=(), name="orig_enc", dtype="int32")

    avg_watch_time     = Input(shape=(), name="avg_watch_time", dtype="float32")
    avg_watchlist      = Input(shape=(), name="avg_watchlist", dtype="float32")
    avg_tickets        = Input(shape=(), name="avg_tickets", dtype="float32")
    mean_interaction_gap = Input(shape=(), name="mean_interaction_gap", dtype="float32")
    tenure_days        = Input(shape=(), name="tenure_days", dtype="float32")
    duration_minutes   = Input(shape=(), name="duration_minutes", dtype="float32")
    content_age_days   = Input(shape=(), name="content_age_days", dtype="float32")

    # Embedding Layers
    user_embedding  = layers.Embedding(input_dim=num_users, output_dim=EMBEDDING_DIM_USER, name="user_embedding")(user_id_input)
    movie_embedding = layers.Embedding(input_dim=num_movies, output_dim=EMBEDDING_DIM_MOVIE, name="movie_embedding")(movie_id_input)

    plan_embedding   = layers.Embedding(input_dim=num_plans, output_dim=EMBEDDING_DIM_SMALL, name="plan_embedding")(plan_enc)
    churn_embedding  = layers.Embedding(input_dim=num_churn_vals, output_dim=EMBEDDING_DIM_SMALL, name="churn_embedding")(churn_enc)
    genre_embedding  = layers.Embedding(input_dim=num_genres, output_dim=EMBEDDING_DIM_SMALL, name="genre_embedding")(genre_enc)
    orig_embedding   = layers.Embedding(input_dim=num_orig_vals, output_dim=EMBEDDING_DIM_SMALL, name="orig_embedding")(orig_enc)

    # Flatten embeddings
    flat_user_emb    = layers.Flatten()(user_embedding)
    flat_movie_emb   = layers.Flatten()(movie_embedding)
    flat_plan_emb    = layers.Flatten()(plan_embedding)
    flat_churn_emb   = layers.Flatten()(churn_embedding)
    flat_genre_emb   = layers.Flatten()(genre_embedding)
    flat_orig_emb    = layers.Flatten()(orig_embedding)

    # Numeric feature dense layers
    rw = layers.Reshape((1,))
    concat_user_numerical = layers.Concatenate(name="user_numerical_concat")([
        rw(avg_watch_time), rw(avg_watchlist), rw(avg_tickets), rw(mean_interaction_gap), rw(tenure_days)
    ])
    concat_content_numerical = layers.Concatenate(name="content_numerical_concat")([
        rw(duration_minutes), rw(content_age_days)
    ])

    user_numerical_dense    = layers.Dense(EMBEDDING_DIM_SMALL, name="user_numerical_dense")(concat_user_numerical)
    content_numerical_dense = layers.Dense(EMBEDDING_DIM_SMALL, name="content_numerical_dense")(concat_content_numerical)

    # Combine all features
    combined_features = layers.Concatenate(name="combined_features")( [
        flat_user_emb, flat_movie_emb,
        flat_plan_emb, flat_churn_emb, flat_genre_emb, flat_orig_emb,
        user_numerical_dense, content_numerical_dense
    ] )

    x = layers.BatchNormalization(name="batch_norm")(combined_features)
    x = layers.Dense(DENSE_UNITS[0], activation="relu", name="dense_0")(x)
    x = layers.Dropout(0.3, name="dropout_0")(x)
    x = layers.Dense(DENSE_UNITS[1], activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    predicted_rating = layers.Dense(1, activation="linear", name="predicted_rating")(x)

    model = Model(
        inputs=[
            user_id_input, movie_id_input, plan_enc, churn_enc, genre_enc, orig_enc,
            avg_watch_time, avg_watchlist, avg_tickets, mean_interaction_gap, tenure_days,
            duration_minutes, content_age_days
        ],
        outputs=[predicted_rating],
        name="HybridEngagementRecModel"
    )

    print("✅ Hybrid Engagement Model built.")
    return model

# ------------------------------------------------------------
# 8. Main Training Pipeline
# ------------------------------------------------------------
def main():
    # 1) Set random seeds
    set_random_seeds(SEED)

    # 2) Ensure artifacts and logs directories
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(LOGS_DIR)

    # 3) Load raw data
    df = load_raw_data(DATA_PATH)

    # 4) Build user & content profiles
    user_profiles    = build_user_profiles(df, num_users=5000)
    content_profiles = build_content_profiles(df, num_contents=900)

    # 5) Build ratings matrix
    ratings_df = build_ratings_matrix(df, user_profiles, content_profiles)

    # 6) Create ID mappings and convert to indices
    user2idx, movie2idx = create_id_mappings(user_profiles, content_profiles)
    ratings_df["user_idx"]  = ratings_df["user_id"].map(user2idx)
    ratings_df["movie_idx"] = ratings_df["content_id"].map(movie2idx)

    # 7) Scale & encode user features
    user_profiles_scaled, user_preprocs = scale_and_encode_user_features(user_profiles)
    user_profiles_scaled.set_index("user_id", inplace=True)

    # 8) Scale & encode content features
    content_profiles_scaled, content_preprocs = scale_and_encode_content_features(content_profiles)
    content_profiles_scaled.set_index("content_id", inplace=True)

    # 9) Merge ratings with user & content features
    merged = ratings_df \
        .merge(user_profiles_scaled.reset_index(), on="user_id", how="inner") \
        .merge(content_profiles_scaled.reset_index(), on="content_id", how="inner")

    print(f"--- Preparing Data for Recommendation Model Training ---")
    print(f"All features for recommender have {len(merged)} samples.")

    # 10) Build feature dictionary for model inputs
    X = {
        "user_id_input": merged["user_idx"].values.astype("int32"),
        "movie_id_input": merged["movie_idx"].values.astype("int32"),
        "plan_enc": LabelEncoder().fit_transform(merged["plan_type"].astype(str)).astype("int32"),
        "churn_enc": LabelEncoder().fit_transform(merged["churned"].astype(str)).astype("int32"),
        "genre_enc": merged["genre_enc"].values.astype("int32"),
        "orig_enc": merged["orig_enc"].values.astype("int32"),
        "avg_watch_time": merged["avg_watch_time"].values.astype("float32"),
        "avg_watchlist": merged["avg_watchlist"].values.astype("float32"),
        "avg_tickets": merged["avg_tickets"].values.astype("float32"),
        "mean_interaction_gap": merged["mean_interaction_gap"].values.astype("float32"),
        "tenure_days": merged["tenure_days"].values.astype("float32"),
        "duration_minutes": merged["duration_minutes"].values.astype("float32"),
        "content_age_days": merged["content_age_days"].values.astype("float32"),
    }
    y = merged["rating"].values.astype("float32")

    # 11) Train/validation split
    num_samples = len(y)
    all_indices = np.arange(num_samples)
    train_idx, val_idx = train_test_split(all_indices, test_size=TEST_SIZE, random_state=SEED)

    X_train = {key: val[train_idx] for key, val in X.items()}
    X_val   = {key: val[val_idx]   for key, val in X.items()}
    y_train = y[train_idx]
    y_val   = y[val_idx]

    print(f"Train set size: {len(y_train)}, Validation set size: {len(y_val)}")
    print("✅ Train/validation split complete.")

    # 12) Build the model
    num_users     = len(user2idx)
    num_movies    = len(movie2idx)
    num_plans     = len(np.unique(X["plan_enc"]))
    num_churn_vals= len(np.unique(X["churn_enc"]))
    num_genres    = len(np.unique(X["genre_enc"]))
    num_orig_vals = len(np.unique(X["orig_enc"]))

    engagement_model = build_hybrid_model(
        num_users=num_users,
        num_movies=num_movies,
        num_plans=num_plans,
        num_churn_vals=num_churn_vals,
        num_genres=num_genres,
        num_orig_vals=num_orig_vals
    )

    # 13) Compile with AdamW + CosineDecayRestarts
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=INITIAL_LR,
        first_decay_steps=FIRST_DECAY_STEPS,
        t_mul=T_MUL,
        m_mul=M_MUL,
        alpha=ALPHA
    )
    optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)

    engagement_model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]  
    )

    # 14) Train the model (no LearningRateScheduler callback needed)
    print(f"--- Training Hybrid Recommendation Model ---")
    history_rec = engagement_model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    print("✅ Training complete.")

    # 15) Save the trained model
    model_save_path = os.path.join(ARTIFACTS_DIR, "flickwise_hybrid_model.keras")
    engagement_model.save(model_save_path)
    print(f"✅ Model saved automatically at: {model_save_path}")

    # 16) Save training history
    history_path = os.path.join(ARTIFACTS_DIR, "training_history.pkl")
    save_pickle(history_rec.history, history_path)


if __name__ == "__main__":
    main()
