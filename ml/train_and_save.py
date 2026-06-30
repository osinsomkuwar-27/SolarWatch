"""
train_and_save.py
Trains the final model on blocks 1+2, saves to ml/models/flare_model_v1.pkl
Run once before starting inference.
"""
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    "soft_xray", "cdte_broadband", "czt_broadband",
    "hard_soft_ratio", "cdte_czt_ratio",
    "slx_d1", "slx_d2", "cdte_d1",
    "slx_d1_smooth_60s", "slx_d1_smooth_300s",
    "slx_roll_mean_5m", "slx_roll_std_5m",
    "slx_roll_mean_30m", "slx_roll_std_30m",
    "cdte_roll_mean_30m", "cdte_roll_std_30m",
    "slx_zscore", "cdte_zscore", "slx_vs_baseline",
    "data_quality",
]

print("Loading dataset...")
df = pd.read_parquet("data/master_dataset_features.parquet")

train = df[df["block"].isin([1, 2])]
print(f"Training on {len(train):,} rows, {train['date'].nunique()} days...")
print(f"Label distribution:\n{train['label'].value_counts()}")

pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(train[FEATURE_COLS], train["label"])

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/flare_model_v1.pkl")
print("Saved: models/flare_model_v1.pkl")