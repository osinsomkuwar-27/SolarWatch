"""
inference.py — runs the trained model on the June 21 2026 demo data
and writes latest_prediction.json every 60 seconds (simulated replay).

For the hackathon demo: replays June 21 2026 data second-by-second,
writing prediction JSON so the backend picks it up automatically.
"""

import json
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import requests
import os

MODEL_PATH      = "models/flare_model_v1.pkl"
DATA_PATH       = "data/master_dataset_features.parquet"

# Dynamic path to the real backend mock folder
BASE_DIR = Path(__file__).parent.parent
OUTPUT_PATH = BASE_DIR / "backend" / "mock" / "latest_prediction.json"
API_URL = os.getenv("API_URL") # E.g., 'https://solar-rzv2.onrender.com'

DEMO_DATE       = "2026-06-21"
REPLAY_INTERVAL = 1  

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

CLASS_NAMES = {0: "quiet", 1: "pre-flare", 2: "flare"}
FLARE_CLASS_MAP = {0: None, 1: "C", 2: "M"}  


def get_data_quality(row) -> dict:
    dq = int(row.get("data_quality", 0))
    if dq == 0:
        return {
            "status": "ok",
            "reason": None,
            "lookback_seconds_available": 1800,
            "lookback_seconds_required": 1800,
        }
    elif dq == 1:
        return {
            "status": "degraded",
            "reason": "masked_overlap",
            "lookback_seconds_available": 600,
            "lookback_seconds_required": 1800,
        }
    else:
        return {
            "status": "unavailable",
            "reason": "instrument_gap",
            "lookback_seconds_available": 0,
            "lookback_seconds_required": 1800,
        }


def build_prediction_json(row: pd.Series, proba: np.ndarray, 
                           pred_class: int, timestamp: str) -> dict:
    nowcast_prob  = float(proba[2])    
    pre_prob      = float(proba[1])
    quiet_prob    = float(proba[0])

    if nowcast_prob > 0.6:
        flare_class = "M"
    elif nowcast_prob > 0.3:
        flare_class = "C"
    else:
        flare_class = None

    d1 = row.get("slx_d1_smooth_60s", 0) or 0
    if pred_class == 1 and d1 > 0:
        onset_minutes = max(5, int(30 - d1 * 10))
    else:
        onset_minutes = None

    label = int(row.get("label", 0))
    phase_map = {0: None, 1: "preflare", 2: "impulsive"}
    flare_phase = phase_map.get(label)

    return {
        "timestamp": timestamp,
        "model_version": "v1.0.0",
        "data_quality": get_data_quality(row),
        "nowcast": {
            "flare_probability": round(nowcast_prob, 4),
            "flare_class": flare_class or "A",
            "confidence": round(max(proba), 4),
            "is_flare_active": pred_class == 2,
        },
        "forecast": {
            "flare_probability_30min": round(min(nowcast_prob * 1.1, 1.0), 4),
            "flare_probability_60min": round(min(nowcast_prob * 0.85, 1.0), 4),
            "predicted_class": flare_class or "A",
            "estimated_onset_minutes": onset_minutes,
        },
        "raw_features": {
            "slx_counts":        _safe(row.get("soft_xray")),
            "cdte_broadband":    _safe(row.get("cdte_broadband")),
            "czt_broadband":     _safe(row.get("czt_broadband")),
            "hardness_ratio":    _safe(row.get("hard_soft_ratio")),
            "hardness_smoothed": _safe(row.get("hard_soft_ratio")),
            "dCR_dt":            _safe(row.get("slx_d1")),
            "d2CR_dt2":          _safe(row.get("slx_d2")),
            "ema_60s":           _safe(row.get("slx_roll_mean_5m")),
            "ema_300s":          _safe(row.get("slx_roll_mean_30m")),
            "neupert_corr":      None,
            "flare_phase":       flare_phase,
            "photon_index_fit":  None,
        },
    }


def _safe(val):
    """Convert numpy float to Python float, return None if NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except:
        return None


def run_demo_replay():
    print(f"Loading model from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)

    print(f"Loading demo data for {DEMO_DATE}...")
    df = pd.read_parquet(DATA_PATH)
    demo = df[df["date"] == DEMO_DATE].copy()
    demo = demo.sort_values("utc").reset_index(drop=True)
    print(f"  {len(demo)} rows loaded for demo day")

    demo_sampled = demo.iloc[::60].reset_index(drop=True)    
    print(f"  {len(demo_sampled)} prediction steps to replay")

    demo_sampled = demo_sampled[
        demo_sampled["utc"] >= "2026-06-21 18:00:00+00:00"
    ].reset_index(drop=True)
    print(f"  Jumping to 18:00 UTC — {len(demo_sampled)} steps to flare peak")

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting replay → writing to {OUTPUT_PATH}")
    print("(Your backend scheduler will pick this up every 60s)")
    print("Press Ctrl+C to stop\n")

    for i, (_, row) in enumerate(demo_sampled.iterrows()):
        X = pd.DataFrame([row[FEATURE_COLS]])
        proba = pipeline.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))

        timestamp = pd.Timestamp(row["utc"]).isoformat()

        payload = build_prediction_json(row, proba, pred_class, timestamp)

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

        # If API_URL is configured, also POST the telemetry to the backend!
        if API_URL:
            try:
                res = requests.post(f"{API_URL.rstrip('/')}/api/telemetry", json=payload, timeout=5)
                api_status = f"HTTP POST: {res.status_code}"
            except Exception as e:
                api_status = f"HTTP POST Failed: {str(e)}"
        else:
            api_status = "Local File Only"

        label_name = CLASS_NAMES[pred_class]
        print(f"[{i+1}/{len(demo_sampled)}] {timestamp[:19]} | "
              f"pred={label_name:<10} | "
              f"flare_prob={proba[2]:.3f} | "
              f"{api_status}")

        time.sleep(1)

    print("\nReplay complete.")


if __name__ == "__main__":
    run_demo_replay()