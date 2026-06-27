import json
import os
from datetime import datetime
from app.schemas import PredictionIn
from app.database import SessionLocal
from app import crud

_last_modified = None


def read_and_ingest(json_path: str):
    global _last_modified

    if not os.path.exists(json_path):
        print(f"[Watcher] File not found: {json_path}")
        return

    modified_at = os.path.getmtime(json_path)

    if _last_modified and modified_at <= _last_modified:
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        prediction = PredictionIn(**data)

        db = SessionLocal()
        try:
            crud.save_prediction(db, prediction)
            _last_modified = modified_at
            print(f"[Watcher] Ingested prediction at {prediction.timestamp}")
        finally:
            db.close()

    except Exception as e:
        print(f"[Watcher] Error ingesting prediction: {e}")


def get_last_modified_at():
    global _last_modified
    if _last_modified:
        return datetime.fromtimestamp(_last_modified)
    return None