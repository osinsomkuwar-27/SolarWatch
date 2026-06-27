from apscheduler.schedulers.background import BackgroundScheduler
from app.services.watcher import read_and_ingest, get_last_modified_at
import os
from dotenv import load_dotenv

load_dotenv()

_scheduler = BackgroundScheduler()
_running   = False

JSON_PATH = os.getenv("PREDICTION_JSON_PATH", "./mock/latest_prediction.json")
INTERVAL  = int(os.getenv("INFERENCE_INTERVAL_SECONDS", "60"))


def start():
    global _running
    _scheduler.add_job(
        func             = lambda: read_and_ingest(JSON_PATH),
        trigger          = "interval",
        seconds          = INTERVAL,
        id               = "prediction_ingest",
        replace_existing = True,
    )
    _scheduler.start()
    _running = True
    print(f"[Scheduler] Started — checking every {INTERVAL}s")


def stop():
    global _running
    _scheduler.shutdown()
    _running = False


def get_state() -> dict:
    return {
        "running":              _running,
        "last_file_modified_at": get_last_modified_at(),
    }