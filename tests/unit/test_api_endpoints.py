import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend directory to sys.path to resolve 'app' imports correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from app.database import Base, get_db
import app.models  # Register models on Base metadata
from app.main import app

# Setup in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
connection = engine.connect()
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Apply the database session override to the FastAPI app
app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="function", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=connection)
    yield
    Base.metadata.drop_all(bind=connection)


@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Solar Flare Prediction API is running"}

def test_api_status_empty(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "scheduler_running" in data
    assert data["last_prediction_at"] is None
    assert data["total_predictions_today"] == 0
    assert data["alert_active"] is False

def test_predictions_empty(client):
    response = client.get("/api/predictions")
    assert response.status_code == 200
    assert response.json() == []

def test_lightcurve_empty(client):
    response = client.get("/api/lightcurve")
    assert response.status_code == 200
    assert response.json() == []

def test_latest_not_found(client):
    response = client.get("/api/latest")
    assert response.status_code == 404
    assert response.json()["detail"] == "No predictions yet"

def test_history_not_found(client):
    response = client.get("/api/history?date=2026-06-21")
    assert response.status_code == 404
    assert "No data for" in response.json()["detail"]

def test_ingestion_and_retrieval(client):
    prediction_payload = {
        "timestamp": "2026-06-21T19:25:00Z",
        "model_version": "v1.0.0",
        "data_quality": {
            "status": "ok",
            "reason": None,
            "lookback_seconds_available": 1800,
            "lookback_seconds_required": 1800
        },
        "nowcast": {
            "flare_probability": 0.87,
            "flare_class": "M",
            "confidence": 0.82,
            "is_flare_active": True
        },
        "forecast": {
            "flare_probability_30min": 0.73,
            "flare_probability_60min": 0.51,
            "predicted_class": "M",
            "estimated_onset_minutes": 12
        },
        "raw_features": {
            "slx_counts": 4200.0,
            "cdte_broadband": None,
            "czt_broadband": None,
            "hardness_ratio": 0.25,
            "hardness_smoothed": 0.23,
            "dCR_dt": 0.003,
            "d2CR_dt2": 0.0001,
            "ema_60s": 11.2,
            "ema_300s": 10.8,
            "neupert_corr": 0.12,
            "flare_phase": "impulsive",
            "photon_index_fit": None
        }
    }

    # Manually save the prediction payload to the test database
    db = TestingSessionLocal()
    from app.schemas import PredictionIn
    from app import crud
    pred_in = PredictionIn(**prediction_payload)
    crud.save_prediction(db, pred_in)
    db.close()

    # Verify latest prediction
    response = client.get("/api/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1.0.0"
    assert data["nowcast"]["flare_class"] == "M"
    assert data["nowcast"]["is_flare_active"] is True
    assert data["raw_features"]["slx_counts"] == 4200.0

    # Verify predictions listing
    response = client.get("/api/predictions")
    assert response.status_code == 200
    assert len(response.json()) == 1

    # Verify lightcurve endpoint
    response = client.get("/api/lightcurve")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["slx_counts"] == 4200.0

    # Verify status reflects prediction
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["alert_active"] is True
    assert data["last_prediction_at"] is not None

    # Verify historical predictions lookup
    response = client.get("/api/history?date=2026-06-21")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_telemetry_post_endpoint(client):
    prediction_payload = {
        "timestamp": "2026-06-21T19:25:00Z",
        "model_version": "v1.0.0",
        "data_quality": {
            "status": "ok",
            "reason": None,
            "lookback_seconds_available": 1800,
            "lookback_seconds_required": 1800
        },
        "nowcast": {
            "flare_probability": 0.87,
            "flare_class": "M",
            "confidence": 0.82,
            "is_flare_active": True
        },
        "forecast": {
            "flare_probability_30min": 0.73,
            "flare_probability_60min": 0.51,
            "predicted_class": "M",
            "estimated_onset_minutes": 12
        },
        "raw_features": {
            "slx_counts": 4200.0,
            "cdte_broadband": None,
            "czt_broadband": None,
            "hardness_ratio": 0.25,
            "hardness_smoothed": 0.23,
            "dCR_dt": 0.003,
            "d2CR_dt2": 0.0001,
            "ema_60s": 11.2,
            "ema_300s": 10.8,
            "neupert_corr": 0.12,
            "flare_phase": "impulsive",
            "photon_index_fit": None
        }
    }

    # Verify post request
    response = client.post("/api/telemetry", json=prediction_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1.0.0"
    assert data["nowcast"]["flare_class"] == "M"

    # Verify it can be retrieved from /api/latest
    response = client.get("/api/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["nowcast"]["flare_class"] == "M"
    assert data["raw_features"]["slx_counts"] == 4200.0


def test_double_slash_middleware(client):
    response = client.get("http://testserver//api/predictions")
    assert response.status_code == 200


