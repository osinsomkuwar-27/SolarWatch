from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
class DataQuality(BaseModel):
    status: str
    reason: Optional[str] = None
    lookback_seconds_available: int
    lookback_seconds_required: int

class Nowcast(BaseModel):
    flare_probability: float
    flare_class: str
    confidence: float
    is_flare_active: bool

class Forecast(BaseModel):
    flare_probability_30min: float
    flare_probability_60min: float
    predicted_class: str
    estimated_onset_minutes: Optional[int] = None

class RawFeatures(BaseModel):
    slx_counts:        Optional[float] = None
    hardness_ratio:    Optional[float] = None
    hardness_smoothed: Optional[float] = None
    dCR_dt:            Optional[float] = None
    d2CR_dt2:          Optional[float] = None
    ema_60s:           Optional[float] = None
    ema_300s:          Optional[float] = None
    neupert_corr:      Optional[float] = None
    flare_phase:       Optional[str]   = None

    cdte_broadband:    Optional[float] = None
    czt_broadband:     Optional[float] = None
    photon_index_fit:  Optional[float] = None

class PredictionIn(BaseModel):
    timestamp:     datetime
    model_version: str = "v1.0.0"
    data_quality:  DataQuality
    nowcast:       Nowcast
    forecast:      Forecast
    raw_features:  RawFeatures

class PredictionOut(BaseModel):
    id:            int
    timestamp:     datetime
    model_version: str
    data_quality:  DataQuality
    nowcast:       Nowcast
    forecast:      Forecast
    raw_features:  RawFeatures
    created_at:    datetime

    class Config:
        from_attributes = True

class LightCurvePoint(BaseModel):
    timestamp:     datetime
    slx_counts:    Optional[float]
    cdte_broadband: Optional[float]
    czt_broadband:  Optional[float]
    hardness_ratio: Optional[float]
    flare_phase:    Optional[str]

    class Config:
        from_attributes = True

class StatusResponse(BaseModel):
    scheduler_running:       bool
    last_prediction_at:      Optional[datetime]
    last_file_modified_at:   Optional[datetime]
    total_predictions_today: int
    alert_active:            bool