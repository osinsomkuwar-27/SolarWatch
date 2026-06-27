from pydantic import BaseModel, model_validator
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

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm(cls, obj):
        return cls(
            id            = obj.id,
            timestamp     = obj.timestamp,
            model_version = obj.model_version,
            created_at    = obj.created_at,
            data_quality  = DataQuality(
                status                     = obj.data_quality_status,
                reason                     = obj.data_quality_reason,
                lookback_seconds_available = obj.lookback_seconds_available,
                lookback_seconds_required  = obj.lookback_seconds_required,
            ),
            nowcast = Nowcast(
                flare_probability = obj.nowcast_probability,
                flare_class       = obj.nowcast_class,
                confidence        = obj.nowcast_confidence,
                is_flare_active   = obj.is_flare_active,
            ),
            forecast = Forecast(
                flare_probability_30min = obj.forecast_prob_30min,
                flare_probability_60min = obj.forecast_prob_60min,
                predicted_class         = obj.forecast_class,
                estimated_onset_minutes = obj.estimated_onset_minutes,
            ),
            raw_features = RawFeatures(
                slx_counts        = obj.slx_counts,
                hardness_ratio    = obj.hardness_ratio,
                hardness_smoothed = obj.hardness_smoothed,
                dCR_dt            = obj.dCR_dt,
                d2CR_dt2          = obj.d2CR_dt2,
                ema_60s           = obj.ema_60s,
                ema_300s          = obj.ema_300s,
                neupert_corr      = obj.neupert_corr,
                flare_phase       = obj.flare_phase,
                cdte_broadband    = obj.cdte_broadband,
                czt_broadband     = obj.czt_broadband,
                photon_index_fit  = obj.photon_index_fit,
            ),
        )

class LightCurvePoint(BaseModel):
    timestamp:      datetime
    slx_counts:     Optional[float]
    cdte_broadband: Optional[float]
    czt_broadband:  Optional[float]
    hardness_ratio: Optional[float]
    flare_phase:    Optional[str]

    model_config = {"from_attributes": True}

class StatusResponse(BaseModel):
    scheduler_running:       bool
    last_prediction_at:      Optional[datetime]
    last_file_modified_at:   Optional[datetime]
    total_predictions_today: int
    alert_active:            bool