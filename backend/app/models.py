from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id                         = Column(Integer, primary_key=True, index=True)
    timestamp                  = Column(DateTime(timezone=True), index=True, nullable=False)
    model_version              = Column(String, default="v1.0.0")

    data_quality_status        = Column(String)
    data_quality_reason        = Column(String, nullable=True)
    lookback_seconds_available = Column(Integer)
    lookback_seconds_required  = Column(Integer)

    nowcast_probability        = Column(Float)
    nowcast_class              = Column(String)
    nowcast_confidence         = Column(Float)
    is_flare_active            = Column(Boolean)

    forecast_prob_30min        = Column(Float)
    forecast_prob_60min        = Column(Float)
    forecast_class             = Column(String)
    estimated_onset_minutes    = Column(Integer, nullable=True)

    slx_counts                 = Column(Float, nullable=True)
    hardness_ratio             = Column(Float, nullable=True)
    hardness_smoothed          = Column(Float, nullable=True)
    dCR_dt                     = Column(Float, nullable=True)
    d2CR_dt2                   = Column(Float, nullable=True)
    ema_60s                    = Column(Float, nullable=True)
    ema_300s                   = Column(Float, nullable=True)
    neupert_corr               = Column(Float, nullable=True)
    flare_phase                = Column(String, nullable=True)

    cdte_broadband             = Column(Float, nullable=True)
    czt_broadband              = Column(Float, nullable=True)
    photon_index_fit           = Column(Float, nullable=True)

    created_at                 = Column(DateTime(timezone=True), server_default=func.now())


class LightCurve(Base):
    __tablename__ = "light_curves"

    id               = Column(Integer, primary_key=True, index=True)
    timestamp        = Column(DateTime(timezone=True), index=True, nullable=False)
    slx_counts       = Column(Float, nullable=True)
    cdte_broadband   = Column(Float, nullable=True)
    czt_broadband    = Column(Float, nullable=True)
    hardness_ratio   = Column(Float, nullable=True)
    flare_phase      = Column(String, nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())