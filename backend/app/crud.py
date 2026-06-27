from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
from datetime import date
from app import models, schemas


def save_prediction(db: Session, prediction: schemas.PredictionIn) -> models.Prediction:
    db_pred = models.Prediction(
        timestamp                  = prediction.timestamp,
        model_version              = prediction.model_version,
        data_quality_status        = prediction.data_quality.status,
        data_quality_reason        = prediction.data_quality.reason,
        lookback_seconds_available = prediction.data_quality.lookback_seconds_available,
        lookback_seconds_required  = prediction.data_quality.lookback_seconds_required,
        nowcast_probability        = prediction.nowcast.flare_probability,
        nowcast_class              = prediction.nowcast.flare_class,
        nowcast_confidence         = prediction.nowcast.confidence,
        is_flare_active            = prediction.nowcast.is_flare_active,
        forecast_prob_30min        = prediction.forecast.flare_probability_30min,
        forecast_prob_60min        = prediction.forecast.flare_probability_60min,
        forecast_class             = prediction.forecast.predicted_class,
        estimated_onset_minutes    = prediction.forecast.estimated_onset_minutes,
        slx_counts                 = prediction.raw_features.slx_counts,
        hardness_ratio             = prediction.raw_features.hardness_ratio,
        hardness_smoothed          = prediction.raw_features.hardness_smoothed,
        dCR_dt                     = prediction.raw_features.dCR_dt,
        d2CR_dt2                   = prediction.raw_features.d2CR_dt2,
        ema_60s                    = prediction.raw_features.ema_60s,
        ema_300s                   = prediction.raw_features.ema_300s,
        neupert_corr               = prediction.raw_features.neupert_corr,
        flare_phase                = prediction.raw_features.flare_phase,
        cdte_broadband             = prediction.raw_features.cdte_broadband,
        czt_broadband              = prediction.raw_features.czt_broadband,
        photon_index_fit           = prediction.raw_features.photon_index_fit,
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)

    save_light_curve_point(db, prediction)
    return db_pred


def save_light_curve_point(db: Session, prediction: schemas.PredictionIn):
    lc = models.LightCurve(
        timestamp      = prediction.timestamp,
        slx_counts     = prediction.raw_features.slx_counts,
        cdte_broadband = prediction.raw_features.cdte_broadband,
        czt_broadband  = prediction.raw_features.czt_broadband,
        hardness_ratio = prediction.raw_features.hardness_ratio,
        flare_phase    = prediction.raw_features.flare_phase,
    )
    db.add(lc)
    db.commit()


def get_latest_prediction(db: Session) -> models.Prediction:
    return db.query(models.Prediction)\
             .order_by(models.Prediction.timestamp.desc())\
             .first()


def get_recent_predictions(db: Session, n: int = 100):
    return db.query(models.Prediction)\
             .order_by(models.Prediction.timestamp.desc())\
             .limit(n).all()


def get_recent_lightcurve(db: Session, n: int = 300):
    return db.query(models.LightCurve)\
             .order_by(models.LightCurve.timestamp.desc())\
             .limit(n).all()


def get_predictions_by_date(db: Session, target_date: date):
    return db.query(models.Prediction)\
             .filter(func.date(models.Prediction.timestamp) == target_date)\
             .order_by(models.Prediction.timestamp.asc())\
             .all()


def count_predictions_today(db: Session) -> int:
    today = date.today()
    return db.query(models.Prediction)\
             .filter(func.date(models.Prediction.timestamp) == today)\
             .count()