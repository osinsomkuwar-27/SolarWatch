from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import date
from typing import List
from app.database import get_db
from app import crud, schemas
from app.services import scheduler as scheduler_service

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/latest", response_model=schemas.PredictionOut)
def get_latest(db: Session = Depends(get_db)):
    pred = crud.get_latest_prediction(db)
    if not pred:
        raise HTTPException(status_code=404, detail="No predictions yet")
    return schemas.PredictionOut.from_orm(pred)


@router.get("/predictions", response_model=List[schemas.PredictionOut])
def get_predictions(
    n: int = Query(default=100, le=1440),
    db: Session = Depends(get_db)
):
    preds = crud.get_recent_predictions(db, n)
    return [schemas.PredictionOut.from_orm(p) for p in preds]


@router.get("/lightcurve", response_model=List[schemas.LightCurvePoint])
def get_lightcurve(
    n: int = Query(default=300, le=3600),
    db: Session = Depends(get_db)
):
    points = crud.get_recent_lightcurve(db, n)
    return list(reversed(points))


@router.get("/history", response_model=List[schemas.PredictionOut])
def get_history(
    date: date = Query(..., description="YYYY-MM-DD"),
    db: Session = Depends(get_db)
):
    results = crud.get_predictions_by_date(db, date)
    if not results:
        raise HTTPException(status_code=404, detail=f"No data for {date}")
    return [schemas.PredictionOut.from_orm(p) for p in results]


@router.get("/status", response_model=schemas.StatusResponse)
def get_status(db: Session = Depends(get_db)):
    latest = crud.get_latest_prediction(db)
    count  = crud.count_predictions_today(db)
    state  = scheduler_service.get_state()
    return schemas.StatusResponse(
        scheduler_running       = state["running"],
        last_prediction_at      = latest.created_at if latest else None,
        last_file_modified_at   = state["last_file_modified_at"],
        total_predictions_today = count,
        alert_active            = latest.is_flare_active if latest else False,
    )


@router.post("/telemetry", response_model=schemas.PredictionOut)
def post_telemetry(prediction: schemas.PredictionIn, db: Session = Depends(get_db)):
    try:
        db_pred = crud.save_prediction(db, prediction)
        return schemas.PredictionOut.from_orm(db_pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))