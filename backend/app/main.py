from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.routes.api import router
from app.services import scheduler

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Solar Flare Prediction API",
    description="BAH 2026 PS-15 — Aditya-L1 flare nowcast and forecast",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    from app.services.watcher import read_and_ingest
    from app.services.scheduler import JSON_PATH
    read_and_ingest(JSON_PATH)
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    scheduler.stop()


@app.get("/")
def root():
    return {"message": "Solar Flare Prediction API is running"}