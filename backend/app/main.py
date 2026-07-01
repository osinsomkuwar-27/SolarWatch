from fastapi import FastAPI, Request
import re
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


@app.middleware("http")
async def clean_double_slashes(request: Request, call_next):
    path = request.scope.get("path", "")
    if "//" in path:
        request.scope["path"] = re.sub(r"/+", "/", path)
    return await call_next(request)


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