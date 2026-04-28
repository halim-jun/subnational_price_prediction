import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .data_loader import store
from .routers import geo, predictions, metrics, features
from .config import TARGETS, TARGET_DISPLAY, HORIZONS, GROUP_COLORS

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load_all()
    yield


app = FastAPI(title="EA Food Price Prediction API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(geo.router)
app.include_router(predictions.router)
app.include_router(metrics.router)
app.include_router(features.router)


@app.get("/api/config")
def get_config():
    return {
        "targets": TARGETS,
        "target_display": TARGET_DISPLAY,
        "horizons": HORIZONS,
        "group_colors": GROUP_COLORS,
        "cv_config": store.cv_config,
    }
