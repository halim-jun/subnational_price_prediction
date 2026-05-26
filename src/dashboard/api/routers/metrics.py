from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..data_loader import store

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/summary")
def get_summary(
    target: str = Query(None),
    horizon: int = Query(None),
):
    """Holdout test metrics."""
    df = store.country_holdout_metrics
    if df is None:
        df = store.holdout_metrics
    if df is None:
        # Fallback to CV metrics
        df = store.cv_aggregated
    if df is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    if target:
        df = df[df["target"] == target]
    if horizon:
        df = df[df["horizon"] == horizon]

    return {"data": df.to_dict(orient="records")}


@router.get("/per-admin")
def get_per_admin(
    target: str = Query(...),
    horizon: int = Query(...),
):
    """Per-admin MAPE table from holdout test."""
    df = store.get_per_admin(target, horizon)
    if df is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return {"data": df.to_dict(orient="records")}


@router.get("/fold-results")
def get_fold_results():
    """Full CV fold results (from Spatio-Temporal CV)."""
    df = store.cv_fold_results
    if df is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return {"data": df.to_dict(orient="records")}
