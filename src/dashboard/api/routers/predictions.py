from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..data_loader import store
from ..config import TARGETS, HORIZONS

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("/map/all")
def get_all_predictions(
    target: str = Query(..., description="Target variable"),
    horizon: int = Query(..., description="Forecast horizon"),
):
    """All monthly predictions for the map slider (fold-averaged)."""
    df = store.get_predictions(target, horizon)
    if df is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    records = df[
        ["admin2_name", "country_name", "date", "actual", "predicted", "error"]
    ].to_dict(orient="records")
    return {"data": records, "count": len(records)}


@router.get("/timeseries")
def get_timeseries(
    target: str = Query(...),
    horizon: int = Query(...),
    admin2_name: str = Query(...),
):
    """Monthly time series for one region."""
    df = store.get_predictions(target, horizon)
    if df is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    region = df[df["admin2_name"] == admin2_name].sort_values("date")
    records = region[
        ["date", "actual", "predicted", "error"]
    ].to_dict(orient="records")
    return {"admin2_name": admin2_name, "data": records}


@router.get("/scatter")
def get_scatter(
    target: str = Query(...),
    horizon: int = Query(...),
):
    """All (actual, predicted) points for scatter plot."""
    df = store.get_predictions(target, horizon)
    if df is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    records = df[
        ["admin2_name", "country_name", "date", "actual", "predicted"]
    ].to_dict(orient="records")
    return {"data": records}


@router.get("/regions")
def get_regions(
    target: str = Query(...),
    horizon: int = Query(...),
):
    """List of unique admin2 regions in predictions."""
    df = store.get_predictions(target, horizon)
    if df is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    regions = sorted(df["admin2_name"].unique().tolist())
    return {"regions": regions}
