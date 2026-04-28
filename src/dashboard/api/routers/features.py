from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..data_loader import store
from ..config import GROUP_COLORS

router = APIRouter(prefix="/api/features", tags=["features"])


@router.get("/importance")
def get_importance(
    target: str = Query(...),
    horizon: int = Query(...),
):
    """Feature group importance averaged across folds."""
    df = store.feature_importance
    if df is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    sub = df[(df["target"] == target) & (df["horizon"] == horizon)]
    if len(sub) == 0:
        return {"data": [], "colors": GROUP_COLORS}

    group_stats = (
        sub.groupby("group")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    records = group_stats.to_dict(orient="records")
    return {"data": records, "colors": GROUP_COLORS}
