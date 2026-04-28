from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..data_loader import store

router = APIRouter(prefix="/api/geo", tags=["geo"])


@router.get("/admin2")
def get_admin2_geojson():
    """Return simplified Admin2 GeoJSON for KEN + SOM."""
    if store.geojson is None:
        return JSONResponse(status_code=404, content={"error": "GeoJSON not loaded"})
    return store.geojson
