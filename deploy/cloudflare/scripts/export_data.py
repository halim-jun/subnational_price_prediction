"""Export model artifacts to static JSON for Cloudflare Pages deployment.

Reads from `artifact/model_output_holdout/`, `artifact/model_output_stcv/`,
and `data/geoboundaries/`, then writes flat JSON files into
`deploy/cloudflare/frontend/public/data/` for the static frontend to fetch.

Run from the project root:
    python deploy/cloudflare/scripts/export_data.py
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_data")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DEPLOY_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / "data/processed/subnational_merged_v3_KEN_SOM.parquet"
GEOBOUNDARIES_DIR = PROJECT_ROOT / "data/geoboundaries"
HOLDOUT_DIR = PROJECT_ROOT / "artifact/model_output_holdout"
STCV_DIR = PROJECT_ROOT / "artifact/model_output_stcv"

OUT_DIR = DEPLOY_DIR / "frontend/public/data"

# ── Domain config (mirrors src/dashboard/api/config.py) ───────────────────────
TARGETS = ["c_food_price_index", "c_maize_fao", "c_sorghum"]
TARGET_DISPLAY = {
    "c_food_price_index": "Food Price Index",
    "c_maize_fao": "Maize (FAO)",
    "c_sorghum": "Sorghum",
}
TARGET_FILE_KEY = {
    "c_food_price_index": "food_price_index",
    "c_maize_fao": "maize",
    "c_sorghum": "sorghum",
}
HORIZONS = [1, 2, 3]
GROUP_COLORS = {
    "Autoregressive": "#E91E63",
    "FLDAS": "#2196F3",
    "Vegetation": "#4CAF50",
    "Climate Index": "#FF9800",
    "Static": "#9C27B0",
    "Conflict": "#F44336",
    "Spatial ID": "#607D8B",
    "Temporal": "#795548",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def write_json(path: Path, payload, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent, allow_nan=False, default=_json_default)
    size_kb = path.stat().st_size / 1024
    log.info("wrote %s (%.1f KB)", path.relative_to(DEPLOY_DIR), size_kb)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Replace NaN/inf with None so JSON is strictly valid.

    pandas' ``where(notna, None)`` round-trips through numpy and silently
    re-introduces NaN for float columns, so we sanitize after to_dict.
    """
    records = df.replace([np.inf, -np.inf], np.nan).to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[k] = None
    return records


# ── Exporters ─────────────────────────────────────────────────────────────────
def export_geojson() -> None:
    """KEN+SOM admin2 boundaries, simplified, filtered to regions present in data."""
    gdfs = []
    for iso in ["KEN", "SOM"]:
        path = GEOBOUNDARIES_DIR / f"gb_{iso}_ADM2.geojson"
        if not path.exists():
            log.warning("missing geoboundary: %s", path)
            continue
        gdf = gpd.read_file(path)
        gdf["country_iso"] = iso
        gdfs.append(gdf)

    if not gdfs:
        raise FileNotFoundError("no geoboundary files found")

    combined = pd.concat(gdfs, ignore_index=True)

    # Filter to admin2 regions actually present in the merged dataset.
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH, columns=["admin2"])
        valid = set(df["admin2"].unique())
        combined = combined[combined["shapeName"].isin(valid)].copy()
    else:
        log.warning("merged parquet not found, exporting all admin2 boundaries")

    combined["geometry"] = combined["geometry"].simplify(
        tolerance=0.01, preserve_topology=True
    )

    geojson = json.loads(combined.to_json())
    write_json(OUT_DIR / "geo/admin2.json", geojson)


def export_config() -> None:
    cv_config = None
    cv_metrics_path = STCV_DIR / "cv_metrics.json"
    if cv_metrics_path.exists():
        try:
            with cv_metrics_path.open() as f:
                cv_config = json.load(f)
        except json.JSONDecodeError:
            log.warning("could not parse %s", cv_metrics_path)

    write_json(
        OUT_DIR / "config.json",
        {
            "targets": TARGETS,
            "target_display": TARGET_DISPLAY,
            "horizons": HORIZONS,
            "group_colors": GROUP_COLORS,
            "cv_config": cv_config,
        },
    )


def export_metrics() -> None:
    # Holdout metrics summary (preferred, mirrors API behavior).
    summary_df: pd.DataFrame | None = None
    holdout_path = HOLDOUT_DIR / "holdout_metrics.csv"
    if holdout_path.exists():
        summary_df = pd.read_csv(holdout_path)
    else:
        cv_path = STCV_DIR / "cv_aggregated_metrics.csv"
        if cv_path.exists():
            summary_df = pd.read_csv(cv_path)

    if summary_df is None:
        log.warning("no metrics summary available")
    else:
        write_json(
            OUT_DIR / "metrics/summary.json",
            {"data": df_to_records(summary_df)},
        )

    # CV fold results (full table).
    fold_path = STCV_DIR / "cv_fold_results.csv"
    if fold_path.exists():
        fold_df = pd.read_csv(fold_path)
        write_json(
            OUT_DIR / "metrics/fold-results.json",
            {"data": df_to_records(fold_df)},
        )


def export_per_admin() -> None:
    for target in TARGETS:
        for h in HORIZONS:
            key = TARGET_FILE_KEY[target]
            path = HOLDOUT_DIR / f"holdout_per_admin_{key}_h{h}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            write_json(
                OUT_DIR / f"metrics/per-admin/{target}_h{h}.json",
                {"data": df_to_records(df)},
            )


def export_predictions() -> None:
    """One JSON per (target, horizon) — feeds map/timeseries/scatter/regions."""
    path = HOLDOUT_DIR / "holdout_predictions.parquet"
    if not path.exists():
        log.warning("no holdout predictions parquet found at %s", path)
        return

    df = pd.read_parquet(path)
    df["error"] = df["predicted"] - df["actual"]
    df["date"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )

    cols = ["admin2_name", "country_name", "date", "actual", "predicted", "error"]

    for target in TARGETS:
        for h in HORIZONS:
            sub = df[(df["target"] == target) & (df["horizon"] == h)]
            if sub.empty:
                continue
            sub = sub[cols].sort_values(["admin2_name", "date"])
            regions = sorted(sub["admin2_name"].unique().tolist())
            write_json(
                OUT_DIR / f"predictions/{target}_h{h}.json",
                {
                    "target": target,
                    "horizon": h,
                    "regions": regions,
                    "data": df_to_records(sub),
                },
            )


def export_feature_importance() -> None:
    path = STCV_DIR / "cv_feature_group_importance.csv"
    if not path.exists():
        log.warning("no feature importance file at %s", path)
        return

    df = pd.read_csv(path)

    for target in TARGETS:
        for h in HORIZONS:
            sub = df[(df["target"] == target) & (df["horizon"] == h)]
            if sub.empty:
                continue
            stats = (
                sub.groupby("group")["importance"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            write_json(
                OUT_DIR / f"features/importance/{target}_h{h}.json",
                {"data": df_to_records(stats), "colors": GROUP_COLORS},
            )


def write_manifest() -> None:
    """Index every produced file with size, useful for debugging deployments."""
    files = []
    for p in sorted(OUT_DIR.rglob("*.json")):
        files.append(
            {
                "path": str(p.relative_to(OUT_DIR)),
                "bytes": p.stat().st_size,
            }
        )
    total = sum(f["bytes"] for f in files)
    write_json(
        OUT_DIR / "manifest.json",
        {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "total_bytes": total,
            "file_count": len(files),
            "files": files,
        },
        indent=2,
    )
    log.info("manifest: %d files, %.2f MB total", len(files), total / 1024 / 1024)


def main() -> None:
    log.info("export root: %s", OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    export_geojson()
    export_config()
    export_metrics()
    export_per_admin()
    export_predictions()
    export_feature_importance()
    write_manifest()

    log.info("done")


if __name__ == "__main__":
    main()
