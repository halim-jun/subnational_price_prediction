import json
import os
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import (
    DATA_PATH,
    GEOBOUNDARIES_DIR,
    OUTPUT_DIR,
    HOLDOUT_DIR,
    TARGETS,
    TARGET_FILE_KEY,
    HORIZONS,
)

logger = logging.getLogger(__name__)


class DataStore:
    """Singleton that loads all data at startup and caches in memory."""

    def __init__(self):
        self.geojson: dict | None = None
        # Holdout predictions (primary — clean train/test split)
        self.predictions: dict[str, pd.DataFrame] = {}
        self.holdout_metrics: pd.DataFrame | None = None
        self.holdout_config: dict | None = None
        self.per_admin: dict[str, pd.DataFrame] = {}
        # CV results (secondary)
        self.cv_aggregated: pd.DataFrame | None = None
        self.cv_fold_results: pd.DataFrame | None = None
        self.feature_importance: pd.DataFrame | None = None
        self.cv_config: dict | None = None

    def load_all(self):
        logger.info("Loading all data...")
        self._load_geojson()
        self._load_holdout_predictions()
        self._load_holdout_metrics()
        self._load_per_admin()
        self._load_cv_metrics()
        self._load_feature_importance()
        self._load_cv_config()
        logger.info("All data loaded.")

    def _load_geojson(self):
        gdfs = []
        for iso in ["KEN", "SOM"]:
            path = os.path.join(GEOBOUNDARIES_DIR, f"gb_{iso}_ADM2.geojson")
            if os.path.exists(path):
                gdf = gpd.read_file(path)
                gdf["country_iso"] = iso
                gdfs.append(gdf)

        combined = pd.concat(gdfs, ignore_index=True)

        # Filter to admin2 regions in data
        if os.path.exists(DATA_PATH):
            df = pd.read_parquet(DATA_PATH, columns=["admin2"])
            valid = set(df["admin2"].unique())
            combined = combined[combined["shapeName"].isin(valid)].copy()

        combined["geometry"] = combined["geometry"].simplify(
            tolerance=0.01, preserve_topology=True
        )

        self.geojson = json.loads(combined.to_json())
        logger.info(f"Loaded GeoJSON: {len(combined)} admin2 features")

    def _load_holdout_predictions(self):
        """Load holdout predictions."""
        path = os.path.join(HOLDOUT_DIR, "holdout_predictions.parquet")
        if not os.path.exists(path):
            logger.warning("No holdout predictions found")
            return

        df = pd.read_parquet(path)

        for target in TARGETS:
            for h in HORIZONS:
                sub = df[(df["target"] == target) & (df["horizon"] == h)].copy()
                if len(sub) == 0:
                    continue

                sub["error"] = sub["predicted"] - sub["actual"]
                sub["date"] = (
                    sub["year"].astype(str)
                    + "-"
                    + sub["month"].astype(str).str.zfill(2)
                )

                cache_key = f"{target}_h{h}"
                self.predictions[cache_key] = sub
                logger.info(
                    f"Loaded holdout predictions: {cache_key} ({len(sub)} rows)"
                )

    def _load_holdout_metrics(self):
        path = os.path.join(HOLDOUT_DIR, "holdout_metrics.csv")
        if os.path.exists(path):
            self.holdout_metrics = pd.read_csv(path)

        path = os.path.join(HOLDOUT_DIR, "holdout_config.json")
        if os.path.exists(path):
            with open(path) as f:
                self.holdout_config = json.load(f)

    def _load_per_admin(self):
        for target in TARGETS:
            for h in HORIZONS:
                key = TARGET_FILE_KEY[target]
                path = os.path.join(
                    HOLDOUT_DIR, f"holdout_per_admin_{key}_h{h}.csv"
                )
                if os.path.exists(path):
                    cache_key = f"{target}_h{h}"
                    self.per_admin[cache_key] = pd.read_csv(path)

    def _load_cv_metrics(self):
        path = os.path.join(OUTPUT_DIR, "cv_aggregated_metrics.csv")
        if os.path.exists(path):
            self.cv_aggregated = pd.read_csv(path)

        path = os.path.join(OUTPUT_DIR, "cv_fold_results.csv")
        if os.path.exists(path):
            self.cv_fold_results = pd.read_csv(path)

    def _load_feature_importance(self):
        path = os.path.join(OUTPUT_DIR, "cv_feature_group_importance.csv")
        if os.path.exists(path):
            self.feature_importance = pd.read_csv(path)

    def _load_cv_config(self):
        path = os.path.join(OUTPUT_DIR, "cv_metrics.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.cv_config = json.load(f)
            except json.JSONDecodeError:
                self.cv_config = None

    def get_predictions(self, target: str, horizon: int) -> pd.DataFrame | None:
        return self.predictions.get(f"{target}_h{horizon}")

    def get_per_admin(self, target: str, horizon: int) -> pd.DataFrame | None:
        return self.per_admin.get(f"{target}_h{horizon}")


store = DataStore()
