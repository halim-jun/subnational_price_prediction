"""
Held-out 2024+ evaluation for month-over-month spike severity prediction.

Usage:
    python src/model/spike/train_holdout.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model import train_model_stcv as base_model
from src.model.spike.features import prepare_spike_frame
from src.model.spike.metrics import compute_spike_metrics
from src.model.spike.train_stcv import train_spike_xgb, _prediction_frame
from src.model.spike.tracking import build_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HOLDOUT_YEAR = 2024
PROJECT_ROOT = base_model.PROJECT_ROOT
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifact/spike_output_holdout")


def run_spike_holdout():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tracker = build_tracker(experiment_name="spike_prediction_holdout")
    df_base = base_model.load_base_data()
    df_base, label_encoders = base_model.encode_categoricals(df_base)

    metrics_rows = []
    prediction_frames = []

    validation_start = pd.Timestamp(year=HOLDOUT_YEAR, month=1, day=1)

    with tracker.start_run(run_name=f"holdout_{HOLDOUT_YEAR}"):
        tracker.log_params(
            {
                "workflow": "holdout",
                "holdout_year": HOLDOUT_YEAR,
                "targets": base_model.TARGETS,
                "horizons": base_model.HORIZONS,
                "objective": "multi:softprob",
                "loss": "class-weighted multiclass log loss",
            }
        )

        for target in base_model.TARGETS:
            for horizon in base_model.HORIZONS:
                logger.info("Preparing holdout spike frame: target=%s horizon=%s", target, horizon)
                df_spike, features, label_col, pct_col = prepare_spike_frame(
                    df_base, target=target, horizon=horizon
                )

                train_mask = (
                    (df_spike["base_date"] < validation_start)
                    & (df_spike["label_date"] < validation_start)
                )
                test_mask = df_spike["base_date"] >= validation_start

                X_train = df_spike.loc[train_mask, features]
                y_train = df_spike.loc[train_mask, label_col].astype(int)
                X_test = df_spike.loc[test_mask, features]
                y_test = df_spike.loc[test_mask, label_col].astype(int)

                if len(X_train) < 50 or len(X_test) < 10 or y_train.nunique() < 2:
                    logger.warning(
                        "Skipping target=%s horizon=%s train=%s test=%s classes=%s",
                        target,
                        horizon,
                        len(X_train),
                        len(X_test),
                        sorted(y_train.unique().tolist()),
                    )
                    continue

                model = train_spike_xgb(X_train, y_train)
                y_proba = model.predict_proba(X_test)
                y_pred = np.argmax(y_proba, axis=1)
                metrics = compute_spike_metrics(y_test.values, y_pred, y_proba)
                metrics_rows.append({"target": target, "horizon": horizon, **metrics})

                with tracker.start_run(
                    run_name=f"{target}_h{horizon}",
                    nested=True,
                ):
                    tracker.log_params(
                        {
                            "target": target,
                            "horizon": horizon,
                            "feature_count": len(features),
                            "train_rows": len(X_train),
                            "test_rows": len(X_test),
                            "train_label_rule": "label_date < holdout_start",
                        }
                    )
                    tracker.log_metrics(metrics)

                preds = _prediction_frame(
                    df_spike.loc[test_mask].copy(),
                    label_col,
                    pct_col,
                    y_pred,
                    y_proba,
                    label_encoders,
                )
                preds["target"] = target
                preds["horizon"] = horizon
                prediction_frames.append(preds)

    if not metrics_rows:
        raise RuntimeError("No spike holdout results produced.")

    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(OUTPUT_DIR, "spike_holdout_metrics.csv"), index=False
    )
    all_preds = pd.concat(prediction_frames, ignore_index=True)
    all_preds.to_parquet(os.path.join(OUTPUT_DIR, "spike_holdout_predictions.parquet"), index=False)

    per_admin_rows = []
    for keys, group in all_preds.groupby(["target", "horizon", "country_name", "admin2_name"]):
        row = {
            "target": keys[0],
            "horizon": keys[1],
            "country_name": keys[2],
            "admin2_name": keys[3],
            "mean_actual_pct_change": float(group["actual_pct_change"].mean()),
            **compute_spike_metrics(group["actual_class"], group["predicted_class"]),
        }
        per_admin_rows.append(row)
    pd.DataFrame(per_admin_rows).to_csv(
        os.path.join(OUTPUT_DIR, "spike_holdout_per_admin_metrics.csv"), index=False
    )

    with open(os.path.join(OUTPUT_DIR, "spike_holdout_config.json"), "w") as f:
        json.dump(
            {
                "holdout_year": HOLDOUT_YEAR,
                "targets": base_model.TARGETS,
                "horizons": base_model.HORIZONS,
                "train_label_rule": "train label_date must be before holdout_start",
                "objective": "multi:softprob",
            },
            f,
            indent=2,
        )
    with tracker.start_run(run_name=f"holdout_{HOLDOUT_YEAR}_summary"):
        tracker.log_params(
            {
                "workflow": "holdout_summary",
                "holdout_year": HOLDOUT_YEAR,
                "result_rows": len(metrics_rows),
            }
        )
        tracker.log_artifacts(
            [
                os.path.join(OUTPUT_DIR, "spike_holdout_metrics.csv"),
                os.path.join(OUTPUT_DIR, "spike_holdout_predictions.parquet"),
                os.path.join(OUTPUT_DIR, "spike_holdout_per_admin_metrics.csv"),
                os.path.join(OUTPUT_DIR, "spike_holdout_config.json"),
            ]
        )
    logger.info("Saved spike holdout artifacts to %s", OUTPUT_DIR)


if __name__ == "__main__":
    run_spike_holdout()
