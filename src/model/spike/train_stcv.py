"""
Spatio-temporal cross-validation for month-over-month spike severity prediction.

This script reuses the existing price-level model's input data, horizon-safe
feature engineering, and spatial/temporal fold generation. Spike-specific code
only creates future price-change labels, trains a weighted multiclass model,
and writes spike evaluation artifacts.

Usage:
    python src/model/spike/train_stcv.py
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model import train_model_stcv as base_model
from src.model.spike.features import prepare_spike_frame
from src.model.spike.metrics import compute_spike_metrics
from src.model.spike.sampling import balanced_sample_weight
from src.model.spike.splits import build_spike_stcv_masks
from src.model.spike.tracking import build_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = base_model.PROJECT_ROOT
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifact/spike_output_stcv")


@dataclass
class SpikeFoldResult:
    spatial_fold_idx: int
    temporal_fold_idx: int
    target: str
    horizon: int
    n_train: int
    n_val: int
    metrics: dict
    val_predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    confusion: pd.DataFrame


def train_spike_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=base_model.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=balanced_sample_weight(y_train), verbose=False)
    return model


def _prediction_frame(val_df, label_col, pct_col, y_pred, y_proba, label_encoders):
    preds = val_df[
        ["admin2", "country_iso", "year", "month", "base_date", "label_date", pct_col, label_col]
    ].copy()
    preds = preds.rename(columns={pct_col: "actual_pct_change", label_col: "actual_class"})
    preds["predicted_class"] = y_pred
    for class_idx in range(4):
        preds[f"prob_class_{class_idx}"] = y_proba[:, class_idx]
    preds["prob_gte_5pct"] = y_proba[:, 1:].sum(axis=1)
    preds["prob_gte_10pct"] = y_proba[:, 2:].sum(axis=1)
    preds["prob_gte_20pct"] = y_proba[:, 3]

    le_admin = label_encoders["admin2"]
    le_country = label_encoders["country_iso"]
    preds["admin2_name"] = le_admin.inverse_transform(preds["admin2"])
    preds["country_name"] = le_country.inverse_transform(preds["country_iso"])
    return preds


def run_spike_stcv_fold(
    df_prepared,
    features,
    label_col,
    pct_col,
    target,
    spatial_fold,
    temporal_fold,
    spatial_fold_idx,
    temporal_fold_idx,
    horizon,
    label_encoders,
):
    train_mask, val_mask = build_spike_stcv_masks(
        df_prepared,
        spatial_fold=spatial_fold,
        temporal_fold=temporal_fold,
        label_encoders=label_encoders,
    )

    X_train = df_prepared.loc[train_mask, features]
    y_train = df_prepared.loc[train_mask, label_col].astype(int)
    X_val = df_prepared.loc[val_mask, features]
    y_val = df_prepared.loc[val_mask, label_col].astype(int)

    if len(X_train) < 50 or len(X_val) < 10 or y_train.nunique() < 2:
        logger.warning(
            "Fold S%s/T%s skipped: train=%s val=%s train_classes=%s",
            spatial_fold_idx,
            temporal_fold_idx,
            len(X_train),
            len(X_val),
            sorted(y_train.unique().tolist()),
        )
        return None

    model = train_spike_xgb(X_train, y_train)
    y_proba = model.predict_proba(X_val)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = compute_spike_metrics(y_val.values, y_pred, y_proba)

    val_df = df_prepared.loc[val_mask].copy()
    for enc_val in sorted(val_df["country_iso"].unique()):
        country_mask = val_df["country_iso"].values == enc_val
        country_name = label_encoders["country_iso"].inverse_transform([enc_val])[0]
        country_metrics = compute_spike_metrics(
            y_val.values[country_mask],
            y_pred[country_mask],
            y_proba[country_mask],
        )
        for key in ["macro_f1", "recall_gte_10pct", "recall_gte_20pct"]:
            metrics[f"{country_name}_{key}"] = country_metrics[key]

    preds = _prediction_frame(val_df, label_col, pct_col, y_pred, y_proba, label_encoders)
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    importance["group"] = importance["feature"].apply(base_model._feature_group)
    cm = pd.DataFrame(
        confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3]),
        index=[f"actual_{i}" for i in range(4)],
        columns=[f"predicted_{i}" for i in range(4)],
    )

    return SpikeFoldResult(
        spatial_fold_idx=spatial_fold_idx,
        temporal_fold_idx=temporal_fold_idx,
        target=target,
        horizon=horizon,
        n_train=len(X_train),
        n_val=len(X_val),
        metrics=metrics,
        val_predictions=preds,
        feature_importance=importance,
        confusion=cm,
    )


def save_spike_stcv_artifacts(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    all_preds = []
    all_importance = []
    all_confusions = []

    for result in results:
        rows.append(
            {
                "target": result.target,
                "horizon": result.horizon,
                "spatial_fold": result.spatial_fold_idx,
                "temporal_fold": result.temporal_fold_idx,
                "n_train": result.n_train,
                "n_val": result.n_val,
                **result.metrics,
            }
        )
        preds = result.val_predictions.copy()
        preds["target"] = result.target
        preds["horizon"] = result.horizon
        preds["spatial_fold"] = result.spatial_fold_idx
        preds["temporal_fold"] = result.temporal_fold_idx
        all_preds.append(preds)

        importance = result.feature_importance.copy()
        importance["target"] = result.target
        importance["horizon"] = result.horizon
        importance["spatial_fold"] = result.spatial_fold_idx
        importance["temporal_fold"] = result.temporal_fold_idx
        all_importance.append(importance)

        confusion = result.confusion.copy()
        confusion["target"] = result.target
        confusion["horizon"] = result.horizon
        confusion["spatial_fold"] = result.spatial_fold_idx
        confusion["temporal_fold"] = result.temporal_fold_idx
        all_confusions.append(confusion.reset_index(names="actual_class"))

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(os.path.join(OUTPUT_DIR, "spike_cv_fold_results.csv"), index=False)

    agg = (
        fold_metrics.groupby(["target", "horizon"])
        .agg(
            n_folds=("n_val", "count"),
            total_val_samples=("n_val", "sum"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            recall_gte_10pct_mean=("recall_gte_10pct", "mean"),
            recall_gte_20pct_mean=("recall_gte_20pct", "mean"),
            mean_absolute_class_error_mean=("mean_absolute_class_error", "mean"),
            multiclass_log_loss_mean=("multiclass_log_loss", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(os.path.join(OUTPUT_DIR, "spike_cv_aggregated_metrics.csv"), index=False)

    pd.concat(all_preds, ignore_index=True).to_parquet(
        os.path.join(OUTPUT_DIR, "spike_cv_predictions.parquet"), index=False
    )
    pd.concat(all_importance, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "spike_cv_feature_importance.csv"), index=False
    )
    pd.concat(all_confusions, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "spike_cv_confusion_matrix.csv"), index=False
    )

    with open(os.path.join(OUTPUT_DIR, "spike_cv_config.json"), "w") as f:
        json.dump(
            {
                "targets": base_model.TARGETS,
                "horizons": base_model.HORIZONS,
                "objective": "multi:softprob",
                "loss": "class-weighted multiclass log loss",
                "train_label_rule": "train label_date must be before validation_start",
            },
            f,
            indent=2,
        )


def run_spike_stcv():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tracker = build_tracker(experiment_name="spike_prediction_stcv")
    logger.info("Loading existing base data and horizon-safe features...")
    df_base = base_model.load_base_data()

    rng = np.random.default_rng(base_model.RANDOM_SEED)
    centroids = base_model.load_admin2_centroids()
    spatial_folds = base_model.generate_spatial_folds(
        centroids,
        base_model.N_SPATIAL_FOLDS,
        base_model.BUFFER_RADIUS_KM,
        rng,
    )
    temporal_folds = base_model.generate_temporal_folds(df_base, base_model.N_TEMPORAL_FOLDS)
    df_base, label_encoders = base_model.encode_categoricals(df_base)

    results = []
    with tracker.start_run(run_name="stcv"):
        tracker.log_params(
            {
                "workflow": "stcv",
                "targets": base_model.TARGETS,
                "horizons": base_model.HORIZONS,
                "n_spatial_folds": len(spatial_folds),
                "n_temporal_folds": len(temporal_folds),
                "objective": "multi:softprob",
                "loss": "class-weighted multiclass log loss",
            }
        )
        for target in base_model.TARGETS:
            for horizon in base_model.HORIZONS:
                logger.info("Preparing spike frame: target=%s horizon=%s", target, horizon)
                df_spike, features, label_col, pct_col = prepare_spike_frame(
                    df_base, target=target, horizon=horizon
                )
                label_counts = df_spike[label_col].value_counts().sort_index().to_dict()
                logger.info("Label distribution: %s", label_counts)

                for s_idx, spatial_fold in enumerate(spatial_folds):
                    for t_idx, temporal_fold in enumerate(temporal_folds):
                        result = run_spike_stcv_fold(
                            df_spike,
                            features,
                            label_col,
                            pct_col,
                            target,
                            spatial_fold,
                            temporal_fold,
                            s_idx,
                            t_idx,
                            horizon,
                            label_encoders,
                        )
                        if result is not None:
                            results.append(result)
                            with tracker.start_run(
                                run_name=f"{target}_h{horizon}_s{s_idx}_t{t_idx}",
                                nested=True,
                            ):
                                tracker.log_params(
                                    {
                                        "target": target,
                                        "horizon": horizon,
                                        "spatial_fold": s_idx,
                                        "temporal_fold": t_idx,
                                        "feature_count": len(features),
                                        "train_rows": result.n_train,
                                        "val_rows": result.n_val,
                                    }
                                )
                                tracker.log_metrics(result.metrics)

    if not results:
        raise RuntimeError("No spike STCV folds produced results.")
    save_spike_stcv_artifacts(results)
    with tracker.start_run(run_name="stcv_summary"):
        tracker.log_params({"workflow": "stcv_summary", "result_folds": len(results)})
        tracker.log_artifacts(
            [
                os.path.join(OUTPUT_DIR, "spike_cv_fold_results.csv"),
                os.path.join(OUTPUT_DIR, "spike_cv_aggregated_metrics.csv"),
                os.path.join(OUTPUT_DIR, "spike_cv_predictions.parquet"),
                os.path.join(OUTPUT_DIR, "spike_cv_feature_importance.csv"),
                os.path.join(OUTPUT_DIR, "spike_cv_confusion_matrix.csv"),
                os.path.join(OUTPUT_DIR, "spike_cv_config.json"),
            ]
        )
    logger.info("Saved spike STCV artifacts to %s", OUTPUT_DIR)


if __name__ == "__main__":
    run_spike_stcv()
