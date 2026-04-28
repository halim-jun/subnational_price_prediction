"""
Held-Out Test Evaluation
========================
Train on all data before HOLDOUT_YEAR, predict on HOLDOUT_YEAR onwards.
This gives a realistic view of how the model performs on truly unseen future data.

Usage:
    python src/model/train_holdout.py
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Reuse everything from the STCV module
from train_model_stcv import (
    PROJECT_ROOT, DATA_PATH, OUTPUT_DIR,
    TARGETS, HORIZONS, CATEGORICAL_FEATURES,
    load_base_data, prepare_for_run, encode_categoricals, train_xgb,
    _feature_group, GROUP_COLORS,
)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HOLDOUT_YEAR = 2024
HOLDOUT_DIR = os.path.join(PROJECT_ROOT, "artifact/model_output_holdout")


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'n': 0}
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        'n': int(len(y_true)),
    }


def run_holdout():
    os.makedirs(HOLDOUT_DIR, exist_ok=True)

    # ── 1. Load & prepare data ──
    logger.info("Loading base data...")
    df_base = load_base_data()
    df_base, label_encoders = encode_categoricals(df_base)

    le_admin = label_encoders['admin2']
    le_country = label_encoders['country_iso']

    all_results = []
    all_predictions = []

    for target in TARGETS:
        for h in HORIZONS:
            logger.info(f"{'='*60}")
            logger.info(f"Target: {target} | Horizon: {h}")

            df_prepared, features, _ = prepare_for_run(df_base, target, h)

            # ── 2. Split: train < HOLDOUT_YEAR, test >= HOLDOUT_YEAR ──
            train_mask = df_prepared['year'] < HOLDOUT_YEAR
            test_mask = df_prepared['year'] >= HOLDOUT_YEAR

            X_train = df_prepared.loc[train_mask, features]
            y_train = df_prepared.loc[train_mask, target]
            X_test = df_prepared.loc[test_mask, features]
            y_test = df_prepared.loc[test_mask, target]

            logger.info(f"  Train: {len(X_train)} rows "
                        f"({df_prepared.loc[train_mask, 'year'].min()}-"
                        f"{df_prepared.loc[train_mask, 'year'].max()})")
            logger.info(f"  Test:  {len(X_test)} rows "
                        f"({df_prepared.loc[test_mask, 'year'].min()}-"
                        f"{df_prepared.loc[test_mask, 'year'].max()})")

            if len(X_train) < 50 or len(X_test) < 10:
                logger.warning("  Too few samples, skipping.")
                continue

            # ── 3. Train ──
            model = train_xgb(X_train, y_train)
            y_pred = model.predict(X_test)

            # ── 4. Overall metrics ──
            metrics = compute_metrics(y_test.values, y_pred)
            logger.info(f"  → R²={metrics['r2']:.4f}  MAPE={metrics['mape']:.2%}  "
                        f"RMSE={metrics['rmse']:.4f}  (n={metrics['n']})")

            # Per-country metrics
            test_df = df_prepared.loc[test_mask].copy()
            for enc_val in sorted(test_df['country_iso'].unique()):
                mask_c = test_df['country_iso'].values == enc_val
                if mask_c.sum() > 0:
                    name = le_country.inverse_transform([enc_val])[0]
                    country_metrics = compute_metrics(
                        y_test.values[mask_c], y_pred[mask_c]
                    )
                    metrics[f'{name}_mape'] = country_metrics['mape']
                    metrics[f'{name}_r2'] = country_metrics['r2']
                    logger.info(f"    {name}: R²={country_metrics['r2']:.4f}  "
                                f"MAPE={country_metrics['mape']:.2%}")

            all_results.append({
                'target': target, 'horizon': h, **metrics,
            })

            # ── 5. Save predictions ──
            preds_df = test_df[['year', 'month', 'admin2', 'country_iso']].copy()
            preds_df['actual'] = y_test.values
            preds_df['predicted'] = y_pred
            preds_df['admin2_name'] = le_admin.inverse_transform(preds_df['admin2'])
            preds_df['country_name'] = le_country.inverse_transform(
                preds_df['country_iso']
            )
            preds_df['target'] = target
            preds_df['horizon'] = h

            all_predictions.append(preds_df)

            # ── 6. Feature importance ──
            imp = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_,
            })
            imp['group'] = imp['feature'].apply(_feature_group)
            group_imp = imp.groupby('group')['importance'].sum().sort_values(
                ascending=False
            )

            logger.info("  Feature importance (by group):")
            for g, v in group_imp.items():
                logger.info(f"    {g}: {v:.4f}")

    # ── 7. Save all artifacts ──
    if not all_results:
        logger.error("No results!")
        return

    # Metrics summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(HOLDOUT_DIR, "holdout_metrics.csv"), index=False
    )

    # All predictions in one file
    all_preds = pd.concat(all_predictions, ignore_index=True)
    all_preds.to_parquet(
        os.path.join(HOLDOUT_DIR, "holdout_predictions.parquet"), index=False
    )

    # Per-admin performance
    for target in TARGETS:
        for h in HORIZONS:
            subset = all_preds[
                (all_preds['target'] == target) & (all_preds['horizon'] == h)
            ]
            if len(subset) == 0:
                continue
            subset = subset.copy()
            subset['abs_error'] = np.abs(subset['actual'] - subset['predicted'])
            subset['pct_error'] = (
                subset['abs_error'] / subset['actual'].clip(lower=1e-6)
            )
            admin_perf = (
                subset.groupby(['country_name', 'admin2_name'])
                .agg(
                    mae=('abs_error', 'mean'),
                    mape=('pct_error', 'mean'),
                    mean_actual=('actual', 'mean'),
                    n=('actual', 'count'),
                )
                .reset_index()
                .sort_values('mape', ascending=False)
            )
            display = target.replace('c_', '').replace('_fao', '')
            admin_perf.to_csv(
                os.path.join(HOLDOUT_DIR, f"holdout_per_admin_{display}_h{h}.csv"),
                index=False,
            )

    # Config JSON
    config = {
        'holdout_year': HOLDOUT_YEAR,
        'train_years': sorted(
            all_preds[all_preds['target'] == TARGETS[0]]['year'].min()
            for _ in [0]  # placeholder
        ),
        'test_years': sorted(all_preds['year'].unique().tolist()),
        'targets': TARGETS,
        'horizons': HORIZONS,
        'results': results_df.to_dict(orient='records'),
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(HOLDOUT_DIR, "holdout_config.json"), 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)

    # ── 8. Summary ──
    print("\n" + "=" * 70)
    print(f"HELD-OUT TEST COMPLETE (train < {HOLDOUT_YEAR}, test >= {HOLDOUT_YEAR})")
    print("=" * 70)

    header = f"{'Target':<22s} {'H':>3s} {'R²':>8s} {'MAPE':>8s} {'RMSE':>10s} {'N':>6s}"
    print(header)
    print("-" * len(header))

    for _, row in results_df.iterrows():
        display = row['target'].replace('c_', '').replace('_fao', '')
        print(f"{display:<22s} {int(row['horizon']):>3d} "
              f"{row['r2']:>8.4f} {row['mape']:>7.2%} "
              f"{row['rmse']:>10.4f} {int(row['n']):>6d}")

    print(f"\nArtifacts saved to: {HOLDOUT_DIR}/")


if __name__ == "__main__":
    run_holdout()
