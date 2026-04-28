"""
Spatio-Temporal Cross-Validation for Food Price Prediction
==========================================================

Cross-validation strategy:
  Spatial:  Leave-Disc-Out — for each fold, a random center point is selected
            and all Admin2 regions whose centroids fall within a buffer radius
            are held out.  This reduces spatial autocorrelation leakage.
  Temporal: Expanding-window forward-only — within each fold the validation
            period is always in the *future* relative to training data, so
            no temporal leakage occurs.

The two dimensions are crossed: each CV fold is a (spatial_fold, temporal_fold)
pair.  Training uses regions OUTSIDE the disc AND time BEFORE the cutoff.
Validation uses regions INSIDE the disc AND time AT/AFTER the cutoff.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/subnational_merged_v3_KEN_SOM.parquet")
GEOBOUNDARIES_DIR = os.path.join(PROJECT_ROOT, "data/geoboundaries")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifact/model_output_stcv")

TARGETS = ['c_food_price_index', 'c_maize_fao', 'c_sorghum']
HORIZONS = [1, 2, 3]

# ── Feature groups (same as train_model.py) ──────────────────────────────────
FLDAS_FEATURES = [
    'Tair_f_3mon_Z', 'SoilMoi00_10cm_3mon_Z', 'SoilMoi10_40cm_3mon_Z',
    'Rainf_f_3mon_Z', 'Evap_3mon_Z', 'Qs_3mon_Z', 'Qsb_3mon_Z',
    'Water_Balance_3mon_Z',
]
VEGETATION_FEATURES = ['NDVI', 'LST', 'VHI']
CLIMATE_INDICES = ['NINO34_Anom', 'IOD_DMI', 'Western_V_Gradient', 'MEI_v2']
STATIC_FEATURES = ['population', 'crop_cover_fraction']
CONFLICT_FEATURES = ['conflict_fatalities', 'conflict_events']
CATEGORICAL_FEATURES = ['country_iso', 'admin2']
TEMPORAL_FEATURES = ['month_sin', 'month_cos']

KNOWN_AT_PREDICTION = TEMPORAL_FEATURES + STATIC_FEATURES + CATEGORICAL_FEATURES
NEEDS_LAG_FEATURES = CONFLICT_FEATURES + FLDAS_FEATURES + VEGETATION_FEATURES + CLIMATE_INDICES
ALL_LAGS = [1, 2, 3, 6, 12]
MIN_LAG_WARMUP = 12

# ── CV configuration ────────────────────────────────────────────────────────
N_SPATIAL_FOLDS = 5       # number of disc-based spatial holdout folds
N_TEMPORAL_FOLDS = 3      # number of expanding-window temporal folds
BUFFER_RADIUS_KM = 350    # radius of the spatial holdout disc
MIN_VAL_REGIONS = 8       # minimum admin2 in validation disc
MAX_VAL_FRACTION = 0.40   # max fraction of admin2 in validation (don't hold out too many)
RANDOM_SEED = 42

COUNTRIES = ['KEN', 'SOM']


# ═══════════════════════════════════════════════════════════════════════════
#  Spatial utilities
# ═══════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_admin2_centroids():
    """Load Admin2 centroids from GeoBoundaries GeoJSON, filtered to those in the dataset."""
    df = pd.read_parquet(DATA_PATH)
    parquet_admins = {iso: set(df[df['country_iso'] == iso]['admin2'].unique())
                      for iso in COUNTRIES}

    records = []
    for iso in COUNTRIES:
        path = os.path.join(GEOBOUNDARIES_DIR, f"gb_{iso}_ADM2.geojson")
        gdf = gpd.read_file(path)
        # Project to a suitable projected CRS for centroid accuracy (UTM zone 37N covers East Africa)
        gdf_proj = gdf.to_crs(epsg=32637)
        gdf_proj['centroid'] = gdf_proj.geometry.centroid
        # Convert centroids back to WGS84 for lat/lon
        centroids_wgs84 = gpd.GeoSeries(gdf_proj['centroid'], crs='EPSG:32637').to_crs(epsg=4326)

        for idx, row in gdf.iterrows():
            name = row['shapeName']
            if name in parquet_admins[iso]:
                c = centroids_wgs84.iloc[idx]
                records.append({
                    'admin2': name,
                    'country_iso': iso,
                    'lat': c.y,
                    'lon': c.x,
                })

    centroids_df = pd.DataFrame(records)
    logger.info(f"Loaded {len(centroids_df)} admin2 centroids "
                f"(KEN={len(centroids_df[centroids_df.country_iso=='KEN'])}, "
                f"SOM={len(centroids_df[centroids_df.country_iso=='SOM'])})")
    return centroids_df


def generate_spatial_folds(centroids_df, n_folds, radius_km, rng):
    """
    Generate spatial folds using the Leave-Disc-Out approach.

    For each fold, pick a random center (from existing admin2 centroids),
    and hold out all admin2 within `radius_km` of that center.

    Returns list of dicts with keys:
      'val_admin2': set of admin2 names in the validation disc
      'train_admin2': set of admin2 names outside the disc
      'center_lat', 'center_lon': disc center coordinates
      'center_admin2': name of the center admin2
    """
    lats = centroids_df['lat'].values
    lons = centroids_df['lon'].values
    names = centroids_df['admin2'].values
    n_total = len(centroids_df)

    folds = []
    used_centers = set()
    max_attempts_per_fold = 50

    for fold_i in range(n_folds):
        for attempt in range(max_attempts_per_fold):
            # Pick a random admin2 centroid as center
            center_idx = rng.integers(0, n_total)
            center_name = names[center_idx]

            if center_name in used_centers:
                continue

            clat, clon = lats[center_idx], lons[center_idx]

            # Compute distances from center to all admin2 centroids
            dists = haversine_km(clat, clon, lats, lons)

            val_mask = dists <= radius_km
            n_val = val_mask.sum()

            # Check constraints
            if n_val < MIN_VAL_REGIONS:
                continue
            if n_val / n_total > MAX_VAL_FRACTION:
                continue

            val_admin2 = set(names[val_mask])
            train_admin2 = set(names[~val_mask])

            # Check no excessive overlap with previous folds' validation sets
            # Allow some overlap but not identical folds
            is_duplicate = False
            for prev in folds:
                overlap = len(val_admin2 & prev['val_admin2'])
                if overlap / len(val_admin2) > 0.7:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            used_centers.add(center_name)
            folds.append({
                'val_admin2': val_admin2,
                'train_admin2': train_admin2,
                'center_lat': clat,
                'center_lon': clon,
                'center_admin2': center_name,
                'n_val': n_val,
                'n_train': n_total - n_val,
            })
            logger.info(f"  Spatial fold {fold_i}: center={center_name} "
                        f"({clat:.2f}, {clon:.2f}), "
                        f"val={n_val} admin2, train={n_total - n_val} admin2")
            break
        else:
            logger.warning(f"  Could not find valid disc for spatial fold {fold_i} "
                           f"after {max_attempts_per_fold} attempts. "
                           f"Reducing MIN_VAL_REGIONS constraint.")
            # Relax constraint and try again
            for attempt in range(max_attempts_per_fold):
                center_idx = rng.integers(0, n_total)
                clat, clon = lats[center_idx], lons[center_idx]
                dists = haversine_km(clat, clon, lats, lons)
                val_mask = dists <= radius_km
                n_val = val_mask.sum()
                if n_val >= 3 and n_val / n_total <= MAX_VAL_FRACTION:
                    val_admin2 = set(names[val_mask])
                    train_admin2 = set(names[~val_mask])
                    folds.append({
                        'val_admin2': val_admin2,
                        'train_admin2': train_admin2,
                        'center_lat': clat,
                        'center_lon': clon,
                        'center_admin2': names[center_idx],
                        'n_val': n_val,
                        'n_train': n_total - n_val,
                    })
                    logger.info(f"  Spatial fold {fold_i} (relaxed): "
                                f"center={names[center_idx]}, val={n_val}")
                    break

    if len(folds) < n_folds:
        logger.warning(f"Only generated {len(folds)}/{n_folds} spatial folds")

    return folds


def generate_temporal_folds(df, n_folds):
    """
    Generate expanding-window temporal folds.

    Each fold has a cutoff year: train on data before cutoff, validate on data at/after cutoff.
    The cutoff years are spaced to give roughly equal validation periods.

    Returns list of dicts with keys:
      'train_years': range of training years (< cutoff)
      'val_years': range of validation years (>= cutoff)
      'cutoff_year': the year threshold
    """
    all_years = sorted(df['year'].unique())
    min_year = min(all_years)
    max_year = max(all_years)

    # Need minimum training period (at least 5 years + warmup)
    min_train_years = 5
    earliest_cutoff = min_year + min_train_years + 1  # +1 for warmup lag

    # Need minimum validation period (at least 2 years)
    min_val_years = 2
    latest_cutoff = max_year - min_val_years + 1

    if earliest_cutoff > latest_cutoff:
        logger.warning("Not enough years for requested temporal folds. Using single fold.")
        return [{'cutoff_year': latest_cutoff,
                 'train_years': [y for y in all_years if y < latest_cutoff],
                 'val_years': [y for y in all_years if y >= latest_cutoff]}]

    # Space cutoff years evenly
    cutoff_years = np.linspace(earliest_cutoff, latest_cutoff, n_folds, dtype=int)
    # Ensure unique cutoff years
    cutoff_years = sorted(set(cutoff_years))

    folds = []
    for cutoff in cutoff_years:
        train_years = [y for y in all_years if y < cutoff]
        val_years = [y for y in all_years if y >= cutoff]
        folds.append({
            'cutoff_year': int(cutoff),
            'train_years': train_years,
            'val_years': val_years,
        })
        logger.info(f"  Temporal fold: train {min(train_years)}-{max(train_years)}, "
                    f"val {min(val_years)}-{max(val_years)} (cutoff={cutoff})")

    return folds


# ═══════════════════════════════════════════════════════════════════════════
#  Feature engineering (reused from train_model.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_all_lag_features(df, target):
    """Build lag features for ALL possible lags."""
    df = df.sort_values(['admin2', 'year', 'month']).copy()

    for m in ALL_LAGS:
        df[f'{target}_lag_{m}'] = df.groupby('admin2')[target].shift(m)

    for shift in [1, 2, 3]:
        prefix = f'{target}_s{shift}'
        shifted = df.groupby('admin2')[target].shift(shift)
        df[f'{prefix}_rmean3'] = shifted.groupby(df['admin2']).transform(
            lambda x: x.rolling(3, min_periods=2).mean()
        )
        df[f'{prefix}_rmean12'] = shifted.groupby(df['admin2']).transform(
            lambda x: x.rolling(12, min_periods=6).mean()
        )
        df[f'{prefix}_rstd12'] = shifted.groupby(df['admin2']).transform(
            lambda x: x.rolling(12, min_periods=6).std()
        )

    df[f'{target}_yoy'] = df[f'{target}_lag_12'] - df.groupby('admin2')[target].shift(13)
    return df


def get_lag_features_for_horizon(target, horizon):
    lag_feats = [f'{target}_lag_{m}' for m in ALL_LAGS if m >= horizon]
    roll_feats = []
    for shift in [1, 2, 3]:
        if shift >= horizon:
            prefix = f'{target}_s{shift}'
            roll_feats += [f'{prefix}_rmean3', f'{prefix}_rmean12', f'{prefix}_rstd12']
    if horizon <= 12:
        roll_feats.append(f'{target}_yoy')
    return lag_feats + roll_feats


def get_exog_features_for_horizon(horizon):
    known = list(KNOWN_AT_PREDICTION)
    lagged = [f'{feat}_Lh{horizon}' for feat in NEEDS_LAG_FEATURES]
    return known + lagged


def load_base_data():
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded: {df.shape}")

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df = df.sort_values(['admin2', 'year', 'month']).reset_index(drop=True)

    for target in TARGETS:
        df = build_all_lag_features(df, target)

    for h in HORIZONS:
        for feat in NEEDS_LAG_FEATURES:
            col_name = f'{feat}_Lh{h}'
            df[col_name] = df.groupby('admin2')[feat].shift(h)

    return df


def prepare_for_run(df_base, target, horizon):
    df = df_base.copy()
    price_lag_feats = get_lag_features_for_horizon(target, horizon)
    exog_feats = get_exog_features_for_horizon(horizon)
    features = exog_feats + price_lag_feats

    required_cols = features + [target]
    before = len(df)
    df = df.dropna(subset=required_cols)
    logger.info(f"  [{target} h={horizon}] Dropped {before - len(df)} rows for warmup, "
                f"{len(df)} remaining")

    return df, features, price_lag_feats


def encode_categoricals(df):
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    # Use last 10% of training data as early-stopping validation
    # (sorted temporally within training set)
    split_idx = int(len(X_train) * 0.9)
    if split_idx < 10 or (len(X_train) - split_idx) < 10:
        # Too few samples for early stopping split — train without it
        model.set_params(early_stopping_rounds=None)
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(
            X_train.iloc[:split_idx], y_train.iloc[:split_idx],
            eval_set=[(X_train.iloc[split_idx:], y_train.iloc[split_idx:])],
            verbose=False,
        )
    return model


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


def _feature_group(f):
    import re
    base = re.sub(r'_Lh\d+$', '', f)
    if base in FLDAS_FEATURES: return 'FLDAS'
    if base in VEGETATION_FEATURES: return 'Vegetation'
    if base in CLIMATE_INDICES: return 'Climate Index'
    if base in STATIC_FEATURES: return 'Static'
    if base in CONFLICT_FEATURES: return 'Conflict'
    if f in CATEGORICAL_FEATURES: return 'Spatial ID'
    if f in TEMPORAL_FEATURES: return 'Temporal'
    if 'lag' in f or 'rmean' in f or 'rstd' in f or 'yoy' in f: return 'Autoregressive'
    return 'Other'


GROUP_COLORS = {
    'Autoregressive': '#E91E63', 'FLDAS': '#2196F3', 'Vegetation': '#4CAF50',
    'Climate Index': '#FF9800', 'Static': '#9C27B0', 'Conflict': '#F44336',
    'Spatial ID': '#607D8B', 'Temporal': '#795548',
}


# ═══════════════════════════════════════════════════════════════════════════
#  Spatio-Temporal CV runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    """Results from one CV fold."""
    spatial_fold_idx: int
    temporal_fold_idx: int
    target: str
    horizon: int
    n_train: int
    n_val: int
    metrics: dict
    val_predictions: pd.DataFrame  # actual, predicted, admin2, year, month
    feature_importance: pd.DataFrame


def run_stcv_fold(df_prepared, features, target,
                  spatial_fold, temporal_fold,
                  spatial_fold_idx, temporal_fold_idx,
                  horizon, label_encoders):
    """
    Run a single spatio-temporal CV fold.

    Train on:  admin2 OUTSIDE disc, year < cutoff
    Validate:  admin2 INSIDE disc,  year >= cutoff
    """
    le_admin = label_encoders['admin2']
    le_country = label_encoders['country_iso']

    # Map admin2 names to encoded values
    val_admin2_encoded = set(le_admin.transform(
        [a for a in spatial_fold['val_admin2'] if a in le_admin.classes_]
    ))
    train_admin2_encoded = set(le_admin.transform(
        [a for a in spatial_fold['train_admin2'] if a in le_admin.classes_]
    ))

    cutoff = temporal_fold['cutoff_year']

    # Build masks
    train_mask = (
        df_prepared['admin2'].isin(train_admin2_encoded) &
        (df_prepared['year'] < cutoff)
    )
    val_mask = (
        df_prepared['admin2'].isin(val_admin2_encoded) &
        (df_prepared['year'] >= cutoff)
    )

    X_train = df_prepared.loc[train_mask, features]
    y_train = df_prepared.loc[train_mask, target]
    X_val = df_prepared.loc[val_mask, features]
    y_val = df_prepared.loc[val_mask, target]

    if len(X_train) < 50 or len(X_val) < 10:
        logger.warning(f"  Fold S{spatial_fold_idx}/T{temporal_fold_idx}: "
                       f"too few samples (train={len(X_train)}, val={len(X_val)}). Skipping.")
        return None

    # Train
    model = train_xgb(X_train, y_train)
    y_pred = model.predict(X_val)

    # Metrics
    metrics = compute_metrics(y_val.values, y_pred)

    # Per-country metrics
    val_df = df_prepared.loc[val_mask].copy()
    for enc_val in sorted(val_df['country_iso'].unique()):
        mask_c = val_df['country_iso'].values == enc_val
        if mask_c.sum() > 0:
            name = le_country.inverse_transform([enc_val])[0]
            metrics[f'{name}_mape'] = float(
                mean_absolute_percentage_error(y_val.values[mask_c], y_pred[mask_c])
            )

    # Predictions dataframe
    val_predictions = val_df[['admin2', 'country_iso', 'year', 'month']].copy()
    val_predictions['actual'] = y_val.values
    val_predictions['predicted'] = y_pred
    val_predictions['admin2_name'] = le_admin.inverse_transform(val_predictions['admin2'])
    val_predictions['country_name'] = le_country.inverse_transform(val_predictions['country_iso'])

    # Feature importance
    imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    imp['group'] = imp['feature'].apply(_feature_group)

    return FoldResult(
        spatial_fold_idx=spatial_fold_idx,
        temporal_fold_idx=temporal_fold_idx,
        target=target,
        horizon=horizon,
        n_train=len(X_train),
        n_val=len(X_val),
        metrics=metrics,
        val_predictions=val_predictions,
        feature_importance=imp,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Artifact generators
# ═══════════════════════════════════════════════════════════════════════════

def save_cv_summary(all_fold_results):
    """Save per-fold and aggregated CV metrics."""
    rows = []
    for fr in all_fold_results:
        rows.append({
            'target': fr.target,
            'horizon': fr.horizon,
            'spatial_fold': fr.spatial_fold_idx,
            'temporal_fold': fr.temporal_fold_idx,
            'n_train': fr.n_train,
            'n_val': fr.n_val,
            **fr.metrics,
        })

    df_folds = pd.DataFrame(rows)
    df_folds.to_csv(os.path.join(OUTPUT_DIR, "cv_fold_results.csv"), index=False)

    # Aggregated: mean ± std across folds per target × horizon
    agg_rows = []
    for target in TARGETS:
        for h in HORIZONS:
            subset = df_folds[(df_folds['target'] == target) & (df_folds['horizon'] == h)]
            if len(subset) == 0:
                continue
            row = {
                'target': target, 'horizon': h, 'n_folds': len(subset),
                'total_val_samples': int(subset['n_val'].sum()),
            }
            for metric in ['rmse', 'mae', 'mape', 'r2']:
                vals = subset[metric].dropna()
                row[f'{metric}_mean'] = float(vals.mean())
                row[f'{metric}_std'] = float(vals.std())
                row[f'{metric}_min'] = float(vals.min())
                row[f'{metric}_max'] = float(vals.max())
            agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, "cv_aggregated_metrics.csv"), index=False)
    logger.info("Saved CV summary")
    return df_folds, df_agg


def save_cv_boxplot(df_folds):
    """Boxplots of CV metrics across folds per target × horizon."""
    metrics_to_plot = ['r2', 'mape', 'rmse']
    fig, axes = plt.subplots(len(TARGETS), len(metrics_to_plot),
                             figsize=(6 * len(metrics_to_plot), 5 * len(TARGETS)))

    for i, target in enumerate(TARGETS):
        for j, metric in enumerate(metrics_to_plot):
            ax = axes[i][j] if len(TARGETS) > 1 else axes[j]
            data_by_h = []
            labels = []
            for h in HORIZONS:
                subset = df_folds[(df_folds['target'] == target) & (df_folds['horizon'] == h)]
                vals = subset[metric].dropna().values
                if len(vals) > 0:
                    data_by_h.append(vals)
                    labels.append(f'h={h}')

            if data_by_h:
                bp = ax.boxplot(data_by_h, tick_labels=labels, patch_artist=True)
                colors = ['#2196F3', '#FF9800', '#F44336']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)
                # Overlay individual fold points
                for k, vals in enumerate(data_by_h):
                    x = np.random.normal(k + 1, 0.04, size=len(vals))
                    ax.scatter(x, vals, alpha=0.7, s=20, zorder=5, color=colors[k])

            display = target.replace('c_', '').replace('_fao', '')
            label_name = metric.upper()
            ax.set_title(f'{display}: {label_name}', fontsize=10)
            ax.set_ylabel(label_name)
            ax.grid(True, alpha=0.3)
            if metric == 'mape':
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    fig.suptitle('Spatio-Temporal CV: Metric Distribution Across Folds', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cv_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved CV boxplot")


def save_spatial_folds_map(centroids_df, spatial_folds):
    """Visualize the spatial folds on a map."""
    n_folds = len(spatial_folds)
    fig, axes = plt.subplots(1, n_folds, figsize=(6 * n_folds, 8))
    if n_folds == 1:
        axes = [axes]

    for ax, (fold_idx, fold) in zip(axes, enumerate(spatial_folds)):
        # Plot all admin2 centroids
        for _, row in centroids_df.iterrows():
            is_val = row['admin2'] in fold['val_admin2']
            color = '#F44336' if is_val else '#2196F3'
            marker = 'o' if row['country_iso'] == 'KEN' else 's'
            ax.scatter(row['lon'], row['lat'], c=color, marker=marker,
                       s=30, alpha=0.7, edgecolors='white', linewidths=0.5)

        # Draw the disc
        theta = np.linspace(0, 2 * np.pi, 100)
        # Approximate: 1 degree ≈ 111 km
        r_deg = BUFFER_RADIUS_KM / 111.0
        clat, clon = fold['center_lat'], fold['center_lon']
        # Adjust longitude radius for latitude
        r_lon = r_deg / np.cos(np.radians(clat))
        ax.plot(clon + r_lon * np.cos(theta), clat + r_deg * np.sin(theta),
                'r--', alpha=0.5, linewidth=1.5)
        ax.plot(clon, clat, 'r*', markersize=15)

        ax.set_title(f"Fold {fold_idx}: {fold['center_admin2']}\n"
                     f"val={fold['n_val']}, train={fold['n_train']}", fontsize=9)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
               markersize=8, label='Train (KEN)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2196F3',
               markersize=8, label='Train (SOM)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
               markersize=8, label='Val (KEN)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F44336',
               markersize=8, label='Val (SOM)'),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=7, loc='lower right')

    fig.suptitle(f'Spatial CV Folds (Leave-Disc-Out, r={BUFFER_RADIUS_KM}km)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "spatial_folds_map.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved spatial folds map")


def save_temporal_folds_diagram(temporal_folds, all_years):
    """Visualize temporal folds as a Gantt-like chart."""
    fig, ax = plt.subplots(figsize=(14, 3 + len(temporal_folds)))

    for i, fold in enumerate(temporal_folds):
        cutoff = fold['cutoff_year']
        train_years = fold['train_years']
        val_years = fold['val_years']

        # Training bar
        ax.barh(i, max(train_years) - min(train_years) + 1,
                left=min(train_years), height=0.6,
                color='#2196F3', alpha=0.7, label='Train' if i == 0 else '')
        # Validation bar
        ax.barh(i, max(val_years) - min(val_years) + 1,
                left=min(val_years), height=0.6,
                color='#F44336', alpha=0.7, label='Validation' if i == 0 else '')
        # Cutoff line
        ax.axvline(cutoff, color='black', linestyle='--', alpha=0.3)
        ax.text(cutoff, i + 0.35, f'cutoff={cutoff}', fontsize=7, ha='center')

    ax.set_yticks(range(len(temporal_folds)))
    ax.set_yticklabels([f'Fold {i}' for i in range(len(temporal_folds))])
    ax.set_xlabel('Year')
    ax.set_title('Temporal CV Folds (Expanding Window, Forward-Only)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2, axis='x')
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "temporal_folds_diagram.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved temporal folds diagram")


def save_all_predictions(all_fold_results):
    """Save raw per-row predictions (actual vs predicted) for all folds."""
    for target in TARGETS:
        for h in HORIZONS:
            preds_list = [
                fr.val_predictions.assign(
                    spatial_fold=fr.spatial_fold_idx,
                    temporal_fold=fr.temporal_fold_idx,
                )
                for fr in all_fold_results
                if fr.target == target and fr.horizon == h
            ]
            if not preds_list:
                continue
            preds = pd.concat(preds_list, ignore_index=True)
            display = target.replace('c_', '').replace('_fao', '')
            preds.to_parquet(
                os.path.join(OUTPUT_DIR, f"cv_predictions_{display}_h{h}.parquet"),
                index=False,
            )
    logger.info("Saved raw predictions")


def save_per_admin_cv_performance(all_fold_results):
    """Aggregate per-admin2 MAPE across all CV folds."""
    all_preds = pd.concat([fr.val_predictions for fr in all_fold_results], ignore_index=True)

    for target in TARGETS:
        for h in HORIZONS:
            subset = all_preds[
                (all_preds.index.isin(
                    [i for i, fr in enumerate(all_fold_results)
                     if fr.target == target and fr.horizon == h
                     for i in fr.val_predictions.index]
                ))
            ]
            # Re-filter from fold results directly
            preds_list = [fr.val_predictions for fr in all_fold_results
                          if fr.target == target and fr.horizon == h]
            if not preds_list:
                continue
            preds = pd.concat(preds_list, ignore_index=True)
            preds['abs_error'] = np.abs(preds['actual'] - preds['predicted'])
            preds['pct_error'] = preds['abs_error'] / preds['actual'].clip(lower=1e-6)

            admin_perf = preds.groupby(['country_name', 'admin2_name']).agg(
                mae=('abs_error', 'mean'),
                mape=('pct_error', 'mean'),
                mean_actual=('actual', 'mean'),
                n=('actual', 'count'),
            ).reset_index().sort_values('mape', ascending=False)

            display = target.replace('c_', '').replace('_fao', '')
            admin_perf.to_csv(os.path.join(OUTPUT_DIR, f"cv_per_admin_{display}_h{h}.csv"),
                              index=False)

    logger.info("Saved per-admin CV performance")


def save_horizon_degradation_cv(df_agg):
    """Performance degradation across horizons with CV error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [('r2_mean', 'r2_std', 'R²'), ('mape_mean', 'mape_std', 'MAPE'),
               ('rmse_mean', 'rmse_std', 'RMSE')]

    for ax, (mean_col, std_col, label) in zip(axes, metrics):
        for target in TARGETS:
            subset = df_agg[df_agg['target'] == target]
            display = target.replace('c_', '').replace('_fao', '').replace('_', ' ')
            style = '-o' if target == 'c_food_price_index' else '--s' if target == 'c_maize_fao' else ':^'
            ax.errorbar(subset['horizon'], subset[mean_col], yerr=subset[std_col],
                        fmt=style, label=display, markersize=8, linewidth=2, capsize=5)

        ax.set_xlabel('Forecast Horizon (months)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} by Horizon (CV mean ± std)')
        ax.set_xticks(HORIZONS)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if 'mape' in mean_col.lower():
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    fig.suptitle('Performance Degradation with CV Error Bars', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cv_horizon_degradation.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved horizon degradation with CV")


def save_feature_importance_cv(all_fold_results):
    """Aggregated feature group importance across CV folds."""
    rows = []
    for fr in all_fold_results:
        group_imp = fr.feature_importance.groupby('group')['importance'].sum()
        for group, imp_val in group_imp.items():
            rows.append({
                'target': fr.target, 'horizon': fr.horizon,
                'spatial_fold': fr.spatial_fold_idx,
                'temporal_fold': fr.temporal_fold_idx,
                'group': group, 'importance': imp_val,
            })

    df_imp = pd.DataFrame(rows)
    df_imp.to_csv(os.path.join(OUTPUT_DIR, "cv_feature_group_importance.csv"), index=False)

    # Plot: mean group importance for h=1 across folds
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(7 * len(TARGETS), 6))
    if len(TARGETS) == 1:
        axes = [axes]

    for ax, target in zip(axes, TARGETS):
        subset = df_imp[(df_imp['target'] == target) & (df_imp['horizon'] == 1)]
        group_stats = subset.groupby('group')['importance'].agg(['mean', 'std']).sort_values(
            'mean', ascending=True)

        colors = [GROUP_COLORS.get(g, '#999') for g in group_stats.index]
        ax.barh(range(len(group_stats)), group_stats['mean'].values, color=colors,
                xerr=group_stats['std'].values, capsize=3)
        ax.set_yticks(range(len(group_stats)))
        ax.set_yticklabels(group_stats.index, fontsize=9)
        display = target.replace('c_', '').replace('_fao', '')
        ax.set_title(f'{display} h=1 (CV mean ± std)', fontsize=10)
        ax.set_xlabel('Importance')

    fig.suptitle('Feature Group Importance (Aggregated Across CV Folds)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cv_feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved CV feature importance")


def save_spatial_performance_heatmap(all_fold_results, centroids_df):
    """Map showing per-admin2 MAPE (color-coded) from CV predictions."""
    for target in TARGETS:
        for h in [1]:  # Just h=1 for clarity
            preds_list = [fr.val_predictions for fr in all_fold_results
                          if fr.target == target and fr.horizon == h]
            if not preds_list:
                continue
            preds = pd.concat(preds_list, ignore_index=True)
            preds['abs_error'] = np.abs(preds['actual'] - preds['predicted'])
            preds['pct_error'] = preds['abs_error'] / preds['actual'].clip(lower=1e-6)

            admin_mape = preds.groupby('admin2_name')['pct_error'].mean()

            fig, ax = plt.subplots(figsize=(10, 10))
            merged = centroids_df.merge(
                admin_mape.reset_index().rename(columns={'pct_error': 'mape'}),
                left_on='admin2', right_on='admin2_name', how='left'
            )

            # Drop admin2 without predictions (weren't in any validation fold)
            merged_valid = merged.dropna(subset=['mape'])
            if len(merged_valid) == 0:
                plt.close(fig)
                continue

            scatter = ax.scatter(
                merged_valid['lon'], merged_valid['lat'],
                c=merged_valid['mape'], cmap='RdYlGn_r',
                s=80, edgecolors='black', linewidths=0.5,
                vmin=0, vmax=merged_valid['mape'].quantile(0.95),
            )
            plt.colorbar(scatter, ax=ax, label='MAPE', format='%.0%%')

            # Label worst admin2
            worst = merged_valid.nlargest(5, 'mape')
            for _, row in worst.iterrows():
                ax.annotate(row['admin2'], (row['lon'], row['lat']),
                            fontsize=6, ha='center', va='bottom',
                            xytext=(0, 5), textcoords='offset points')

            display = target.replace('c_', '').replace('_fao', '')
            ax.set_title(f'{display} h={h}: Spatial MAPE (CV aggregated)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.2)

            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f"cv_spatial_mape_{display}_h{h}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

    logger.info("Saved spatial performance heatmap")


def save_all_metrics_json(df_agg, all_fold_results, spatial_folds, temporal_folds):
    """Save comprehensive metrics JSON."""
    metrics = {
        'cv_config': {
            'n_spatial_folds': N_SPATIAL_FOLDS,
            'n_temporal_folds': N_TEMPORAL_FOLDS,
            'buffer_radius_km': BUFFER_RADIUS_KM,
            'total_folds': len(spatial_folds) * len(temporal_folds),
            'targets': TARGETS,
            'horizons': HORIZONS,
        },
        'spatial_folds': [
            {
                'center_admin2': f['center_admin2'],
                'center_lat': f['center_lat'],
                'center_lon': f['center_lon'],
                'n_val': f['n_val'],
                'n_train': f['n_train'],
                'val_admin2': sorted(f['val_admin2']),
            }
            for f in spatial_folds
        ],
        'temporal_folds': [
            {
                'cutoff_year': f['cutoff_year'],
                'train_years': f['train_years'],
                'val_years': f['val_years'],
            }
            for f in temporal_folds
        ],
        'aggregated_results': {},
    }

    for _, row in df_agg.iterrows():
        target = row['target']
        h = int(row['horizon'])
        if target not in metrics['aggregated_results']:
            metrics['aggregated_results'][target] = {}
        metrics['aggregated_results'][target][f'h{h}'] = {
            'n_folds': int(row['n_folds']),
            'total_val_samples': int(row['total_val_samples']),
            'r2': {'mean': row['r2_mean'], 'std': row['r2_std'],
                    'min': row['r2_min'], 'max': row['r2_max']},
            'mape': {'mean': row['mape_mean'], 'std': row['mape_std'],
                     'min': row['mape_min'], 'max': row['mape_max']},
            'rmse': {'mean': row['rmse_mean'], 'std': row['rmse_std'],
                     'min': row['rmse_min'], 'max': row['rmse_max']},
            'mae': {'mean': row['mae_mean'], 'std': row['mae_std'],
                    'min': row['mae_min'], 'max': row['mae_max']},
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

    path = os.path.join(OUTPUT_DIR, "cv_metrics.json")
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved CV metrics -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate_stcv():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Load admin2 centroids ──
    logger.info("Loading admin2 centroids...")
    centroids_df = load_admin2_centroids()

    # ── 2. Generate spatial folds ──
    logger.info(f"Generating {N_SPATIAL_FOLDS} spatial folds (r={BUFFER_RADIUS_KM}km)...")
    spatial_folds = generate_spatial_folds(centroids_df, N_SPATIAL_FOLDS, BUFFER_RADIUS_KM, rng)

    # ── 3. Load and prepare data ──
    logger.info("Loading and preparing base data...")
    df_base = load_base_data()
    # Exclude 2025 from CV — reserve for true out-of-sample evaluation
    n_before = len(df_base)
    df_base = df_base[df_base['year'] < 2025].copy()
    logger.info(f"Excluded 2025 data from CV: {n_before} -> {len(df_base)} rows")
    df_base, label_encoders = encode_categoricals(df_base)

    # ── 4. Generate temporal folds ──
    logger.info(f"Generating {N_TEMPORAL_FOLDS} temporal folds...")
    all_years = sorted(df_base['year'].unique())
    temporal_folds = generate_temporal_folds(df_base, N_TEMPORAL_FOLDS)

    # ── 5. Save fold visualizations ──
    save_spatial_folds_map(centroids_df, spatial_folds)
    save_temporal_folds_diagram(temporal_folds, all_years)

    # ── 6. Run all CV folds ──
    total_folds = len(spatial_folds) * len(temporal_folds) * len(TARGETS) * len(HORIZONS)
    logger.info(f"Running {total_folds} total folds "
                f"({len(spatial_folds)} spatial × {len(temporal_folds)} temporal "
                f"× {len(TARGETS)} targets × {len(HORIZONS)} horizons)...")

    all_fold_results = []
    fold_count = 0

    for target in TARGETS:
        for h in HORIZONS:
            logger.info(f"{'='*60}")
            logger.info(f"Target: {target} | Horizon: {h}")

            # Prepare features for this target × horizon
            df_prepared, features, lag_feats = prepare_for_run(df_base, target, h)

            for s_idx, s_fold in enumerate(spatial_folds):
                for t_idx, t_fold in enumerate(temporal_folds):
                    fold_count += 1
                    logger.info(f"  Fold {fold_count}/{total_folds}: "
                                f"S{s_idx}/T{t_idx} ({target} h={h})")

                    result = run_stcv_fold(
                        df_prepared, features, target,
                        s_fold, t_fold,
                        s_idx, t_idx,
                        h, label_encoders,
                    )

                    if result is not None:
                        all_fold_results.append(result)
                        m = result.metrics
                        logger.info(f"    → R²={m['r2']:.4f}  MAPE={m['mape']:.2%}  "
                                    f"RMSE={m['rmse']:.4f}  "
                                    f"(train={result.n_train}, val={result.n_val})")

    if not all_fold_results:
        logger.error("No valid fold results! Check CV configuration.")
        return

    logger.info(f"\nCompleted {len(all_fold_results)} valid folds out of {total_folds} total")

    # ── 7. Generate all artifacts ──
    logger.info("Generating artifacts...")
    df_folds, df_agg = save_cv_summary(all_fold_results)
    save_cv_boxplot(df_folds)
    save_horizon_degradation_cv(df_agg)
    save_feature_importance_cv(all_fold_results)
    save_all_predictions(all_fold_results)
    save_per_admin_cv_performance(all_fold_results)
    save_spatial_performance_heatmap(all_fold_results, centroids_df)
    save_all_metrics_json(df_agg, all_fold_results, spatial_folds, temporal_folds)

    # ── 8. Console summary ──
    print("\n" + "=" * 80)
    print("SPATIO-TEMPORAL CV COMPLETE")
    print(f"  {len(spatial_folds)} spatial folds × {len(temporal_folds)} temporal folds "
          f"= {len(spatial_folds) * len(temporal_folds)} fold combinations per target×horizon")
    print(f"  {len(all_fold_results)} valid folds completed")
    print("=" * 80)

    header = (f"{'Target':<22s} {'H':>3s} {'R² mean':>10s} {'R² std':>8s} "
              f"{'MAPE mean':>10s} {'MAPE std':>9s} {'RMSE mean':>10s} {'N folds':>8s}")
    print(header)
    print("-" * len(header))

    for _, row in df_agg.iterrows():
        display = row['target'].replace('c_', '').replace('_fao', '')
        print(f"{display:<22s} {int(row['horizon']):>3d} "
              f"{row['r2_mean']:>10.4f} {row['r2_std']:>8.4f} "
              f"{row['mape_mean']:>9.2%} {row['mape_std']:>9.2%} "
              f"{row['rmse_mean']:>10.4f} {int(row['n_folds']):>8d}")

    print(f"\nArtifacts saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    train_and_evaluate_stcv()
