
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/subnational_merged_v3_KEN_SOM.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifact/model_output")

TARGETS = ['c_food_price_index', 'c_maize_fao', 'c_sorghum']
HORIZONS = [1, 2, 3]  # months ahead
SPLIT_YEAR = 2024

# ── Feature groups (same rationale as before) ────────────────────────────────
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

# Features that are KNOWN at prediction time (no lag needed)
KNOWN_AT_PREDICTION = TEMPORAL_FEATURES + STATIC_FEATURES + CATEGORICAL_FEATURES

# Features that are NOT known h months ahead → must be lagged by h
# (at prediction time we don't know future conflict, climate, vegetation)
NEEDS_LAG_FEATURES = CONFLICT_FEATURES + FLDAS_FEATURES + VEGETATION_FEATURES + CLIMATE_INDICES

# Lag configuration per horizon:
#   h=1: can use lag_1,3,6,12 + rolling from t
#   h=2: can use lag_2,3,6,12 + rolling from t-1
#   h=3: can use lag_3,6,12   + rolling from t-2
ALL_LAGS = [1, 2, 3, 6, 12]
MIN_LAG_WARMUP = 12  # months to drop at start for lag construction


def build_all_lag_features(df, target):
    """Build lag features for ALL possible lags. Horizon-specific filtering happens later."""
    df = df.sort_values(['admin2', 'year', 'month']).copy()

    for m in ALL_LAGS:
        df[f'{target}_lag_{m}'] = df.groupby('admin2')[target].shift(m)

    # Rolling stats anchored at different offsets (shift=1 means "up to last month")
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

    # YoY change (always available: compares lag_12 to lag_13)
    df[f'{target}_yoy'] = df[f'{target}_lag_12'] - df.groupby('admin2')[target].shift(13)

    return df


def get_lag_features_for_horizon(target, horizon):
    """Return the lag/rolling feature names available at a given forecast horizon."""
    # Lags: can only use lag_k where k >= horizon
    lag_feats = [f'{target}_lag_{m}' for m in ALL_LAGS if m >= horizon]

    # Rolling: shift must be >= horizon
    roll_feats = []
    for shift in [1, 2, 3]:
        if shift >= horizon:
            prefix = f'{target}_s{shift}'
            roll_feats += [f'{prefix}_rmean3', f'{prefix}_rmean12', f'{prefix}_rstd12']

    # YoY always uses lag_12 vs lag_13 → available for h<=12
    if horizon <= 12:
        roll_feats.append(f'{target}_yoy')

    return lag_feats + roll_feats


def load_base_data():
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded: {df.shape}")

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Sort once for all lag operations
    df = df.sort_values(['admin2', 'year', 'month']).reset_index(drop=True)

    # Build target price lag features
    for target in TARGETS:
        df = build_all_lag_features(df, target)

    # Build lagged exogenous features for each horizon.
    # At horizon h, we only know exogenous values from h months ago or earlier.
    # e.g. predicting 3 months ahead → use conflict/climate/vegetation from 3+ months ago.
    for h in HORIZONS:
        for feat in NEEDS_LAG_FEATURES:
            col_name = f'{feat}_Lh{h}'
            df[col_name] = df.groupby('admin2')[feat].shift(h)

    return df


def get_exog_features_for_horizon(horizon):
    """Return exogenous feature names available at a given horizon."""
    # Static/temporal/spatial: always known at prediction time
    known = list(KNOWN_AT_PREDICTION)
    # Lagged exogenous: shifted by horizon months
    lagged = [f'{feat}_Lh{horizon}' for feat in NEEDS_LAG_FEATURES]
    return known + lagged


def prepare_for_run(df_base, target, horizon):
    """Prepare features and drop NaN rows for a specific target+horizon combination."""
    df = df_base.copy()
    price_lag_feats = get_lag_features_for_horizon(target, horizon)
    exog_feats = get_exog_features_for_horizon(horizon)
    features = exog_feats + price_lag_feats

    # Drop rows with NaN in any feature or target
    required_cols = features + [target]
    before = len(df)
    df = df.dropna(subset=required_cols)
    logger.info(f"  [{target} h={horizon}] Dropped {before - len(df)} rows for warmup, "
                f"{len(df)} remaining, {len(price_lag_feats)} price lags + "
                f"{len([f for f in exog_feats if '_Lh' in f])} lagged exog features")

    return df, features, price_lag_feats


def encode_categoricals(df):
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


def temporal_split(df, features, target):
    train = df[df['year'] < SPLIT_YEAR]
    test = df[df['year'] >= SPLIT_YEAR]
    return train[features], train[target], test[features], test[target], test


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
    split_idx = int(len(X_train) * 0.9)
    model.fit(
        X_train.iloc[:split_idx], y_train.iloc[:split_idx],
        eval_set=[(X_train.iloc[split_idx:], y_train.iloc[split_idx:])],
        verbose=False,
    )
    return model


def compute_metrics(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'n': int(len(y_true)),
    }


def _feature_group(f):
    # Strip _Lh{n} suffix to get the original feature name for group lookup
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


# ── Single run: train + evaluate + collect results ───────────────────────────

def run_single(df_base, target, horizon, label_encoders):
    """Run one target × horizon combination. Returns metrics, model, predictions."""
    df_run, features, lag_feats = prepare_for_run(df_base, target, horizon)
    X_train, y_train, X_test, y_test, test_df = temporal_split(df_run, features, target)

    model = train_xgb(X_train, y_train)
    y_pred = model.predict(X_test)

    # Overall metrics
    overall = compute_metrics(y_test, y_pred)

    # Per-country
    le = label_encoders['country_iso']
    country_metrics = {}
    for enc_val in sorted(test_df['country_iso'].unique()):
        mask = test_df['country_iso'].values == enc_val
        name = le.inverse_transform([enc_val])[0]
        country_metrics[name] = compute_metrics(y_test.values[mask], y_pred[mask])

    # Per-admin
    le_admin = label_encoders['admin2']
    admin_results = test_df[['admin2', 'country_iso', 'year', 'month']].copy()
    admin_results['actual'] = y_test.values
    admin_results['predicted'] = y_pred
    admin_results['abs_error'] = np.abs(admin_results['actual'] - admin_results['predicted'])
    admin_results['pct_error'] = admin_results['abs_error'] / admin_results['actual'].clip(lower=1e-6)
    admin_results['admin2_name'] = le_admin.inverse_transform(admin_results['admin2'])
    admin_results['country_name'] = le.inverse_transform(admin_results['country_iso'])

    # Feature importance
    imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    imp['group'] = imp['feature'].apply(_feature_group)
    imp = imp.sort_values('importance', ascending=False)

    return {
        'target': target, 'horizon': horizon,
        'overall': overall, 'country': country_metrics,
        'model': model, 'features': features, 'lag_feats': lag_feats,
        'importance': imp, 'predictions': admin_results,
    }


# ── Artifact generators ──────────────────────────────────────────────────────

def save_horizon_degradation(all_results):
    """Q1: How does performance degrade across horizons per target?"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_names = ['r2', 'mape', 'rmse']
    metric_labels = ['R²', 'MAPE', 'RMSE']

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        for target in TARGETS:
            vals = []
            for h in HORIZONS:
                key = (target, h)
                vals.append(all_results[key]['overall'][metric])
            style = '-o' if target == 'c_food_price_index' else '--s' if target == 'c_maize_fao' else ':^'
            display = target.replace('c_', '').replace('_fao', '').replace('_', ' ')
            ax.plot(HORIZONS, vals, style, label=display, markersize=8, linewidth=2)

        ax.set_xlabel('Forecast Horizon (months)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} by Horizon')
        ax.set_xticks(HORIZONS)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric == 'mape':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    fig.suptitle('Performance Degradation Across Forecast Horizons', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "horizon_degradation.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Table
    rows = []
    for target in TARGETS:
        for h in HORIZONS:
            r = all_results[(target, h)]
            row = {'target': target, 'horizon': h, **r['overall']}
            for country, cm in r['country'].items():
                for k, v in cm.items():
                    row[f'{country}_{k}'] = v
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "horizon_summary.csv"), index=False)
    logger.info("Saved horizon degradation")


def save_per_admin_worst(all_results, label_encoders):
    """Q2: Which admins/countries are worst at each horizon?"""
    fig, axes = plt.subplots(len(TARGETS), len(HORIZONS), figsize=(6 * len(HORIZONS), 5 * len(TARGETS)))

    for i, target in enumerate(TARGETS):
        for j, h in enumerate(HORIZONS):
            ax = axes[i][j] if len(TARGETS) > 1 else axes[j]
            preds = all_results[(target, h)]['predictions']

            admin_perf = preds.groupby(['country_name', 'admin2_name']).agg(
                mape=('abs_error', lambda x: (x / preds.loc[x.index, 'actual'].clip(lower=1e-6)).mean()),
            ).reset_index().sort_values('mape', ascending=False)

            top_n = 10
            worst = admin_perf.head(top_n)
            colors = ['#F44336' if c == 'SOM' else '#2196F3' for c in worst['country_name']]
            ax.barh(range(len(worst)), worst['mape'] * 100, color=colors)
            ax.set_yticks(range(len(worst)))
            ax.set_yticklabels([f"{r['admin2_name']}" for _, r in worst.iterrows()], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel('MAPE (%)')
            display = target.replace('c_', '').replace('_fao', '')
            ax.set_title(f'{display} h={h}', fontsize=10)

    fig.suptitle('Worst-Performing Admin2 Regions by Target & Horizon', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "worst_admins_grid.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved worst-admin analysis")


def save_temporal_error_pattern(all_results):
    """Q3: Which time periods have highest error? Crisis detection."""
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(14, 5 * len(TARGETS)), sharex=True)
    if len(TARGETS) == 1:
        axes = [axes]

    for ax, target in zip(axes, TARGETS):
        for h in HORIZONS:
            preds = all_results[(target, h)]['predictions']
            monthly = preds.groupby(['year', 'month']).apply(
                lambda g: mean_absolute_percentage_error(g['actual'], g['predicted']),
                include_groups=False,
            ).reset_index(name='mape')
            x_labels = [f"{int(r['year'])}-{int(r['month']):02d}" for _, r in monthly.iterrows()]
            x_idx = range(len(x_labels))
            ax.plot(x_idx, monthly['mape'] * 100, 'o-', label=f'h={h}', markersize=3, linewidth=1.5)

        display = target.replace('c_', '').replace('_fao', '')
        ax.set_title(f'{display}: Monthly MAPE on Test Set')
        ax.set_ylabel('MAPE (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "temporal_error_pattern.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved temporal error pattern")


def save_feature_importance_shift(all_results):
    """Q4: How does feature importance shift when short-term lags are removed?"""
    fig, axes = plt.subplots(len(TARGETS), len(HORIZONS), figsize=(7 * len(HORIZONS), 6 * len(TARGETS)))

    for i, target in enumerate(TARGETS):
        for j, h in enumerate(HORIZONS):
            ax = axes[i][j] if len(TARGETS) > 1 else axes[j]
            imp = all_results[(target, h)]['importance']
            group_imp = imp.groupby('group')['importance'].sum().sort_values(ascending=True)

            colors = [GROUP_COLORS.get(g, '#999') for g in group_imp.index]
            ax.barh(range(len(group_imp)), group_imp.values, color=colors)
            ax.set_yticks(range(len(group_imp)))
            ax.set_yticklabels(group_imp.index, fontsize=9)
            display = target.replace('c_', '').replace('_fao', '')
            ax.set_title(f'{display} h={h}', fontsize=10)
            ax.set_xlabel('Importance')

    fig.suptitle('Feature Group Importance Shift Across Horizons', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "importance_shift.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Detailed CSV: top features per run
    all_imp_rows = []
    for target in TARGETS:
        for h in HORIZONS:
            imp = all_results[(target, h)]['importance'].copy()
            imp['target'] = target
            imp['horizon'] = h
            all_imp_rows.append(imp)
    pd.concat(all_imp_rows).to_csv(os.path.join(OUTPUT_DIR, "all_feature_importance.csv"), index=False)
    logger.info("Saved feature importance shift")


def save_spike_analysis(all_results):
    """Q5: Does the model struggle more with price spikes or stable periods?"""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 6))
    if len(TARGETS) == 1:
        axes = [axes]

    for ax, target in zip(axes, TARGETS):
        # Use h=1 for clearest signal
        preds = all_results[(target, 1)]['predictions'].copy()
        preds['pct_error'] = preds['abs_error'] / preds['actual'].clip(lower=1e-6)

        # Classify periods by price level relative to per-admin distribution
        admin_stats = preds.groupby('admin2_name')['actual'].agg(['mean', 'std'])
        preds = preds.merge(admin_stats, left_on='admin2_name', right_index=True)
        preds['z_price'] = (preds['actual'] - preds['mean']) / preds['std'].clip(lower=1e-6)

        bins = [-np.inf, -1, -0.5, 0.5, 1, np.inf]
        labels_bin = ['Crash\n(<-1σ)', 'Low\n(-1 to -0.5σ)', 'Normal\n(±0.5σ)',
                      'High\n(0.5 to 1σ)', 'Spike\n(>1σ)']
        preds['regime'] = pd.cut(preds['z_price'], bins=bins, labels=labels_bin)

        regime_err = preds.groupby('regime', observed=True)['pct_error'].agg(['mean', 'count'])
        colors_bar = ['#2196F3', '#64B5F6', '#9E9E9E', '#FF9800', '#F44336']
        bars = ax.bar(range(len(regime_err)), regime_err['mean'] * 100, color=colors_bar[:len(regime_err)])

        # Annotate count
        for idx, (_, row) in enumerate(regime_err.iterrows()):
            ax.text(idx, row['mean'] * 100 + 0.1, f'n={int(row["count"])}',
                    ha='center', fontsize=8, color='gray')

        ax.set_xticks(range(len(regime_err)))
        ax.set_xticklabels(regime_err.index, fontsize=9)
        ax.set_ylabel('MAPE (%)')
        display = target.replace('c_', '').replace('_fao', '')
        ax.set_title(f'{display}: Error by Price Regime (h=1)')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "spike_analysis.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved spike analysis")


def save_actual_vs_predicted_grid(all_results, label_encoders):
    """Scatter plots: actual vs predicted for all target × horizon combos."""
    fig, axes = plt.subplots(len(TARGETS), len(HORIZONS), figsize=(6 * len(HORIZONS), 5 * len(TARGETS)))

    le = label_encoders['country_iso']
    for i, target in enumerate(TARGETS):
        for j, h in enumerate(HORIZONS):
            ax = axes[i][j] if len(TARGETS) > 1 else axes[j]
            preds = all_results[(target, h)]['predictions']
            r = all_results[(target, h)]

            for enc_val in sorted(preds['country_iso'].unique()):
                mask = preds['country_iso'].values == enc_val
                name = le.inverse_transform([enc_val])[0]
                ax.scatter(preds.loc[mask, 'actual'], preds.loc[mask, 'predicted'],
                           alpha=0.3, s=10, label=name)

            lims = [preds['actual'].min(), preds['actual'].max()]
            ax.plot(lims, lims, 'r--', lw=1)
            display = target.replace('c_', '').replace('_fao', '')
            ax.set_title(f"{display} h={h}\nR²={r['overall']['r2']:.3f} MAPE={r['overall']['mape']:.1%}", fontsize=9)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle('Actual vs Predicted: All Targets × Horizons', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted_grid.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved actual vs predicted grid")


def save_sample_timeseries(all_results, label_encoders):
    """Time series for 4 sample admins across all horizons, for each target."""
    le_admin = label_encoders['admin2']

    for target in TARGETS:
        # Pick 4 admins: best KEN, worst KEN, best SOM, worst SOM (h=1)
        preds_h1 = all_results[(target, 1)]['predictions']
        admin_mape = preds_h1.groupby(['country_name', 'admin2_name']).apply(
            lambda g: mean_absolute_percentage_error(g['actual'], g['predicted']),
            include_groups=False,
        ).reset_index(name='mape')

        samples = []
        for country in ['KEN', 'SOM']:
            c_admins = admin_mape[admin_mape['country_name'] == country].sort_values('mape')
            if len(c_admins) >= 2:
                samples.append(c_admins.iloc[0])   # best
                samples.append(c_admins.iloc[-1])   # worst

        fig, axes = plt.subplots(len(samples), 1, figsize=(14, 3.5 * len(samples)), sharex=True)
        if len(samples) <= 1:
            axes = [axes]

        for ax, sample in zip(axes, samples):
            admin_name = sample['admin2_name']
            country = sample['country_name']

            # Plot actual (from h=1 — same actual for all horizons)
            sub = preds_h1[preds_h1['admin2_name'] == admin_name].sort_values(['year', 'month'])
            x_labels = [f"{int(r['year'])}-{int(r['month']):02d}" for _, r in sub.iterrows()]
            x_idx = range(len(x_labels))
            ax.plot(x_idx, sub['actual'].values, 'o-', label='Actual', color='black',
                    markersize=4, linewidth=2)

            # Plot predicted for each horizon
            h_colors = {1: '#2196F3', 2: '#FF9800', 3: '#F44336'}
            for h in HORIZONS:
                preds_h = all_results[(target, h)]['predictions']
                sub_h = preds_h[preds_h['admin2_name'] == admin_name].sort_values(['year', 'month'])
                if not sub_h.empty:
                    ax.plot(x_idx[:len(sub_h)], sub_h['predicted'].values, 's--',
                            label=f'h={h}', color=h_colors[h], markersize=3, alpha=0.8)

            ax.set_title(f"{admin_name} ({country})", fontsize=10)
            ax.legend(fontsize=8, ncol=4)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

        display = target.replace('c_', '').replace('_fao', '')
        fig.suptitle(f'{display}: Sample Time Series Across Horizons', fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"timeseries_{display}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    logger.info("Saved sample time series")


def save_feature_correlation(df_base, all_results):
    """Feature & target correlation heatmap (using raw unlagged columns)."""
    raw_feats = (FLDAS_FEATURES + VEGETATION_FEATURES + CLIMATE_INDICES +
                 CONFLICT_FEATURES + STATIC_FEATURES)
    # Use unencoded target values from df_base
    cols = [c for c in raw_feats + TARGETS if c in df_base.columns]
    corr = df_base[cols].corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size': 7})
    ax.set_title('Feature & Target Correlation Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"), dpi=150)
    plt.close(fig)

    for t in TARGETS:
        if t in corr.columns:
            t_corr = corr[t].drop(TARGETS, errors='ignore').sort_values(key=abs, ascending=False)
            display = t.replace('c_', '').replace('_fao', '')
            t_corr.to_csv(os.path.join(OUTPUT_DIR, f"target_corr_{display}.csv"))
    logger.info("Saved feature correlation")


def save_residual_analysis_all(all_results, label_encoders):
    """Residual analysis for all 3 targets at h=1."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for idx, target in enumerate(TARGETS):
        preds = all_results[(target, 1)]['predictions']
        residuals = preds['actual'].values - preds['predicted'].values
        display = target.replace('c_', '').replace('_fao', '')

        ax = axes[idx, 0]
        ax.scatter(preds['predicted'], residuals, alpha=0.3, s=8, c='steelblue')
        ax.axhline(0, color='r', ls='--', lw=1)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Residual')
        ax.set_title(f'{display}: Residuals vs Predicted')

        ax = axes[idx, 1]
        ax.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='r', ls='--', lw=1)
        ax.set_title(f'{display}: Distribution (μ={np.mean(residuals):.4f}, σ={np.std(residuals):.4f})')

        ax = axes[idx, 2]
        month_err = preds.copy()
        month_err['abs_err'] = np.abs(residuals)
        monthly = month_err.groupby('month')['abs_err'].mean()
        ax.bar(monthly.index, monthly.values, color='steelblue')
        ax.set_xticks(range(1, 13)); ax.set_xlabel('Month'); ax.set_ylabel('MAE')
        ax.set_title(f'{display}: Error by Month')

    fig.suptitle('Residual Analysis (h=1, All Targets)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "residual_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved residual analysis")


def save_per_admin_all(all_results, label_encoders):
    """Per-admin performance CSV + plot for all 3 targets at h=1."""
    le_admin = label_encoders['admin2']
    le_country = label_encoders['country_iso']

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    top_n = 15

    for idx, target in enumerate(TARGETS):
        preds = all_results[(target, 1)]['predictions'].copy()
        display = target.replace('c_', '').replace('_fao', '')

        admin_perf = preds.groupby(['country_name', 'admin2_name']).agg(
            mae=('abs_error', 'mean'),
            mape=('pct_error', 'mean'),
            mean_actual=('actual', 'mean'),
            n=('actual', 'count'),
        ).reset_index().sort_values('mape', ascending=False)
        admin_perf.to_csv(os.path.join(OUTPUT_DIR, f"per_admin_{display}.csv"), index=False)

        for j, (subset, title) in enumerate([
            (admin_perf.head(top_n), 'Worst'),
            (admin_perf.tail(top_n), 'Best'),
        ]):
            ax = axes[idx, j]
            colors = ['#F44336' if c == 'SOM' else '#2196F3' for c in subset['country_name']]
            ax.barh(range(len(subset)), subset['mape'] * 100, color=colors)
            ax.set_yticks(range(len(subset)))
            ax.set_yticklabels(
                [f"{r['admin2_name']} ({r['country_name']})" for _, r in subset.iterrows()],
                fontsize=7)
            ax.invert_yaxis(); ax.set_xlabel('MAPE (%)')
            ax.set_title(f'{display} h=1: {title} {top_n} Admins')

    fig.suptitle('Per-Admin Performance (h=1, All Targets)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "per_admin_performance.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved per-admin performance")


def save_feature_group_pie(all_results):
    """Feature group pie chart for all 3 targets at h=1."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, target in zip(axes, TARGETS):
        imp = all_results[(target, 1)]['importance']
        group_imp = imp.groupby('group')['importance'].sum().sort_values(ascending=False)
        colors = [GROUP_COLORS.get(g, '#999') for g in group_imp.index]
        display = target.replace('c_', '').replace('_fao', '')
        ax.pie(group_imp.values, labels=group_imp.index, colors=colors,
               autopct=lambda p: f'{p:.1f}%' if p > 1 else '', startangle=90,
               textprops={'fontsize': 8})
        ax.set_title(f'{display} (h=1)')
    fig.suptitle('Feature Group Importance Share', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_group_pie.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved feature group pie")


def save_all_metrics(all_results):
    """Save comprehensive metrics JSON."""
    metrics = {}
    for target in TARGETS:
        metrics[target] = {}
        for h in HORIZONS:
            r = all_results[(target, h)]
            metrics[target][f'h{h}'] = {
                'overall': r['overall'],
                'country': r['country'],
                'n_features': len(r['features']),
                'n_lag_features': len(r['lag_feats']),
                'lag_features_used': r['lag_feats'],
                'best_iteration': int(r['model'].best_iteration),
            }

    metrics['config'] = {
        'targets': TARGETS,
        'horizons': HORIZONS,
        'split_year': SPLIT_YEAR,
        'known_at_prediction': KNOWN_AT_PREDICTION,
        'needs_lag_features': NEEDS_LAG_FEATURES,
    }

    path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics -> {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def train_and_evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_base = load_base_data()

    # Encode categoricals once (shared across all runs)
    df_base, label_encoders = encode_categoricals(df_base)

    # Run all target × horizon combinations
    all_results = {}
    for target in TARGETS:
        for h in HORIZONS:
            logger.info(f"{'='*60}")
            logger.info(f"Training: {target} | horizon={h}")
            result = run_single(df_base, target, h, label_encoders)
            all_results[(target, h)] = result

            m = result['overall']
            logger.info(f"  → R²={m['r2']:.4f}  MAPE={m['mape']:.2%}  RMSE={m['rmse']:.4f}")

    # ── Generate all artifacts ──
    save_all_metrics(all_results)
    save_horizon_degradation(all_results)
    save_per_admin_worst(all_results, label_encoders)
    save_temporal_error_pattern(all_results)
    save_feature_importance_shift(all_results)
    save_spike_analysis(all_results)
    save_actual_vs_predicted_grid(all_results, label_encoders)
    save_sample_timeseries(all_results, label_encoders)
    save_feature_correlation(df_base, all_results)
    save_residual_analysis_all(all_results, label_encoders)
    save_per_admin_all(all_results, label_encoders)
    save_feature_group_pie(all_results)

    # Save models
    for target in TARGETS:
        for h in HORIZONS:
            display = target.replace('c_', '').replace('_fao', '')
            all_results[(target, h)]['model'].save_model(
                os.path.join(OUTPUT_DIR, f"model_{display}_h{h}.json")
            )

    # ── Console summary ──
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE — 3 Targets × 3 Horizons = 9 Models")
    print("=" * 80)

    header = f"{'Target':<22s} {'Horizon':>7s} {'R²':>8s} {'MAPE':>8s} {'RMSE':>10s} {'KEN_MAPE':>10s} {'SOM_MAPE':>10s}"
    print(header)
    print("-" * len(header))
    for target in TARGETS:
        for h in HORIZONS:
            r = all_results[(target, h)]
            display = target.replace('c_', '').replace('_fao', '')
            ken = r['country'].get('KEN', {}).get('mape', float('nan'))
            som = r['country'].get('SOM', {}).get('mape', float('nan'))
            print(f"{display:<22s} h={h:>3d}    {r['overall']['r2']:>8.4f} {r['overall']['mape']:>7.2%} "
                  f"{r['overall']['rmse']:>10.4f} {ken:>9.2%} {som:>9.2%}")
        print()

    # Feature importance summary for h=1 vs h=3
    print("Feature group importance shift (h=1 → h=3):")
    for target in TARGETS:
        display = target.replace('c_', '').replace('_fao', '')
        imp1 = all_results[(target, 1)]['importance'].groupby('group')['importance'].sum()
        imp3 = all_results[(target, 3)]['importance'].groupby('group')['importance'].sum()
        print(f"\n  {display}:")
        all_groups = sorted(set(imp1.index) | set(imp3.index))
        for g in all_groups:
            v1 = imp1.get(g, 0)
            v3 = imp3.get(g, 0)
            arrow = "↑" if v3 > v1 + 0.01 else "↓" if v3 < v1 - 0.01 else "→"
            print(f"    {g:<18s} {v1:.1%} → {v3:.1%} {arrow}")

    print(f"\nArtifacts saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    train_and_evaluate()
