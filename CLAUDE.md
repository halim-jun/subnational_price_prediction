# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

East Africa food price prediction project integrating climate data (SPI, FLDAS, vegetation indices), macroeconomic indicators (World Bank), conflict data (ACLED), food security data (FEWS NET IPC), and geospatial data to predict subnational (Admin2-level) food prices in Kenya and Somalia.

This is a very thorough data science work where a random bypassing tricks (if error, raise and pass, if no value just drop it) kind of approach can be devastating.
Always come up with a very deeply thorough and meticulous data science viewpoint where you carefully examine errors, edge cases and test cases.

## Environment Setup

- Python 3.11.9 (see `.python-version`)
- Virtual environment: `source venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Environment variables in `.env`: NASA credentials (`BEARER_TOKEN`), used by night lights download

## Key Commands

### Model Training

```bash
python src/model/train_model_stcv.py    # Primary: Spatio-Temporal CV (outputs to artifact/model_output_stcv/)
python src/model/train_holdout.py       # Held-out 2024+ evaluation (outputs to artifact/model_output_holdout/)
python src/model/train_model.py         # Legacy baseline temporal split (outputs to artifact/model_output/)
```

### Dashboard

```bash
uvicorn src.dashboard.api.main:app --reload --port 8000   # FastAPI backend at localhost:8000
cd src/dashboard/frontend && npm run dev                    # Next.js frontend at localhost:3000
```

### Cloudflare Pages Prototype Build (private demo only — see `docs/cloudflare-prototype.md`)

```bash
python scripts/export_static_data.py                          # FastAPI → public/data/*.json
cd src/dashboard/frontend
NEXT_PUBLIC_STATIC_MODE=true npm run build                    # static export → out/
# Cloudflare Pages env vars required: SITE_PASSWORD, AUTH_SECRET
```

Prototype build replaces FastAPI calls with static JSON and gates everything
behind a shared-password Cloudflare Pages Function (`functions/_middleware.ts`).
Not for production — use Cloudflare Access / Auth0 etc. when promoting beyond demo.

### Data Pipeline (not needed for regular work — data is pre-processed in `data/`)

```bash
python src/data_pipeline/spi/run_spi_generation.py --download-chirps   # CHIRPS → SPI
python src/data_pipeline/macro/process_wb_data.py                       # World Bank indicators
python src/data_pipeline/fews_net/download.py                           # FEWS NET IPC data
python src/data_pipeline/night_lights/download.py                       # NASA Black Marble
python src/data_pipeline/merge/run_subnational_merge.py                 # Master merge → parquet
```

### Quality Checks

```bash
python src/data_pipeline/merge/check_merge_quality.py
python src/data_pipeline/merge/check_admin_coverage.py
```

### Formatting & Linting

```bash
black src/
flake8 src/
pytest
```

## Architecture

### Data Flow

```
Raw sources (APIs/downloads)
  → src/data_pipeline/ modules (8 subdirs: spi, macro, fews_net, crop_mask, night_lights, wfp, dynamic_world, merge)
  → data/processed/subnational_merged_v3_KEN_SOM.parquet
  → src/model/train_model_stcv.py (primary) → artifact/model_output_stcv/
  → src/dashboard/ (Streamlit visualization)
```

### Modeling Pipeline (src/model/)

Three training scripts share identical XGBoost config and feature engineering but differ in evaluation strategy:

**`train_model_stcv.py`** (~1100 lines) — Primary evaluation. Spatio-Temporal CV with Leave-Disc-Out spatial folds (5 folds, 350km radius) × expanding-window temporal folds (3 folds) = 15 fold combinations per target×horizon. Prevents both spatial autocorrelation leakage and temporal leakage.

**`train_holdout.py`** (~240 lines) — Simulates real deployment. Trains on all data < 2024, evaluates on 2024+. Tracks per-country metrics (Kenya vs Somalia separately).

**`train_model.py`** (~650 lines) — Legacy exploratory baseline. Simple temporal split with detailed diagnostic outputs (spike analysis, residual analysis, temporal error patterns, worst-performing admin2 regions).

**Targets:** Food Price Index, Maize (FAO), Sorghum
**Horizons:** 1, 2, 3 months ahead

**Lag strategy (critical for forecasting integrity):**
- Price features: lag_1,2,3,6,12 + rolling means/stds + year-over-year change
- Exogenous features (conflict, FLDAS, vegetation, climate): ALL lagged by h months to avoid data leakage
- 12-month warmup: drops first 12 rows per admin2 to build lag features

**XGBoost config (consistent across all scripts):**
n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=30

**8 feature groups:** FLDAS (8 vars), Vegetation (3), Climate Index (4), Static (2: crop_fraction, population), Conflict (2: event_count, fatalities), Spatial IDs, Temporal (sin/cos month encoding)

### Master Merge (src/data_pipeline/merge/)

`subnational_merger.py` joins 8 data sources spatially (via GeoBoundaries Admin2 GeoJSON) and temporally (monthly skeleton: year × month × admin2). Key merging strategies:
- Spatial joins: points → polygons for price markets & conflict events
- Fuzzy matching: admin1 city names → admin2 (e.g., "Nairobi" → "Starehe") with manual fallback overrides
- Zonal statistics: raster → polygon for population
- Output: `data/processed/subnational_merged_v3_KEN_SOM.parquet` (50+ columns)

### Dashboard (src/dashboard/)

Next.js 16 + React 19 frontend with FastAPI REST backend.

**Frontend** (`src/dashboard/frontend/`): Tailwind CSS, Recharts, Leaflet maps, Zustand state management.
- **Overview** — Holdout test metrics (R², MAPE, RMSE), best/worst performing regions
- **Map** — Interactive Leaflet choropleth (predicted/actual/error views) with date slider
- **Time Series** — Per-region line charts, predicted vs actual scatter, MoM change analysis

**Backend** (`src/dashboard/api/`): FastAPI with endpoints for metrics, geo, predictions, features.

### Spatial Reference System

- GeoBoundaries GeoJSON files in `data/geoboundaries/` serve as the canonical Admin2 boundary reference
- All datasets are spatially joined to Admin2 regions using geopandas
- Country codes use ISO3: KEN (Kenya), SOM (Somalia), ETH (Ethiopia)

### Data Directory Layout

- `data/raw/` — original downloaded data (climate, worldbank_commodity, wfp, acled)
- `data/processed/` — pipeline outputs (spi, external, night_lights, final merged parquet)
- `data/geoboundaries/` — administrative boundary GeoJSON files
- `data/climate_indices/`, `data/fldas/`, `data/vegetation/` — domain-specific processed data

### Artifact Directory Layout

- `artifact/model_output_stcv/` — Primary CV results (cv_fold_results.csv, cv_aggregated_metrics.csv, cv_predictions_*.parquet, cv_per_admin_*.csv, cv_feature_group_importance.csv, visualizations)
- `artifact/model_output_holdout/` — 2024+ holdout evaluation (holdout_metrics.csv, holdout_predictions.parquet, holdout_per_admin_*.csv)
- `artifact/model_output/` — Legacy baseline (13+ diagnostic PNGs/CSVs)

## Code Conventions

- Path resolution: scripts use `sys.path.append(project_root)` with dynamically computed project root
- Logging: Python `logging` module with StreamHandler + FileHandler
- File naming: snake_case throughout
- Data formats: Parquet for tabular data, NetCDF for gridded climate data, GeoJSON for boundaries
