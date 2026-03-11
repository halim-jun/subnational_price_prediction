# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

East Africa food price prediction project integrating climate data (SPI, FLDAS, vegetation indices), macroeconomic indicators (World Bank), conflict data (ACLED), food security data (FEWS NET IPC), and geospatial data to predict subnational (Admin2-level) food prices in Kenya and Somalia.
This is a very thorough data science work where a random bypassing tricks (if error, raise and pass, if no value just drop it) kind of approach can be devastating.
Always come up with a very deeply through and meticulous data science view point where you carefully examine errors, edge cases and test cases.

## Environment Setup

- Python 3.11.9 (see `.python-version`)
- Virtual environment: `source venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Environment variables in `.env`: NASA credentials (`BEARER_TOKEN`), used by night lights download

## Key Commands

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
Raw sources (APIs/downloads) → data_pipeline modules → processed data → master merge → modeling
```

1. **Data Pipeline** (`src/data_pipeline/`): Each subdirectory handles one data source (spi, macro, fews_net, crop_mask, night_lights, wfp, dynamic_world). Each has its own entry point script.
2. **Master Merge** (`src/data_pipeline/merge/`): `subnational_merger.py` joins all data sources spatially (via GeoBoundaries Admin2 GeoJSON) and temporally (monthly). Output: `data/processed/subnational_merged_v3_KEN_SOM.parquet`
3. **Modeling** (`src/notebook/subnational_level_prediction_baseline.ipynb`): Primary modeling notebook using XGBoost. Also `src/model/train_model.py` and `src/analysis/eth_sarimax_model.py` for SARIMA.
4. **Notebooks** (`src/notebook/`): `data_*.ipynb` notebooks handle raw data processing for climate indices, FLDAS, and vegetation. `process_*.ipynb` notebooks handle intermediate processing. `subnational_merge_notebook.ipynb` is the latest merge logic (v3).

### Spatial Reference System

- GeoBoundaries GeoJSON files in `data/geoboundaries/` serve as the canonical Admin2 boundary reference
- All datasets are spatially joined to Admin2 regions using geopandas
- Country codes use ISO3: KEN (Kenya), SOM (Somalia), ETH (Ethiopia)

### Data Directory Layout

- `data/raw/` — original downloaded data (climate, worldbank_commodity, wfp, acled)
- `data/processed/` — pipeline outputs (spi, external, night_lights, final merged parquet)
- `data/geoboundaries/` — administrative boundary GeoJSON files
- `data/climate_indices/`, `data/fldas/`, `data/vegetation/` — domain-specific processed data

## Code Conventions

- Path resolution: scripts use `sys.path.append(project_root)` with dynamically computed project root
- Logging: Python `logging` module with StreamHandler + FileHandler
- File naming: snake_case throughout
- Data formats: Parquet for tabular data, NetCDF for gridded climate data, GeoJSON for boundaries
