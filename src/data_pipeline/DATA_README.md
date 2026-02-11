# Data Pipeline Documentation

This directory contains the data processing pipelines for the project, organized by data source and domain.

## 📂 Directory Structure

```
src/data_pipeline/
├── macro/          # Macroeconomic and External Data Processing
├── spi/            # Precipitation (CHIRPS) & SPI Calculation
├── fews_net/       # Food Security Data (IPC)
├── crop_mask/      # Crop Mask (ASAP)
├── night_lights/   # Night Lights (Black Marble)
├── spatial_weighting/ # Spatial Weighting (SPAM)
└── utils/          # Utilities and Diagnostics
```

---

## 1. 🌦️ SPI Pipeline (`src/data_pipeline/spi/`)

Handles the download, processing, and enrichment of precipitation data from CHIRPS.

| Script | Input Data | Output Data | Description |
|--------|------------|-------------|-------------|
| **`run_spi_generation.py`** | `data/raw/climate/chirps/*.nc` | `data/processed/spi/05_spi_final/*.nc` | **Main Orchestrator**. Runs the full pipeline: Clipping, Filling, and SPI Calculation using `generate_spi_python.py`. |
| `generate_spi_python.py` | NetCDF (Raw) | NetCDF (Processed) | Core logic for SPI calculation using `climate-indices` library. |
| `convert_nc_to_csv.py` | `data/processed/spi/05_spi_final/*.nc` | `data/processed/spi/06_spi_csv/*.csv` | Converts the calculated SPI NetCDF files into easy-to-use CSV format. |
| `enrich_all_spi.py` | `data/processed/spi/06_spi_csv/*.csv` | `data/processed/spi/07_enriched/*.csv` | Adds administrative boundaries (Country, Region, Zone) to the SPI CSVs using GeoBoundaries data. |
| **`generate_weighted_indices.py`** | SPI CSV + SPAM Data | CSV (Admin2 Weighted) | Generates SPI indices weighted by crop production areas per region. |
| `climate_aggregator.py` | - | - | Core logic module for weighted average calculation. |

### 🚀 Usage
```bash
# Run the full pipeline (Generation -> Conversion)
python src/data_pipeline/spi/run_spi_generation.py
```

---

## 2. 📈 Macro Pipeline (`src/data_pipeline/macro/`)

Handles macroeconomic indicators like World Bank commodity prices and exchange rates.

| Script | Input Data | Output Data | Description |
|--------|------------|-------------|-------------|
| **`process_wb_data.py`** | `data/raw/worldbank_commodity/*.xlsx` | `data/processed/external/worldbank_indices.csv` | Extracts Energy, Food, and Fertilizer indices from World Bank "Pink Sheet" data. |
| `merge_external_data.py` | `worldbank_indices.csv`, etc. | `data/processed/external/external_variables_merged.csv` | Merges various external indicators (WB Data, Exchange Rates, FAO Index) into a single master time-series file. |

### 🚀 Usage
```bash
# Process World Bank Data
python src/data_pipeline/macro/process_wb_data.py

# Merge all external data
python src/data_pipeline/macro/merge_external_data.py
```

---

## 3. 🌽 Food Security Pipeline (`src/data_pipeline/fews_net/`)

Downloads and processes food security data (IPC Phase) from FEWS NET.

| Script | Description |
|--------|-------------|
| **`download.py`** | Downloads IPC data for Kenya, Ethiopia, and Somalia via FEWS NET API. |
| `merge.py` | Merges split CSV files into one. |

### 🚀 Usage
```bash
python src/data_pipeline/fews_net/download.py
```

---

## 4. 🌾 Crop Mask Pipeline (`src/data_pipeline/crop_mask/`)

Processes ASAP Crop Mask (TIFF) and maps it to administrative regions.

| Script | Description |
|--------|-------------|
| **`tiff_to_parquet.py`** | Extracts crop pixels from large TIFF files and converts them to Parquet. |
| `map_to_admin.py` | Maps Parquet point data to administrative regions (Admin2). |
| `visualize.py` | Visualizes TIFF files and saves them as PNG. |

### 🚀 Usage
```bash
python src/data_pipeline/crop_mask/tiff_to_parquet.py
python src/data_pipeline/crop_mask/map_to_admin.py
```

---

## 5. 🌃 Night Lights Pipeline (`src/data_pipeline/night_lights/`)

Downloads and aggregates NASA Black Marble (VNP46A4) data.

| Script | Description |
|--------|-------------|
| **`download.py`** | Downloads annual night light data. (Requires `BEARER_TOKEN` in `.env`) |
| `process.py` | Aggregates downloaded data by administrative region. |

### 🚀 Usage
```bash
python src/data_pipeline/night_lights/download.py
python src/data_pipeline/night_lights/process.py
```

---

## 6. ⚖️ Spatial Weighting Pipeline (`src/data_pipeline/spatial_weighting/`)

Calculates climate indices weighted by crop production areas, rather than simple spatial averages.

| Script | Description |
|--------|-------------|
| **`spam_loader.py`** | Loads SPAM 2020 data (Crop Distribution). |

**Note**: `climate_aggregator.py` and `generate_weighted_indices.py` have been moved to `src/data_pipeline/spi/`.

