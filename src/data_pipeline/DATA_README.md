# Data Pipeline Documentation

This directory contains the data processing pipelines for the project, organized by data source and domain.

## 📂 Directory Structure

```
src/data_pipeline/
├── macro/          # Macroeconomic and External Data Processing
└── spi/            # Precipitation (CHIRPS) & SPI Calculation
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
