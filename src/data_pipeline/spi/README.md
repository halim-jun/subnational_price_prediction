# SPI (Standardized Precipitation Index) Generation Guide

## 📌 Overview

This directory contains tools to generate **Standardized Precipitation Index (SPI)** from CHIRPS precipitation data.
SPI is a meteorological drought index that represents how wet or dry a period is compared to the long-term average, standardized to a normal distribution.

---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install climate-indices scipy
```

### 2. Run Pipeline

The `run_spi_generation.py` script handles everything: downloading data, clipping to East Africa, and calculating SPI.

```bash
# General usage
python run_spi_generation.py --download-chirps

# Recommended: 30-year calibration (WMO Standard)
python run_spi_generation.py \
  --download-chirps \
  --year-start 2016 --year-end 2024 \
  --calibration-start 1991 --calibration-end 2020
```

---

## 📂 File Structure

```
src/data_pipeline/spi/
├── README.md                     # This file
├── run_spi_generation.py         # Main entry point script
├── generate_spi_python.py        # Core SPI calculation logic (Pure Python)
├── convert_nc_to_csv.py          # Helper to convert NetCDF output to CSV
└── enrich_all_spi.py             # Helper to add administrative boundaries
```

---

## 🔄 Pipeline Steps

1.  **Clip**: Extracts East Africa region from global CHIRPS data.
2.  **Fill**: Interpolates missing values near coastlines.
3.  **Metadata**: Standardizes units to `mm` and fixes time attributes.
4.  **Reorder**: Transposes dimensions to `(lat, lon, time)` for calculation.
5.  **Calculate**: Computes SPI using Gamma distribution (via `climate-indices`).
6.  **Finalize**: Reorders back to `(time, lat, lon)` and saves as CF-compliant NetCDF.

---

## 📊 Outputs

Files are saved in `data/processed/spi/05_spi_final/`:

-   `east_africa_spi_gamma_01_month.nc`
-   `east_africa_spi_gamma_03_month.nc`
-   ...and so on for other scales.

---

## 🔧 Troubleshooting

### 1. "SPI calculation failed" (Calibration Period Mismatch)
**Error:** `Command '['spi', ...]' returned non-zero exit status 1.`
**Cause:** The specified calibration period (e.g., 1991-2020) falls outside the available data range.
**Fix:** The script now attempts to auto-adjust. If it fails, manually ensure your `--year-start` includes the calibration period, or adjust `--calibration-start`.

### 2. "climate-indices not found"
**Error:** `FileNotFoundError: 'spi' command not found`
**Fix:** Install the package:
```bash
pip install climate-indices
```

### 3. Memory Errors
**Cause:** Processing a huge area or time range.
**Fix:** Reduce the scope using `--lon-min/max` or `--year-start/end`, or calculate fewer scales at a time:
```bash
python run_spi_generation.py --scales 3 6 12
```

### 4. Missing Data Warning
**Warning:** `High percentage of missing values`
**Fix:** The pipeline includes an automatic filling step. Ensure it ran correctly. If using custom data, you may need to preprocess it to fill NaNs over land.
