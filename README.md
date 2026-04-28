# 🌾 East Africa Food Price & Prediction Project

This project aims to analyze and predict food prices in East Africa (focusing on Ethiopia) by integrating climate data (SPI), macroeconomic indicators, and historical price data.

## 📂 Project Structure

The source code is organized in `src/` as follows:

```
src/
├── data_pipeline/      # Data Acquisition & Processing
│   ├── spi/            # Precipitation data (CHIRPS) & SPI calculation
│   ├── macro/          # Macroeconomic indicators (World Bank, Exchange Rates)
│   ├── fews_net/       # Food Security Data
│   ├── crop_mask/      # Crop Mask Data
│   └── night_lights/   # Night Lights Data
│
├── notebook/           # Jupyter Notebooks
│   ├── national_level_analysis.ipynb
│   └── subnational_level_prediction_baseline.ipynb
│
├── utils/              # Helper Scripts
│   └── fix_file_naming.py
├── data/
│   ├── raw/
│   │   ├── climate/            # CHIRPS Precipitation data
│   │   ├── worldbank_commodity/# World Bank Pink Sheet data
│   │   └── wfp/                # WFP Food Price data (Manual/Pre-downloaded)
│   └── processed/
│       ├── spi/                # Calculated SPI indices
│       └── external/           # Processed Macro indicators

```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ recommended.
- Required packages are listed in `requirements.txt`.

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/halim-jun/subnational_price_prediction
cd subnational_price_prediction

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate  # Windows

# Create processing directories
mkdir -p data

# Download  data
https://drive.google.com/file/d/1VnybB0lOMB3Dhg_wE4bZrurr0vZnI77-/view?usp=sharing

place the folder under the root (price_prediction_clean/data/..)

# Install dependencies
pip install -r requirements.txt


```


## 📊 Dashboard

Interactive Next.js dashboard with a FastAPI backend for exploring model performance and predictions.

```bash
# 1. Start the FastAPI backend
uvicorn src.dashboard.api.main:app --reload --port 8000

# 2. Start the Next.js frontend (in a separate terminal)
cd src/dashboard/frontend
npm install   # first time only
npm run dev
```

Frontend runs at http://localhost:3000, API at http://localhost:8000.

The frontend proxies `/api/*` to the backend URL set by `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). To run the API on a different port, set the env var before `npm run dev`, e.g. `NEXT_PUBLIC_API_URL=http://localhost:8001 npm run dev`.

Pages:
- **Overview** — Holdout test metrics (R², MAPE, RMSE), best/worst performing regions
- **Map** — Interactive Leaflet choropleth (predicted/actual/error views) with date slider
- **Time Series** — Per-region line charts, predicted vs actual scatter plots, MoM change analysis

## 📓 Notebooks

Interactive analysis is available in the `src/notebook/` directory.

*   `national_level_analysis.ipynb`: High-level trends and national aggregates.
*   `subnational_level_prediction_baseline.ipynb`: Granular, region-specific price predictions.

Major modeling work is done in `subnational_level_prediction_baseline.ipynb`
---




## 🛠️ Data Pipeline

It is unnecessary to run the data pipeline as the data is already processed and stored in the data folder.

1.  **Macro Economic Data**:
    ```bash
    python src/data_pipeline/macro/process_wb_data.py
    python src/data_pipeline/macro/merge_external_data.py
    ```

2.  **Climate Data (SPI)**:
    ```bash
    # Download CHIRPS data and generate SPI
    python src/data_pipeline/spi/run_spi_generation.py --download-chirps
    ```
    *(See `src/data_pipeline/spi/README.md` for detailed SPI instructions)*


---



## ✅ Implemented Data Pipelines

### 1. Climate Data (SPI) ✅
- **Location**: `src/data_pipeline/spi/`
- **Source**: CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
- **Key Scripts**:
  - `run_spi_generation.py`: Automated downloader and SPI calculator.
  - `generate_spi_python.py`: Core SPI computation logic (Gamma distribution).
  - `enrich_all_spi.py`: Adds administrative boundaries (Country, Region, Zone).
- **Features**:
  - Automatic download from CHC UCSB servers.
  - 30-year calibration (1991-2020) for robust drought indexing.
  - NetCDF to CSV conversion for modeling integration.

### 2. Macroeconomic Data ✅
- **Location**: `src/data_pipeline/macro/`
- **Source**: World Bank Commodity Markets (Pink Sheet)
- **Key Scripts**:
  - `process_wb_data.py`: Extracts Energy, Food, and Fertilizer indices.
  - `merge_external_data.py`: Merges various economic indicators into a master dataset.
- **Key Indicators**:
  - Energy Index (Oil, Gas, Coal)
  - Food Price Index
  - Fertilizer Index
- **Input**: Excel file (`data/raw/worldbank_commodity/*.xlsx`)

### 3. Food Price Data (WFP)
- **Source**: WFP VAM (Vulnerability Analysis and Mapping)
- **Status**: Raw data available in `data/raw/wfp/`.
- **Note**: Currently used as the target variable for forecasting models.

### 4. Crop Mask Data (ASAP)
- **Source**: ASAP Crop Mask (TIFF format)
- **Processing**:
    - The raw TIFF file (`data/crop_mask/asap_mask_crop_v04.tif`) is extremely large (29k x 80k pixels).
    - It has been processed to extract only **non-zero** pixels (pixels indicating crop presence).
    - **Transformation**: `src/data_pipeline/crop_mask/tiff_to_parquet.py` reads the TIFF using `rasterio`, calculates Latitude/Longitude for each active pixel, and saves the result as a Parquet file.
    - **Output**: The script generates `/tmp/asap_mask_crop_v04.parquet`.
    - **Note**: You should move this file to `data/crop_mask/` manually due to write permissions:
      ```bash
      mv /tmp/asap_mask_crop_v04.parquet data/crop_mask/
      ```
    - **Format**: Parquet file with columns: `longitude`, `latitude`, `value`.
    - **Spatial Mapping (Optional)**:
      To map these points to administrative regions (Admin Level 2), run the provided script. This requires `geopandas`.
      ```bash
      # Install geospatial dependencies
      pip install geopandas shapely pyarrow

      # Run the mapping script
      python src/data_pipeline/crop_mask/map_to_admin.py
      ```
      - **Output**: `data/crop_mask/admin_mapped/` (Parquet files partitioned by chunk).



### 5. Night Light Data (NASA Black Marble)
- **Source**: NASA Black Marble (VNP46A4 - Annual Nighttime Lights)
- **Scripts**:
    1. **Download**: `src/data_pipeline/night_lights/download.py`
       - Downloads annual NetCDF data.
       - Output: `data/night_lights/{Country}/{Country}_VNP46A4_{Year}.nc`
    2. **Processing**: `src/data_pipeline/night_lights/process.py`
       - Aggregates mean night light intensity by Admin2 regions (using GeoBoundaries).
       - Output: `data/processed/night_lights_admin2.parquet`
- **Usage**:
  ```bash
  # 1. Download Data (Requires BEARER_TOKEN in .env)
  python src/data_pipeline/night_lights/download.py

  # 2. Process to Admin2 Level
  python src/data_pipeline/night_lights/process.py
  ```



### 6. Food Security Data (FEWS NET)
- **Source**: FEWS NET IPC Phase Data (API)
- **Countries**: Somalia (SO), Kenya (KE), Ethiopia (ET)
- **Scripts**: 
    - `src/data_pipeline/fews_net/download.py`: Downloads data for each country separately (to handle API stability) and merges them.
- **Output**: `fewsnet/food_security/fewsnet_ipc_data.csv`
- **Usage**:
  ```bash
  python src/data_pipeline/fews_net/download.py
  ```

## 📊 Data Pipeline Usage

### Running the SPI Pipeline
```bash
# Download CHIRPS and generate SPI (Full Process)
python src/data_pipeline/spi/run_spi_generation.py --download-chirps
```

### Running the Macro Pipeline
```bash
# Process World Bank Economic Indicators
python src/data_pipeline/macro/process_wb_data.py
python src/data_pipeline/macro/merge_external_data.py
```

