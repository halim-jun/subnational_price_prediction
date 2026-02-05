# 🌾 East Africa Food Price & Prediction Project

This project aims to analyze and predict food prices in East Africa (focusing on Ethiopia) by integrating climate data (SPI), macroeconomic indicators, and historical price data.

## 📂 Project Structure

The source code is organized in `src/` as follows:

```
src/
├── data_pipeline/      # Data Acquisition & Processing
│   ├── spi/            # Precipitation data (CHIRPS) & SPI calculation
│   └── macro/          # Macroeconomic indicators (World Bank, Exchange Rates)
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
    - **Transformation**: `scripts/tiff_to_parquet.py` reads the TIFF using `rasterio`, calculates Latitude/Longitude for each active pixel, and saves the result as a Parquet file.
    - **Output**: `/tmp/asap_mask_crop_v04.parquet` (Saved to `/tmp` for permission reasons).
    - **Format**: Parquet file with columns: `longitude`, `latitude`, `value`.


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

