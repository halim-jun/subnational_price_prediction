# 🌾 East Africa Food Price & Prediction Project

This project aims to analyze and predict food prices in East Africa (focusing on Ethiopia) by integrating climate data (SPI), macroeconomic indicators, and historical price data. It uses SARIMAX models to forecast prices at both national and sub-national levels.

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
git clone https://github.com/halim-jun/price_prediction
cd wpf_colla_v2

# Create processing directories
mkdir -p data/raw data/processed

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Data Pipeline

To prepare the dataset for analysis, run the data pipelines:

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


## 📓 Notebooks

Interactive analysis is available in the `src/notebook/` directory.

*   `national_level_analysis.ipynb`: High-level trends and national aggregates.
*   `subnational_level_prediction_baseline.ipynb`: Granular, region-specific price predictions.

---
