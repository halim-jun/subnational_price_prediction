import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data/processed/subnational_merged_v3_KEN_SOM.parquet"
)
GEOBOUNDARIES_DIR = os.path.join(PROJECT_ROOT, "data/geoboundaries")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "artifact/model_output_stcv")
HOLDOUT_DIR = os.path.join(PROJECT_ROOT, "artifact/model_output_holdout")

TARGETS = ["c_food_price_index", "c_maize_fao", "c_sorghum"]

TARGET_DISPLAY = {
    "c_food_price_index": "Food Price Index",
    "c_maize_fao": "Maize (FAO)",
    "c_sorghum": "Sorghum",
}

TARGET_FILE_KEY = {
    "c_food_price_index": "food_price_index",
    "c_maize_fao": "maize",
    "c_sorghum": "sorghum",
}

HORIZONS = [1, 2, 3]

GROUP_COLORS = {
    "Autoregressive": "#E91E63",
    "FLDAS": "#2196F3",
    "Vegetation": "#4CAF50",
    "Climate Index": "#FF9800",
    "Static": "#9C27B0",
    "Conflict": "#F44336",
    "Spatial ID": "#607D8B",
    "Temporal": "#795548",
}
