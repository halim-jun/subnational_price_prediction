import pandas as pd
import os
from pathlib import Path

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # src/data_pipeline/macro -> src/data_pipeline -> src -> root

INPUT_FILE = PROJECT_ROOT / "data/raw/worldbank_commodity/CMO-Historical-Data-Monthly.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/external"
OUTPUT_FILE = OUTPUT_DIR / "worldbank_indices.csv"

def process_wb_data():
    """
    Reads World Bank Pink Sheet Excel file, extracts Energy, Food, and Fertilizer Indices.
    Converts YYYYMdd format dates to YYYY-MM-DD.
    Saves to CSV.
    """
    print(f"Reading {INPUT_FILE}...")
    
    try:
        # Determine start row (data starts at '1960M01')
        df_sheet = pd.read_excel(INPUT_FILE, sheet_name='Monthly Indices', header=None)
        start_row_idx = df_sheet[df_sheet[0].astype(str) == '1960M01'].index[0]
        
        # Column Indices (0-based) based on inspection:
        # 0: Date
        # 1: Total Index (skip)
        # 2: Energy
        # 6: Food
        # 12: Fertilizers
        
        sub_df = df_sheet.iloc[start_row_idx:, [0, 2, 6, 12]].copy()
        sub_df.columns = ['date_str', 'energy_index', 'food_index', 'fertilizer_index']
        
        # Drop empty rows
        sub_df = sub_df.dropna(how='all', subset=['energy_index', 'food_index', 'fertilizer_index'])

        # Convert Date '1960M01' -> '1960-01-01'
        def parse_date(d_str):
            try:
                return pd.to_datetime(str(d_str), format='%YM%m')
            except:
                return None

        sub_df['date'] = sub_df['date_str'].apply(parse_date)
        sub_df = sub_df.dropna(subset=['date'])
        
        # Enforce numeric
        for col in ['energy_index', 'food_index', 'fertilizer_index']:
            sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
            
        sub_df = sub_df.sort_values('date')
        
        final_df = sub_df[['date', 'energy_index', 'food_index', 'fertilizer_index']]
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved processed indices to {OUTPUT_FILE}")
        print(final_df.head())
        print(final_df.tail())

    except Exception as e:
        print(f"Error processing WB data: {e}")

if __name__ == "__main__":
    process_wb_data()
