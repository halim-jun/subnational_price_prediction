import pandas as pd
import os
from functools import reduce
from pathlib import Path

# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data/processed/external"
OUTPUT_FILE = PROCESSED_DIR / "external_variables_merged.csv"

def merge_external_data():
    print("Merging external variables...")
    
    # List of expected files
    # Note: process_wb_data.py now produces worldbank_indices.csv (All WB indices)
    # process_exchange_rate.py produces exchange_rate_usd_etb.csv
    # process_fao_index.py produces fao_food_price_index.csv (Optional now)
    
    files = {
        'wb_indices': 'worldbank_indices.csv',
        'exchange_rate': 'exchange_rate_usd_etb.csv',
        'fao_index': 'fao_food_price_index.csv'
    }
    
    dfs = []
    
    for key, filename in files.items():
        filepath = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            dfs.append(df)
        else:
            print(f"Warning: {filename} not found. Skipping.")
    
    if not dfs:
        print("No data to merge.")
        return

    # Outer join on date
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
    
    merged_df = merged_df.sort_values('date')
    
    # Filter to reasonable range 
    # e.g. from 2000 onwards for modeling relevance
    # merged_df = merged_df[merged_df['date'] >= '2000-01-01']
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved merged data to {OUTPUT_FILE}")
    print(merged_df.head())
    print(merged_df.tail())

if __name__ == "__main__":
    merge_external_data()
