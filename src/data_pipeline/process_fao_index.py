import pandas as pd
import os
import glob

INPUT_FILES = glob.glob("data/raw/macro/*.csv") # Likely share same folder or subfolder
# Actually user might put it in data/raw/wfp or data/raw/macro? 
# Let's assume data/raw/macro for now as requested.

OUTPUT_DIR = "data/processed/external"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fao_food_price_index.csv")

def process_fao_index():
    print("Processing FAO Index...")
    # This might conflict with exchange rate if both are just 'random csvs'
    # Ideally look for filename keyword
    
    fao_files = [f for f in INPUT_FILES if 'fao' in f.lower() or 'food' in f.lower()]
    
    if not fao_files:
        print("No FAO/Food Index CSV files found in data/raw/macro/.")
        return

    target_file = fao_files[0]
    print(f"Using file: {target_file}")

    try:
        # FAO usually has a specific format, often rows of years/months if from web, or standard csv
        df = pd.read_csv(target_file)
        
        # Check standard FAO columns: Date, Food Price Index
        # Often: 'Date', 'Food Price Index'
        
        cols = df.columns.str.lower()
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        # FAO specific text often: 'Food Price Index'
        val_col = next((c for c in df.columns if 'food' in c.lower() and 'index' in c.lower()), None)
        
        if not val_col:
             # Try generic 'Index' or 'Value' if we are sure it's the FAO file
             val_col = next((c for c in df.columns if 'value' in c.lower() or 'index' in c.lower()), None)

        if not date_col or not val_col:
            print(f"Could not identify Date/Value columns in {target_file}. Found: {list(df.columns)}")
            return

        df['date'] = pd.to_datetime(df[date_col])
        df['fao_use_index'] = pd.to_numeric(df[val_col], errors='coerce')
        
        df = df.dropna(subset=['date', 'fao_use_index'])
        df = df.sort_values('date')
        
        # Resample to MS just in case
        df.set_index('date', inplace=True)
        monthly_df = df['fao_use_index'].resample('MS').mean().reset_index()
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        monthly_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved processed FAO index to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error processing FAO index: {e}")

if __name__ == "__main__":
    process_fao_index()
