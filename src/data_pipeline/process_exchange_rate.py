import pandas as pd
import os
import glob

# INPUT_DIR = "data/raw/macro" # User should put the file here
# Look for any csv in data/raw/macro if specific name not found?
INPUT_FILES = glob.glob("data/raw/macro/*.csv")

OUTPUT_DIR = "data/processed/external"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exchange_rate_usd_etb.csv")

def process_exchange_rate():
    print("Processing Exchange Rate...")
    if not INPUT_FILES:
        print("No CSV files found in data/raw/macro/ for Exchange Rate.")
        return

    # heuristic: use the first file found, or look for specific keywords
    target_file = INPUT_FILES[0] 
    print(f"Using file: {target_file}")

    try:
        df = pd.read_csv(target_file)
        
        # Heuristic column mapping
        # We need 'Date' and 'Rate'
        # Common names: 'Date', 'Time', 'Period' -> date
        # 'Price', 'Close', 'Value', 'Rate' -> value
        
        cols = df.columns.str.lower()
        
        date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
        val_col = next((c for c in df.columns if any(x in c.lower() for x in ['price', 'close', 'rate', 'value'])), None)

        if not date_col or not val_col:
            print(f"Could not identify Date/Value columns in {target_file}. Found: {list(df.columns)}")
            return

        df['date'] = pd.to_datetime(df[date_col])
        df['exchange_rate'] = pd.to_numeric(df[val_col], errors='coerce')
        
        df = df.dropna(subset=['date', 'exchange_rate'])
        df = df.sort_values('date')
        
        # Resample to month start if daily?
        # Check frequency
        # If many days per month, take mean or end? usually monthly avg is good.
        # Let's resample to MS (Month Start) and take mean
        df.set_index('date', inplace=True)
        monthly_df = df['exchange_rate'].resample('MS').mean().reset_index()
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        monthly_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved processed exchange rate to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error processing exchange rate: {e}")

if __name__ == "__main__":
    process_exchange_rate()
