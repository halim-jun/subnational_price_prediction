import pandas as pd
import json
import os
import sys

# Add path to utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
try:
    from utils.google_drive import read_drive_file
except ImportError:
    # Try local import if running from root
    sys.path.append('src')
    from utils.google_drive import read_drive_file

def run_analysis_from_drive():
    # 0. Try Drive (User Request)
    drive_url = 'https://drive.google.com/file/d/1In0phmpmTLejavhAz8MsdhfPPDXPz0gM/view?usp=drive_link'
    df = None
    try:
        print(f"Attempting to read from Drive: {drive_url}")
        df = read_drive_file(drive_url)
        print("Successfully read data from Drive!")
    except Exception as e:
        print(f"Drive load failed: {e}")
        return

    # 2. Process
    if df is not None:
        # 3. Filter for Target Countries
        target_countries = ['Kenya', 'Ethiopia', 'Somalia']
        
        # Standardize column name checking
        country_col = 'country' if 'country' in df.columns else 'adm0_name'
        
        if country_col in df.columns:
            # Filter rows
            country_filtered_df = df[
                (df[country_col].isin(target_countries))
            ]
            print(f"Filtered columns for {target_countries}.")
            print(f"Original rows: {len(df)}, Filtered rows: {len(country_filtered_df)}")
            
            # 4. Select Sorghum Related Columns
            sorghum_cols = [c for c in df.columns if 'sorghum' in c.lower()]
            print(f"Found {len(sorghum_cols)} Sorghum related columns: {sorghum_cols}")
            
            # Define Context Columns
            context_cols = [c for c in ['country', 'ISO3', 'adm1_name', 'mkt_name', 'DATES', 'year', 'month', 'currency'] if c in df.columns]
            
            # Combine columns
            final_cols = list(set(context_cols + sorghum_cols))
            
            # Create Final DataFrame
            final_df = country_filtered_df[final_cols]
            
            # Display result
            print("Head of Final Data:")
            print(final_df.head())
        else:
            print(f"Error: Country column '{country_col}' not found in dataset. Columns: {df.columns}")

if __name__ == "__main__":
    run_analysis_from_drive()
