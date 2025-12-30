import pandas as pd
from pathlib import Path
import re

PRICE_PATH = Path("data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv")

def extract_weight(unit_str):
    if not isinstance(unit_str, str):
        return 1.0
    
    # Regex to find the first number (integer or float)
    match = re.search(r'(\d+(\.\d+)?)', unit_str)
    if match:
        return float(match.group(1))
    
    return 1.0

def verify_units():
    print(f"Loading {PRICE_PATH}...")
    df = pd.read_csv(PRICE_PATH)
    
    display_df = df.copy()
    display_df = display_df[~display_df['commodity'].str.contains("Milling cost", case=False)]
    
    # 1. Inspect all unique units AFTER filtering
    print("\n--- Unique Units & Extracted Weights (After Filtering Milling Cost) ---")
    unique_units = display_df['unit'].unique()
    
    results = []
    for u in unique_units:
        w = extract_weight(u)
        results.append({'Unit': u, 'Extracted Weight': w})
        
    res_df = pd.DataFrame(results)
    
    # Show those that parsed successfully (Weight != 1.0)
    parsed = res_df[res_df['Extracted Weight'] != 1.0]
    print(f"\nParsed {len(parsed)} unit types successfully (Weight != 1). Examples:")
    print(parsed.head(10))
    
    # Show those that defaulted to 1.0
    defaulted = res_df[res_df['Extracted Weight'] == 1.0]
    print(f"\nDefaulted {len(defaulted)} unit types to 1.0. These might be non-KG units or 1 KG:")
    print(defaulted)
    
    # Diagnosis: Which commodities use "LCU/3.5kg" or "L"?
    print("\n--- Usage of Defaulted Units ---")
    for u in defaulted['Unit']:
        users = df[df['unit'] == u]['commodity'].unique()
        print(f"Unit '{u}' is used by: {users}")
    
    # 2. Check for "L" (Liters) or other common measures
    print("\n--- Potential Non-KG Units ---")
    non_kg_keywords = ['L', 'LITER', 'ML', 'G', 'GRAM', 'PCS', 'HEAD', 'BUNCH', 'DOZEN']
    
    potential_misses = []
    for u in unique_units:
        u_upper = str(u).upper()
        if 'KG' not in u_upper:
            potential_misses.append(u)
            
    print(f"Units without 'KG' in name ({len(potential_misses)} types):")
    print(potential_misses)
    
    # 3. Apply to full dataset to see impact
    df['weight_kg'] = df['unit'].apply(extract_weight)
    
    # Check if any weights are 0 or negative (error case)
    errors = df[df['weight_kg'] <= 0]
    if not errors.empty:
        print(f"\nCRITICAL: Found {len(errors)} rows with weight <= 0!")
        print(errors[['unit', 'weight_kg']].head())
    else:
        print("\nNo weights <= 0 found.")

if __name__ == "__main__":
    verify_units()
