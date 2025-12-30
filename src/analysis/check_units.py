import pandas as pd
from pathlib import Path

PRICE_PATH = Path("data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv")

def check_units():
    print("Checking Units for Key Commodities...")
    df = pd.read_csv(PRICE_PATH)
    eth = df[df['countryiso3'] == 'ETH']
    
    targets = ['Maize', 'Sorghum', 'Wheat', 'Teff', 'Beans']
    
    for t in targets:
        print(f"\n--- {t} ---")
        subset = eth[eth['commodity'].str.contains(t, case=False)]
        print(subset['unit'].value_counts())
        
        # Check if price magnitude correlates with unit?
        for unit in subset['unit'].unique():
            u_df = subset[subset['unit'] == unit]
            mean_p = u_df['usdprice'].mean()
            print(f"  Unit: {unit} | Mean Price (USD): {mean_p:.2f} | Count: {len(u_df)}")

if __name__ == "__main__":
    check_units()
