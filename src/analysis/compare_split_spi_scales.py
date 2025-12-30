import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
# We need to reload the raw SPI aggregations because the 'merged' file only kept the BEST scale (24).
# So we load the SPI cache and Price data again to reconstruct.
SPI_CACHE = Path("data/processed/spi/06_spi_csv/eth_agg_spi_all_scales.csv")
PRICE_PATH = Path("data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv")
WB_PATH = Path("data/processed/external/worldbank_indices.csv")
OUTPUT_DIR = Path("data/analysis_results/eth_analysis")

def compare_split_scales():
    print("Comparing Split SPI Across Timescales (1, 3, 6, 12, 24) for Maize...")
    
    # 1. Load Data
    spi_df = pd.read_csv(SPI_CACHE)
    if 'month' in spi_df.columns:
        spi_df['month'] = pd.to_datetime(spi_df['month'])
        
    wb_df = pd.read_csv(WB_PATH)
    wb_df['date'] = pd.to_datetime(wb_df['date'])
    wb_df['month'] = wb_df['date'].dt.to_period('M').dt.to_timestamp()
    
    df = pd.read_csv(PRICE_PATH)
    df = df[df['countryiso3'] == 'ETH']
    
    # Normalize 100 KG
    mask = df['unit'] == '100 KG'
    df.loc[mask, 'usdprice'] = df.loc[mask, 'usdprice'] / 100
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # filter Sorghum
    maize_df = df[df['commodity'].str.contains('Sorghum', case=False)].groupby('month')['usdprice'].median().reset_index()
    
    # Merge
    merged = pd.merge(maize_df, spi_df, on='month', how='inner')
    merged = pd.merge(merged, wb_df, on='month', how='left')
    
    results = []
    
    scales = [1, 3, 6, 12, 24]
    
    for s in scales:
        col = f'spi_{s}'
        if col not in merged.columns: continue
        
        # Create Split Vars
        # Drought: val if val < -0.5 else 0
        merged[f'drought_{s}'] = merged[col].apply(lambda x: x if x < -0.5 else 0)
        # Flood: val if val > 0.5 else 0
        merged[f'flood_{s}'] = merged[col].apply(lambda x: x if x > 0.5 else 0)
        
        # Prep Reg
        reg_df = merged.dropna(subset=['usdprice', f'drought_{s}', f'flood_{s}', 'energy_index', 'food_index']).copy()
        
        reg_df['log_price'] = np.log(reg_df['usdprice'] + 1e-6)
        reg_df['log_energy'] = np.log(reg_df['energy_index'] + 1e-6)
        reg_df['log_food'] = np.log(reg_df['food_index'] + 1e-6)
        
        formula = f"log_price ~ drought_{s} + flood_{s} + log_energy + log_food"
        
        try:
            model = sm.OLS.from_formula(formula, data=reg_df).fit()
            results.append({
                'scale': s,
                'r2': model.rsquared,
                'aic': model.aic,
                'coef_drought': model.params[f'drought_{s}'],
                'p_drought': model.pvalues[f'drought_{s}'],
                'coef_flood': model.params[f'flood_{s}'],
                'p_flood': model.pvalues[f'flood_{s}']
            })
        except:
            pass
            
    res_df = pd.DataFrame(results)
    print("\nComparison Results (Split SPI):")
    print(res_df)
    
    # Plot R2
    plt.figure(figsize=(8, 5))
    sns.barplot(data=res_df, x='scale', y='r2', palette='viridis')
    plt.title('Predictive Power (R²) of Split SPI Variables by Timescale\n(Higher is Better)')
    plt.xlabel('SPI Timescale (Months)')
    plt.ylabel('R-Squared')
    plt.savefig(OUTPUT_DIR / "split_spi_scale_comparison_sorghum.png")
    print(f"Saved plot to {OUTPUT_DIR / 'split_spi_scale_comparison_sorghum.png'}")

if __name__ == "__main__":
    compare_split_scales()
