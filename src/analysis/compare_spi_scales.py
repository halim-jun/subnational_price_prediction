import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
SPI_DIR = Path("data/processed/spi/06_spi_csv")
OUTPUT_DIR = Path("data/analysis_results/eth_analysis")

def compare_scales():
    print("Comparing SPI Scales for Maize...")
    
    # 1. Load Data (Using the previously cached aggregation if possible, or re-load)
    # We'll use the cache created by eth_price_impact_analysis.py
    spi_cache = SPI_DIR / "eth_agg_spi_all_scales.csv"
    if not spi_cache.exists():
        print("Error: SPI cache not found. Please run eth_price_impact_analysis.py first.")
        return
        
    spi_df = pd.read_csv(spi_cache)
    if 'month' in spi_df.columns:
        spi_df['month'] = pd.to_datetime(spi_df['month'])

    # 2. Load Price (Maize only for clarity)
    price_path = "data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv"
    df = pd.read_csv(price_path)
    df = df[df['countryiso3'] == 'ETH']
    
    # Normalize 100 KG
    mask = df['unit'] == '100 KG'
    df.loc[mask, 'usdprice'] = df.loc[mask, 'usdprice'] / 100
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    maize_df = df[df['commodity'].str.contains('Sorghum', case=False)].groupby('month')['usdprice'].median().reset_index()
    
    # 3. Merge
    merged = pd.merge(maize_df, spi_df, on='month', how='inner')
    
    # 4. Calculate Correlations
    scales = [1, 3, 6, 12, 24]
    corrs = {}
    
    for s in scales:
        col = f'spi_{s}'
        if col in merged.columns:
            # We care about impact on PRICE. 
            # Drought (Negative SPI) -> High Price. 
            # So Correlation should be NEGATIVE.
            c = merged['usdprice'].corr(merged[col])
            corrs[s] = c
            
    # 5. Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(corrs.keys()), y=list(corrs.values()), palette='coolwarm_r')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel('SPI Timescale (Months)')
    plt.ylabel('Correlation with Sorghum Price (USD)')
    plt.title('Which SPI Scale predicts Price best?\n(Stronger Negative Correlation = Better predictor of Drought Impact)')
    
    # Add labels
    for i, v in enumerate(corrs.values()):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom' if v > 0 else 'top')
        
    plt.tight_layout()
    save_path = OUTPUT_DIR / "spi_scale_comparison_sorghum.png"
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    print("Correlations:", corrs)

if __name__ == "__main__":
    compare_scales()
