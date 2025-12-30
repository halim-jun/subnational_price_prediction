import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
DATA_DIR = Path("data/analysis_results/eth_analysis")
OUTPUT_DIR = Path("data/analysis_results/eth_analysis/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_price_vs_spi():
    # Load data by re-running the aggregation part or loading cache if we saved intermediates.
    # The analysis script saved no intermediates, but `eth_agg_spi_all_scales.csv` exists.
    # We need to reload Price data.
    
    print("Loading Data for Plotting...")
    
    # 1. SPI
    spi_df = pd.read_csv("data/processed/spi/06_spi_csv/eth_agg_spi_all_scales.csv")
    spi_df['month'] = pd.to_datetime(spi_df['month'])
    
    # 2. Price (Re-load logic roughly)
    price_path = "data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv"
    df = pd.read_csv(price_path)
    df = df[df['countryiso3'] == 'ETH']
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # Maize
    maize_df = df[df['commodity'].str.contains('Maize', case=False)].groupby('month')['usdprice'].median().reset_index()
    
    # Merge
    merged = pd.merge(maize_df, spi_df, on='month', how='inner')
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot PPS (Price)
    ax1.plot(merged['month'].values, merged['usdprice'].values, color='tab:red', label='Maize Price (USD)', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD/kg)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    # Plot SPI (Right Axis)
    ax2 = ax1.twinx()
    # Use SPI 24 as it was 'Best'
    ax2.plot(merged['month'].values, merged['spi_24'].values, color='tab:blue', label='SPI 24-Month', linewidth=2, linestyle='--')
    ax2.set_ylabel('SPI (24-Month)', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # Add horizontal lines for Drought/Flood
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(-0.5, color='orange', linestyle=':', alpha=0.5, label='Drought Threshold')
    ax2.axhline(0.5, color='green', linestyle=':', alpha=0.5, label='Flood Threshold')
    
    # Shading
    ax2.fill_between(merged['month'], merged['spi_24'], 0.5, where=(merged['spi_24'] > 0.5), interpolate=True, color='green', alpha=0.1)
    ax2.fill_between(merged['month'], merged['spi_24'], -0.5, where=(merged['spi_24'] < -0.5), interpolate=True, color='orange', alpha=0.1)
    
    plt.title('Ethiopia: Maize Price vs SPI (24-Month Scale)')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "eth_maize_price_vs_spi24.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_price_vs_spi()
