import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Config
SPI_DIR = Path("data/processed/spi/06_spi_csv")
PRICE_PATH = Path("data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv")
OUTPUT_DIR = Path("data/analysis_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPI_TIMESCALES = [1, 3, 6, 12, 24]

def load_and_prep_price_data():
    """Loads and aggregates price data to Country-Month level for major crops."""
    print("Loading Price Data...")
    df = pd.read_csv(PRICE_PATH)
    
    # Standardize Date
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # Filter for major commodities
    target_commodities = ['Maize', 'Sorghum', 'Wheat', 'Rice', 'Beans', 'Teff']
    mask = df['commodity'].apply(lambda x: any(c in x for c in target_commodities))
    df = df[mask].copy()
    
    # Exclude non-food items (e.g., Milling cost)
    df = df[~df['commodity'].str.contains("Milling cost", case=False)]
    
    # Simplify Commodity Names
    def simplify_name(name):
        for c in target_commodities:
            if c in name:
                return c
        return name
    
    df['commodity_group'] = df['commodity'].apply(simplify_name)
    
    # Normalize Unit (handle 100 KG etc if present, though basic wfp data often checked individually)
    # The snippet from previous context showed simple loading. Let's assume usdprice is usable
    # but a quick check on unit:
    # If unit is 100 KG, usually price is for 100 KG. We want per KG or consistent unit.
    # For correlation, scaling unit doesn't change correlation, so raw usdprice is fine unless mixed units.
    # We will assume within a country-commodity, units are consistent or we rely on 'usdprice'.
    
    # Extract numerical weight from unit (e.g., "100 KG" -> 100, "KG" -> 1)
    def extract_weight(unit_str):
        if not isinstance(unit_str, str):
            return 1.0
        
        # Regex to find the first number (integer or float)
        # Matches "90", "3.5", "100" at start or inside string
        match = re.search(r'(\d+(\.\d+)?)', unit_str)
        if match:
            return float(match.group(1))
        
        # If no number found, assume 1.0 (e.g. "KG", "L")
        return 1.0

    df['weight_kg'] = df['unit'].apply(extract_weight)
    df['price_per_kg'] = df['usdprice'] / df['weight_kg']

    # Aggregate to Country-Month level (Mean Price per KG)
    price_agg = df.groupby(['countryiso3', 'month', 'commodity_group'])['price_per_kg'].mean().reset_index()
    return price_agg

def load_and_prep_spi_data():
    """Loads SPI data for all timescales and aggregates to Country-Month using chunks. Caches result."""
    cache_path = Path("data/processed/spi/agg_spi_country_month.csv")
    
    if cache_path.exists():
        print("Loading cached SPI data...")
        return pd.read_csv(cache_path)

    spi_dfs = []
    
    for scale in SPI_TIMESCALES:
        print(f"Processing SPI {scale} month...")
        file_path = SPI_DIR / f"east_africa_spi_gamma_{scale}_month_with_boundaries.csv"
        
        if not file_path.exists():
            print(f"Warning: File not found {file_path}")
            continue
            
        spi_col = f"spi_gamma_{scale}_month"
        usecols = ['time', 'country_iso', spi_col]
        dtypes = {'country_iso': str, spi_col: float}
        
        try:
            chunk_size = 1000000 
            agg_chunks = []
            
            for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunk_size, dtype=dtypes):
                chunk['time'] = pd.to_datetime(chunk['time'])
                chunk['month'] = chunk['time'].dt.to_period('M').dt.to_timestamp()
                
                # Group by Country-Month within chunk
                chunk_agg = chunk.groupby(['country_iso', 'month'])[spi_col].agg(['sum', 'count']).reset_index()
                agg_chunks.append(chunk_agg)
            
            # Combine chunks
            full_agg = pd.concat(agg_chunks)
            final_agg = full_agg.groupby(['country_iso', 'month']).sum().reset_index()
            final_agg['value'] = final_agg['sum'] / final_agg['count']
            final_agg['scale'] = scale
            
            spi_dfs.append(final_agg[['country_iso', 'month', 'scale', 'value']])
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if spi_dfs:
        all_spi = pd.concat(spi_dfs)
        pivot_spi = all_spi.pivot_table(index=['country_iso', 'month'], columns='scale', values='value').reset_index()
        pivot_spi.columns = ['countryiso3', 'month'] + [f'spi_{c}' for c in pivot_spi.columns[2:]]
        
        print(f"Saving cache to {cache_path}")
        pivot_spi.to_csv(cache_path, index=False)
        return pivot_spi
    else:
        return pd.DataFrame()

def run_split_analysis():
    # 1. Load Data
    price_df = load_and_prep_price_data()
    spi_df = load_and_prep_spi_data()
    
    if price_df.empty or spi_df.empty:
        print("Data load failed.")
        return

    # Ensure Date format
    if 'month' in spi_df.columns:
        spi_df['month'] = pd.to_datetime(spi_df['month'])
    
    results = []
    
    # 2. Iterate and Calculate split correlations
    for commodity in price_df['commodity_group'].unique():
        # print(f"Processing {commodity}...")
        comm_df = price_df[price_df['commodity_group'] == commodity].copy()
        
        merged = pd.merge(comm_df, spi_df, on=['countryiso3', 'month'], how='inner')
        
        valid_cols = [c for c in merged.columns if c.startswith('spi_')]
        
        for scale_col in valid_cols:
            scale = int(scale_col.split('_')[1])
            
            # Filter valid rows
            valid = merged.dropna(subset=['price_per_kg', scale_col])
            if len(valid) < 20: continue
            
            # Splits
            drought_df = valid[valid[scale_col] < 0]
            flood_df = valid[valid[scale_col] > 0] # strictly greater than 0
            
            # Correlations
            overall_corr = valid['price_per_kg'].corr(valid[scale_col])
            
            drought_corr = np.nan
            if len(drought_df) >= 10:
                drought_corr = drought_df['price_per_kg'].corr(drought_df[scale_col])
                
            flood_corr = np.nan
            if len(flood_df) >= 10:
                flood_corr = flood_df['price_per_kg'].corr(flood_df[scale_col])
                
            results.append({
                'commodity': commodity,
                'scale': scale,
                'n_total': len(valid),
                'n_drought': len(drought_df),
                'n_flood': len(flood_df),
                'corr_overall': overall_corr,
                'corr_drought': drought_corr,
                'corr_flood': flood_corr
            })
            
    # 3. Save & Visualize
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No results generated.")
        return

    csv_path = OUTPUT_DIR / "split_spi_correlation_results.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved split correlation results to {csv_path}")
    
    
    # Identify Best Fit Scale per Commodity Separately for Drought and Flood
    print("\nBest Fit Scale by Max Absolute Correlation (Separate):")
    best_results = []
    for comm in res_df['commodity'].unique():
        comm_res = res_df[res_df['commodity'] == comm].copy()
        
        # Best Drought
        comm_res['abs_drought'] = comm_res['corr_drought'].abs().fillna(0)
        best_drought_row = comm_res.loc[comm_res['abs_drought'].idxmax()]
        
        # Best Flood
        comm_res['abs_flood'] = comm_res['corr_flood'].abs().fillna(0)
        best_flood_row = comm_res.loc[comm_res['abs_flood'].idxmax()]
        
        best_results.append({
            'Commodity': comm,
            'Best_Drought_Scale': best_drought_row['scale'],
            'Drought_Corr': best_drought_row['corr_drought'],
            'Best_Flood_Scale': best_flood_row['scale'],
            'Flood_Corr': best_flood_row['corr_flood']
        })
    
    best_df = pd.DataFrame(best_results)
    print(best_df)

    # Top commodities by observation count for plotting
    top_comms = res_df.groupby('commodity')['n_total'].sum().nlargest(4).index.tolist()
    plot_df = res_df[res_df['commodity'].isin(top_comms)]
    
    # Melt for plotting
    melted = plot_df.melt(id_vars=['commodity', 'scale'], 
                          value_vars=['corr_overall', 'corr_drought', 'corr_flood'],
                          var_name='condition', value_name='correlation')
    
    melted['condition'] = melted['condition'].replace({
        'corr_overall': 'Overall',
        'corr_drought': 'Drought (SPI<0)',
        'corr_flood': 'Flood (SPI>0)'
    })
    
    # Plot
    plt.figure(figsize=(14, 8))
    g = sns.FacetGrid(melted, col="commodity", col_wrap=2, height=4, aspect=1.5, sharey=True)
    g.map_dataframe(sns.barplot, x="scale", y="correlation", hue="condition", 
                    palette={'Overall': 'gray', 
                             'Drought (SPI<0)': 'red', 
                             'Flood (SPI>0)': 'blue'},
                    alpha=0.8)
    g.add_legend()
    g.set_axis_labels("SPI Timescale (Months)", "Correlation")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Price Correlations: Drought vs Flood (Best Fit Selection)')
    
    plot_path = OUTPUT_DIR / "split_spi_correlation_plot.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    run_split_analysis()
