import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Config
SPI_DIR = Path("data/processed/spi/06_spi_csv")
WB_PATH = Path("data/processed/external/worldbank_indices.csv")
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
    
    # Filter for major commodities (Maize, Sorghum, Wheat) - adjusting based on actual names if needed
    # Let's check unique commodities first if this were interactive, but for now we'll pick common ones.
    # WFP data often has "Maize", "Maize (white)", etc. 
    # For now, let's just take the top 5 most frequent commodities to be safe, or search for "Maize"
    target_commodities = ['Maize', 'Sorghum', 'Wheat', 'Rice', 'Beans']
    
    # specific filtering often needed, but let's try strict matching first or partial
    mask = df['commodity'].apply(lambda x: any(c in x for c in target_commodities))
    df = df[mask]
    
    # Simplify Commodity Names for aggregation
    def simplify_name(name):
        for c in target_commodities:
            if c in name:
                return c
        return name
    
    df['commodity_group'] = df['commodity'].apply(simplify_name)
    
    # Aggregate to Country-Month level (Mean Price)
    # Using 'usdprice' to be comparable across countries
    price_agg = df.groupby(['countryiso3', 'month', 'commodity_group'])['usdprice'].mean().reset_index()
    return price_agg

def load_and_prep_wb_data():
    """Loads World Bank indices."""
    print("Loading World Bank Data...")
    df = pd.read_csv(WB_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # Calculate log or keeps as is? Regression usually uses log for prices.
    # We will log transform in the regression step.
    return df[['month', 'energy_index', 'food_index', 'fertilizer_index']]

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
        
        # Define types to speed up parsing and avoid warnings
        # Note: time is parsed as object then converted.
        dtypes = {'country_iso': str, spi_col: float}
        
        try:
            # Check columns first
            header = pd.read_csv(file_path, nrows=0)
            if spi_col not in header.columns:
                 print(f"Warning: Column {spi_col} not found in {file_path}")
                 continue
                 
            chunk_size = 1000000 
            agg_chunks = []
            
            # Use engine='c' and dtype
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
            print(f"  Finished processing SPI {scale} month.")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if spi_dfs:
        all_spi = pd.concat(spi_dfs)
        # Pivot to have columns spi_1, spi_3, etc.
        pivot_spi = all_spi.pivot_table(index=['country_iso', 'month'], columns='scale', values='value').reset_index()
        pivot_spi.columns = ['countryiso3', 'month'] + [f'spi_{c}' for c in pivot_spi.columns[2:]]
        
        print(f"Saving cache to {cache_path}")
        pivot_spi.to_csv(cache_path, index=False)
        return pivot_spi
    else:
        return pd.DataFrame()

def run_analysis():
    # 1. Load Data
    price_df = load_and_prep_price_data()
    wb_df = load_and_prep_wb_data()
    spi_df = load_and_prep_spi_data() # Returns single DF now
    
    if price_df.empty:
        print("Error: No price data found.")
        return
    if spi_df.empty:
        print("Error: No SPI data found.")
        return

    # Master list of results
    results = []
    
    # Pre-merge SPI and WB (Control vars)
    # The spi_df already has spi_1, spi_3, etc.
    # Convert 'month' to datetime if lost in CSV roundtrip
    if 'month' in spi_df.columns:
        spi_df['month'] = pd.to_datetime(spi_df['month'])
    
    # Process per Commodity
    for commodity in price_df['commodity_group'].unique():
        print(f"\nAnalyzing Commodity: {commodity}")
        comm_df = price_df[price_df['commodity_group'] == commodity].copy()
        
        # Merge all data
        merged = pd.merge(comm_df, spi_df, on=['countryiso3', 'month'], how='inner')
        merged = pd.merge(merged, wb_df, on='month', how='left')
        
        # Iterate scales
        correlations = {}
        valid_scales = [c for c in merged.columns if c.startswith('spi_')]
        
        for scale_col in valid_scales:
             valid_data = merged.dropna(subset=['usdprice', scale_col])
             if len(valid_data) < 20: continue
             correlations[scale_col] = valid_data['usdprice'].corr(valid_data[scale_col])
        
        if not correlations:
            print(f"  No valid data for {commodity}")
            continue
            
        # Best Scale
        best_scale_col = min(correlations, key=correlations.get) # Most negative correlation
        best_corr = correlations[best_scale_col]
        best_scale = int(best_scale_col.split('_')[1])
        
        print(f"  Best SPI Scale: {best_scale} months (Corr: {best_corr:.3f})")
        
        # Regression
        reg_df = merged.dropna(subset=['usdprice', best_scale_col, 'energy_index', 'food_index'])
        if len(reg_df) < 20: continue

        # Variables
        reg_df['log_price'] = np.log(reg_df['usdprice'] + 1e-6)
        reg_df['log_energy'] = np.log(reg_df['energy_index'] + 1e-6)
        reg_df['log_food_idx'] = np.log(reg_df['food_index'] + 1e-6)
        
        if reg_df['countryiso3'].nunique() > 1:
            formula = f"log_price ~ {best_scale_col} + log_energy + log_food_idx + C(countryiso3)"
        else:
            formula = f"log_price ~ {best_scale_col} + log_energy + log_food_idx"

        try:
            model = sm.OLS.from_formula(formula, data=reg_df).fit()
            
            # Save Summary
            summary_str = model.summary().as_text()
            file_safe_comm = commodity.replace("/", "_").replace(" ", "_")
            with open(OUTPUT_DIR / f"regression_summary_{file_safe_comm}.txt", "w") as f:
                f.write(summary_str)
            
            results.append({
                'commodity': commodity,
                'best_spi_scale': best_scale,
                'correlation': best_corr,
                'spi_coefficient': model.params.get(best_scale_col, np.nan),
                'spi_pvalue': model.pvalues.get(best_scale_col, np.nan),
                'n_obs': int(model.nobs),
                'r_squared': model.rsquared
            })
            
        except Exception as e:
            print(f"  Regression failed: {e}")

    # Save Results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_DIR / "price_impact_summary.csv", index=False)
        print("\nAnalysis Complete. Summary saved to data/analysis_results/price_impact_summary.csv")
        print(res_df)
    else:
        print("No results generated.")

if __name__ == "__main__":
    run_analysis()
