import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Config
SPI_DIR = Path("data/processed/spi/06_spi_csv")
WB_PATH = Path("data/processed/external/worldbank_indices.csv")
PRICE_PATH = Path("data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv")
OUTPUT_DIR = Path("src/analysis/analysis_results/eth_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPI_TIMESCALES = [1, 3, 6, 12, 24]

def load_eth_price_data():
    """Loads price data, filters for Ethiopia, and aggregates to National Median."""
    print("Loading ETH Price Data...")
    df = pd.read_csv(PRICE_PATH)
    
    # Filter Ethiopia
    df = df[df['countryiso3'] == 'ETH'].copy()
    
    # Standardize Date
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # Major Commodities (Simplify names)
    target_commodities = ['Maize', 'Sorghum', 'Wheat', 'Teff', 'Barley'] # Teff is key for ETH
    
    def simplify_name(name):
        for c in target_commodities:
            if c.lower() in name.lower():
                return c
        return name # Keep original if not matched, or filter?
    
    df['commodity_group'] = df['commodity'].apply(simplify_name)
    # Filter to only the simplified targets + maybe others common in ETH?
    # Let's keep all but analyze top ones.
    
    price_agg = df.groupby(['month', 'commodity_group'])['usdprice'].median().reset_index()
    return price_agg

def normalize_units(df):
    """Normalizes price to per KG."""
    # 100 KG -> Divide Price by 100
    mask_100 = df['unit'] == '100 KG'
    if mask_100.any():
        print(f"  Normalizing {mask_100.sum()} rows from 100 KG to KG...")
        df.loc[mask_100, 'usdprice'] = df.loc[mask_100, 'usdprice'] / 100
        df.loc[mask_100, 'unit'] = 'KG'
        
    # Check for other units? (L, etc)
    # For now, just fix the major distortion (100x).
    return df

def load_eth_price_data():
    """Loads price data, filters for Ethiopia, and aggregates to National Median."""
    print("Loading ETH Price Data...")
    df = pd.read_csv(PRICE_PATH)
    
    # Filter Ethiopia
    df = df[df['countryiso3'] == 'ETH'].copy()
    
    # Normalize Units BEFORE Aggregation
    df = normalize_units(df)
    
    # Standardize Date
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    # Major Commodities (Simplify names)
    target_commodities = ['Maize', 'Sorghum', 'Wheat', 'Teff', 'Barley', 'Beans'] # Added Beans
    
    def simplify_name(name):
        for c in target_commodities:
            if c.lower() in name.lower():
                return c
        return name 
    
    df['commodity_group'] = df['commodity'].apply(simplify_name)
    
    # Aggregate: Median Price per Month per Commodity (National Level)
    price_agg = df.groupby(['month', 'commodity_group'])['usdprice'].median().reset_index()
    return price_agg

def load_wb_data():
    """Loads World Bank indices."""
    print("Loading World Bank Data...")
    df = pd.read_csv(WB_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    return df[['month', 'energy_index', 'food_index']]

def load_eth_spi_data():
    """Loads SPI data, filters for ETH, aggregates to National Mean."""
    spi_data = {}
    cache_path = SPI_DIR / "eth_agg_spi_all_scales.csv"
    
    if cache_path.exists():
        print("Loading cached ETH SPI data...")
        return pd.read_csv(cache_path)

    for scale in SPI_TIMESCALES:
        print(f"Processing SPI {scale} month for ETH...")
        # Prefer the pre-filtered file if available (e.g. 3 month)
        # But for consistency, let's use the main files if we can, or check specific ETH files.
        # User only pointed out 'eth_spi_gamma_3_month.csv'.
        # We will try to read from the large files but strictly filter for ETH.
        
        file_path = SPI_DIR / f"east_africa_spi_gamma_{scale}_month_with_boundaries.csv"
        if not file_path.exists():
            # Check for alternative name or skip
            if scale == 3 and (SPI_DIR / "eth_spi_gamma_3_month.csv").exists():
                 file_path = SPI_DIR / "eth_spi_gamma_3_month.csv"
            else:
                 print(f"  File not found {file_path}")
                 continue
        
        spi_col = f"spi_gamma_{scale}_month"
        usecols = ['time', 'country_iso', spi_col]
        dtypes = {'country_iso': str, spi_col: float}
        
        try:
            # Check cols
            header = pd.read_csv(file_path, nrows=0)
            if spi_col not in header.columns:
                 # In 'eth_spi_gamma_3_month.csv', cols might be slightly different.
                 # Checked head: 'spi_gamma_3_month' exists. 'country_iso' exists.
                 pass
            
            chunk_agg = []
            chunk_size = 1000000
            
            for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunk_size, dtype=dtypes):
                # Filter ETH
                eth_chunk = chunk[chunk['country_iso'] == 'ETH'].copy()
                if eth_chunk.empty:
                    continue
                
                eth_chunk['time'] = pd.to_datetime(eth_chunk['time'])
                eth_chunk['month'] = eth_chunk['time'].dt.to_period('M').dt.to_timestamp()
                
                # Agg
                gr = eth_chunk.groupby('month')[spi_col].agg(['sum', 'count']).reset_index()
                chunk_agg.append(gr)
                
            if chunk_agg:
                full = pd.concat(chunk_agg)
                final = full.groupby('month').sum().reset_index()
                final[f'spi_{scale}'] = final['sum'] / final['count']
                
                spi_data[scale] = final[['month', f'spi_{scale}']]
                print(f"  Loaded {len(final)} months for SPI {scale}")
            else:
                print(f"  No ETH data found in SPI {scale}")
                
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            continue

    # Merge all scales
    if not spi_data:
        return pd.DataFrame()
    
    base_df = None
    for scale, df in spi_data.items():
        if base_df is None:
            base_df = df
        else:
            base_df = pd.merge(base_df, df, on='month', how='outer')
            
    # Save cache
    base_df.to_csv(cache_path, index=False)
    return base_df

def create_spi_vars(df, scale):
    """Creates Drought and Flood variables."""
    col = f'spi_{scale}'
    # Drought: spi if spi < -0.5 else 0
    df[f'spi_drought_{scale}'] = df[col].apply(lambda x: x if x < -0.5 else 0)
    # Flood: spi if spi > 0.5 else 0
    df[f'spi_flood_{scale}'] = df[col].apply(lambda x: x if x > 0.5 else 0)
    return df

def run_analysis():
    # Load
    price_df = load_eth_price_data()
    wb_df = load_wb_data()
    spi_df = load_eth_spi_data()
    
    if spi_df.empty:
        print("No SPI Data found.")
        return
        
    # Convert month cols
    if 'month' in spi_df.columns:
        spi_df['month'] = pd.to_datetime(spi_df['month'])
    
    results = []
    
    # Analyze by Commodity
    # Determine best scale globally or per commodity? 
    # Usually consistent scale is better for comparison, but let's see corr.
    
    all_merged_data = []

    for com in price_df['commodity_group'].unique():
        # Clean Com name for files
        com_safe = com.replace("/", "_").replace(" ", "")
        print(f"\nAnalyzing {com}...")
        
        cvar = price_df[price_df['commodity_group'] == com].copy()
        
        # Merge
        merged = pd.merge(cvar, spi_df, on='month', how='inner')
        merged = pd.merge(merged, wb_df, on='month', how='left')
        
        valid_scales = [c for c in merged.columns if c.startswith('spi_') and 'drought' not in c and 'flood' not in c]
        valid_scales = [int(s.split('_')[1]) for s in valid_scales]
        
        # Correlation with raw SPI to pick 'best fit' timescale?
        # Or should we correlate with the drought var? 
        # Let's correlate with raw SPI first to find the relevant timeline.
        corrs = {}
        for s in valid_scales:
            dat = merged.dropna(subset=['usdprice', f'spi_{s}'])
            if len(dat) > 20:
                # Expect negative correlation for Drought->Price (Low SPI -> High Price)
                corrs[s] = dat['usdprice'].corr(dat[f'spi_{s}'])
        
        if not corrs:
            continue
            
        # Select scale with strongest NEGATIVE correlation (assuming drought risk is primary driver)
        # But if Flood is driver, it might be positive? 
        # Let's pick max absolute correlation? No, usually we care about "Anomalies".
        # User prompt: "Drought OR Flood impact". 
        # Let's start with the scale that has the strongest magnitude correlation.
        best_scale = max(corrs, key=lambda k: abs(corrs[k]))
        print(f"  Best Scale: {best_scale} month (Corr: {corrs[best_scale]:.3f})")
        
        # Create Vars for ALL scales (so downstream models can switch)
        for s in valid_scales:
            merged = create_spi_vars(merged, s)
            
        # Mark best scale for export (reference)
        merged['selected_scale'] = best_scale
        all_merged_data.append(merged)
        
        # Regression
        # log(P) ~ Drought_Var + Flood_Var + log(Energy) + log(Food)
        # Note: Drought_Var is negative numbers. So Coef should be NEGATIVE (More negative drought -> Higher price => Negative * Negative = Positive effect??)
        # Wait: 
        # If SPI = -2 (Drought), and Price goes UP.
        # Reg: Price = Beta * SPI. 
        # High Price = Beta * (-2). Beta must be NEGATIVE.
        # So we expect Beta_Drought < 0.
        
        # Flood: SPI = +2. Price goes UP (maybe?).
        # High Price = Beta * (+2). Beta must be POSITIVE.
        
        adat = merged.dropna(subset=['usdprice', f'spi_drought_{best_scale}', f'spi_flood_{best_scale}', 'energy_index', 'food_index'])
        if len(adat) < 20: continue
        
        adat['log_price'] = np.log(adat['usdprice'] + 1e-6)
        adat['log_energy'] = np.log(adat['energy_index'] + 1e-6)
        adat['log_food'] = np.log(adat['food_index'] + 1e-6)
        
        formula = f"log_price ~ spi_drought_{best_scale} + spi_flood_{best_scale} + log_energy + log_food"
        
        try:
            model = sm.OLS.from_formula(formula, data=adat).fit()
            
            with open(OUTPUT_DIR / f"eth_reg_{com_safe}.txt", "w") as f:
                f.write(model.summary().as_text())
                
            results.append({
                'commodity': com,
                'scale': best_scale,
                'corr_raw': corrs[best_scale],
                'coef_drought': model.params.get(f'spi_drought_{best_scale}', 0),
                'pval_drought': model.pvalues.get(f'spi_drought_{best_scale}', 1),
                'coef_flood': model.params.get(f'spi_flood_{best_scale}', 0),
                'pval_flood': model.pvalues.get(f'spi_flood_{best_scale}', 1),
                'r2': model.rsquared
            })
        except Exception as e:
            print(f"  Reg failed: {e}")
            
    if all_merged_data:
        full_df = pd.concat(all_merged_data, ignore_index=True)
        # Save merged data
        save_path = OUTPUT_DIR / "eth_merged_data.csv"
        full_df.to_csv(save_path, index=False)
        print(f"Merged Dataset saved to {save_path}")

    if results:
        pd.DataFrame(results).to_csv(OUTPUT_DIR / "eth_price_impact_summary.csv", index=False)
        print("\nResults saved to src/analysis/analysis_results/eth_analysis/eth_price_impact_summary.csv")
        print(pd.DataFrame(results))

if __name__ == "__main__":
    run_analysis()
