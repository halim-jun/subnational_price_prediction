import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, ccf
from pathlib import Path
import seaborn as sns

DATA_PATH = Path("data/analysis_results/eth_analysis/eth_merged_data.csv")
OUTPUT_DIR = Path("data/analysis_results/eth_analysis")

def check_preprocessing_needs():
    print("Checking Data Quality & Preprocessing Needs...")
    df = pd.read_csv(DATA_PATH)
    df['month'] = pd.to_datetime(df['month'])
    
    # Filter Sorghum as proxy
    maize = df[df['commodity_group'].str.contains('Sorghum', case=False)].copy()
    maize = maize.sort_values('month').set_index('month')
    maize = maize.asfreq('MS')
    
    # 1. Missing Values
    missing = maize['usdprice'].isna().sum()
    print(f"\n1. Missing SORGHUM Prices: {missing} / {len(maize)} months")
    if missing > 0:
        print("   - Action: Interpolation is needed (already in model script).")
    
    # 2. Stationarity (ADF Test)
    # Check on Log Price
    log_price = np.log(maize['usdprice'].interpolate() + 1e-6)
    
    adf_result = adfuller(log_price.dropna())
    print(f"\n2. Stationarity (ADF Test) on Log Price:")
    print(f"   ADF Statistic: {adf_result[0]:.4f}")
    print(f"   p-value: {adf_result[1]:.4f}")
    
    if adf_result[1] > 0.05:
        print("   - Result: Non-Stationary (p > 0.05).")
        print("   - Action: SARIMAX must use d=1 (Integration).")
    else:
        print("   - Result: Stationary.")
    
    # Check First Difference
    diff_price = log_price.diff().dropna()
    adf_diff = adfuller(diff_price)
    print(f"   ADF on Differenced Price (d=1): p-value {adf_diff[1]:.4f}")
    
    # 3. Lag Analysis (Cross Correlation)
    # Check lagged impact of SPI (6-month split variables generally best, but let's check global best scale from file)
    # The file has 'selected_scale', let's use that (usually 24 for maize in previous run, or 6 in split comparison)
    # Let's check the specific SPI variable available in the merged file.
    # We'll calculate correlation for lags 0 to 12.
    
    # Validation: Check Lag for SPI 6 explicitly as we are switching to it.
    scale = 6 
    spi_col = f'spi_{scale}' # Raw SPI for lag check
    
    if spi_col not in maize.columns:
        # Fallback if raw col missing, use drought col
        spi_col = f'spi_drought_{scale}'
    
    target = log_price
    feature = maize[spi_col].fillna(0) # Standardized factor, 0 is mean
    
    # Align
    common = pd.concat([target, feature], axis=1).dropna()
    target = common.iloc[:, 0]
    feature = common.iloc[:, 1]
    
    # CCF
    lags = 12
    # ccf returns correlation at lag k (feature leads target?)
    # We want to know if PAST SPI affects FUTURE Price.
    # statsmodels ccf(x, y): calculates corr(x_t, y_{t-k})? 
    # Usually: ccf(y, x) -> correlation of y(t) and x(t-k).
    # Let's verify manually to be safe.
    
    corrs = []
    for lag in range(lags + 1):
        # Shift feature forward (Lag 1 means value from t-1 is aligned with t)
        feat_shifted = feature.shift(lag)
        c = target.corr(feat_shifted)
        corrs.append(c)
        
    print(f"\n3. Lagged Correlations (Price vs {spi_col} at t-lag):")
    lag_df = pd.DataFrame({'lag': range(lags+1), 'corr': corrs})
    print(lag_df)
    
    plt.figure(figsize=(8, 4))
    sns.barplot(data=lag_df, x='lag', y='corr', palette='coolwarm')
    plt.title(f'Cross-Correlation: {spi_col} (Lead) vs Price')
    plt.xlabel('Lag (Months)')
    plt.ylabel('Correlation')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.savefig(OUTPUT_DIR / "lag_analysis_sorghum.png")
    print(f"   - Saved Lag Plot to {OUTPUT_DIR / 'lag_analysis_sorghum.png'}")
    
    # Recommend Action
    max_lag = lag_df.loc[lag_df['corr'].abs().idxmax()]
    print(f"\n   - Strongest Lag: {int(max_lag['lag'])} months (Corr: {max_lag['corr']:.3f})")
    if int(max_lag['lag']) > 0:
        print(f"   - Recommendation: Add SPI lagged by {int(max_lag['lag'])} months as a feature.")

if __name__ == "__main__":
    check_preprocessing_needs()
