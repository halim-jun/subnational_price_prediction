import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# Config
DATA_PATH = Path("src/analysis/analysis_results/eth_analysis/eth_merged_data.csv")
OUTPUT_DIR = Path("src/analysis/analysis_results/eth_analysis")

def compare_definitions():
    print("Comparing SPI Definitions (Raw vs Split)...")
    
    # Load Data
    if not DATA_PATH.exists():
        print("Merged data not found. Run eth_price_impact_analysis.py first.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # Filter for Sorghum
    maize_df = df[df['commodity_group'].str.contains('Sorghum', case=False)].copy()
    
    # Use 24-month scale (Best identified previously)
    scale = 24
    cols = ['usdprice', f'spi_{scale}', f'spi_drought_{scale}', f'spi_flood_{scale}', 'energy_index', 'food_index']
    
    # Drop NAs
    model_df = maize_df[cols].dropna()
    
    # Log transforms
    model_df['log_price'] = np.log(model_df['usdprice'] + 1e-6)
    model_df['log_energy'] = np.log(model_df['energy_index'] + 1e-6)
    model_df['log_food'] = np.log(model_df['food_index'] + 1e-6)
    
    # Model A: Raw SPI
    print("\n--- Model A: Raw SPI ---")
    formula_a = f"log_price ~ spi_{scale} + log_energy + log_food"
    model_a = sm.OLS.from_formula(formula_a, data=model_df).fit()
    print(f"R-squared: {model_a.rsquared:.4f}")
    print(f"AIC: {model_a.aic:.4f}")
    print(model_a.params)
    
    # Model B: Split SPI (Drought + Flood)
    print("\n--- Model B: Split SPI (Drought + Flood) ---")
    formula_b = f"log_price ~ spi_drought_{scale} + spi_flood_{scale} + log_energy + log_food"
    model_b = sm.OLS.from_formula(formula_b, data=model_df).fit()
    print(f"R-squared: {model_b.rsquared:.4f}")
    print(f"AIC: {model_b.aic:.4f}")
    print(model_b.params)
    
    # Comparison Plot
    results = pd.DataFrame({
        'Model': ['Raw SPI', 'Split SPI'],
        'R-Squared': [model_a.rsquared, model_b.rsquared],
        'AIC': [model_a.aic, model_b.aic]
    })
    
    print("\nComparison Summary:")
    print(results)
    
    # Save comparison text
    with open(OUTPUT_DIR / "spi_definition_comparison.txt", "w") as f:
        f.write(f"Model A (Raw) R2: {model_a.rsquared:.4f}\n")
        f.write(f"Model B (Split) R2: {model_b.rsquared:.4f}\n")
        f.write("\nModel A Summary:\n")
        f.write(model_a.summary().as_text())
        f.write("\n\nModel B Summary:\n")
        f.write(model_b.summary().as_text())

if __name__ == "__main__":
    compare_definitions()
