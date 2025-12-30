import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Config
DATA_PATH = Path("data/analysis_results/eth_analysis/eth_merged_data.csv")
OUTPUT_DIR = Path("data/analysis_results/eth_analysis/forecasts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target Commodities
TARGETS = ['Maize', 'Wheat', 'Sorghum', 'Beans']

def run_sarimax_modeling():
    print("Loading Consolidated Data for Modeling...")
    df = pd.read_csv(DATA_PATH)
    df['month'] = pd.to_datetime(df['month'])
    
    results = []
    
    for com in TARGETS:
        com_safe = com.replace(" ", "")
        print(f"\nModeling {com}...")
        
        # Filter Data
        comm_df = df[df['commodity_group'].str.contains(com, case=False, na=False)].copy()
        
        if comm_df.empty:
            print(f"No data for {com}")
            continue
            
        # Ensure Time Series Frequency and Handle Duplicates
        # If multiple entries for same month, take mean (shouldn't happen if properly aggregated, but safety first)
        comm_df = comm_df.groupby('month').first().reset_index()
        
        comm_df = comm_df.sort_values('month').set_index('month')
        comm_df = comm_df.asfreq('MS') # Month Start
        
        # Interpolate missing prices if any (small gaps)
        comm_df['usdprice'] = comm_df['usdprice'].interpolate(method='linear')
        
        # Log Transform Target
        comm_df['log_price'] = np.log(comm_df['usdprice'] + 1e-6)
        
        # Determine best scale
        # Correlation-based selection (Raw)
        correlations = comm_df[[c for c in comm_df.columns if 'spi_' in c and 'drought' not in c]].corrwith(comm_df['usdprice'])
        best_scale_col = correlations.idxmin() # Strongest negative correlation
        try:
            best_scale = int(best_scale_col.split('_')[1])
        except:
            best_scale = 24
                
        print(f"  Using Best SPI Scale: {best_scale}")
        
        # Exogenous Vars
        # Feature Engineering: Add Lag (Commodity Specific)
        # Maize: Lag 7, Sorghum: Lag 9 (Best with Scale 24)
        if 'Sorghum' in com:
            lag_k = 9
        else:
            lag_k = 7
            
        comm_df[f'spi_drought_{best_scale}_lag{lag_k}'] = comm_df[f'spi_drought_{best_scale}'].shift(lag_k)
        comm_df[f'spi_flood_{best_scale}_lag{lag_k}'] = comm_df[f'spi_flood_{best_scale}'].shift(lag_k)
        
        # Also log transform Indices
        comm_df['log_energy'] = np.log(comm_df['energy_index'] + 1e-6)
        comm_df['log_food'] = np.log(comm_df['food_index'] + 1e-6)
        
        # Final Exog List (Use LAGGED SPI)
        # Note: We need to use FILLNA for the initial lagged Nans if we want to use the whole series, 
        # but standard practice is to drop the first K rows.
        final_exog = [f'spi_drought_{best_scale}_lag{lag_k}', f'spi_flood_{best_scale}_lag{lag_k}', 'log_energy', 'log_food']
        
        # Clean nulls (This will drop the first 7 months)
        model_df = comm_df[['log_price'] + final_exog].dropna()
        
        if len(model_df) < 24:
            print("  Not enough data.")
            continue
            
        # Train/Test Split (Last 12 Months as Test)
        test_size = 12
        train = model_df.iloc[:-test_size]
        test = model_df.iloc[-test_size:]
        
        # Fit SARIMAX
        # Order: (1, 1, 1) x (0, 0, 0, 12) basic starter.
        
        order = (1, 1, 1)
        seasonal_order = (1, 0, 0, 12)
        
        try:
            model = SARIMAX(train['log_price'], 
                            exog=train[final_exog],
                            order=order, 
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            
            fit = model.fit(disp=False)
            
            # Forecast
            forecast = fit.get_forecast(steps=test_size, exog=test[final_exog])
            pred_log = forecast.predicted_mean
            pred_price = np.exp(pred_log)
            
            actual_price = np.exp(test['log_price'])
            
            # Error metrics
            rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
            print(f"  RMSE: {rmse:.4f}")
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(train.index.values, np.exp(train['log_price']).values, label='Train')
            plt.plot(test.index.values, actual_price.values, label='Test (Actual)')
            plt.plot(test.index.values, pred_price.values, label='Forecast (SARIMAX)', linestyle='--')
            
            # CI
            conf_int = forecast.conf_int()
            lower = np.exp(conf_int.iloc[:, 0])
            upper = np.exp(conf_int.iloc[:, 1])
            plt.fill_between(test.index.values, lower.values, upper.values, color='green', alpha=0.1)
            
            plt.title(f"{com} Price Forecast (RMSE: {rmse:.2f})\nExog: SPI (Lag {lag_k}), Energy, Food")
            plt.legend()
            save_path = OUTPUT_DIR / f"forecast_{com_safe}.png"
            plt.savefig(save_path)
            plt.close()
            
            # Save Coeffs
            summary_path = OUTPUT_DIR / f"summary_{com_safe}.txt"
            with open(summary_path, "w") as f:
                f.write(fit.summary().as_text())
                
            results.append({
                'commodity': com,
                'rmse': rmse,
                'plot_path': str(save_path)
            })
            
        except Exception as e:
            print(f"  Model fit failed: {e}")

    # Summary
    if results:
        pd.DataFrame(results).to_csv(OUTPUT_DIR / "sarimax_summary.csv", index=False)
        print("\nSARIMAX Modeling Complete. Forecasts saved to data/analysis_results/eth_analysis/forecasts")

if __name__ == "__main__":
    run_sarimax_modeling()
