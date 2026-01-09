
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import requests
import geopandas as gpd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Settings
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# Paths
PRICE_DATA_PATH = "data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv"
SPI_DATA_PATH = "data/processed/spi/06_spi_csv/east_africa_spi_gamma_6_month_with_boundaries.csv"
CACHE_FILE = "data/processed/admin2_temp_cache.csv"

def main():
    # 1. Load Price Data
    print("Loading Price Data...")
    if not os.path.exists(PRICE_DATA_PATH):
        print(f"Error: Price data not found at {PRICE_DATA_PATH}")
        return

    df_price = pd.read_csv(PRICE_DATA_PATH)
    df_price['date'] = pd.to_datetime(df_price['date'])
    print(f"Columns: {df_price.columns}")

    # Standardize Units
    def extract_weight(unit_str):
        try:
            unit_str = str(unit_str).lower()
            if 'kg' in unit_str:
                numeric_part = ''.join([c for c in unit_str if c.isdigit() or c == '.'])
                if numeric_part:
                    return float(numeric_part)
                return 1.0 # Default to 1 if just "KG"
            elif 'l' in unit_str: 
                numeric_part = ''.join([c for c in unit_str if c.isdigit() or c == '.'])
                if numeric_part:
                    return float(numeric_part)
                return 1.0 
        except:
            pass
        return None

    df_price['weight_kg'] = df_price['unit'].apply(extract_weight)
    df_price = df_price[df_price['weight_kg'] > 0].copy()
    df_price['price_per_kg'] = df_price['usdprice'] / df_price['weight_kg']

    print(f"Price Data Loaded: {df_price.shape}")

    # Filter for key commodities
    target_commodities = ['Maize', 'Sorghum', 'Wheat', 'Rice', 'Beans', 'Teff']
    mask = df_price['commodity'].apply(lambda x: any(c in x for c in target_commodities))
    df_price = df_price[mask].copy()

    # Simplify Commodity Names
    def simplify_name(name):
        for c in target_commodities:
            if c in name: return c
        return name
    df_price['commodity_group'] = df_price['commodity'].apply(simplify_name)

    # Select key columns & Aggregate
    # Use 'countryiso3' instead of 'country'
    price_clean = df_price[['countryiso3', 'admin1', 'admin2', 'market', 'latitude', 'longitude', 'date', 'commodity_group', 'price_per_kg']].copy()
    # Rename for consistency
    price_clean.rename(columns={'countryiso3': 'country_iso'}, inplace=True)
    
    admin2_price = price_clean.groupby(['country_iso', 'admin2', 'date', 'commodity_group'])['price_per_kg'].mean().reset_index()

    print(f"Aggregated Admin2 Price Data: {admin2_price.shape}")


    # 2. Load & Aggregate SPI Data
    print("Loading SPI Data (6-month)...")
    if not os.path.exists(SPI_DATA_PATH):
        print(f"Error: SPI data not found at {SPI_DATA_PATH}")
        return

    try:
        cols_to_use = ['lat', 'lon', 'time', 'spi_gamma_6_month', 'country_iso', 'admin2']
        # Read a sample first or optimized read if too large. For now, try standard read.
        df_spi = pd.read_csv(SPI_DATA_PATH, usecols=cols_to_use)
        df_spi['date'] = pd.to_datetime(df_spi['time'])
        df_spi = df_spi.dropna(subset=['admin2'])
        
        # Calculate Centroids
        admin2_centroids = df_spi.groupby(['country_iso', 'admin2'])[['lat', 'lon']].mean().reset_index()
        
        # Aggregate SPI
        admin2_spi = df_spi.groupby(['country_iso', 'admin2', 'date'])['spi_gamma_6_month'].mean().reset_index()
        admin2_spi.rename(columns={'spi_gamma_6_month': 'spi_6m'}, inplace=True)
        
        print(f"Aggregated SPI Data: {admin2_spi.shape}")
        
    except Exception as e:
        print(f"Error loading SPI: {e}")
        return


    # 3. Load Temperature Data - SKIPPED AS PER USER REQUEST
    print("Skipping Temperature Data...")

    # 4. Merge Data
    print("Merging Datasets...")

    # DEBUG: Check keys
    print("Price Dates (Sample):", admin2_price['date'].head().values)
    print("SPI Dates (Sample):", admin2_spi['date'].head().values)
    
    # Normalize dates to 1st of month if needed
    admin2_price['date'] = admin2_price['date'].dt.to_period('M').dt.to_timestamp()
    admin2_spi['date'] = admin2_spi['date'].dt.to_period('M').dt.to_timestamp()

    print("Normalized Dates.")
    print("Price Admin2s:", admin2_price['admin2'].nunique())
    print("SPI Admin2s:", admin2_spi['admin2'].nunique())
    
    # Check intersection
    common_admins = set(admin2_price['admin2']).intersection(set(admin2_spi['admin2']))
    print(f"Common Admin2s: {len(common_admins)}")
    
    # Merge Price + SPI
    merged_df = pd.merge(admin2_price, admin2_spi, on=['country_iso', 'admin2', 'date'], how='inner') 
    print(f"Merge (Price+SPI) Size: {merged_df.shape}")
    
    if merged_df.empty:
        print("DEBUG: Merge failed. Check Country ISOs or Dates.")
        print("Price ISOs:", admin2_price['country_iso'].unique())
        print("SPI ISOs:", admin2_spi['country_iso'].unique())
    
    # Drop rows with missing critical data
    model_df = merged_df.dropna(subset=['price_per_kg', 'spi_6m']).copy()

    print(f"Merged Dataset Size: {model_df.shape}")


    # 5. Feature Engineering
    print("Feature Engineering...")
    model_df.sort_values(['country_iso', 'admin2', 'commodity_group', 'date'], inplace=True)

    # Lags
    for lag in [1, 2, 3, 6]:
        model_df[f'price_lag_{lag}'] = model_df.groupby(['country_iso', 'admin2', 'commodity_group'])['price_per_kg'].shift(lag)

    model_df['month'] = model_df['date'].dt.month
    model_df = model_df.dropna()

    print(f"Final Model Dataset: {model_df.shape}")
    
    # Save processed dataset for inspection
    model_df.to_csv("data/processed/admin2_model_dataset.csv", index=False)


    # 6. Modeling
    print("Training Model...")
    
    # Categorical Feature Encoding
    categorical_cols = ['commodity_group', 'country_iso', 'admin2']
    print(f"Encoding categorical features: {categorical_cols}")
    
    for col in categorical_cols:
        # Convert to category and then to numeric codes
        model_df[col] = model_df[col].astype('category').cat.codes

    features = ['spi_6m', 'month', 'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_6'] + categorical_cols
    target = 'price_per_kg'

    split_date = '2024-01-01'
    train = model_df[model_df['date'] < split_date]
    test = model_df[model_df['date'] >= split_date]

    if train.empty or test.empty:
        print("Error: Train or Test set is empty. Check dates or data availability.")
        return

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print("--- Model Performance ---")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_preds)):.4f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.4f}")
    print(f"Test MAE: {mean_absolute_error(y_test, test_preds):.4f}")
    print(f"Test R2: {r2_score(y_test, test_preds):.4f}")

    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Feature Importance ---")
    print(importances)
    
    # --- Visualization for Presentation ---
    print("Generating Visualizations...")
    os.makedirs("src/analysis/analysis_results/eth_notebook_results", exist_ok=True)
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index.tolist(), palette="viridis")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("src/analysis/analysis_results/eth_notebook_results/feature_importance.png")
    print("Saved feature_importance.png")
    
    # 2. Actual vs Predicted Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test.values, test_preds, alpha=0.1, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Price (USD/KG)")
    plt.ylabel("Predicted Price (USD/KG)")
    plt.title("Actual vs Predicted Prices (Test Set)")
    plt.tight_layout()
    plt.savefig("src/analysis/analysis_results/eth_notebook_results/actual_vs_predicted.png")
    print("Saved actual_vs_predicted.png")
    
    # 3. Time Series Example (Best Admin2)
    # Find an admin2 with good amount of test data
    sample_admin2 = test['admin2'].mode()[0]
    sample_comm = test[test['admin2'] == sample_admin2]['commodity_group'].mode()[0] 
    
    subset = model_df[(model_df['admin2'] == sample_admin2) & (model_df['commodity_group'] == sample_comm)].sort_values('date')
    
    if not subset.empty:
        # Re-predict for the whole subset 
        # Note: subset features must be processed same as training
        subset_preds = model.predict(subset[features])
        
        plt.figure(figsize=(14, 6))
        plt.plot(subset['date'].values, subset['price_per_kg'].values, label='Actual Price', marker='o', markersize=3)
        plt.plot(subset['date'].values, subset_preds, label='Predicted Price', linestyle='--', alpha=0.8)
        plt.axvline(pd.to_datetime(split_date), color='r', linestyle=':', label='Train/Test Split')
        plt.title(f"Price Prediction Example: Region {sample_admin2} - Commodity {sample_comm}")
        plt.xlabel("Date")
        plt.ylabel("Price (USD/KG)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("src/analysis/analysis_results/eth_notebook_results/timeseries_example.png")
        print("Saved timeseries_example.png")

if __name__ == "__main__":
    main()
