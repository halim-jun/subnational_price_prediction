import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_pipeline.spatial_weighting.spam_loader import load_spam_data

# Config
SPAM_DIR = Path("data/raw/spam2020/Global_CSV/spam2020V2r0_global_harvested_area")
# Defines the SPI input file (CSV format)
SPI_CSV = Path("data/processed/spi/06_spi_csv/east_africa_spi_gamma_6_month_with_boundaries.csv")
OUTPUT_CSV = Path("data/processed/weighted_spi_indices_admin2.csv")

TARGET_CROPS = {
    'Maize': 'MAIZ',
    'Wheat': 'WHEA',
    'Sorghum': 'SORG',
    # 'Beans': 'BEAN' # Use if needed
}

def generate_subnational_weighted_indices(
    spi_csv_path: Path,
    spam_dir: Path,
    output_csv_path: Path = None,
    target_crops: dict = None,
    country_iso_filter: list = None,
):
    """
    Generates subnational (Admin2) weighted climate indices by combining SPI data (CSV)
    and SPAM crop distribution data (Raster/CSV).

    Args:
        spi_csv_path (Path): Path to the large SPI CSV file with admin boundaries.
        spam_dir (Path): Directory containing SPAM CSV files.
        output_csv_path (Path, optional): Path to save the resulting CSV. Returns DataFrame if None.
        target_crops (dict, optional): Dictionary of {CropName: CropCode}, e.g. {'Maize': 'MAIZ'}.
        country_iso_filter (list): List of Country ISO3 codes to filter SPI data (e.g., ['ETH', 'KEN', 'SOM']).
    
    Returns:
        pd.DataFrame: The resulting weighted indices dataframe.
    """
    if target_crops is None:
        target_crops = {'Maize': 'MAIZ', 'Wheat': 'WHEA', 'Sorghum': 'SORG'}
    
    if country_iso_filter is None:
        country_iso_filter = ['ETH']
    
    # Ensure list
    if isinstance(country_iso_filter, str):
        country_iso_filter = [country_iso_filter]

    print(f"--- Generating Subnational Weighted Indices for {country_iso_filter} ---")
    
    # 1. Load Climate Data (SPI from CSV)
    print(f"Loading SPI Data from CSV: {spi_csv_path}")
    if not spi_csv_path.exists():
        print(f"Error: SPI file not found at {spi_csv_path}")
        return None
    
    # Load CSV with Admin columns
    usecols = ['lat', 'lon', 'time', 'spi_gamma_6_month', 'country_iso', 'admin1', 'admin2']
    
    try:
        print(f"Reading CSV (filtering for {country_iso_filter})...")
        dfs = []
        chunksize = 10 ** 6
        for chunk in pd.read_csv(spi_csv_path, usecols=usecols, chunksize=chunksize):
            # Optimised check for multiple ISOs
            country_subset = chunk[chunk['country_iso'].isin(country_iso_filter)]
            if not country_subset.empty:
                dfs.append(country_subset)
        
        if not dfs:
            print(f"No data found for {country_iso_filter}.")
            return None

        df_spi = pd.concat(dfs, ignore_index=True)
        df_spi['time'] = pd.to_datetime(df_spi['time'])
        
        print(f"SPI Data Loaded. Shape: {df_spi.shape}")
        print(f"Unique Locations: {len(df_spi[['lat', 'lon']].drop_duplicates())}")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Prepare unique coordinates for interpolation
    unique_coords = df_spi[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    
    target_lats = xr.DataArray(unique_coords['lat'].values, dims='point')
    target_lons = xr.DataArray(unique_coords['lon'].values, dims='point')
    
    # 2. Loop over Crops and Interpolate Weights
    for crop_name, crop_code in target_crops.items():
        print(f"\nProcessing {crop_name} ({crop_code})...")
        
        try:
            # Load SPAM Weights (Global - efficiently relies on interpolation to pick relevant points)
            # Passing region_filter=None to load global data (needed for multi-country support)
            da_weight = load_spam_data(
                spam_dir=spam_dir, 
                crop_code=crop_code, 
                variable='H',       # Harvested Area
                tech_type='TA',     # Total All technologies
                region_filter=None  # Load Global to support multiple countries
            )
            
            # Interpolate Weights to SPI locations
            print("  Interpolating weights to SPI locations...")
            weights_at_points = da_weight.interp(lat=target_lats, lon=target_lons, method='linear')
            
            weights_res = weights_at_points.to_dataframe(name='weight').reset_index()
            
            unique_coords[f'weight_{crop_code}'] = weights_res['weight'].values
            unique_coords[f'weight_{crop_code}'] = unique_coords[f'weight_{crop_code}'].fillna(0)
            
        except Exception as e:
            print(f"  Skipping {crop_name}: {e}")
            continue

    # Merge weights back to main dataframe
    print("\nMerging weights and calculating indices...")
    df_merged = pd.merge(df_spi, unique_coords, on=['lat', 'lon'], how='left')
    
    # 3. Calculate Weighted Averages per Admin Region
    results = []
    group_cols = ['country_iso', 'admin1', 'admin2', 'time']
    
    for crop_name, crop_code in target_crops.items():
        weight_col = f'weight_{crop_code}'
        if weight_col not in df_merged.columns:
            continue
            
        print(f"  Aggregating {crop_name}...")
        
        # Calculate numerator
        df_merged[f'w_spi_{crop_code}'] = df_merged['spi_gamma_6_month'] * df_merged[weight_col]
        
        # Groupby Sum
        grouped = df_merged.groupby(group_cols)[[f'w_spi_{crop_code}', weight_col]].sum()
        
        # Calculate weighted mean
        grouped[f'spi_weighted_{crop_name.lower()}'] = grouped[f'w_spi_{crop_code}'] / grouped[weight_col]
        
        # Handle division by zero
        grouped.loc[grouped[weight_col] == 0, f'spi_weighted_{crop_name.lower()}'] = np.nan
        
        results.append(grouped[[f'spi_weighted_{crop_name.lower()}']])
        
    if results:
        final_result = pd.concat(results, axis=1).reset_index()
        print(f"\nResult Shape: {final_result.shape}")
        
        if output_csv_path:
            print(f"Saving to {output_csv_path}")
            final_result.to_csv(output_csv_path, index=False)
            print("Done.")
            
        return final_result
    else:
        print("No results generated.")
        return None

def main():
    # Example Custom Usage
    target_crops = {
        'Maize': 'MAIZ',
        'Wheat': 'WHEA',
        'Sorghum': 'SORG'
    }
    
    generate_subnational_weighted_indices(
        spi_csv_path=SPI_CSV,
        spam_dir=SPAM_DIR,
        output_csv_path=Path("data/processed/weighted_spi_indices_east_africa.csv"),
        country_iso_filter=['ETH', 'KEN', 'SOM'], # Multi-country support
        target_crops=target_crops
    )

if __name__ == "__main__":
    main()
