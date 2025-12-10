"""
Merge WFP price data with CHIRPS precipitation data using H3 spatial index.

This script:
1. Loads WFP sorghum price data and CHIRPS precipitation
2. Assigns H3 hexagon indices to both datasets
3. Aggregates by H3 and date
4. Merges datasets on H3 index and date
5. Calculates additional features (inflation rate, etc.)

Usage:
    python src/preprocessing/merge_h3_data.py --h3-resolution 5
    python src/preprocessing/merge_h3_data.py --h3-resolution 4 --output-suffix "_h3_4"
"""

import pandas as pd
import numpy as np
import h3
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir='data/raw'):
    """Load WFP and CHIRPS data."""
    data_dir = Path(data_dir)
    
    print("📂 Loading data...")
    
    # Load WFP sorghum price data
    wfp_files = list((data_dir / 'wfp').glob('*.csv'))
    if not wfp_files:
        raise FileNotFoundError("No WFP CSV files found in data/raw/wfp/")
    
    print(f"  Found {len(wfp_files)} WFP file(s)")
    wfp_data = pd.concat([pd.read_csv(f) for f in wfp_files], ignore_index=True)
    
    # Filter for sorghum
    sorghum_data = wfp_data[wfp_data['commodity'].str.lower().str.contains('sorghum', na=False)].copy()
    print(f"  Loaded {len(sorghum_data):,} sorghum price records")
    
    # Load CHIRPS precipitation (use processed parquet file)
    chirps_file = Path('data/processed/chirps_eastern_africa_geopandas.parquet')
    if not chirps_file.exists():
        # Try alternative location
        chirps_file = data_dir / 'climate' / 'chirps_precipitation.csv'
        if not chirps_file.exists():
            raise FileNotFoundError(f"CHIRPS file not found")
        chirps_data = pd.read_csv(chirps_file)
    else:
        chirps_data = pd.read_parquet(chirps_file)
    
    print(f"  Loaded {len(chirps_data):,} CHIRPS precipitation records")
    
    return sorghum_data, chirps_data


def normalize_dates(df, date_col='date'):
    """
    Normalize dates to Year-Month (first day of month).
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
    
    Returns:
        DataFrame with normalized dates
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].dt.to_period('M').dt.to_timestamp()
    return df


def add_h3_index(df, lat_col, lon_col, h3_resolution=5):
    """
    Add H3 hexagon index to DataFrame.
    
    Args:
        df: DataFrame with lat/lon columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        h3_resolution: H3 resolution level (3-15)
    
    Returns:
        DataFrame with h3_index column
    """
    df = df.copy()
    
    print(f"  Adding H3 index (resolution {h3_resolution})...")
    print(f"    ~{h3.average_hexagon_area(h3_resolution, unit='km^2'):.0f} km² per hexagon")
    
    # Drop rows with missing coordinates
    df = df.dropna(subset=[lat_col, lon_col])
    
    # Add H3 index
    df['h3_index'] = df.apply(
        lambda row: h3.latlng_to_cell(row[lat_col], row[lon_col], h3_resolution),
        axis=1
    )
    
    return df


def aggregate_by_h3_date(df, group_cols=['h3_index', 'date'], 
                         agg_dict=None, include_counts=True):
    """
    Aggregate data by H3 index and date.
    
    Args:
        df: DataFrame with h3_index and date
        group_cols: Columns to group by
        agg_dict: Dictionary of column -> aggregation function
        include_counts: Whether to include count of records
    
    Returns:
        Aggregated DataFrame
    """
    if agg_dict is None:
        # Default aggregation
        agg_dict = {}
        for col in df.columns:
            if col not in group_cols and pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'mean'
    
    print(f"  Aggregating by {group_cols}...")
    grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    if include_counts:
        counts = df.groupby(group_cols).size().reset_index(name='n_records')
        grouped = grouped.merge(counts, on=group_cols, how='left')
    
    return grouped


def merge_datasets(price_df, precip_df, on=['h3_index', 'date'], how='inner'):
    """
    Merge price and precipitation datasets.
    
    Args:
        price_df: Price DataFrame with h3_index and date
        precip_df: Precipitation DataFrame with h3_index and date
        on: Columns to merge on
        how: Type of merge ('inner', 'left', 'outer')
    
    Returns:
        Merged DataFrame
    """
    print(f"\n🔗 Merging datasets ({how} join)...")
    print(f"  Price data: {len(price_df):,} rows, {price_df['h3_index'].nunique()} unique H3 cells")
    print(f"  Precip data: {len(precip_df):,} rows, {precip_df['h3_index'].nunique()} unique H3 cells")
    
    merged = price_df.merge(precip_df, on=on, how=how, suffixes=('', '_precip'))
    
    print(f"  Merged: {len(merged):,} rows, {merged['h3_index'].nunique()} unique H3 cells")
    print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    return merged


def calculate_inflation_rate(df, price_col='per_unit_price', group_col='h3_index'):
    """
    Calculate month-over-month inflation rate (log returns).
    
    Args:
        df: DataFrame with price column
        price_col: Name of price column
        group_col: Column to group by for lag calculation
    
    Returns:
        DataFrame with inflation_rate column
    """
    print("\n📊 Calculating inflation rate...")
    
    df = df.copy().sort_values([group_col, 'date'])
    
    # Calculate log returns
    df['price_lag1'] = df.groupby(group_col)[price_col].shift(1)
    df['inflation_rate'] = np.log(df[price_col] / df['price_lag1'])
    
    # Clean up
    df = df.drop('price_lag1', axis=1)
    
    # Replace inf/-inf with NaN
    df['inflation_rate'] = df['inflation_rate'].replace([np.inf, -np.inf], np.nan)
    
    print(f"  Calculated inflation rate for {df['inflation_rate'].notna().sum():,} records")
    
    return df


def process_h3_merge(h3_resolution=5, output_suffix=''):
    """
    Main processing pipeline.
    
    Args:
        h3_resolution: H3 resolution level (3-15)
        output_suffix: Suffix to add to output filename
    
    Returns:
        Processed DataFrame
    """
    print("="*80)
    print(f"H3 GEOSPATIAL DATA MERGING (Resolution {h3_resolution})")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    sorghum_price, chirps_precip = load_data()
    
    # ===================================================================
    # Process Price Data
    # ===================================================================
    print("\n📊 Processing price data...")
    
    # Normalize dates
    price_prep = normalize_dates(sorghum_price, 'date')
    print(f"  Date range: {price_prep['date'].min()} to {price_prep['date'].max()}")
    
    # Add H3 index
    price_prep = add_h3_index(price_prep, 'latitude', 'longitude', h3_resolution)
    
    # Aggregate by H3 and date
    # Calculate mean price and count markets per H3 cell
    price_agg = aggregate_by_h3_date(
        price_prep,
        group_cols=['h3_index', 'date', 'countryiso3'],
        agg_dict={
            'usdprice': 'mean',
            'latitude': 'mean',  # Representative coordinates
            'longitude': 'mean'
        },
        include_counts=True
    )
    
    # Rename for clarity
    price_agg = price_agg.rename(columns={
        'usdprice': 'per_unit_price',
        'n_records': 'n_markets'
    })
    
    print(f"  Aggregated to {len(price_agg):,} H3-date combinations")
    
    # ===================================================================
    # Process Precipitation Data
    # ===================================================================
    print("\n🌧️  Processing precipitation data...")
    
    # Normalize dates
    precip_prep = chirps_precip.copy()
    precip_prep['date'] = pd.to_datetime(precip_prep['date'], format='%Y-%m')
    print(f"  Date range: {precip_prep['date'].min()} to {precip_prep['date'].max()}")
    
    # Add H3 index
    precip_prep = add_h3_index(precip_prep, 'lat', 'lon', h3_resolution)
    
    # Aggregate by H3 and date (mean precipitation across pixels in each H3 cell)
    precip_agg = aggregate_by_h3_date(
        precip_prep,
        group_cols=['h3_index', 'date'],
        agg_dict={'precipitation': 'mean'},
        include_counts=False
    )
    
    print(f"  Aggregated to {len(precip_agg):,} H3-date combinations")
    
    # ===================================================================
    # Merge Datasets
    # ===================================================================
    merged = merge_datasets(price_agg, precip_agg, on=['h3_index', 'date'], how='inner')
    
    # ===================================================================
    # Calculate Additional Features
    # ===================================================================
    merged = calculate_inflation_rate(merged, 'per_unit_price', 'h3_index')
    
    # ===================================================================
    # Save Output
    # ===================================================================
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_filename = f'sorghum_price_with_precipitation_h3_{h3_resolution}{output_suffix}.csv'
    output_path = output_dir / output_filename
    
    merged.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE")
    print("="*80)
    print(f"\n📁 Output saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Total rows: {len(merged):,}")
    print(f"   Unique H3 cells: {merged['h3_index'].nunique():,}")
    print(f"   Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"   Countries: {sorted(merged['countryiso3'].unique())}")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Avg price per H3-month: ${merged['per_unit_price'].mean():.2f}")
    print(f"   Avg markets per H3-month: {merged['n_markets'].mean():.1f}")
    print(f"   Avg precipitation: {merged['precipitation'].mean():.1f} mm")
    print(f"   Records with inflation data: {merged['inflation_rate'].notna().sum():,}")
    
    # Null counts
    null_counts = merged.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n⚠️  Null values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"   {col}: {count:,} ({count/len(merged)*100:.1f}%)")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return merged


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Merge WFP price and CHIRPS precipitation data using H3 spatial index'
    )
    parser.add_argument(
        '--h3-resolution',
        type=int,
        default=5,
        choices=range(3, 16),
        help='H3 resolution level (3-15). Default: 5 (~252 km² per hexagon)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='',
        help='Suffix to add to output filename (optional)'
    )
    
    args = parser.parse_args()
    
    # Process
    process_h3_merge(
        h3_resolution=args.h3_resolution,
        output_suffix=args.output_suffix
    )


if __name__ == '__main__':
    main()

