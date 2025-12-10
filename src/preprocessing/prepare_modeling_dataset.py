"""
Prepare modeling dataset for geotemporal food price forecasting.

This script creates features for modeling sorghum price inflation with:
- Flood/drought indicators (3-month and 6-month windows)
- Price volatility metrics
- Spatial and temporal features
- Proper train/test split to prevent information leakage

Author: Modeling Pipeline
Date: 2025-10-22
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class GeotemporalDataPreparator:
    """
    Prepare geotemporal dataset for food price modeling.
    
    This class handles:
    1. Feature engineering (flood/drought indicators, volatility)
    2. Target variable creation (YoY and MoM inflation)
    3. Spatial-temporal train/test split
    """
    
    def __init__(self, data_path: str, output_dir: str):
        """
        Initialize the preparator.
        
        Args:
            data_path: Path to the input CSV file
            output_dir: Directory to save processed datasets
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        print(f"Loaded {len(self.df):,} rows")
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Filter to only rows with actual price data
        self.df = self.df[self.df['per_unit_price'].notna()].copy()
        print(f"After filtering to rows with price data: {len(self.df):,} rows")
        
    def create_precipitation_statistics(self):
        """
        Calculate 3-year rolling mean and std for precipitation.
        This is used as baseline for flood/drought detection.
        """
        print("\n[1/7] Calculating 3-year precipitation statistics...")
        
        # Sort by h3_index and date
        self.df = self.df.sort_values(['h3_index', 'date']).reset_index(drop=True)
        
        # Calculate 3-year (36-month) rolling statistics
        # Use a minimum of 24 months to start calculating
        self.df['precip_3yr_mean'] = self.df.groupby('h3_index')['precipitation'].transform(
            lambda x: x.rolling(window=36, min_periods=24).mean()
        )
        
        self.df['precip_3yr_std'] = self.df.groupby('h3_index')['precipitation'].transform(
            lambda x: x.rolling(window=36, min_periods=24).std()
        )
        
        print(f"  - Created 3-year precipitation baseline statistics")
        
    def create_flood_drought_indicators(self):
        """
        Create flood and drought indicators.
        
        Flood: precipitation > (3yr_mean + 1.5 * 3yr_std)
        Drought: precipitation < (3yr_mean - 1.5 * 3yr_std)
        
        Then create rolling windows:
        - flood/drought in last 3 months
        - flood/drought in last 6 months
        """
        print("\n[2/7] Creating flood and drought indicators...")
        
        # Define flood and drought thresholds (1.5 standard deviations)
        self.df['is_flood_month'] = (
            self.df['precipitation'] > 
            (self.df['precip_3yr_mean'] + 1.5 * self.df['precip_3yr_std'])
        ).astype(int)
        
        self.df['is_drought_month'] = (
            self.df['precipitation'] < 
            (self.df['precip_3yr_mean'] - 1.5 * self.df['precip_3yr_std'])
        ).astype(int)
        
        # Handle NaN in baseline stats (early periods)
        self.df['is_flood_month'] = self.df['is_flood_month'].fillna(0).astype(int)
        self.df['is_drought_month'] = self.df['is_drought_month'].fillna(0).astype(int)
        
        # Create rolling window indicators
        # 3-month window: any flood/drought in last 3 months?
        self.df['flood_3m'] = self.df.groupby('h3_index')['is_flood_month'].transform(
            lambda x: x.rolling(window=3, min_periods=1).max()
        ).astype(int)
        
        self.df['drought_3m'] = self.df.groupby('h3_index')['is_drought_month'].transform(
            lambda x: x.rolling(window=3, min_periods=1).max()
        ).astype(int)
        
        # 6-month window: any flood/drought in last 6 months?
        self.df['flood_6m'] = self.df.groupby('h3_index')['is_flood_month'].transform(
            lambda x: x.rolling(window=6, min_periods=1).max()
        ).astype(int)
        
        self.df['drought_6m'] = self.df.groupby('h3_index')['is_drought_month'].transform(
            lambda x: x.rolling(window=6, min_periods=1).max()
        ).astype(int)
        
        print(f"  - Flood events (3m): {self.df['flood_3m'].sum():,}")
        print(f"  - Flood events (6m): {self.df['flood_6m'].sum():,}")
        print(f"  - Drought events (3m): {self.df['drought_3m'].sum():,}")
        print(f"  - Drought events (6m): {self.df['drought_6m'].sum():,}")
        
    def create_price_volatility(self):
        """
        Calculate price volatility based on previous 2 years of data.
        
        Volatility = std(price) over rolling 24-month window
        """
        print("\n[3/7] Calculating price volatility (24-month rolling)...")
        
        self.df['price_volatility_24m'] = self.df.groupby('h3_index')['per_unit_price'].transform(
            lambda x: x.rolling(window=24, min_periods=12).std()
        )
        
        # Also calculate coefficient of variation for normalized volatility
        self.df['price_cv_24m'] = self.df.groupby('h3_index').apply(
            lambda x: (x['per_unit_price'].rolling(window=24, min_periods=12).std() / 
                      x['per_unit_price'].rolling(window=24, min_periods=12).mean())
        ).reset_index(level=0, drop=True)
        
        print(f"  - Created volatility metrics")
        
    def create_target_variables(self):
        """
        Create target variables for modeling:
        1. MoM (Month-over-Month) inflation: log(price_t / price_{t-1})
        2. YoY (Year-over-Year) inflation: log(price_t / price_{t-12})
        """
        print("\n[4/7] Creating target variables...")
        
        # Sort to ensure proper lag calculation
        self.df = self.df.sort_values(['h3_index', 'date']).reset_index(drop=True)
        
        # MoM inflation (already exists as 'inflation_rate', but recalculate for clarity)
        self.df['price_lag1'] = self.df.groupby('h3_index')['per_unit_price'].shift(1)
        self.df['inflation_mom'] = np.log(self.df['per_unit_price'] / self.df['price_lag1'])
        
        # YoY inflation
        self.df['price_lag12'] = self.df.groupby('h3_index')['per_unit_price'].shift(12)
        self.df['inflation_yoy'] = np.log(self.df['per_unit_price'] / self.df['price_lag12'])
        
        # Drop intermediate lag columns
        self.df = self.df.drop(['price_lag1', 'price_lag12'], axis=1)
        
        print(f"  - MoM inflation range: [{self.df['inflation_mom'].min():.4f}, {self.df['inflation_mom'].max():.4f}]")
        print(f"  - YoY inflation range: [{self.df['inflation_yoy'].min():.4f}, {self.df['inflation_yoy'].max():.4f}]")
        
    def add_temporal_features(self):
        """
        Add temporal features:
        - Month (1-12)
        - Quarter (1-4)
        - Year
        """
        print("\n[5/7] Adding temporal features...")
        
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['year'] = self.df['date'].dt.year
        
        print(f"  - Added month, quarter, year features")
        
    def create_final_dataset(self):
        """
        Clean up and select final features for modeling.
        """
        print("\n[6/7] Creating final dataset...")
        
        # Select final columns
        feature_cols = [
            # Identifiers
            'date', 'h3_index', 'countryiso3',
            
            # Spatial info
            'n_markets',
            
            # Climate features
            'precipitation',
            'flood_3m', 'flood_6m',
            'drought_3m', 'drought_6m',
            
            # Price features
            'per_unit_price',
            'price_volatility_24m',
            'price_cv_24m',
            
            # Temporal features
            'month', 'quarter', 'year',
            
            # Target variables
            'inflation_mom',
            'inflation_yoy'
        ]
        
        self.df_final = self.df[feature_cols].copy()
        
        # Remove rows where we can't calculate targets (first months, etc.)
        # For YoY, we need at least 12 months of history
        # For MoM, we need at least 1 month
        # For volatility, we need at least 12 months
        
        # Only keep rows from 2018 onwards to ensure enough history
        self.df_final = self.df_final[self.df_final['year'] >= 2018].copy()
        
        # Remove rows with missing target variables
        self.df_final = self.df_final.dropna(subset=['inflation_mom', 'inflation_yoy'])
        
        print(f"  - Final dataset: {len(self.df_final):,} rows")
        print(f"  - Date range: {self.df_final['date'].min()} to {self.df_final['date'].max()}")
        print(f"  - Unique h3 cells: {self.df_final['h3_index'].nunique():,}")
        print(f"  - Countries: {sorted(self.df_final['countryiso3'].unique())}")
        
        return self.df_final
    
    def spatial_temporal_split(self, test_start_date='2024-01-01', 
                               spatial_test_ratio=0.2,
                               random_seed=42):
        """
        Create train/test split that prevents both spatial and temporal leakage.
        
        Strategy:
        1. Temporal split: Use data before test_start_date for training
        2. Spatial split: Randomly hold out some h3_index cells for spatial validation
        
        This creates three sets:
        - Train: temporal_train + spatial_train
        - Temporal test: same cells as train, but future dates
        - Spatial test: held-out cells, all dates in test period
        
        Args:
            test_start_date: Date to start temporal test set
            spatial_test_ratio: Fraction of h3 cells to hold out for spatial test
            random_seed: Random seed for reproducibility
        """
        print("\n[7/7] Creating spatial-temporal train/test split...")
        
        test_start = pd.to_datetime(test_start_date)
        
        # Get all unique h3 indices
        unique_h3 = self.df_final['h3_index'].unique()
        
        # Randomly split h3 indices into train and spatial-test
        np.random.seed(random_seed)
        n_spatial_test = int(len(unique_h3) * spatial_test_ratio)
        spatial_test_h3 = np.random.choice(unique_h3, size=n_spatial_test, replace=False)
        spatial_train_h3 = np.setdiff1d(unique_h3, spatial_test_h3)
        
        # Create splits
        # 1. Temporal train: spatial_train cells + dates before test_start
        temporal_train = self.df_final[
            (self.df_final['h3_index'].isin(spatial_train_h3)) & 
            (self.df_final['date'] < test_start)
        ].copy()
        
        # 2. Temporal test: spatial_train cells + dates >= test_start
        temporal_test = self.df_final[
            (self.df_final['h3_index'].isin(spatial_train_h3)) & 
            (self.df_final['date'] >= test_start)
        ].copy()
        
        # 3. Spatial test: spatial_test cells + dates >= test_start
        spatial_test = self.df_final[
            (self.df_final['h3_index'].isin(spatial_test_h3)) & 
            (self.df_final['date'] >= test_start)
        ].copy()
        
        # 4. Full train (for models that can handle it): all data before test_start
        full_train = self.df_final[self.df_final['date'] < test_start].copy()
        
        print(f"\n  Split summary:")
        print(f"  - Temporal train: {len(temporal_train):,} rows")
        print(f"    * H3 cells: {temporal_train['h3_index'].nunique():,}")
        print(f"    * Date range: {temporal_train['date'].min()} to {temporal_train['date'].max()}")
        
        print(f"\n  - Temporal test: {len(temporal_test):,} rows")
        print(f"    * H3 cells: {temporal_test['h3_index'].nunique():,}")
        print(f"    * Date range: {temporal_test['date'].min()} to {temporal_test['date'].max()}")
        
        print(f"\n  - Spatial test: {len(spatial_test):,} rows")
        print(f"    * H3 cells: {spatial_test['h3_index'].nunique():,}")
        print(f"    * Date range: {spatial_test['date'].min()} to {spatial_test['date'].max()}")
        
        print(f"\n  - Full train (alternative): {len(full_train):,} rows")
        print(f"    * H3 cells: {full_train['h3_index'].nunique():,}")
        
        # Verify no leakage
        assert len(set(spatial_train_h3) & set(spatial_test_h3)) == 0, "Spatial leakage detected!"
        assert temporal_train['date'].max() < test_start, "Temporal leakage in train!"
        assert temporal_test['date'].min() >= test_start, "Temporal leakage in test!"
        
        print(f"\n  ✓ No spatial or temporal leakage detected")
        
        return {
            'temporal_train': temporal_train,
            'temporal_test': temporal_test,
            'spatial_test': spatial_test,
            'full_train': full_train,
            'spatial_train_h3': spatial_train_h3,
            'spatial_test_h3': spatial_test_h3
        }
    
    def save_datasets(self, splits):
        """
        Save all datasets to files.
        
        Args:
            splits: Dictionary of split datasets
        """
        print("\n" + "="*60)
        print("Saving datasets...")
        print("="*60)
        
        # Save full processed dataset
        full_path = self.output_dir / 'modeling_dataset_full.parquet'
        self.df_final.to_parquet(full_path, index=False)
        print(f"\n✓ Saved full dataset: {full_path}")
        print(f"  Size: {full_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Save train/test splits
        for name, df in splits.items():
            if isinstance(df, pd.DataFrame):
                path = self.output_dir / f'{name}.parquet'
                df.to_parquet(path, index=False)
                print(f"\n✓ Saved {name}: {path}")
                print(f"  Rows: {len(df):,}")
                print(f"  Size: {path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Save h3 index lists for reference
        import json
        h3_splits = {
            'spatial_train_h3': list(splits['spatial_train_h3']),
            'spatial_test_h3': list(splits['spatial_test_h3'])
        }
        h3_path = self.output_dir / 'h3_spatial_splits.json'
        with open(h3_path, 'w') as f:
            json.dump(h3_splits, f, indent=2)
        print(f"\n✓ Saved H3 spatial splits: {h3_path}")
        
        # Create and save data summary
        summary = {
            'created_at': datetime.now().isoformat(),
            'source_file': str(self.data_path),
            'total_rows': len(self.df_final),
            'date_range': {
                'start': str(self.df_final['date'].min()),
                'end': str(self.df_final['date'].max())
            },
            'unique_h3_cells': int(self.df_final['h3_index'].nunique()),
            'countries': sorted(self.df_final['countryiso3'].unique().tolist()),
            'features': self.df_final.columns.tolist(),
            'splits': {
                name: {
                    'rows': len(df),
                    'h3_cells': int(df['h3_index'].nunique()),
                    'date_range': {
                        'start': str(df['date'].min()),
                        'end': str(df['date'].max())
                    }
                }
                for name, df in splits.items() if isinstance(df, pd.DataFrame)
            },
            'null_counts': self.df_final.isnull().sum().to_dict()
        }
        
        summary_path = self.output_dir / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Saved dataset summary: {summary_path}")
        
        print("\n" + "="*60)
        print("Dataset preparation complete!")
        print("="*60)


def main():
    """Main execution function."""
    
    print("="*60)
    print("Geotemporal Food Price Modeling Dataset Preparation")
    print("="*60)
    
    # Configuration
    data_path = 'data/processed/sorghum_price_with_precipitation_h3_5.csv'
    output_dir = 'data/processed/modeling'
    
    # Initialize preparator
    preparator = GeotemporalDataPreparator(data_path, output_dir)
    
    # Execute pipeline
    preparator.create_precipitation_statistics()
    preparator.create_flood_drought_indicators()
    preparator.create_price_volatility()
    preparator.create_target_variables()
    preparator.add_temporal_features()
    preparator.create_final_dataset()
    
    # Create train/test splits
    splits = preparator.spatial_temporal_split(
        test_start_date='2024-01-01',
        spatial_test_ratio=0.2,
        random_seed=42
    )
    
    # Save everything
    preparator.save_datasets(splits)
    
    print("\n✅ All done! Ready for modeling.")
    print("\nNext steps:")
    print("1. Load datasets from data/processed/modeling/")
    print("2. Build models considering spatial autocorrelation")
    print("3. Evaluate on both temporal_test (same locations) and spatial_test (new locations)")
    

if __name__ == '__main__':
    main()

