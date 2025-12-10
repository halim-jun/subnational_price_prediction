"""
Data preparation utilities for geotemporal modeling.

Shared data loading and preprocessing functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import h3
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreparator:
    """Prepare data for modeling with NaN handling and spatial features."""
    
    def __init__(self, data_dir='data/processed/modeling'):
        """Load datasets."""
        self.data_dir = Path(data_dir)
        self.load_data()
        
    def load_data(self):
        """Load train and test datasets."""
        print("Loading datasets...")
        self.temporal_train = pd.read_parquet(self.data_dir / 'temporal_train.parquet')
        self.temporal_test = pd.read_parquet(self.data_dir / 'temporal_test.parquet')
        self.spatial_test = pd.read_parquet(self.data_dir / 'spatial_test.parquet')
        
        print(f"  Temporal train: {len(self.temporal_train):,} rows")
        print(f"  Temporal test: {len(self.temporal_test):,} rows")
        print(f"  Spatial test: {len(self.spatial_test):,} rows")
        
    def add_spatial_lag_features(self, df, k_ring=1):
        """
        Add spatial lag features based on H3 neighbors with proper temporal lag.
        
        IMPORTANT: Uses PREVIOUS MONTH's neighbor data to prevent data leakage!
        When predicting month t, we use month t-1's neighbor values.
        
        For each h3_index, calculate average price volatility of neighboring cells
        from the PREVIOUS month.
        """
        print(f"  Adding spatial lag features (k_ring={k_ring}, with 1-month lag)...")
        
        df = df.copy().sort_values(['h3_index', 'date'])
        
        # Create a mapping of (h3_index, date) -> volatility
        df['year_month'] = df['date'].dt.to_period('M')
        
        spatial_lags = []
        spatial_diffs = []
        
        for idx, row in df.iterrows():
            h3_idx = row['h3_index']
            current_period = row['year_month']
            
            # Get PREVIOUS month's period
            try:
                prev_period = current_period - 1
            except:
                # First month, no previous data
                spatial_lags.append(row['price_volatility_24m'])
                spatial_diffs.append(0)
                continue
            
            # Get neighbors
            try:
                neighbors = h3.k_ring(h3_idx, k_ring)
                neighbors = neighbors - {h3_idx}  # Exclude self
                
                # Get PREVIOUS month's data for neighbors
                prev_month_data = df[df['year_month'] == prev_period]
                neighbor_data = prev_month_data[prev_month_data['h3_index'].isin(neighbors)]
                
                if len(neighbor_data) > 0:
                    spatial_lag = neighbor_data['price_volatility_24m'].mean()
                    spatial_diff = row['price_volatility_24m'] - spatial_lag
                else:
                    # No neighbor data available, use own value
                    spatial_lag = row['price_volatility_24m']
                    spatial_diff = 0
            except Exception as e:
                spatial_lag = row['price_volatility_24m']
                spatial_diff = 0
            
            spatial_lags.append(spatial_lag)
            spatial_diffs.append(spatial_diff)
        
        df['volatility_spatial_lag'] = spatial_lags
        df['volatility_spatial_diff'] = spatial_diffs
        df = df.drop('year_month', axis=1)
        
        return df
    
    def add_temporal_lag_features(self, df, target='inflation_mom'):
        """
        Add temporal lag features for time series modeling.
        
        Creates:
        - Lag 1, 2, 3 months of inflation
        - 3-month rolling average of inflation
        - 3-month rolling std of inflation
        - Price change trend
        - Precipitation lags
        """
        print(f"  Adding temporal lag features...")
        
        df = df.copy().sort_values(['h3_index', 'date'])
        
        # Inflation lags (1, 2, 3 months)
        for lag in [1, 2, 3]:
            df[f'{target}_lag{lag}'] = df.groupby('h3_index')[target].shift(lag)
        
        # Inflation rolling statistics (3 months)
        df[f'{target}_rolling_mean_3m'] = df.groupby('h3_index')[target].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        df[f'{target}_rolling_std_3m'] = df.groupby('h3_index')[target].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().shift(1)
        )
        
        # Price volatility lag
        df['price_volatility_lag1'] = df.groupby('h3_index')['price_volatility_24m'].shift(1)
        
        # Precipitation lags (1, 2, 3 months)
        for lag in [1, 2, 3]:
            df[f'precipitation_lag{lag}'] = df.groupby('h3_index')['precipitation'].shift(lag)
        
        # Precipitation 3-month rolling average
        df['precipitation_rolling_mean_3m'] = df.groupby('h3_index')['precipitation'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        return df
    
    def prepare_features(self, target='inflation_mom', add_spatial=True, add_temporal=True, scale=False):
        """
        Prepare feature matrices and remove NaN values.
        
        Args:
            target: 'inflation_mom' or 'inflation_yoy'
            add_spatial: Whether to add spatial lag features
            add_temporal: Whether to add temporal lag features
            scale: Whether to standardize features
            
        Returns:
            Dictionary with X_train, y_train, X_test, y_test, etc.
        """
        print(f"\nPreparing features for {target}...")
        
        # Add temporal lag features first (before spatial, as it needs the target variable)
        if add_temporal:
            self.temporal_train = self.add_temporal_lag_features(self.temporal_train, target)
            self.temporal_test = self.add_temporal_lag_features(self.temporal_test, target)
            self.spatial_test = self.add_temporal_lag_features(self.spatial_test, target)
        
        # Add spatial features if requested
        if add_spatial:
            self.temporal_train = self.add_spatial_lag_features(self.temporal_train)
            self.temporal_test = self.add_spatial_lag_features(self.temporal_test)
            self.spatial_test = self.add_spatial_lag_features(self.spatial_test)
        
        # Define base features (per_unit_price removed to prevent data leakage)
        base_features = [
            'precipitation',
            'flood_3m', 'flood_6m',
            'drought_3m', 'drought_6m',
            'price_volatility_24m',
            'price_cv_24m',
            'month', 'quarter',
            'n_markets'
        ]
        
        # Add temporal lag features
        if add_temporal:
            temporal_features = [
                f'{target}_lag1', f'{target}_lag2', f'{target}_lag3',
                f'{target}_rolling_mean_3m', f'{target}_rolling_std_3m',
                'price_volatility_lag1',
                'precipitation_lag1', 'precipitation_lag2', 'precipitation_lag3',
                'precipitation_rolling_mean_3m'
            ]
            base_features.extend(temporal_features)
        
        if add_spatial:
            base_features.extend(['volatility_spatial_lag', 'volatility_spatial_diff'])
        
        # One-hot encode country
        train = pd.get_dummies(self.temporal_train, columns=['countryiso3'], prefix='country')
        test = pd.get_dummies(self.temporal_test, columns=['countryiso3'], prefix='country')
        spatial_test = pd.get_dummies(self.spatial_test, columns=['countryiso3'], prefix='country')
        
        # Get country columns
        country_cols = [col for col in train.columns if col.startswith('country_')]
        all_features = base_features + country_cols
        
        # Align columns
        for col in all_features:
            if col not in test.columns:
                test[col] = 0
            if col not in spatial_test.columns:
                spatial_test[col] = 0
        
        # Remove rows with NaN in features or target
        train_clean = train[all_features + [target]].dropna()
        test_clean = test[all_features + [target]].dropna()
        spatial_test_clean = spatial_test[all_features + [target]].dropna()
        
        print(f"  After removing NaN:")
        print(f"    Train: {len(train)} -> {len(train_clean)} rows")
        print(f"    Test: {len(test)} -> {len(test_clean)} rows")
        print(f"    Spatial test: {len(spatial_test)} -> {len(spatial_test_clean)} rows")
        
        # Extract X, y
        X_train = train_clean[all_features].values
        y_train = train_clean[target].values
        
        X_test = test_clean[all_features].values
        y_test = test_clean[target].values
        
        X_spatial = spatial_test_clean[all_features].values
        y_spatial = spatial_test_clean[target].values
        
        # Scale if requested
        scaler = None
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_spatial = scaler.transform(X_spatial)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_spatial': X_spatial,
            'y_spatial': y_spatial,
            'feature_names': all_features,
            'scaler': scaler,
            'n_features': len(all_features)
        }


def calculate_metrics(y_true, y_pred, prefix=''):
    """Calculate regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        f'{prefix}rmse': rmse,
        f'{prefix}mae': mae,
        f'{prefix}r2': r2,
        f'{prefix}mape': mape
    }


def print_metrics(metrics, model_name):
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"  Train RMSE: {metrics.get('train_rmse', 0):.4f}")
    print(f"  Test RMSE:  {metrics.get('test_rmse', 0):.4f}")
    print(f"  Spatial RMSE: {metrics.get('spatial_rmse', 0):.4f}")
    print(f"  Test MAE:   {metrics.get('test_mae', 0):.4f}")
    print(f"  Test R²:    {metrics.get('test_r2', 0):.4f}")
    print(f"{'='*60}")

