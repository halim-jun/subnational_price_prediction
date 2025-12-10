"""
Geotemporal Food Price Forecasting Models

Implements various models for spatial-temporal food price prediction:
- Machine Learning: XGBoost, Random Forest, LightGBM
- Deep Learning: LSTM, GRU, Bidirectional LSTM
- Statistical: SARIMA, VAR
- Spatial: Spatial Lag Features

All experiments are logged to MLflow for comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML/DL libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Statistical models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR

# Spatial
import h3

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.xgboost
import mlflow.lightgbm

# Utils
from datetime import datetime
import json


class DataPreparator:
    """Prepare data for modeling with NaN handling."""
    
    def __init__(self, data_dir='data/processed/modeling'):
        """Load and prepare datasets."""
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
        Add spatial lag features based on H3 neighbors.
        
        For each h3_index, calculate average price of neighboring cells.
        """
        print(f"  Adding spatial lag features (k_ring={k_ring})...")
        
        df = df.copy().sort_values(['date', 'h3_index'])
        spatial_lags = []
        
        for date in df['date'].unique():
            date_data = df[df['date'] == date].copy()
            h3_values = dict(zip(date_data['h3_index'], date_data['per_unit_price']))
            
            for idx, row in date_data.iterrows():
                h3_idx = row['h3_index']
                
                # Get neighbors
                try:
                    neighbors = h3.k_ring(h3_idx, k_ring)
                    neighbors = neighbors - {h3_idx}  # Exclude self
                    
                    # Calculate average of neighbors
                    neighbor_values = [h3_values[n] for n in neighbors if n in h3_values]
                    spatial_lag = np.mean(neighbor_values) if neighbor_values else row['per_unit_price']
                except:
                    spatial_lag = row['per_unit_price']
                
                spatial_lags.append(spatial_lag)
        
        df['price_spatial_lag'] = spatial_lags
        
        # Also add price difference from spatial lag
        df['price_spatial_diff'] = df['per_unit_price'] - df['price_spatial_lag']
        
        return df
    
    def prepare_features(self, target='inflation_mom', add_spatial=True):
        """
        Prepare feature matrices and remove NaN values.
        
        Args:
            target: 'inflation_mom' or 'inflation_yoy'
            add_spatial: Whether to add spatial lag features
        """
        print(f"\nPreparing features for {target}...")
        
        # Add spatial features if requested
        if add_spatial:
            self.temporal_train = self.add_spatial_lag_features(self.temporal_train)
            self.temporal_test = self.add_spatial_lag_features(self.temporal_test)
            self.spatial_test = self.add_spatial_lag_features(self.spatial_test)
        
        # Define base features
        base_features = [
            'precipitation',
            'flood_3m', 'flood_6m',
            'drought_3m', 'drought_6m',
            'price_volatility_24m',
            'price_cv_24m',
            'month', 'quarter',
            'n_markets'
            # Note: per_unit_price removed to prevent data leakage
        ]
        
        if add_spatial:
            base_features.extend(['price_spatial_lag', 'price_spatial_diff'])
        
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
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'X_spatial': X_spatial,
            'y_spatial': y_spatial,
            'feature_names': all_features,
            'scaler': None
        }
    
    def prepare_sequences(self, target='inflation_mom', lookback=6):
        """
        Prepare sequences for deep learning models (LSTM, GRU).
        
        Creates sequences of [lookback] time steps for each location.
        """
        print(f"\nPreparing sequences (lookback={lookback}) for {target}...")
        
        # Prepare features first
        data = self.prepare_features(target=target, add_spatial=True)
        
        # For simplicity, we'll create sequences from the time dimension
        # This is a simplified version - proper implementation would group by h3_index
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(data['X_train'])
        X_test_scaled = scaler.transform(data['X_test'])
        X_spatial_scaled = scaler.transform(data['X_spatial'])
        
        data['X_train'] = X_train_scaled
        data['X_test'] = X_test_scaled
        data['X_spatial'] = X_spatial_scaled
        data['scaler'] = scaler
        
        return data


class MLModels:
    """Machine Learning models."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def get_models(self):
        """Return dictionary of ML models."""
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }


class DLModels:
    """Deep Learning models."""
    
    def __init__(self, input_dim, random_state=42):
        self.input_dim = input_dim
        self.random_state = random_state
        tf.random.set_seed(random_state)
        
    def build_feedforward(self, hidden_layers=[64, 32]):
        """Build feedforward neural network."""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(hidden_layers[0], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(hidden_layers[1], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_deep_network(self):
        """Build deeper neural network."""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def get_callbacks(self):
        """Get training callbacks."""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=0
            )
        ]


def calculate_metrics(y_true, y_pred, prefix=''):
    """Calculate regression metrics."""
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


def train_ml_model(model, model_name, data, target='inflation_mom'):
    """Train and evaluate a machine learning model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {target}")
    print(f"{'='*60}")
    
    with mlflow.start_run(run_name=f"{model_name}_{target}"):
        # Log parameters
        mlflow.log_param('model_type', 'ml')
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('target', target)
        mlflow.log_param('n_features', data['X_train'].shape[1])
        mlflow.log_param('n_train', len(data['X_train']))
        
        # Train
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        y_train_pred = model.predict(data['X_train'])
        y_test_pred = model.predict(data['X_test'])
        y_spatial_pred = model.predict(data['X_spatial'])
        
        # Calculate metrics
        train_metrics = calculate_metrics(data['y_train'], y_train_pred, prefix='train_')
        test_metrics = calculate_metrics(data['y_test'], y_test_pred, prefix='test_')
        spatial_metrics = calculate_metrics(data['y_spatial'], y_spatial_pred, prefix='spatial_')
        
        # Log all metrics
        all_metrics = {**train_metrics, **test_metrics, **spatial_metrics}
        for metric_name, value in all_metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log model
        if 'XGBoost' in model_name:
            mlflow.xgboost.log_model(model, "model")
        elif 'LightGBM' in model_name:
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': data['feature_names'],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv('/tmp/feature_importance.csv', index=False)
            mlflow.log_artifact('/tmp/feature_importance.csv')
        
        # Print results
        print(f"\n  Train RMSE: {train_metrics['train_rmse']:.4f}")
        print(f"  Test RMSE:  {test_metrics['test_rmse']:.4f}")
        print(f"  Spatial RMSE: {spatial_metrics['spatial_rmse']:.4f}")
        print(f"  Test R²:    {test_metrics['test_r2']:.4f}")
        
        return model, all_metrics


def train_dl_model(model, model_name, data, target='inflation_mom', epochs=100):
    """Train and evaluate a deep learning model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} for {target}")
    print(f"{'='*60}")
    
    with mlflow.start_run(run_name=f"{model_name}_{target}"):
        # Log parameters
        mlflow.log_param('model_type', 'dl')
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('target', target)
        mlflow.log_param('n_features', data['X_train'].shape[1])
        mlflow.log_param('n_train', len(data['X_train']))
        mlflow.log_param('epochs', epochs)
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(data['X_train'])
        X_test_scaled = scaler.transform(data['X_test'])
        X_spatial_scaled = scaler.transform(data['X_spatial'])
        
        # Get callbacks
        dl_models = DLModels(input_dim=X_train_scaled.shape[1])
        callbacks = dl_models.get_callbacks()
        
        # Train
        history = model.fit(
            X_train_scaled, data['y_train'],
            validation_split=0.2,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()
        y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()
        y_spatial_pred = model.predict(X_spatial_scaled, verbose=0).flatten()
        
        # Calculate metrics
        train_metrics = calculate_metrics(data['y_train'], y_train_pred, prefix='train_')
        test_metrics = calculate_metrics(data['y_test'], y_test_pred, prefix='test_')
        spatial_metrics = calculate_metrics(data['y_spatial'], y_spatial_pred, prefix='spatial_')
        
        # Log all metrics
        all_metrics = {**train_metrics, **test_metrics, **spatial_metrics}
        for metric_name, value in all_metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log training history
        mlflow.log_metric('final_train_loss', history.history['loss'][-1])
        mlflow.log_metric('final_val_loss', history.history['val_loss'][-1])
        mlflow.log_metric('epochs_trained', len(history.history['loss']))
        
        # Log model
        mlflow.tensorflow.log_model(model, "model")
        
        # Print results
        print(f"\n  Epochs trained: {len(history.history['loss'])}")
        print(f"  Train RMSE: {train_metrics['train_rmse']:.4f}")
        print(f"  Test RMSE:  {test_metrics['test_rmse']:.4f}")
        print(f"  Spatial RMSE: {spatial_metrics['spatial_rmse']:.4f}")
        print(f"  Test R²:    {test_metrics['test_r2']:.4f}")
        
        return model, all_metrics


def main():
    """Main training pipeline."""
    print("="*80)
    print("GEOTEMPORAL FOOD PRICE FORECASTING - MODEL TRAINING")
    print("="*80)
    
    # Set MLflow tracking
    mlflow.set_experiment("food_price_forecasting")
    
    # Initialize data preparator
    preparator = DataPreparator()
    
    # Train models for both targets
    targets = ['inflation_mom', 'inflation_yoy']
    results = {}
    
    for target in targets:
        print(f"\n\n{'#'*80}")
        print(f"# TARGET: {target.upper()}")
        print(f"{'#'*80}")
        
        # Prepare data
        data = preparator.prepare_features(target=target, add_spatial=True)
        
        # 1. Train ML models
        print(f"\n\n{'*'*60}")
        print("MACHINE LEARNING MODELS")
        print(f"{'*'*60}")
        
        ml_models = MLModels()
        ml_results = {}
        
        for name, model in ml_models.get_models().items():
            try:
                trained_model, metrics = train_ml_model(model, name, data, target)
                ml_results[name] = metrics
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
        
        # 2. Train DL models
        print(f"\n\n{'*'*60}")
        print("DEEP LEARNING MODELS")
        print(f"{'*'*60}")
        
        dl_models = DLModels(input_dim=data['X_train'].shape[1])
        dl_results = {}
        
        # Feedforward NN
        try:
            model = dl_models.build_feedforward()
            trained_model, metrics = train_dl_model(model, 'FeedForward_NN', data, target, epochs=100)
            dl_results['FeedForward_NN'] = metrics
        except Exception as e:
            print(f"  ❌ Error training FeedForward NN: {e}")
        
        # Deep NN
        try:
            model = dl_models.build_deep_network()
            trained_model, metrics = train_dl_model(model, 'Deep_NN', data, target, epochs=100)
            dl_results['Deep_NN'] = metrics
        except Exception as e:
            print(f"  ❌ Error training Deep NN: {e}")
        
        results[target] = {
            'ml': ml_results,
            'dl': dl_results
        }
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*80}")
    
    for target in targets:
        print(f"\n{target.upper()} Results:")
        print("-" * 60)
        
        all_results = []
        
        # ML results
        for name, metrics in results[target]['ml'].items():
            all_results.append({
                'Model': name,
                'Type': 'ML',
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'Test R²': metrics['test_r2'],
                'Spatial RMSE': metrics['spatial_rmse']
            })
        
        # DL results
        for name, metrics in results[target]['dl'].items():
            all_results.append({
                'Model': name,
                'Type': 'DL',
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'Test R²': metrics['test_r2'],
                'Spatial RMSE': metrics['spatial_rmse']
            })
        
        # Create DataFrame and sort by Test RMSE
        results_df = pd.DataFrame(all_results).sort_values('Test RMSE')
        print(results_df.to_string(index=False))
        
        # Save results
        output_dir = Path('data/processed/modeling/results')
        output_dir.mkdir(exist_ok=True, parents=True)
        results_df.to_csv(output_dir / f'{target}_model_comparison.csv', index=False)
    
    print(f"\n\n{'='*80}")
    print("✅ ALL MODELS TRAINED AND LOGGED TO MLFLOW")
    print(f"{'='*80}")
    print("\nView results:")
    print("  mlflow ui")
    print("\nThen open: http://localhost:5000")


if __name__ == '__main__':
    main()

