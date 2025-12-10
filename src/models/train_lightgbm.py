"""
Train LightGBM model for geotemporal food price forecasting.

Usage:
    python src/models/train_lightgbm.py
"""

import sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import DataPreparator, calculate_metrics, print_metrics


def train_lightgbm(target='inflation_mom', add_spatial=True):
    """Train LightGBM model."""
    
    print(f"\n{'='*80}")
    print(f"Training LightGBM for {target.upper()}")
    print(f"{'='*80}")
    
    # Prepare data
    preparator = DataPreparator()
    data = preparator.prepare_features(target=target, add_spatial=add_spatial, scale=False)
    
    # Set MLflow experiment
    mlflow.set_experiment("food_price_forecasting")
    
    with mlflow.start_run(run_name=f"LightGBM_{target}"):
        # Log parameters
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        mlflow.log_param('model_type', 'ml')
        mlflow.log_param('model_name', 'LightGBM')
        mlflow.log_param('target', target)
        mlflow.log_param('n_features', data['n_features'])
        mlflow.log_param('n_train', len(data['X_train']))
        mlflow.log_param('add_spatial', add_spatial)
        
        for k, v in params.items():
            mlflow.log_param(k, v)
        
        # Train model
        print("\nTraining LightGBM...")
        model = lgb.LGBMRegressor(**params)
        model.fit(
            data['X_train'], data['y_train'],
            eval_set=[(data['X_test'], data['y_test'])],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Predictions
        y_train_pred = model.predict(data['X_train'])
        y_test_pred = model.predict(data['X_test'])
        y_spatial_pred = model.predict(data['X_spatial'])
        
        # Calculate metrics
        train_metrics = calculate_metrics(data['y_train'], y_train_pred, prefix='train_')
        test_metrics = calculate_metrics(data['y_test'], y_test_pred, prefix='test_')
        spatial_metrics = calculate_metrics(data['y_spatial'], y_spatial_pred, prefix='spatial_')
        
        all_metrics = {**train_metrics, **test_metrics, **spatial_metrics}
        
        # Log metrics
        for metric_name, value in all_metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log model
        mlflow.lightgbm.log_model(model, "model")
        
        # Log feature importance
        import pandas as pd
        importance_df = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv('/tmp/lightgbm_feature_importance.csv', index=False)
        mlflow.log_artifact('/tmp/lightgbm_feature_importance.csv')
        
        print("\nTop 10 Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Print results
        print_metrics(all_metrics, 'LightGBM')
        
        return model, all_metrics


if __name__ == '__main__':
    # Train for both targets
    for target in ['inflation_mom', 'inflation_yoy']:
        train_lightgbm(target=target, add_spatial=True)
    
    print("\n✅ LightGBM training complete!")
    print("View results: mlflow ui")

