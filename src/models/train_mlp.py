"""
Train MLP (Multi-Layer Perceptron) for geotemporal food price forecasting.

Uses scikit-learn's MLPRegressor instead of TensorFlow to avoid compatibility issues.

Usage:
    python src/models/train_mlp.py
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
import mlflow
import mlflow.sklearn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import DataPreparator, calculate_metrics, print_metrics


def train_mlp(target='inflation_mom', add_spatial=True):
    """Train MLP model."""
    
    print(f"\n{'='*80}")
    print(f"Training MLP Neural Network for {target.upper()}")
    print(f"{'='*80}")
    
    # Prepare data (with scaling for neural networks)
    preparator = DataPreparator()
    data = preparator.prepare_features(target=target, add_spatial=add_spatial, scale=True)
    
    # Set MLflow experiment
    mlflow.set_experiment("food_price_forecasting")
    
    with mlflow.start_run(run_name=f"MLP_{target}"):
        # Log parameters
        params = {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,  # L2 penalty
            'batch_size': 32,
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 30,
            'random_state': 42
        }
        
        mlflow.log_param('model_type', 'dl')
        mlflow.log_param('model_name', 'MLP')
        mlflow.log_param('target', target)
        mlflow.log_param('n_features', data['n_features'])
        mlflow.log_param('n_train', len(data['X_train']))
        mlflow.log_param('add_spatial', add_spatial)
        mlflow.log_param('scaled', True)
        
        for k, v in params.items():
            mlflow.log_param(k, v)
        
        # Train model
        print("\nTraining MLP Neural Network...")
        print(f"Architecture: Input({data['n_features']}) -> 128 -> 64 -> 32 -> Output(1)")
        
        model = MLPRegressor(**params, verbose=True)
        model.fit(data['X_train'], data['y_train'])
        
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
        
        # Log training info
        mlflow.log_metric('iterations', model.n_iter_)
        mlflow.log_metric('final_loss', model.loss_)
        
        print(f"\nTraining completed in {model.n_iter_} iterations")
        print(f"Final loss: {model.loss_:.6f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print_metrics(all_metrics, 'MLP')
        
        return model, all_metrics


if __name__ == '__main__':
    # Train for both targets
    for target in ['inflation_mom', 'inflation_yoy']:
        train_mlp(target=target, add_spatial=True)
    
    print("\n✅ MLP training complete!")
    print("View results: mlflow ui")

