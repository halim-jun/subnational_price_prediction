"""
Train Neural Network models for geotemporal food price forecasting.

Usage:
    python src/models/train_neural_network.py
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mlflow
import mlflow.tensorflow

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import DataPreparator, calculate_metrics, print_metrics

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)


def build_feedforward_nn(input_dim):
    """Build feedforward neural network."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_deep_nn(input_dim):
    """Build deeper neural network with batch normalization."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def get_callbacks():
    """Get training callbacks."""
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=1
        )
    ]


def train_neural_network(target='inflation_mom', model_type='feedforward', add_spatial=True):
    """Train neural network model."""
    
    model_name = f"{'Deep' if model_type == 'deep' else 'FeedForward'}_NN"
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} for {target.upper()}")
    print(f"{'='*80}")
    
    # Prepare data (with scaling for neural networks)
    preparator = DataPreparator()
    data = preparator.prepare_features(target=target, add_spatial=add_spatial, scale=True)
    
    # Set MLflow experiment
    mlflow.set_experiment("food_price_forecasting")
    
    with mlflow.start_run(run_name=f"{model_name}_{target}"):
        # Log parameters
        mlflow.log_param('model_type', 'dl')
        mlflow.log_param('model_name', model_name)
        mlflow.log_param('target', target)
        mlflow.log_param('n_features', data['n_features'])
        mlflow.log_param('n_train', len(data['X_train']))
        mlflow.log_param('add_spatial', add_spatial)
        mlflow.log_param('scaled', True)
        mlflow.log_param('epochs', 200)
        mlflow.log_param('batch_size', 32)
        
        # Build model
        print(f"\nBuilding {model_name}...")
        if model_type == 'deep':
            model = build_deep_nn(data['n_features'])
        else:
            model = build_feedforward_nn(data['n_features'])
        
        print(model.summary())
        
        # Train model
        print("\nTraining...")
        callbacks = get_callbacks()
        
        history = model.fit(
            data['X_train'], data['y_train'],
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Predictions
        y_train_pred = model.predict(data['X_train'], verbose=0).flatten()
        y_test_pred = model.predict(data['X_test'], verbose=0).flatten()
        y_spatial_pred = model.predict(data['X_spatial'], verbose=0).flatten()
        
        # Calculate metrics
        train_metrics = calculate_metrics(data['y_train'], y_train_pred, prefix='train_')
        test_metrics = calculate_metrics(data['y_test'], y_test_pred, prefix='test_')
        spatial_metrics = calculate_metrics(data['y_spatial'], y_spatial_pred, prefix='spatial_')
        
        all_metrics = {**train_metrics, **test_metrics, **spatial_metrics}
        
        # Log metrics
        for metric_name, value in all_metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log training history
        mlflow.log_metric('final_train_loss', history.history['loss'][-1])
        mlflow.log_metric('final_val_loss', history.history['val_loss'][-1])
        mlflow.log_metric('epochs_trained', len(history.history['loss']))
        
        # Log model
        mlflow.tensorflow.log_model(model, "model")
        
        # Print results
        print_metrics(all_metrics, model_name)
        
        return model, all_metrics


if __name__ == '__main__':
    # Train feedforward NN for both targets
    for target in ['inflation_mom', 'inflation_yoy']:
        train_neural_network(target=target, model_type='feedforward', add_spatial=True)
    
    # Train deep NN for both targets
    for target in ['inflation_mom', 'inflation_yoy']:
        train_neural_network(target=target, model_type='deep', add_spatial=True)
    
    print("\n✅ Neural Network training complete!")
    print("View results: mlflow ui")

