"""
Train LSTM model for geotemporal food price forecasting.

Uses LSTM to capture temporal dependencies with categorical embeddings.

Usage:
    python src/models/train_lstm.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mlflow
import mlflow.tensorflow
import wandb
from wandb.keras import WandbCallback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import DataPreparator, calculate_metrics, print_metrics

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)


def prepare_lstm_data(preparator, target='inflation_mom', lookback=6):
    """
    Prepare sequential data for LSTM.
    
    Creates sequences of [lookback] time steps for each h3_index.
    
    Args:
        preparator: DataPreparator instance
        target: Target variable
        lookback: Number of time steps to look back
        
    Returns:
        Dictionary with prepared sequences and metadata
    """
    print(f"\nPreparing LSTM sequences (lookback={lookback})...")
    
    # Add features
    preparator.temporal_train = preparator.add_temporal_lag_features(
        preparator.temporal_train, target
    )
    preparator.temporal_test = preparator.add_temporal_lag_features(
        preparator.temporal_test, target
    )
    preparator.spatial_test = preparator.add_temporal_lag_features(
        preparator.spatial_test, target
    )
    
    preparator.temporal_train = preparator.add_spatial_lag_features(
        preparator.temporal_train
    )
    preparator.temporal_test = preparator.add_spatial_lag_features(
        preparator.temporal_test
    )
    preparator.spatial_test = preparator.add_spatial_lag_features(
        preparator.spatial_test
    )
    
    # Numeric features
    numeric_features = [
        'precipitation',
        'flood_3m', 'flood_6m', 'drought_3m', 'drought_6m',
        'price_volatility_24m', 'price_cv_24m',
        f'{target}_lag1', f'{target}_lag2', f'{target}_lag3',
        f'{target}_rolling_mean_3m', f'{target}_rolling_std_3m',
        'price_volatility_lag1',
        'precipitation_lag1', 'precipitation_lag2', 'precipitation_lag3',
        'precipitation_rolling_mean_3m',
        'volatility_spatial_lag', 'volatility_spatial_diff',
        'n_markets'
    ]
    
    # Categorical features
    categorical_features = ['month', 'quarter', 'countryiso3']
    
    # Create sequences for each h3_index
    def create_sequences(df, lookback):
        """Create sequences grouped by h3_index."""
        df = df.sort_values(['h3_index', 'date']).copy()
        
        X_numeric_list = []
        X_month_list = []
        X_quarter_list = []
        X_country_list = []
        y_list = []
        
        for h3_idx in df['h3_index'].unique():
            h3_data = df[df['h3_index'] == h3_idx].copy()
            
            # Skip if not enough data
            if len(h3_data) < lookback + 1:
                continue
            
            # Drop NaN
            h3_data = h3_data.dropna(subset=numeric_features + [target])
            
            if len(h3_data) < lookback + 1:
                continue
            
            # Create sequences
            for i in range(lookback, len(h3_data)):
                # Numeric features sequence
                seq_numeric = h3_data.iloc[i-lookback:i][numeric_features].values
                
                # Categorical features (use last time step)
                month = h3_data.iloc[i]['month']
                quarter = h3_data.iloc[i]['quarter']
                country = h3_data.iloc[i]['countryiso3']
                
                # Target
                y_val = h3_data.iloc[i][target]
                
                X_numeric_list.append(seq_numeric)
                X_month_list.append(month)
                X_quarter_list.append(quarter)
                X_country_list.append(country)
                y_list.append(y_val)
        
        if len(X_numeric_list) == 0:
            return None
        
        X_numeric = np.array(X_numeric_list)
        X_month = np.array(X_month_list)
        X_quarter = np.array(X_quarter_list)
        X_country = np.array(X_country_list)
        y = np.array(y_list)
        
        return X_numeric, X_month, X_quarter, X_country, y
    
    # Create sequences for train
    train_data = create_sequences(preparator.temporal_train, lookback)
    if train_data is None:
        raise ValueError("Not enough training data")
    
    X_train_num, X_train_month, X_train_quarter, X_train_country, y_train = train_data
    
    # Create sequences for test
    test_data = create_sequences(preparator.temporal_test, lookback)
    if test_data is None:
        raise ValueError("Not enough test data")
    
    X_test_num, X_test_month, X_test_quarter, X_test_country, y_test = test_data
    
    # Create sequences for spatial test
    spatial_data = create_sequences(preparator.spatial_test, lookback)
    if spatial_data is None:
        X_spatial_num, X_spatial_month, X_spatial_quarter, X_spatial_country, y_spatial = (
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        )
    else:
        X_spatial_num, X_spatial_month, X_spatial_quarter, X_spatial_country, y_spatial = spatial_data
    
    # Normalize numeric features
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    # Reshape for scaling
    n_train, n_steps, n_features = X_train_num.shape
    X_train_num_reshaped = X_train_num.reshape(-1, n_features)
    X_train_num_scaled = scaler.fit_transform(X_train_num_reshaped)
    X_train_num = X_train_num_scaled.reshape(n_train, n_steps, n_features)
    
    if len(X_test_num) > 0:
        n_test = X_test_num.shape[0]
        X_test_num_reshaped = X_test_num.reshape(-1, n_features)
        X_test_num_scaled = scaler.transform(X_test_num_reshaped)
        X_test_num = X_test_num_scaled.reshape(n_test, n_steps, n_features)
    
    if len(X_spatial_num) > 0:
        n_spatial = X_spatial_num.shape[0]
        X_spatial_num_reshaped = X_spatial_num.reshape(-1, n_features)
        X_spatial_num_scaled = scaler.transform(X_spatial_num_reshaped)
        X_spatial_num = X_spatial_num_scaled.reshape(n_spatial, n_steps, n_features)
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    # Country encoder
    country_encoder = LabelEncoder()
    all_countries = np.concatenate([
        X_train_country, X_test_country, 
        X_spatial_country if len(X_spatial_country) > 0 else []
    ])
    country_encoder.fit(all_countries)
    
    X_train_country_encoded = country_encoder.transform(X_train_country)
    X_test_country_encoded = country_encoder.transform(X_test_country)
    X_spatial_country_encoded = (
        country_encoder.transform(X_spatial_country) 
        if len(X_spatial_country) > 0 else np.array([])
    )
    
    print(f"  Train sequences: {len(X_train_num)}")
    print(f"  Test sequences: {len(X_test_num)}")
    print(f"  Spatial test sequences: {len(X_spatial_num)}")
    print(f"  Sequence shape: ({lookback}, {n_features})")
    print(f"  Countries: {list(country_encoder.classes_)}")
    
    return {
        'X_train_num': X_train_num,
        'X_train_month': X_train_month,
        'X_train_quarter': X_train_quarter,
        'X_train_country': X_train_country_encoded,
        'y_train': y_train,
        'X_test_num': X_test_num,
        'X_test_month': X_test_month,
        'X_test_quarter': X_test_quarter,
        'X_test_country': X_test_country_encoded,
        'y_test': y_test,
        'X_spatial_num': X_spatial_num,
        'X_spatial_month': X_spatial_month,
        'X_spatial_quarter': X_spatial_quarter,
        'X_spatial_country': X_spatial_country_encoded,
        'y_spatial': y_spatial,
        'n_features': n_features,
        'n_countries': len(country_encoder.classes_),
        'scaler': scaler,
        'country_encoder': country_encoder
    }


def build_lstm_model(n_features, n_countries, lookback=6):
    """
    Build LSTM model with categorical embeddings.
    
    Architecture:
    - Numeric input: LSTM layers
    - Categorical inputs: Embedding layers
    - Concatenate all
    - Dense output
    """
    # Numeric sequence input
    numeric_input = layers.Input(shape=(lookback, n_features), name='numeric_input')
    
    # LSTM layers
    x = layers.LSTM(64, return_sequences=True)(numeric_input)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    # Categorical inputs with embeddings
    month_input = layers.Input(shape=(1,), name='month_input')
    month_embedding = layers.Embedding(
        input_dim=13,  # 1-12 months + 0 for unknown
        output_dim=4,
        name='month_embedding'
    )(month_input)
    month_flat = layers.Flatten()(month_embedding)
    
    quarter_input = layers.Input(shape=(1,), name='quarter_input')
    quarter_embedding = layers.Embedding(
        input_dim=5,  # 1-4 quarters + 0
        output_dim=2,
        name='quarter_embedding'
    )(quarter_input)
    quarter_flat = layers.Flatten()(quarter_embedding)
    
    country_input = layers.Input(shape=(1,), name='country_input')
    country_embedding = layers.Embedding(
        input_dim=n_countries + 1,  # +1 for unknown
        output_dim=4,
        name='country_embedding'
    )(country_input)
    country_flat = layers.Flatten()(country_embedding)
    
    # Concatenate all features
    concatenated = layers.Concatenate()([
        x, month_flat, quarter_flat, country_flat
    ])
    
    # Dense layers
    dense = layers.Dense(32, activation='relu')(concatenated)
    dense = layers.Dropout(0.2)(dense)
    output = layers.Dense(1, activation='linear', name='output')(dense)
    
    # Create model
    model = models.Model(
        inputs=[numeric_input, month_input, quarter_input, country_input],
        outputs=output,
        name='LSTM_Geotemporal'
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_lstm(target='inflation_mom', lookback=6):
    """Train LSTM model."""
    
    print(f"\n{'='*80}")
    print(f"Training LSTM for {target.upper()}")
    print(f"{'='*80}")
    
    # Prepare data
    preparator = DataPreparator()
    data = prepare_lstm_data(preparator, target=target, lookback=lookback)
    
    # Initialize wandb
    wandb.init(
        project="food-price-forecasting",
        name=f"LSTM_{target}",
        config={
            "model_type": "lstm",
            "target": target,
            "lookback": lookback,
            "n_features": data['n_features'],
            "n_train": len(data['X_train_num']),
            "n_countries": data['n_countries'],
            "epochs": 200,
            "batch_size": 32,
            "learning_rate": 0.001,
            "lstm_units": [64, 32],
            "dropout": 0.3,
            "use_embeddings": True,
            "normalized": True
        },
        reinit=True
    )
    
    # Set MLflow experiment
    mlflow.set_experiment("food_price_forecasting")
    
    with mlflow.start_run(run_name=f"LSTM_{target}"):
        # Log parameters
        mlflow.log_param('model_type', 'dl')
        mlflow.log_param('model_name', 'LSTM')
        mlflow.log_param('target', target)
        mlflow.log_param('lookback', lookback)
        mlflow.log_param('n_features', data['n_features'])
        mlflow.log_param('n_train', len(data['X_train_num']))
        mlflow.log_param('n_countries', data['n_countries'])
        mlflow.log_param('use_embeddings', True)
        mlflow.log_param('normalized', True)
        
        # Build model
        print(f"\nBuilding LSTM model...")
        model = build_lstm_model(
            n_features=data['n_features'],
            n_countries=data['n_countries'],
            lookback=lookback
        )
        
        print(model.summary())
        
        # Callbacks
        callbacks = [
            WandbCallback(
                monitor='val_loss',
                save_model=False,  # MLflow will handle model saving
                log_weights=True,
                log_gradients=False
            ),
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
        
        # Train
        print("\n" + "="*60)
        print("STARTING TRAINING...")
        print("="*60)
        print(f"Total samples: {len(data['X_train_num'])}")
        print(f"Training samples: ~{int(len(data['X_train_num']) * 0.8)}")
        print(f"Validation samples: ~{int(len(data['X_train_num']) * 0.2)}")
        print(f"Max epochs: 200 (with early stopping)")
        print("="*60 + "\n")
        
        history = model.fit(
            [data['X_train_num'], data['X_train_month'], 
             data['X_train_quarter'], data['X_train_country']],
            data['y_train'],
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1  # Show progress bar
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print(f"Epochs trained: {len(history.history['loss'])}")
        print(f"Final train loss: {history.history['loss'][-1]:.6f}")
        print(f"Final val loss: {history.history['val_loss'][-1]:.6f}")
        print(f"Best val loss: {min(history.history['val_loss']):.6f}")
        print("="*60 + "\n")
        
        # Predictions
        y_train_pred = model.predict(
            [data['X_train_num'], data['X_train_month'],
             data['X_train_quarter'], data['X_train_country']],
            verbose=0
        ).flatten()
        
        y_test_pred = model.predict(
            [data['X_test_num'], data['X_test_month'],
             data['X_test_quarter'], data['X_test_country']],
            verbose=0
        ).flatten()
        
        if len(data['X_spatial_num']) > 0:
            y_spatial_pred = model.predict(
                [data['X_spatial_num'], data['X_spatial_month'],
                 data['X_spatial_quarter'], data['X_spatial_country']],
                verbose=0
            ).flatten()
        else:
            y_spatial_pred = np.array([])
            data['y_spatial'] = np.array([])
        
        # Calculate metrics
        train_metrics = calculate_metrics(data['y_train'], y_train_pred, prefix='train_')
        test_metrics = calculate_metrics(data['y_test'], y_test_pred, prefix='test_')
        
        if len(y_spatial_pred) > 0:
            spatial_metrics = calculate_metrics(data['y_spatial'], y_spatial_pred, prefix='spatial_')
        else:
            spatial_metrics = {}
        
        all_metrics = {**train_metrics, **test_metrics, **spatial_metrics}
        
        # Log metrics
        for metric_name, value in all_metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log training history to MLflow
        mlflow.log_metric('final_train_loss', history.history['loss'][-1])
        mlflow.log_metric('final_val_loss', history.history['val_loss'][-1])
        mlflow.log_metric('epochs_trained', len(history.history['loss']))
        
        # Log final metrics to wandb
        wandb.log({
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss']),
            'best_val_loss': min(history.history['val_loss'])
        })
        
        # Log model
        mlflow.tensorflow.log_model(model, "model")
        
        # Log test metrics to wandb
        wandb.log(all_metrics)
        
        # Print results
        print_metrics(all_metrics, 'LSTM')
        
        # Finish wandb run
        wandb.finish()
        
        return model, all_metrics


if __name__ == '__main__':
    # Train for both targets
    for target in ['inflation_mom', 'inflation_yoy']:
        try:
            train_lstm(target=target, lookback=6)
        except Exception as e:
            print(f"\n❌ Error training LSTM for {target}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ LSTM training complete!")
    print("View results: mlflow ui")

