"""
Train PyTorch LSTM model for geotemporal food price forecasting.

Uses PyTorch LSTM with categorical embeddings and wandb tracking.

Usage:
    python src/models/train_lstm_pytorch.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
import wandb
import h3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import DataPreparator, calculate_metrics, print_metrics

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class FoodPriceDataset(Dataset):
    """PyTorch Dataset for food price sequences."""
    
    def __init__(self, X_num, X_month, X_quarter, X_country, y):
        self.X_num = torch.FloatTensor(X_num)
        self.X_month = torch.LongTensor(X_month)
        self.X_quarter = torch.LongTensor(X_quarter)
        self.X_country = torch.LongTensor(X_country)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_month[idx],
            self.X_quarter[idx],
            self.X_country[idx],
            self.y[idx]
        )


class LSTMGeotemporalModel(nn.Module):
    """
    PyTorch LSTM model with categorical embeddings.
    
    Architecture:
    - Numeric features: LSTM layers
    - Categorical features: Embedding layers
    - Concatenate and predict
    """
    
    def __init__(self, n_features, n_countries, lookback=6, 
                 lstm_hidden=64, lstm_layers=2, dropout=0.3):
        super(LSTMGeotemporalModel, self).__init__()
        
        # LSTM for numeric sequence
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Embeddings for categorical features
        self.month_embedding = nn.Embedding(13, 4)  # 1-12 + padding
        self.quarter_embedding = nn.Embedding(5, 2)  # 1-4 + padding
        self.country_embedding = nn.Embedding(n_countries + 1, 4)
        
        # Fully connected layers
        fc_input_size = lstm_hidden + 4 + 2 + 4  # LSTM output + embeddings
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x_num, x_month, x_quarter, x_country):
        # LSTM for sequences
        lstm_out, _ = self.lstm(x_num)
        lstm_last = lstm_out[:, -1, :]  # Take last time step
        
        # Embeddings
        month_emb = self.month_embedding(x_month).squeeze(1)
        quarter_emb = self.quarter_embedding(x_quarter).squeeze(1)
        country_emb = self.country_embedding(x_country).squeeze(1)
        
        # Concatenate all features
        combined = torch.cat([lstm_last, month_emb, quarter_emb, country_emb], dim=1)
        
        # Predict
        output = self.fc(combined)
        return output.squeeze()


def prepare_lstm_data(preparator, target='inflation_mom', lookback=6):
    """
    Prepare sequential data for LSTM.
    
    Creates sequences of [lookback] time steps for each h3_index.
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
            
            if len(h3_data) < lookback + 1:
                continue
            
            h3_data = h3_data.dropna(subset=numeric_features + [target])
            
            if len(h3_data) < lookback + 1:
                continue
            
            for i in range(lookback, len(h3_data)):
                seq_numeric = h3_data.iloc[i-lookback:i][numeric_features].values
                month = h3_data.iloc[i]['month']
                quarter = h3_data.iloc[i]['quarter']
                country = h3_data.iloc[i]['countryiso3']
                y_val = h3_data.iloc[i][target]
                
                X_numeric_list.append(seq_numeric)
                X_month_list.append(month)
                X_quarter_list.append(quarter)
                X_country_list.append(country)
                y_list.append(y_val)
        
        if len(X_numeric_list) == 0:
            return None
        
        return (
            np.array(X_numeric_list),
            np.array(X_month_list),
            np.array(X_quarter_list),
            np.array(X_country_list),
            np.array(y_list)
        )
    
    # Create sequences
    train_data = create_sequences(preparator.temporal_train, lookback)
    if train_data is None:
        raise ValueError("Not enough training data")
    
    X_train_num, X_train_month, X_train_quarter, X_train_country, y_train = train_data
    
    test_data = create_sequences(preparator.temporal_test, lookback)
    if test_data is None:
        raise ValueError("Not enough test data")
    
    X_test_num, X_test_month, X_test_quarter, X_test_country, y_test = test_data
    
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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mae = 0
    
    for X_num, X_month, X_quarter, X_country, y in dataloader:
        X_num = X_num.to(device)
        X_month = X_month.to(device)
        X_quarter = X_quarter.to(device)
        X_country = X_country.to(device)
        y = y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_num, X_month, X_quarter, X_country)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mae += torch.mean(torch.abs(outputs - y)).item()
    
    return total_loss / len(dataloader), total_mae / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_mae = 0
    
    with torch.no_grad():
        for X_num, X_month, X_quarter, X_country, y in dataloader:
            X_num = X_num.to(device)
            X_month = X_month.to(device)
            X_quarter = X_quarter.to(device)
            X_country = X_country.to(device)
            y = y.to(device)
            
            outputs = model(X_num, X_month, X_quarter, X_country)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs - y)).item()
    
    return total_loss / len(dataloader), total_mae / len(dataloader)


def predict(model, dataset, device, batch_size=32):
    """Make predictions."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    
    with torch.no_grad():
        for X_num, X_month, X_quarter, X_country, _ in dataloader:
            X_num = X_num.to(device)
            X_month = X_month.to(device)
            X_quarter = X_quarter.to(device)
            X_country = X_country.to(device)
            
            outputs = model(X_num, X_month, X_quarter, X_country)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)


def train_lstm(target='inflation_mom', lookback=6, h3_resolution=5, data_path=None):
    """
    Train PyTorch LSTM model.
    
    Args:
        target: 'inflation_mom' or 'inflation_yoy'
        lookback: Number of time steps to look back
        h3_resolution: H3 resolution level used in data
        data_path: Path to modeling data directory (optional)
    """
    
    print(f"\n{'='*80}")
    print(f"Training PyTorch LSTM for {target.upper()}")
    print(f"H3 Resolution: {h3_resolution}")
    print(f"{'='*80}")
    
    # Prepare data
    preparator = DataPreparator(data_path) if data_path else DataPreparator()
    data = prepare_lstm_data(preparator, target=target, lookback=lookback)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 200
    patience = 30
    
    # Initialize wandb
    wandb.init(
        project="food-price-forecasting",
        name=f"PyTorch_LSTM_H3{h3_resolution}_{target}",
        tags=[f"h3_{h3_resolution}", "pytorch", "lstm", target],
        config={
            # Data info
            "h3_resolution": h3_resolution,
            "h3_cell_size_km2": int(h3.average_hexagon_area(h3_resolution, unit='km^2')),
            "data_source": data_path or "data/processed/modeling/",
            "target": target,
            "n_train": len(data['X_train_num']),
            "n_test": len(data['X_test_num']),
            "n_spatial_test": len(data['X_spatial_num']),
            "n_features": data['n_features'],
            "n_countries": data['n_countries'],
            "countries": list(data['country_encoder'].classes_),
            
            # Model architecture
            "model_type": "pytorch_lstm",
            "lookback": lookback,
            "lstm_hidden": 64,
            "lstm_layers": 2,
            "dropout": 0.3,
            "use_embeddings": True,
            "month_emb_dim": 4,
            "quarter_emb_dim": 2,
            "country_emb_dim": 4,
            
            # Training config
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss_function": "MSE",
            "normalized": True,
            "device": str(device),
            
            # Feature engineering
            "spatial_lag": True,
            "temporal_lag": True,
            "flood_drought_indicators": True
        },
        reinit=True
    )
    
    # Set MLflow experiment
    mlflow.set_experiment("food_price_forecasting")
    
    with mlflow.start_run(run_name=f"PyTorch_LSTM_H3{h3_resolution}_{target}"):
        # Log parameters
        mlflow.log_param('model_type', 'pytorch_lstm')
        mlflow.log_param('h3_resolution', h3_resolution)
        mlflow.log_param('h3_cell_size_km2', int(h3.average_hexagon_area(h3_resolution, unit='km^2')))
        mlflow.log_param('target', target)
        mlflow.log_param('lookback', lookback)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('n_countries', data['n_countries'])
        
        # Create datasets
        train_dataset = FoodPriceDataset(
            data['X_train_num'], data['X_train_month'],
            data['X_train_quarter'], data['X_train_country'],
            data['y_train']
        )
        
        test_dataset = FoodPriceDataset(
            data['X_test_num'], data['X_test_month'],
            data['X_test_quarter'], data['X_test_country'],
            data['y_test']
        )
        
        # Split train into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Model
        model = LSTMGeotemporalModel(
            n_features=data['n_features'],
            n_countries=data['n_countries'],
            lookback=lookback
        ).to(device)
        
        print(f"\nModel Architecture:")
        print(model)
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )
        
        # Training loop
        print("\n" + "="*60)
        print("STARTING TRAINING...")
        print("="*60)
        print(f"Train samples: {train_size}")
        print(f"Val samples: {val_size}")
        print(f"Test samples: {len(test_dataset)}")
        print("="*60 + "\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_mae = validate(model, val_loader, criterion, device)
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Log to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.4f} | "
                  f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        print(f"Best validation loss: {best_val_loss:.6f}")
        print("="*60 + "\n")
        
        # Final predictions
        y_train_pred = predict(model, train_dataset, device, batch_size)
        y_test_pred = predict(model, test_dataset, device, batch_size)
        
        if len(data['X_spatial_num']) > 0:
            spatial_dataset = FoodPriceDataset(
                data['X_spatial_num'], data['X_spatial_month'],
                data['X_spatial_quarter'], data['X_spatial_country'],
                data['y_spatial']
            )
            y_spatial_pred = predict(model, spatial_dataset, device, batch_size)
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
            wandb.log({metric_name: value})
        
        # Log final stats
        mlflow.log_metric('best_val_loss', best_val_loss)
        wandb.log({'best_val_loss': best_val_loss})
        
        # Save model
        model_path = f'/tmp/lstm_{target}.pth'
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        # Print results
        print_metrics(all_metrics, 'PyTorch LSTM')
        
        # Finish wandb
        wandb.finish()
        
        return model, all_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PyTorch LSTM for food price forecasting')
    parser.add_argument('--h3-resolution', type=int, default=5, 
                       help='H3 resolution level (default: 5)')
    parser.add_argument('--lookback', type=int, default=6,
                       help='Number of time steps to look back (default: 6)')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to modeling data directory (optional)')
    parser.add_argument('--target', type=str, choices=['inflation_mom', 'inflation_yoy', 'both'],
                       default='both', help='Target variable to predict (default: both)')
    
    args = parser.parse_args()
    
    # Train for specified targets
    targets = ['inflation_mom', 'inflation_yoy'] if args.target == 'both' else [args.target]
    
    for target in targets:
        try:
            train_lstm(
                target=target, 
                lookback=args.lookback,
                h3_resolution=args.h3_resolution,
                data_path=args.data_path
            )
        except Exception as e:
            print(f"\n❌ Error training LSTM for {target}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ PyTorch LSTM training complete!")
    print(f"\nH3 Resolution: {args.h3_resolution} (~{h3.average_hexagon_area(args.h3_resolution, unit='km^2'):.0f} km² per hexagon)")
    print("\nView results:")
    print("  - wandb: https://wandb.ai")
    print("  - MLflow: mlflow ui → http://localhost:5000")

