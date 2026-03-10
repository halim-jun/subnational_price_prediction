
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate():
    print("Loading data...")
    try:
        # Assuming script is run from project root, or adjusting path if needed
        # Let's try to load from relative path first
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(project_root, "data/merged/20260212df.csv")
        
        if not os.path.exists(data_path):
             # Fallback: try loading relative to current working directory (if run from project root)
             data_path = "data/merged/20260212df.csv"

        df = pd.read_csv(data_path)
        print(f"Data loaded from: {data_path}")
    except FileNotFoundError:
        print("Error: Data file not found at data/merged/20260212df.csv")
        return

    # --- Preprocessing ---
    print("Preprocessing data...")
    
    # Define features and target
    features = [
        'year', 'month',
        'population', 'crop_cover_fraction',
        'conflict_fatalities', 'conflict_events',
        'energy_index', 'food_index', 'fertilizer_index',
        'night_light_mean',
        'country_iso', 'admin1', 'admin2_canonical'
    ]
    
    target = 'c_maize'

    # Check if all features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features in dataset: {missing_features}")
        features = [f for f in features if f in df.columns]

    # Keep only relevant columns
    cols_to_keep = features + [target]
    df = df[cols_to_keep]

    # 1. Drop rows with missing values (Target or Features)
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows with missing values (target or features).")
    print(f"Remaining rows: {len(df)}")
    
    # 2. Categorical Encoding
    categorical_cols = ['country_iso', 'admin1', 'admin2_canonical']
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Convert to string to handle mixed types if any
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"Encoded {col}")

    # --- Spatio-Temporal Split ---
    print("Performing Spatio-Temporal Split...")
    
    # Sort by time to ensure correct splitting logic physically (though we filter by year)
    df = df.sort_values(by=['year', 'month'])
    
    if len(df) == 0:
        print("Error: No data remaining after filtering.")
        return

    max_year = df['year'].max()
    split_year = max_year - 1
    
    print(f"Data Year Range: {df['year'].min()} - {max_year}")
    print(f"Splitting data: Train < {split_year}, Test >= {split_year}")
    
    train_df = df[df['year'] < split_year]
    test_df = df[df['year'] >= split_year]
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: One of the splits is empty. Check your split logic or data.")
        return

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # --- Modeling ---
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    # --- Evaluation ---
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"\n--- Model Performance on Test Set (Years >= {split_year}) ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")
    print(f"R2 Score: {r2:.4f} (Max: 1.0)")
    
    # Feature Importance
    print("\n--- Feature Importance ---")
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    print(feat_imp)
    
    # --- Detailed Error Analysis ---
    print("\n--- Detailed Error Analysis ---")
    print(f"Test Target (y_test) Stats:\n{y_test.describe()}")
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Prices')
    plt.tight_layout()
    
    output_path_pred = "actual_vs_predicted.png"
    if 'project_root' in locals():
         output_path_pred = os.path.join(project_root, "actual_vs_predicted.png")
    plt.savefig(output_path_pred)
    print(f"Actual vs Predicted plot saved to {output_path_pred}")

    # Plot Residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residuals Analysis')
    plt.tight_layout()
    
    output_path_res = "residuals.png"
    if 'project_root' in locals():
         output_path_res = os.path.join(project_root, "residuals.png")
    plt.savefig(output_path_res)
    print(f"Residuals plot saved to {output_path_res}")
    
    # High vs Low Value Analysis
    median_val = y_test.median()
    low_mask = y_test < median_val
    high_mask = y_test >= median_val
    
    print(f"\n--- Segmentation Analysis (Split by Median: {median_val:.2f}) ---")
    
    if low_mask.any():
        mape_low = mean_absolute_percentage_error(y_test[low_mask], y_pred[low_mask])
        print(f"Low Value Group (< Median) MAPE: {mape_low:.4f} ({mape_low*100:.2f}%)")
        
    if high_mask.any():
        mape_high = mean_absolute_percentage_error(y_test[high_mask], y_pred[high_mask])
        print(f"High Value Group (>= Median) MAPE: {mape_high:.4f} ({mape_high*100:.2f}%)")

    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    
    # Save to project root if possible, else current working directory
    output_path = "feature_importance.png"
    if 'project_root' in locals():
         output_path = os.path.join(project_root, "feature_importance.png")
         
    plt.savefig(output_path)
    print(f"Feature importance plot saved to {output_path}")

if __name__ == "__main__":
    train_and_evaluate()
