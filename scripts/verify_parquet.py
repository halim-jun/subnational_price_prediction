
import pandas as pd
import os

parquet_path = "/tmp/asap_mask_crop_v04.parquet"

if os.path.exists(parquet_path):
    print(f"File found at {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        print("Successfully read Parquet file.")
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Head:")
        print(df.head())
        print("\nValue counts:")
        print(df['value'].value_counts().head())
        
        file_size = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Error reading Parquet: {e}")
else:
    print(f"File not found at {parquet_path}")
