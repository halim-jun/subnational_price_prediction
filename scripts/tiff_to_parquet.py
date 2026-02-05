
import rasterio
import pandas as pd
import numpy as np
import os

def tiff_to_parquet(tiff_path, parquet_path):
    """
    Reads a TIFF file, extracts non-zero pixels with their lat/lon coordinates,
    and saves the result to a Parquet file.
    """
    if not os.path.exists(tiff_path):
        print(f"Error: File not found at {tiff_path}")
        return

    try:
        print(f"Opening {tiff_path}...")
        with rasterio.open(tiff_path) as src:
            # Read the entire array (assuming memory is sufficient for this specific file size/machine)
            # If memory issues arise, we would need to chunk this.
            # 29346 * 80640 pixels is ~2.3 billion pixels. 
            # If 8-bit, that's 2.3 GB. It might be tight or okay depending on available RAM.
            # Let's try reading it. If it fails, we'll implement chunking.
            print("Reading data into memory...")
            data = src.read(1)
            
            # Get the transform for coordinate calculation
            transform = src.transform
            
            print("Finding non-zero indices...")
            # Get indices where data is not 0 (and not masked if applicable)
            # Using numpy.nonzero is efficient
            rows, cols = np.nonzero(data)
            
            if len(rows) == 0:
                print("No non-zero data found.")
                return

            print(f"Found {len(rows)} non-zero pixels.")
            
            # Extract values
            values = data[rows, cols]
            
            print("Calculating coordinates...")
            # rasterio.transform.xy handles arrays of rows/cols
            # It returns xs, ys
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
            
            # Create DataFrame
            print("Creating DataFrame...")
            df = pd.DataFrame({
                'longitude': xs,
                'latitude': ys,
                'value': values
            })
            
            # Optimize dtypes
            df['longitude'] = df['longitude'].astype('float32')
            df['latitude'] = df['latitude'].astype('float32')
            # value type depends on source, likely uint8 or similar
            
            print(f"Saving to {parquet_path}...")
            df.to_parquet(parquet_path, index=False)
            print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    tiff_file = os.path.join("data", "crop_mask", "asap_mask_crop_v04.tif")
    # Save to tmp to avoid permission issues
    parquet_file = "/tmp/asap_mask_crop_v04.parquet"
    
    print(f"Target file: {parquet_file}")
    tiff_to_parquet(tiff_file, parquet_file)
