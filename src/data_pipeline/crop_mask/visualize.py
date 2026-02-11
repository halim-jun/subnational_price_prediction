
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
from rasterio.enums import Resampling

def visualize_tiff(tiff_path, output_path, scale_factor=0.1):
    """
    Visualizes a TIFF file and saves the output as a PNG.
    Uses downsampling to handle large files.
    """
    if not os.path.exists(tiff_path):
        print(f"Error: File not found at {tiff_path}")
        return

    try:
        with rasterio.open(tiff_path) as src:
            print(f"Opened {tiff_path}")
            print(f"Original Shape: {src.shape}")
            
            # Calculate new shape
            new_height = int(src.height * scale_factor)
            new_width = int(src.width * scale_factor)
            print(f"Resampling to: {new_height} x {new_width}")
            
            # Read the first band with resampling
            data = src.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.nearest
            )
            
            # Create a figure
            plt.figure(figsize=(12, 8))
            
            # Masking nodata
            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            
            # Use a categorical colormap if this is a mask (likely discrete values)
            # Inspect unique values to confirm
            if np.ma.is_masked(data):
                unique_vals = np.unique(data.compressed())
            else:
                unique_vals = np.unique(data)
            print(f"Unique values in mask: {unique_vals}")
            
            # Use 'tab20' or 'viridis' depending on nature of data. 
            # If it's a mask, 'tab20' might be good to distinguish classes.
            cmap = 'viridis' 
            if len(unique_vals) < 20:
                cmap = 'tab20'
                
            im = plt.imshow(data, cmap=cmap, interpolation='nearest')
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Class Value')
            plt.title(f"Crop Mask Visualization (Downsampled {scale_factor*100}%)")
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            
            # Add basic lat/lon axes ticks (approximation based on bounds)
            # This is improved if we use proper extent in imshow
            plt.imshow(data, cmap=cmap, extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
            
            # Save the figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
            plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    tiff_file = os.path.join("data", "crop_mask", "asap_mask_crop_v04.tif")
    output_file = "asap_mask_visualization.png"
    
    # Use a stronger downsampling for speed (0.05 = 5%)
    # 29k * 0.05 ~ 1500 px high. Reasonable.
    visualize_tiff(tiff_file, output_file, scale_factor=0.05)
