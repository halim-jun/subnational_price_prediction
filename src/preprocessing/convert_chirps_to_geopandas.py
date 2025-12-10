"""
Convert all CHIRPS TIF files to a single GeoDataFrame
"""
import os
import glob
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
from datetime import datetime
from tqdm import tqdm

def parse_chirps_filename(filepath):
    """Extract date from CHIRPS filename: chirps-v2.0.YYYY.MMDD.tiff"""
    filename = os.path.basename(filepath)
    parts = filename.replace('.tiff', '').split('.')
    year = int(parts[2])
    month_range = parts[3]
    start_month = int(month_range[:2])
    return datetime(year, start_month, 1), f"{year}-{start_month:02d}"

def tif_to_gdf(tif_path, date_str, subsample=10):
    """
    Convert single TIF file to GeoDataFrame
    
    Parameters:
    -----------
    tif_path : str
        Path to TIF file
    date_str : str
        Date string for this file
    subsample : int
        Subsample every N pixels (default: 10 = every 10th pixel)
        Set to 1 for all pixels (WARNING: very large!)
    """
    with rasterio.open(tif_path) as src:
        # Read band
        band = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Create subsampled grid
        rows, cols = np.meshgrid(
            np.arange(0, band.shape[0], subsample),
            np.arange(0, band.shape[1], subsample),
            indexing='ij'
        )
        
        # Flatten
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        values_flat = band[rows_flat, cols_flat]
        
        # Filter NoData (usually negative or very small values)
        valid_mask = values_flat > -9999
        rows_valid = rows_flat[valid_mask]
        cols_valid = cols_flat[valid_mask]
        values_valid = values_flat[valid_mask]
        
        # Convert pixel coordinates to geographic coordinates (lon, lat)
        # xy() returns (x, y) where x=longitude, y=latitude for EPSG:4326
        xs, ys = rasterio.transform.xy(
            transform, 
            rows_valid, 
            cols_valid, 
            offset='center'
        )
        
        # Convert to numpy arrays
        lons = np.array(xs)
        lats = np.array(ys)
        
        # Create GeoDataFrame with explicit EPSG:4326 (WGS84)
        # Point(longitude, latitude) - x is lon, y is lat
        gdf = gpd.GeoDataFrame(
            {
                'precipitation': values_valid,
                'date': date_str,
                'lon': lons,
                'lat': lats
            },
            geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
            crs='EPSG:4326'  # Explicitly set WGS84
        )
        
        return gdf

def convert_all_chirps(chirps_dir, output_path, subsample=10):
    """
    Convert all CHIRPS TIF files and save as single GeoParquet/GeoJSON
    
    Parameters:
    -----------
    chirps_dir : str
        Directory containing CHIRPS TIF files
    output_path : str
        Output file path (.gpkg, .geojson, or .parquet)
    subsample : int
        Subsample factor (10 = every 10th pixel)
    """
    # Find all TIF files
    tiff_files = sorted(glob.glob(os.path.join(chirps_dir, "*.tiff")))
    
    if not tiff_files:
        print(f"❌ No TIF files found in {chirps_dir}")
        return
    
    print(f"📁 Found {len(tiff_files)} CHIRPS TIF files")
    print(f"🔄 Converting with subsample={subsample} (1 = all pixels, 10 = every 10th)")
    print(f"💾 Output: {output_path}")
    print()
    
    # Convert all files
    all_gdfs = []
    
    for tif_path in tqdm(tiff_files, desc="Converting TIF files"):
        try:
            # Parse date
            date_obj, date_str = parse_chirps_filename(tif_path)
            
            # Convert to GeoDataFrame
            gdf = tif_to_gdf(tif_path, date_str, subsample=subsample)
            all_gdfs.append(gdf)
            
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(tif_path)}: {e}")
            continue
    
    # Concatenate all GeoDataFrames
    print("\n🔗 Concatenating all GeoDataFrames...")
    combined_gdf = pd.concat(all_gdfs, ignore_index=True)
    
    print(f"\n✅ Combined GeoDataFrame created!")
    print(f"   Total rows: {len(combined_gdf):,}")
    print(f"   Date range: {combined_gdf['date'].min()} to {combined_gdf['date'].max()}")
    print(f"   CRS: {combined_gdf.crs}")
    print(f"   Precipitation range: {combined_gdf['precipitation'].min():.2f} - {combined_gdf['precipitation'].max():.2f}")
    
    # Save to file
    print(f"\n💾 Saving to {output_path}...")
    
    if output_path.endswith('.gpkg'):
        combined_gdf.to_file(output_path, driver='GPKG')
    elif output_path.endswith('.geojson'):
        combined_gdf.to_file(output_path, driver='GeoJSON')
    elif output_path.endswith('.parquet'):
        combined_gdf.to_parquet(output_path)
    else:
        # Default to GeoParquet
        combined_gdf.to_parquet(output_path)
    
    print(f"✅ Saved successfully!")
    
    return combined_gdf

if __name__ == "__main__":
    # Configuration
    CHIRPS_DIR = "./data/raw/climate/chirps/"
    OUTPUT_PATH = "./data/processed/chirps_all_geopandas.parquet"  # or .gpkg, .geojson
    SUBSAMPLE = 3  # 1 = all pixels (LARGE!), 5 = every 5th pixel, 10 = every 10th
    
    # Convert
    gdf = convert_all_chirps(
        chirps_dir=CHIRPS_DIR,
        output_path=OUTPUT_PATH,
        subsample=SUBSAMPLE
    )
    
    # Preview
    if gdf is not None:
        print("\n📊 Preview:")
        print(gdf.head(10))
        print(f"\n📈 Summary by date:")
        print(gdf.groupby('date')['precipitation'].agg(['count', 'mean', 'min', 'max']))

