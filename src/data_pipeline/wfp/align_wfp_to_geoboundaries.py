
import pandas as pd
import geopandas as gpd
import requests
import os
import sys
from pathlib import Path

# Config
# Ensure we map from project root or relative
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_RAW = BASE_DIR / "data/raw/wfp"
DATA_PROCESSED = BASE_DIR / "data/processed/wfp"
GB_DIR = BASE_DIR / "data/geoboundaries"

# Input WFP File (Select the largest/newest one)
WFP_INPUT_FILE = DATA_RAW / "wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv"
OUTPUT_FILE = DATA_PROCESSED / f"geoboundaries_{WFP_INPUT_FILE.name}"

start_year = 2019 # From filename
ISO_CODES = [
    'DJI', 'ERI', 'ETH', 'KEN', 'SOM', 'SSD', 'SDN', 'UGA', 'RWA', 'BDI', 'TZA', 
    'ZMB', 'MWI', 'MOZ', 'MDG'
]
GB_API_URL = "https://www.geoboundaries.org/api/current/gbOpen/{}/{}/"

def download_gb_file(iso, level):
    """Downloads GeoBoundaries GeoJSON if not exists."""
    lvl_str = f"ADM{level}"
    GB_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"gb_{iso}_{lvl_str}.geojson"
    filepath = GB_DIR / filename
    
    if filepath.exists() and filepath.stat().st_size > 1000:
        return filepath

    api_url = GB_API_URL.format(iso, lvl_str)
    try:
        r = requests.get(api_url, timeout=10)
        if r.status_code != 200: return None
        
        meta = r.json()
        dl_url = meta.get('gjDownloadURL')
        if not dl_url: return None
            
        print(f"Downloading {dl_url}...")
        r_dl = requests.get(dl_url, stream=True, timeout=30)
        if r_dl.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in r_dl.iter_content(chunk_size=32768):
                    f.write(chunk)
            return filepath
    except Exception as e:
        print(f"Error downloading {iso} {lvl_str}: {e}")
    
    return None

def load_layers():
    """Loads all ADM layers into separate merged GDFs."""
    print("Loading boundary layers...")
    gdfs_l0, gdfs_l1, gdfs_l2 = [], [], []
    
    for iso in ISO_CODES:
        # L0 (Country)
        f0 = download_gb_file(iso, 0)
        if f0:
            try:
                gdf = gpd.read_file(f0)
                if 'shapeGroup' not in gdf.columns: gdf['shapeGroup'] = iso
                # Standardize column for ISO
                gdf['gb_country_iso'] = gdf['shapeGroup']  
                gdfs_l0.append(gdf[['gb_country_iso', 'geometry']])
            except: pass

        # L1 (Admin1)
        f1 = download_gb_file(iso, 1)
        if f1:
            try:
                gdf = gpd.read_file(f1).rename(columns={'shapeName': 'gb_admin1'})
                # Need country context for uniqueness? Spatial join handles location but helpful to have
                gdfs_l1.append(gdf[['gb_admin1', 'geometry']])
            except: pass
            
        # L2 (Admin2)
        f2 = download_gb_file(iso, 2)
        if f2:
            try:
                gdf = gpd.read_file(f2).rename(columns={'shapeName': 'gb_admin2'})
                gdfs_l2.append(gdf[['gb_admin2', 'geometry']])
            except: pass
            
    # Concatenate
    layer0 = pd.concat(gdfs_l0, ignore_index=True) if gdfs_l0 else None
    layer1 = pd.concat(gdfs_l1, ignore_index=True) if gdfs_l1 else None
    layer2 = pd.concat(gdfs_l2, ignore_index=True) if gdfs_l2 else None
    
    print("Layers loaded.")
    return layer0, layer1, layer2

def align_wfp_data():
    print("--- Aligment of WFP Price Data to GeoBoundaries ---")
    
    if not WFP_INPUT_FILE.exists():
        print(f"Error: Input file not found: {WFP_INPUT_FILE}")
        return

    # 1. Load WFP
    print(f"Loading WFP CSV: {WFP_INPUT_FILE}")
    df = pd.read_csv(WFP_INPUT_FILE)
    print(f"Initial Shape: {df.shape}")
    
    # Filter valid lat/lon
    df_valid = df.dropna(subset=['latitude', 'longitude']).copy()
    print(f"Rows with valid coordinates: {len(df_valid)}")
    
    # Create GeoDataFrame
    gdf_pts = gpd.GeoDataFrame(
        df_valid, 
        geometry=gpd.points_from_xy(df_valid.longitude, df_valid.latitude), 
        crs="EPSG:4326"
    )
    
    # 2. Load Boundaries
    l0, l1, l2 = load_layers()
    
    # 3. Spatial Joins
    print("Performing Spatial Joins...")
    
    # Join Country (Correct ISO)
    # Using 'inner' or 'left'? Use left to keep points, but we only want matches that fall in our region.
    # WFP has 'countryiso3' but let's trust geometry.
    
    if l0 is not None:
        print("  Joining Admin0 (Country)...")
        gdf_pts = gpd.sjoin(gdf_pts, l0, how='left', predicate='within')
        # Clean up sjoin columns
        if 'index_right' in gdf_pts.columns: gdf_pts = gdf_pts.drop(columns=['index_right'])
    
    if l1 is not None:
        print("  Joining Admin1...")
        gdf_pts = gpd.sjoin(gdf_pts, l1, how='left', predicate='within')
        if 'index_right' in gdf_pts.columns: gdf_pts = gdf_pts.drop(columns=['index_right'])

    if l2 is not None:
        print("  Joining Admin2...")
        gdf_pts = gpd.sjoin(gdf_pts, l2, how='left', predicate='within')
        if 'index_right' in gdf_pts.columns: gdf_pts = gdf_pts.drop(columns=['index_right'])
        
    # 4. Post-processing
    # Check coverage
    matched = gdf_pts.dropna(subset=['gb_country_iso', 'gb_admin2'])
    print(f"Matched {len(matched)} / {len(df_valid)} rows to GeoBoundaries regions.")
    
    # Create aligned columns
    # We prioritize the GeoBoundaries names
    # original columns: admin1, admin2, countryiso3
    # new columns: admin1_aligned, admin2_aligned, country_iso_aligned
    
    gdf_pts['country_iso_aligned'] = gdf_pts['gb_country_iso']
    gdf_pts['admin1_aligned'] = gdf_pts['gb_admin1']
    gdf_pts['admin2_aligned'] = gdf_pts['gb_admin2']
    
    # Drop geometry for CSV export
    df_result = pd.DataFrame(gdf_pts.drop(columns=['geometry', 'gb_country_iso', 'gb_admin1', 'gb_admin2']))
    
    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {OUTPUT_FILE}")
    df_result.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    align_wfp_data()
