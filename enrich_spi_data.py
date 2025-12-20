
import pandas as pd
import geopandas as gpd
import requests
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Extended list + some Central Africa just in case
ISO_CODES = [
    'DJI', 'ERI', 'ETH', 'KEN', 'SOM', 'SSD', 'SDN', 'UGA', 'RWA', 'BDI', 'TZA', 
    'ZMB', 'MWI', 'MOZ', 'MDG', 'COD', 'CAF'
]
# Base URL for GADM 4.1
GADM_BASE_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{}_{}.json"
DATA_DIR = "data/external/gadm"
INPUT_FILE = "data/processed/spi/06_spi_csv/east_africa_spi_gamma_3_month.csv"
OUTPUT_FILE = "data/processed/spi/06_spi_csv/east_africa_spi_gamma_3_month_with_boundaries.csv"

def get_session():
    """Returns a requests session with retries."""
    session = requests.Session()
    retry = Retry(connect=5, read=5, redirect=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def download_country_boundary(iso):
    """Downloads GADM data for a country, trying levels 3 down to 0."""
    os.makedirs(DATA_DIR, exist_ok=True)
    session = get_session()
    
    # Try levels 3, 2, 1, 0
    for level in range(3, -1, -1):
        filename = f"gadm41_{iso}_{level}.json"
        filepath = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(filepath):
            # Check if file is valid JSON (empty check)
            if os.path.getsize(filepath) > 100:
                print(f"Using existing {filename} (Level {level})")
                return filepath, level
            else:
                print(f"Existing file {filename} seems invalid. Redownloading...")
        
        url = GADM_BASE_URL.format(iso, level)
        print(f"Trying to download {url}...")
        try:
            r = session.get(url, stream=True, timeout=60)
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        f.write(chunk)
                print(f"Downloaded {filename}.")
                return filepath, level
            else:
                print(f"Level {level} not available for {iso} (Status {r.status_code})")
        except Exception as e:
            print(f"Error downloading {iso} level {level}: {e}")
            
    return None, None

def load_and_merge_boundaries():
    gdfs = []
    for iso in ISO_CODES:
        print(f"Fetching {iso}...")
        filepath, level = download_country_boundary(iso)
        if filepath:
            try:
                gdf = gpd.read_file(filepath)
                # Standardize columns
                cols = {}
                # GADM 4.1 often uses GID_0, NAME_0, etc.
                # Just specific mapping
                if 'GID_0' in gdf.columns: cols['GID_0'] = 'country_iso'
                if 'NAME_1' in gdf.columns: cols['NAME_1'] = 'admin1'
                if 'NAME_2' in gdf.columns: cols['NAME_2'] = 'admin2'
                if 'NAME_3' in gdf.columns: cols['NAME_3'] = 'admin3'
                # Sometimes user wants ISO codes for sub-admins? User asked "admin1,2,3". Names are better.
                
                # Rename available
                gdf = gdf.rename(columns=cols)
                
                # Ensure generic columns exist
                target_cols = ['country_iso', 'admin1', 'admin2', 'admin3']
                for c in target_cols:
                    if c not in gdf.columns:
                        gdf[c] = None
                
                gdfs.append(gdf[target_cols + ['geometry']])
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
    
    if not gdfs:
        raise ValueError("No boundaries loaded!")
        
    print("Merging boundaries...")
    full_gdf = pd.concat(gdfs, ignore_index=True)
    return full_gdf

def main():
    # 1. Prepare Boundaries
    boundaries = load_and_merge_boundaries()
    print(f"Combined boundaries have {len(boundaries)} polygons.")
    
    # 2. Process CSV
    chunksize = 1000000 
    cache = {} 
    
    first_chunk = True
    processed_count = 0
    
    print(f"Processing {INPUT_FILE}...")
    try:
        reader = pd.read_csv(INPUT_FILE, chunksize=chunksize)
        for i, chunk in enumerate(reader):
            print(f"Processing chunk {i} ({chunksize} rows)...")
            
            # Identify unique locations
            unique_locs = chunk[['lat', 'lon']].drop_duplicates()
            unknown_locs = []
            
            for idx, row in unique_locs.iterrows():
                key = (row['lat'], row['lon'])
                if key not in cache:
                    unknown_locs.append(row)
            
            if unknown_locs:
                print(f"  Resolving {len(unknown_locs)} new locations...")
                unknown_df = pd.DataFrame(unknown_locs)
                unknown_gdf = gpd.GeoDataFrame(
                    unknown_df, 
                    geometry=gpd.points_from_xy(unknown_df.lon, unknown_df.lat), 
                    crs="EPSG:4326"
                )
                
                # Spatial Join - optimized
                # Buffer slightly? No, points should fall in.
                joined = gpd.sjoin(unknown_gdf, boundaries, how="left", predicate="within")
                
                # Deduplicate if point falls in multiple (nested or overlap) - though we joined left.
                # sjoin usually duplicates point rows if multiple polygons match.
                # We need one match. 'within' usually unique for admin boundaries.
                # But GADM might have overlaps?
                joined = joined[~joined.index.duplicated(keep='first')]
                
                # Update cache
                # Since we did left join, index should match unknown_gdf
                # But we must align with unique_locs or just iter over joined
                for idx, row in joined.iterrows():
                     # idx is index from unknown_df
                     # Need lat/lon from original or row
                     # joined has lat/lon columns because they are preserved from left
                     lat = row['lat']
                     lon = row['lon']
                     
                     cache[(lat, lon)] = {
                        'country_iso': row['country_iso'] if pd.notna(row['country_iso']) else None,
                        'admin1': row['admin1'] if pd.notna(row['admin1']) else None,
                        'admin2': row['admin2'] if pd.notna(row['admin2']) else None,
                        'admin3': row['admin3'] if pd.notna(row['admin3']) else None,
                     }

            # Map to chunk
            # Construct columns
            # Using lists is faster than apply
            c_iso, c_a1, c_a2, c_a3 = [], [], [], []
            
            for lat, lon in zip(chunk['lat'], chunk['lon']):
                val = cache.get((lat, lon), {})
                c_iso.append(val.get('country_iso'))
                c_a1.append(val.get('admin1'))
                c_a2.append(val.get('admin2'))
                c_a3.append(val.get('admin3'))
            
            chunk['country_iso'] = c_iso
            chunk['admin1'] = c_a1
            chunk['admin2'] = c_a2
            chunk['admin3'] = c_a3
            
            # Write
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
            first_chunk = False
            processed_count += len(chunk)
            
        print(f"Done! Processed {processed_count} rows. Saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
