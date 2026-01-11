
import pandas as pd
import geopandas as gpd
import requests
import os
import glob
import time

ISO_CODES = [
    'DJI', 'ERI', 'ETH', 'KEN', 'SOM', 'SSD', 'SDN', 'UGA', 'RWA', 'BDI', 'TZA', 
    'ZMB', 'MWI', 'MOZ', 'MDG'
]
GB_API_URL = "https://www.geoboundaries.org/api/current/gbOpen/{}/{}/"
DATA_DIR = "../../../data/geoboundaries"
INPUT_DIR = "../../../data/processed/spi/06_spi_csv"

# List of input files to process (exclude already processed or outputs)
INPUT_FILES = [
    "east_africa_spi_gamma_1_month.csv",
    "east_africa_spi_gamma_2_month.csv",
    # "east_africa_spi_gamma_3_month.csv", # Already done
    "east_africa_spi_gamma_6_month.csv",
    "east_africa_spi_gamma_9_month.csv",
    "east_africa_spi_gamma_12_month.csv",
    "east_africa_spi_gamma_18_month.csv",
    "east_africa_spi_gamma_24_month.csv",
]

def download_gb_file(iso, level):
    """Downloads GeoBoundaries GeoJSON if not exists."""
    lvl_str = f"ADM{level}"
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"gb_{iso}_{lvl_str}.geojson"
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
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
    gdfs_l0, gdfs_l1, gdfs_l2, gdfs_l3 = [], [], [], []
    
    for iso in ISO_CODES:
        # L0
        f0 = download_gb_file(iso, 0)
        if f0:
            try:
                gdf = gpd.read_file(f0)
                if 'shapeGroup' not in gdf.columns: gdf['shapeGroup'] = iso
                gdf = gdf[['shapeGroup', 'geometry']].rename(columns={'shapeGroup': 'country_iso'})
                gdfs_l0.append(gdf)
            except: pass

        # L1
        f1 = download_gb_file(iso, 1)
        if f1:
            try:
                gdf = gpd.read_file(f1).rename(columns={'shapeName': 'admin1'})[['admin1', 'geometry']]
                gdfs_l1.append(gdf)
            except: pass
            
        # L2
        f2 = download_gb_file(iso, 2)
        if f2:
            try:
                gdf = gpd.read_file(f2).rename(columns={'shapeName': 'admin2'})[['admin2', 'geometry']]
                gdfs_l2.append(gdf)
            except: pass
            
        # L3
        f3 = download_gb_file(iso, 3)
        if f3:
            try:
                gdf = gpd.read_file(f3).rename(columns={'shapeName': 'admin3'})[['admin3', 'geometry']]
                gdfs_l3.append(gdf)
            except: pass
            
    layer0 = pd.concat(gdfs_l0, ignore_index=True) if gdfs_l0 else None
    layer1 = pd.concat(gdfs_l1, ignore_index=True) if gdfs_l1 else None
    layer2 = pd.concat(gdfs_l2, ignore_index=True) if gdfs_l2 else None
    layer3 = pd.concat(gdfs_l3, ignore_index=True) if gdfs_l3 else None
    
    print("Layers loaded.")
    return layer0, layer1, layer2, layer3

def main():
    l0, l1, l2, l3 = load_layers()
    
    # We can reuse the cache across files since the grid is likely identical.
    # This will save HUGE amount of time.
    cache = {} 
    
    for filename in INPUT_FILES:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(INPUT_DIR, filename.replace(".csv", "_with_boundaries.csv"))
        
        if not os.path.exists(input_path):
            print(f"Skipping {filename} (not found)")
            continue
            
        if os.path.exists(output_path):
            print(f"Skipping {output_path} (already exists)")
            continue
            
        print(f"Processing {filename}...")
        
        chunksize = 2000000 # Increased chunk size for speed
        total_processed = 0
        first_chunk = True
        
        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            # 1. Identify new unique locations
            unique_locs = chunk[['lat', 'lon']].drop_duplicates()
            unknown_locs = []
            
            # Check cache
            # Optimization: Vectorized cache check?
            # Creating a set of keys from chunk is fast.
            # But let's stick to simple logic first, the grid is static.
            # First file fills cache, subsequent files just use it.
            
            for _, row in unique_locs.iterrows():
                if (row['lat'], row['lon']) not in cache:
                    unknown_locs.append(row)
            
            if unknown_locs:
                print(f"  Resolving {len(unknown_locs)} new locations...")
                unknown_df = pd.DataFrame(unknown_locs)
                gdf_pts = gpd.GeoDataFrame(
                    unknown_df, 
                    geometry=gpd.points_from_xy(unknown_df.lon, unknown_df.lat), 
                    crs="EPSG:4326"
                )
                
                # Helper
                def join_layer(points, layer, col_name):
                    if layer is None: return points
                    if 'index_right' in points.columns: points = points.drop(columns=['index_right'])
                    j = gpd.sjoin(points, layer, how='left', predicate='within')
                    j = j[~j.index.duplicated(keep='first')]
                    if col_name not in j.columns: j[col_name] = None
                    return j
                
                # Spatial Joins
                res = join_layer(gdf_pts, l0, 'country_iso')
                res = join_layer(res, l1, 'admin1')
                res = join_layer(res, l2, 'admin2')
                res = join_layer(res, l3, 'admin3')
                
                # Update Cache
                for _, row in res.iterrows():
                    cache[(row['lat'], row['lon'])] = {
                        'country_iso': row['country_iso'],
                        'admin1': row['admin1'],
                        'admin2': row['admin2'],
                        'admin3': row['admin3']
                    }
                    
            # 2. Map Data
            # Construct lists
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
            
            # 3. Write
            mode = 'w' if first_chunk else 'a'
            chunk.to_csv(output_path, mode=mode, header=first_chunk, index=False)
            first_chunk = False
            total_processed += len(chunk)
            print(f"  Processed {total_processed} rows...")
            
        print(f"Finished {filename}.")

if __name__ == "__main__":
    main()
