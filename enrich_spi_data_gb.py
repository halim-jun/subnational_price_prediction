
import pandas as pd
import geopandas as gpd
import requests
import os
import time

ISO_CODES = [
    'DJI', 'ERI', 'ETH', 'KEN', 'SOM', 'SSD', 'SDN', 'UGA', 'RWA', 'BDI', 'TZA', 
    'ZMB', 'MWI', 'MOZ', 'MDG'
]
GB_API_URL = "https://www.geoboundaries.org/api/current/gbOpen/{}/{}/"
DATA_DIR = "data/external/geoboundaries"
INPUT_FILE = "data/processed/spi/06_spi_csv/east_africa_spi_gamma_3_month.csv"
OUTPUT_FILE = "data/processed/spi/06_spi_csv/east_africa_spi_gamma_3_month_with_boundaries.csv"

def download_gb_file(iso, level):
    """Downloads GeoBoundaries GeoJSON."""
    lvl_str = f"ADM{level}"
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"gb_{iso}_{lvl_str}.geojson"
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return filepath

    api_url = GB_API_URL.format(iso, lvl_str)
    print(f"Fetching metadata {api_url}...")
    try:
        r = requests.get(api_url, timeout=10)
        if r.status_code != 200:
            print(f"  {lvl_str} not available for {iso}")
            return None
        
        meta = r.json()
        dl_url = meta.get('gjDownloadURL')
        if not dl_url:
            return None
            
        print(f"  Downloading {dl_url}...")
        r_dl = requests.get(dl_url, stream=True, timeout=30)
        if r_dl.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in r_dl.iter_content(chunk_size=32768):
                    f.write(chunk)
            print(f"  Saved {filename}")
            return filepath
    except Exception as e:
        print(f"  Error downloading {iso} {lvl_str}: {e}")
    
    return None

def load_layers():
    """Loads all ADM layers into separate merged GDFs."""
    gdfs_l1 = []
    gdfs_l2 = []
    gdfs_l3 = []
    gdfs_l0 = []
    
    for iso in ISO_CODES:
        # L0 for ISO
        f0 = download_gb_file(iso, 0)
        if f0:
            try:
                gdf = gpd.read_file(f0)
                # Keep ISO. 'shapeGroup' or 'shapeISO'
                # usually shapeGroup is ISO3
                if 'shapeGroup' not in gdf.columns: gdf['shapeGroup'] = iso
                gdf = gdf[['shapeGroup', 'geometry']].rename(columns={'shapeGroup': 'country_iso'})
                gdfs_l0.append(gdf)
            except: pass

        # L1
        f1 = download_gb_file(iso, 1)
        if f1:
            try:
                gdf = gpd.read_file(f1)
                gdf = gdf[['shapeName', 'geometry']].rename(columns={'shapeName': 'admin1'})
                gdfs_l1.append(gdf)
            except: pass
            
        # L2
        f2 = download_gb_file(iso, 2)
        if f2:
            try:
                gdf = gpd.read_file(f2)
                gdf = gdf[['shapeName', 'geometry']].rename(columns={'shapeName': 'admin2'})
                gdfs_l2.append(gdf)
            except: pass
            
        # L3
        f3 = download_gb_file(iso, 3)
        if f3:
            try:
                gdf = gpd.read_file(f3)
                gdf = gdf[['shapeName', 'geometry']].rename(columns={'shapeName': 'admin3'})
                gdfs_l3.append(gdf)
            except: pass
            
    # Concat
    print("Merging layers...")
    layer0 = pd.concat(gdfs_l0, ignore_index=True) if gdfs_l0 else None
    layer1 = pd.concat(gdfs_l1, ignore_index=True) if gdfs_l1 else None
    layer2 = pd.concat(gdfs_l2, ignore_index=True) if gdfs_l2 else None
    layer3 = pd.concat(gdfs_l3, ignore_index=True) if gdfs_l3 else None
    
    return layer0, layer1, layer2, layer3

def main():
    l0, l1, l2, l3 = load_layers()
    
    processed_count = 0
    chunksize = 1000000
    cache = {} 
    
    first_chunk = True
    
    print(f"Processing {INPUT_FILE}...")
    for chunk in pd.read_csv(INPUT_FILE, chunksize=chunksize):
        # Identify unique locations
        unique_locs = chunk[['lat', 'lon']].drop_duplicates()
        unknown_locs = []
        for _, row in unique_locs.iterrows():
            if (row['lat'], row['lon']) not in cache:
                unknown_locs.append(row)
        
        if unknown_locs:
            print(f"Resolving {len(unknown_locs)} new locations...")
            unknown_df = pd.DataFrame(unknown_locs)
            gdf_pts = gpd.GeoDataFrame(
                unknown_df, 
                geometry=gpd.points_from_xy(unknown_df.lon, unknown_df.lat), 
                crs="EPSG:4326"
            )
            
            # Prepare results dict keys
            # Initialize with lat,lon
            # We will merge results.
            
            # Helper to join
            def join_layer(points, layer, col_name):
                if layer is None: return points
                if 'index_right' in points.columns:
                    points = points.drop(columns=['index_right'])
                
                j = gpd.sjoin(points, layer, how='left', predicate='within')
                j = j[~j.index.duplicated(keep='first')] # Keep one match
                # Add col if missing
                if col_name not in j.columns:
                    j[col_name] = None
                return j
            
            # Join L0
            res = join_layer(gdf_pts, l0, 'country_iso')
            # Join L1
            res = join_layer(res, l1, 'admin1')
            # Join L2
            res = join_layer(res, l2, 'admin2')
            # Join L3
            res = join_layer(res, l3, 'admin3')
            
            # Update cache
            for idx, row in res.iterrows():
                lat, lon = row['lat'], row['lon']
                cache[(lat, lon)] = {
                    'country_iso': row['country_iso'] if pd.notna(row['country_iso']) else None,
                    'admin1': row['admin1'] if pd.notna(row['admin1']) else None,
                    'admin2': row['admin2'] if pd.notna(row['admin2']) else None,
                    'admin3': row['admin3'] if pd.notna(row['admin3']) else None,
                }
        
        # Apply mapping
        mapped_data = []
        
        # Vectorized map is tricky, list comp is fine
        countries = []
        adm1s = []
        adm2s = []
        adm3s = []
        
        for lat, lon in zip(chunk['lat'], chunk['lon']):
            c = cache.get((lat, lon), {})
            countries.append(c.get('country_iso'))
            adm1s.append(c.get('admin1'))
            adm2s.append(c.get('admin2'))
            adm3s.append(c.get('admin3'))
            
        chunk['country_iso'] = countries
        chunk['admin1'] = adm1s
        chunk['admin2'] = adm2s
        chunk['admin3'] = adm3s
        
        mode = 'w' if first_chunk else 'a'
        chunk.to_csv(OUTPUT_FILE, mode=mode, header=first_chunk, index=False)
        first_chunk = False
        processed_count += len(chunk)
        print(f"Processed {processed_count} rows.")

    print(f"Done. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
