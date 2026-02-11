
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
import glob
from shapely.geometry import Point
import re

def parse_boundary_filename(filename):
    """Parses country ISO and ADM level from filename e.g. gb_ETH_ADM2.geojson"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 3:
        iso = parts[1]
        adm_level = parts[2].replace('.geojson', '')
        return iso, adm_level
    return None, None

def build_hierarchy_lookup(boundaries_dir):
    """
    Builds a hierarchy lookup table for ADM2 -> ADM1 -> ADM0.
    Returns a DataFrame with ADM2 keys and full hierarchy columns.
    """
    print("Building administrative hierarchy (ADM0 -> ADM1 -> ADM2)...")
    
    # 1. Load all boundaries by level
    files = glob.glob(os.path.join(boundaries_dir, "*.geojson"))
    adm_gdfs = {'ADM0': {}, 'ADM1': {}, 'ADM2': {}}
    
    for f in files:
        iso, level = parse_boundary_filename(f)
        if iso and level in adm_gdfs:
            print(f"Loading {iso} {level}...")
            gdf = gpd.read_file(f)
            # Standardize columns
            if 'shapeName' not in gdf.columns: gdf['shapeName'] = "Unknown"
            if 'shapeID' not in gdf.columns: gdf['shapeID'] = "Unknown"
            if 'shapeISO' not in gdf.columns: gdf['shapeISO'] = "Unknown"
            
            # Reproject centroid calculation to avoid warnings if strictly needed, 
            # but for simple containment checking 4326 is usually accepted in this context or warned.
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
                
            adm_gdfs[level][iso] = gdf

    # 2. Link Levels
    # We will iterate through countries that have ADM2
    hierarchy_rows = []
    
    countries_with_adm2 = list(adm_gdfs['ADM2'].keys())
    
    for iso in countries_with_adm2:
        print(f"Linking hierarchy for {iso}...")
        gdf_adm2 = adm_gdfs['ADM2'][iso]
        
        # Prepare ADM2 info
        # We need centroids to find parents (point-in-polygon is faster/reliable for hierarchy)
        # Using representative_point() is safer than centroid for irregular shapes
        gdf_adm2['geometry_point'] = gdf_adm2.geometry.representative_point()
        
        # Join ADM2 -> ADM1
        if iso in adm_gdfs['ADM1']:
            gdf_adm1 = adm_gdfs['ADM1'][iso][['shapeName', 'shapeID', 'geometry']]
            joined1 = gpd.sjoin(
                gpd.GeoDataFrame(gdf_adm2[['shapeName', 'shapeID', 'shapeISO', 'geometry_point']], geometry='geometry_point', crs="EPSG:4326"),
                gdf_adm1,
                how='left',
                predicate='within',
                rsuffix='_ADM1'
            )
            # Cleanup joined columns
            # shapeName_left is ADM2, shapeName_right is ADM1
            # We rename carefully
            # The sjoin output will have shapeName_left (ADM2), shapeID_left (ADM2), shapeName_right (ADM1), shapeID_right (ADM1)
        else:
            # Fallback if no ADM1
            print(f"  Warning: No ADM1 for {iso}")
            continue # Can't build full hierarchy

        # Join ADM2 (enriched with ADM1) -> ADM0
        # We can use the same ADM2 representative points to find ADM0, usually consistent
        if iso in adm_gdfs['ADM0']:
             gdf_adm0 = adm_gdfs['ADM0'][iso][['shapeName', 'shapeID', 'shapeISO', 'geometry']]
             # We use the previous join result, which is a GeoDataFrame of points
             joined2 = gpd.sjoin(
                 joined1,
                 gdf_adm0,
                 how='left',
                 predicate='within',
                 rsuffix='_ADM0'
             )
        else:
             print(f"  Warning: No ADM0 for {iso}")
             continue
             
        # Now we have a flattened dataframe with all info
        # Columns likely: shapeName_left (ADM2), shapeName_right (ADM1), shapeName (ADM0 - from last join)
        # Check actual columns
        # Rename for clarity
        # The dataframe 'joined2' has columns from ADM2, ADM1 join, AND ADM0 join
        
        # Mappings:
        # From ADM2 (original): shapeName_left -> ADM2_Name, shapeID_left -> ADM2_ID
        # From ADM1 (first join): shapeName_right -> ADM1_Name, shapeID_right -> ADM1_ID
        # From ADM0 (second join): shapeName -> ADM0_Name, shapeISO -> ADM0_ISO
        
        # Let's clean up
        data = joined2.copy()
        
        # Safely rename
        rename_map = {
            'shapeName_left': 'shapeName_ADM2',
            'shapeID_left': 'shapeID_ADM2',
            'shapeISO_left': 'shapeISO_ADM2',
            'shapeName_right': 'shapeName_ADM1',
            'shapeID_right': 'shapeID_ADM1',
            'shapeName': 'shapeName_ADM0',
            'shapeID': 'shapeID_ADM0',
            'shapeISO': 'shapeISO_ADM0'
        }
        
        # Adjust for column name collisions from sjoin suffixes which can be tricky
        # Logic: 
        # joined1: [shapeName (ADM2), shapeName_ADM1 (ADM1)] approx
        
        # Let's trust explicit column selection if we can, or just inspect
        # A safer way is to just keep specific columns
        
        # Simplification:
        # We assume the join worked.
        # We need a table: shapeID_ADM2 (Key) -> [All other columns]
        
        # Let's iterate and build a dedicated dict/frame
        # It's cleaner to just rename what we have
        
        # Note: sjoin adds suffixes to existing columns if overlap.
        # ADM1 join: 'shapeName' overlaps. -> shapeName_left (ADM2), shapeName_right (ADM1)
        # ADM0 join: 'shapeName' from ADM0 overlaps with something? 
        # NO, joined1 has 'shapeName_right'. ADM0 has 'shapeName'. No collision with 'right', but 'left' collision?
        # Actually joined1 has 'shapeName_left'. ADM0 has 'shapeName'.
        # This will produce 'shapeName_left' (from joined1) and 'shapeName_right' (from ADM0)??
        # It gets messy.
        
        # Better strategy: Rename inputs BEFORE join.
        pass

    # RESTART HIERARCHY STRATEGY: RENAME FIRST
    print("  Linking with robust renaming...")
    master_lookup = pd.DataFrame()
    
    for iso in countries_with_adm2:
        if iso not in adm_gdfs['ADM0'] or iso not in adm_gdfs['ADM1']:
            continue
            
        # Prepare ADM2 (The Base)
        base = adm_gdfs['ADM2'][iso].copy()
        base.columns = [f"{c}_ADM2" if c in ['shapeName', 'shapeID', 'shapeISO', 'shapeGroup', 'shapeType'] else c for c in base.columns]
        base['geometry_point'] = base.geometry.representative_point()
        
        # Prepare ADM1
        adm1 = adm_gdfs['ADM1'][iso][['shapeName', 'shapeID', 'shapeISO', 'geometry']].copy()
        adm1.columns = [f"{c}_ADM1" if c != 'geometry' else c for c in adm1.columns]
        
        # Prepare ADM0
        adm0 = adm_gdfs['ADM0'][iso][['shapeName', 'shapeID', 'shapeISO', 'geometry']].copy()
        adm0.columns = [f"{c}_ADM0" if c != 'geometry' else c for c in adm0.columns]

        # LINK 2 -> 1
        base_pts = gpd.GeoDataFrame(base.drop(columns='geometry'), geometry='geometry_point', crs=base.crs)
        joined1 = gpd.sjoin(base_pts, adm1, how='left', predicate='within')
        
        # LINK (2+1) -> 0
        joined2 = gpd.sjoin(joined1.drop(columns='index_right'), adm0, how='left', predicate='within')
        
        # Clean up
        cols = [c for c in joined2.columns if 'ADM' in c]
        # Ensure we have the base ADM2 ID
        if 'shapeID_ADM2' in cols:
            iso_lookup = pd.DataFrame(joined2[cols])
            master_lookup = pd.concat([master_lookup, iso_lookup], ignore_index=True)
            
    print(f"Hierarchy built for {len(master_lookup)} districts.")
    return master_lookup

def map_crop_mask_to_admin(input_parquet, output_dir, boundaries_dir):
    # 1. Build Lookup
    lookup_df = build_hierarchy_lookup(boundaries_dir)
    
    if lookup_df.empty:
        print("Empty hierarchy lookup. Check boundary files.")
        return

    # 2. Load ADM2 Geometries for the main Spatial Join
    # We re-load ADM2 boundaries just for their geometry + shapeID to join against points
    print("Loading ADM2 geometries for point mapping...")
    all_adm2_files = glob.glob(os.path.join(boundaries_dir, "*_ADM2.geojson"))
    adm2_gdfs = []
    for f in all_adm2_files:
        gdf = gpd.read_file(f)[['shapeID', 'geometry']]
        if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
        adm2_gdfs.append(gdf)
        
    if not adm2_gdfs:
        print("No ADM2 files found.")
        return
        
    master_adm2_geom = pd.concat(adm2_gdfs, ignore_index=True)
    # shapeID in master_adm2_geom will match shapeID_ADM2 in lookup_df
    
    # 3. Process Parquet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading {input_parquet}...")
    parquet_file = pq.ParquetFile(input_parquet)
    batch_size = 5_000_000 
    batch_count = 0
    
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        print(f"Processing batch {i}...")
        df_chunk = batch.to_pandas()
        
        # Create GeoDataFrame
        gdf_chunk = gpd.GeoDataFrame(
            df_chunk, 
            geometry=gpd.points_from_xy(df_chunk.longitude, df_chunk.latitude),
            crs="EPSG:4326"
        )
        
        # Spatial Join: Points -> ADM2 Geometries
        # This gives us 'shapeID' (which is the ADM2 ID)
        joined_chunk = gpd.sjoin(gdf_chunk, master_adm2_geom, how="left", predicate="within")
        
        # Now Merge with Hierarchy Lookup on shapeID == shapeID_ADM2
        # joined_chunk has 'shapeID' from master_adm2_geom
        merged_chunk = pd.merge(
            joined_chunk.drop(columns=['geometry', 'index_right']),
            lookup_df,
            left_on='shapeID',
            right_on='shapeID_ADM2',
            how='left'
        )
        
        # Clean up redundant columns
        if 'shapeID' in merged_chunk.columns and 'shapeID_ADM2' in merged_chunk.columns:
            merged_chunk = merged_chunk.drop(columns=['shapeID']) # Keep ADM2 specific one
            
        # Save
        chunk_path = os.path.join(output_dir, f"part_{i:04d}.parquet")
        merged_chunk.to_parquet(chunk_path, index=False)
        batch_count += 1
        
    print(f"Completed! Output saved to {output_dir}")

if __name__ == "__main__":
    input_file = "data/crop_mask/asap_mask_crop_v04.parquet"
    if not os.path.exists(input_file):
        print(f"Warning: Input file not found at {input_file}.")
        
    output_directory = "data/crop_mask/admin_mapped"
    boundaries_directory = "data/geoboundaries"
    
    map_crop_mask_to_admin(input_file, output_directory, boundaries_directory)
