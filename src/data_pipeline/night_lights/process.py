import os
import glob
import logging
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats
import numpy as np
from shapely.geometry import box

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_country_night_lights(country_name, admin_gdf, input_dir):
    """
    Process night lights for a single country using TIFF tiles.
    """
    results = []
    
    # Find all source files for the country (prefer TIF, fallback to H5)
    country_input_dir = os.path.join(input_dir, country_name)
    tif_files = glob.glob(os.path.join(country_input_dir, "*.tif"))
    h5_files = glob.glob(os.path.join(country_input_dir, "*.h5"))
    
    # Combine sources, preferring tif if duplicates present (though we group by year anyway)
    all_files = tif_files + h5_files
    
    if not all_files:
        logger.warning(f"No TIFF or HDF5 files found for {country_name} in {country_input_dir}")
        return []

    logger.info(f"Found {len(tif_files)} TIFF and {len(h5_files)} HDF5 files for {country_name}.")

    # Group files by year
    # Filename format example: VNP46A4.A2012001.h21v09.002.2025086173009.tif
    files_by_year = {}
    for f in all_files:
        try:
            basename = os.path.basename(f)
            # Extract "A2012001" part
            parts = basename.split('.')
            if len(parts) > 1 and parts[1].startswith('A'):
                year_doy = parts[1] # e.g., A2012001
                year = int(year_doy[1:5])
                
                if year not in files_by_year:
                    files_by_year[year] = []
                # Avoid duplicates: if we have both .tif and .h5 for same tile, prefer .tif?
                # Actually, check by tile ID (hXXvYY)
                files_by_year[year].append(f)
        except Exception as e:
            logger.warning(f"Skipping file {f}: {e}")
            
    # Deduplicate within year (hXXvYY) - prefer .tif
    final_files_by_year = {}
    for year, files in files_by_year.items():
        tile_map = {}
        for f in files:
            # simple check for hXXvYY in filename
            # VNP46A4.A2012001.h21v09...
            import re
            match = re.search(r'h\d{2}v\d{2}', os.path.basename(f))
            if match:
                tile_id = match.group(0)
                ext = os.path.splitext(f)[1]
                
                # If we already have this tile, keep existing if it is .tif, replace if new is .tif
                if tile_id in tile_map:
                    current_ext = os.path.splitext(tile_map[tile_id])[1]
                    if ext == '.tif' and current_ext == '.h5':
                        tile_map[tile_id] = f
                else:
                    tile_map[tile_id] = f
            else:
                 # No tile ID, just keep it
                 tile_map[f] = f
        
        final_files_by_year[year] = list(tile_map.values())

    logger.info(f"Processing years: {sorted(final_files_by_year.keys())}")

    for year, files in sorted(final_files_by_year.items()):
        logger.info(f"Processing {country_name} - {year} ({len(files)} tiles)...")
        
        try:
            # Open all source files
            src_files_to_mosaic = []
            source_crs = None
            source_nodata = None
            
            # Context managers to keep open
            from rasterio.vrt import WarpedVRT
            
            for fp in files:
                ext = os.path.splitext(fp)[1]
                src = None
                
                if ext == '.h5':
                    # READ SUBDATASET
                    try:
                        # Need to find subdataset path.
                        # Rasterio can open HDF5 and list subdatasets.
                        # Usually: HDF5:"file.h5"://HDFEOS/GRIDS/VNP_Grid_DNB/Data_Fields/AllAngle_Composite_Snow_Free
                        with rasterio.open(fp) as tmp_src:
                             subs = tmp_src.subdatasets
                             target = [s for s in subs if 'AllAngle_Composite_Snow_Free' in s]
                             if target:
                                 src = rasterio.open(target[0])
                             else:
                                 logger.warning(f"No suitable subdataset found in {fp}")
                                 continue
                    except Exception as e:
                        logger.warning(f"Failed to open HDF5 {fp}: {e}")
                        continue
                else:
                    src = rasterio.open(fp)
                
                # Check CRS and Wrap if missing
                if src:
                    if src.crs is None:
                        # VNP46A4 is EPSG:4326
                        # Use WarpedVRT to assign CRS without warping data (src_crs=dst_crs)
                        vrt = WarpedVRT(src, crs="EPSG:4326")
                        src_files_to_mosaic.append(vrt)
                        current_crs = "EPSG:4326"
                    else:
                        src_files_to_mosaic.append(src)
                        current_crs = src.crs
                        
                    if source_crs is None:
                        source_crs = current_crs
                    if source_nodata is None:
                        source_nodata = src.nodata

            # Use default nodata if not present in metadata
            if source_nodata is None:
                source_nodata = 65535 # Common for VNP46A4
            
            if not src_files_to_mosaic:
                 logger.warning(f"No valid source files loaded for {year}")
                 continue

            # Merge (Mosaic) tiles
            mosaic, out_trans = merge(src_files_to_mosaic)
            
            # Helper to close src files
            for src in src_files_to_mosaic:
                src.close()

            # The mosaic is a numpy array (bands, rows, cols) -> (1, rows, cols)
            # We need just (rows, cols) for zonal_stats raster
            data_array = mosaic[0]
            
            # Prepare affine transform
            affine = out_trans
            
            # REPROJECT VECTOR TO RASTER CRS
            if admin_gdf.crs != source_crs:
                logger.info(f"Reprojecting admin boundaries from {admin_gdf.crs} to {source_crs}")
                admin_gdf_proj = admin_gdf.to_crs(source_crs)
            else:
                admin_gdf_proj = admin_gdf
            
            # Run Zonal Statistics
            # attributes allow us to keep ID info in the result list (optional, but we iterate by index mostly)
            # all_touched=True : KEY FIX for small geometries or slight misalignments
            stats = zonal_stats(
                vectors=admin_gdf_proj,
                raster=data_array,
                affine=affine,
                stats=['mean', 'count'],
                nodata=source_nodata,
                all_touched=True 
            )
            
            valid_count = 0
            nan_count = 0
            
            # Combine stats with Admin info
            for idx, stat in enumerate(stats):
                region_info = admin_gdf.iloc[idx] # Use original GDF for metadata
                
                # Try to get best available name/ID
                name = region_info.get('shapeName') or region_info.get('ADM2_NAME') or region_info.get('Name') or 'Unknown'
                region_id = region_info.get('shapeISO') or region_info.get('ADM2_CODE') or str(idx)
                admin1 = region_info.get('admin1') or 'Unknown'
                
                # Check valid data
                val = stat.get('mean')
                if val is None or val == source_nodata:
                    val = np.nan
                    nan_count += 1
                else:
                    valid_count += 1
                    
                results.append({
                    'country_iso': region_info.get('shapeGroup') or country_name[:3].upper(),
                    'admin1': admin1,
                    'admin2_canonical': name,
                    #'admin2_id': region_id, # Optional, notebook mainly uses names
                    'year': year,
                    'night_light_mean': val,
                    'pixel_count': stat.get('count', 0)
                })
            
            logger.info(f"Year {year}: {valid_count} valid regions, {nan_count} NaNs.")

        except Exception as e:
            logger.error(f"Failed to process year {year} for {country_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    return results

def load_boundaries_with_admin1(country, adm2_path):
    """
    Loads ADM2 boundaries and spatially joins with ADM1 boundaries to get admin1 names.
    Replicates the logic from subnational_merge_notebook.ipynb.
    """
    try:
        # Load Admin2
        gdf = gpd.read_file(adm2_path)
        
        # Standardize Admin2 Name
        rename_map = {
            'shapeName_ADM2': 'shapeName',
            'ADM2_NAME': 'shapeName',
            'admin2Name': 'shapeName',
            'shapeName': 'shapeName' 
        }
        gdf.rename(columns=rename_map, inplace=True)
        
        # Ensure shapeISO exists using the country name map if needed, 
        # but here we can just pass it or rely on existing columns.
        # The notebook passed iso3_list, here we have country name. 
        # We'll handle iso mapping in the process loop or here.
        
        # Try to find sibling ADM1 file
        # Assumption: adm2_path matches ...ADM2... which references ...ADM1...
        # e.g. data/geoboundaries/gb_KEN_ADM2.geojson -> ...ADM1...
        adm1_path = adm2_path.replace('ADM2', 'ADM1')
        
        if os.path.exists(adm1_path):
            logger.info(f"Loading sibling Admin1 file: {adm1_path}")
            try:
                adm1_gdf = gpd.read_file(adm1_path)
                # Standardize Admin1 Name
                adm1_rename = {
                    'shapeName': 'admin1',
                    'shapeName_ADM1': 'admin1',
                    'ADM1_NAME': 'admin1'
                }
                adm1_gdf.rename(columns=adm1_rename, inplace=True)
                
                if 'admin1' in adm1_gdf.columns:
                    # Ensure CRS match for sjoin
                    if gdf.crs != adm1_gdf.crs:
                        adm1_gdf = adm1_gdf.to_crs(gdf.crs)
                        
                    # Spatial Join
                    # Use centroid for safer join if polygons are messy, 
                    # but 'intersects' usually works if hierarchies are clean.
                    # Notebook uses: joined = gpd.sjoin(gdf, adm1_gdf[['admin1', 'geometry']], how='left', predicate='intersects')
                    # And drops duplicates.
                    
                    joined = gpd.sjoin(gdf, adm1_gdf[['admin1', 'geometry']], how='left', predicate='intersects')
                    joined = joined.drop_duplicates(subset=['shapeName'])
                    
                    if 'admin1' in joined.columns:
                        gdf = gdf.merge(joined[['shapeName', 'admin1']], on='shapeName', how='left')
                        logger.info(f"[{country}] Added Admin1 info via spatial join.")
                    else:
                        logger.warning(f"[{country}] Spatial join produced no admin1 column.")
                else:
                    logger.warning(f"[{country}] Admin1 file has no standard name column.")
            except Exception as e:
                logger.warning(f"[{country}] Failed to process Admin1 file: {e}")
        else:
             logger.warning(f"[{country}] No Admin1 sibling file found at {adm1_path}")
             
        return gdf

    except Exception as e:
        logger.error(f"Error loading boundaries for {country}: {e}")
        return None

def main():
    # Configuration
    INPUT_DIR = "data/night_lights"
    OUTPUT_FILE = "data/processed/night_lights_admin2.parquet" 
    
    # Mapping Country Name to GeoJSON file
    COUNTRY_MAPPING = {
        "Ethiopia": "data/geoboundaries/gb_ETH_ADM2.geojson",
        "Kenya": "data/geoboundaries/gb_KEN_ADM2.geojson",
        "Somalia": "data/geoboundaries/gb_SOM_ADM2.geojson"
    }

    all_results = []

    for country, geojson_path in COUNTRY_MAPPING.items():
        if not os.path.exists(geojson_path):
            logger.warning(f"GeoJSON not found for {country}: {geojson_path}. Skipping.")
            continue
            
        logger.info(f"Loading Admin2 boundaries for {country}...")
        
        # Load boundaries with Admin1 info
        admin_gdf = load_boundaries_with_admin1(country, geojson_path)
        
        if admin_gdf is None or admin_gdf.empty:
            continue
            
        # Ensure we have the necessary columns for processing
        # process_country_night_lights expects 'shapeName' (standardized in loader)
        # and we want 'admin1' if available.
        if 'admin1' not in admin_gdf.columns:
            admin_gdf['admin1'] = 'Unknown'

        # Force EPSG:4326 for initial state if not set (though loader reads file crs)
        # The key is process_country_night_lights will reproject it to raster CRS anyway.
        
        country_results = process_country_night_lights(country, admin_gdf, INPUT_DIR)
        all_results.extend(country_results)

    if all_results:
        logger.info(f"Saving aggregated results to {OUTPUT_FILE}...")
        df = pd.DataFrame(all_results)
        
        # Ensure output directory exists (data/processed usually exists)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        # Reorder columns for cleanliness
        cols = ['country_iso', 'admin1', 'admin2_canonical', 'year', 'night_light_mean', 'pixel_count']
        # If any are missing, fill/don't crash
        final_cols = [c for c in cols if c in df.columns]
        df = df[final_cols]
        
        df.to_parquet(OUTPUT_FILE, index=False)
        logger.info("Processing complete!")
        print(df.head())
        print(f"Total rows: {len(df)}")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()
