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
    
    # Find all TIFF files for the country
    country_input_dir = os.path.join(input_dir, country_name)
    tif_files = glob.glob(os.path.join(country_input_dir, "*.tif"))
    
    if not tif_files:
        logger.warning(f"No TIFF files found for {country_name} in {country_input_dir}")
        return []

    logger.info(f"Found {len(tif_files)} TIFF files for {country_name}.")

    # Group files by year
    # Filename format example: VNP46A4.A2012001.h21v09.002.2025086173009.tif
    files_by_year = {}
    for f in tif_files:
        try:
            basename = os.path.basename(f)
            # Extract "A2012001" part
            parts = basename.split('.')
            if len(parts) > 1 and parts[1].startswith('A'):
                year_doy = parts[1] # e.g., A2012001
                year = int(year_doy[1:5])
                
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append(f)
        except Exception as e:
            logger.warning(f"Skipping file {f}: {e}")

    logger.info(f"Processing years: {sorted(files_by_year.keys())}")

    for year, files in sorted(files_by_year.items()):
        logger.info(f"Processing {country_name} - {year} ({len(files)} tiles)...")
        
        try:
            # Open all source files
            src_files_to_mosaic = []
            for fp in files:
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)
            
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
            
            # NOTE: Assuming CRS is EPSG:4326 as standard for VNP46
            # If tiles have different CRS, we might need to reproject gdf.
            # Usually Black Marble is 4326.
            
            # Handle NoData (usually 65535 for uint16 in VNP46)
            # Masking it to None or NaN for stats
            # zonal_stats handles 'nodata' param.
            
            # Run Zonal Statistics
            stats = zonal_stats(
                vectors=admin_gdf,
                raster=data_array,
                affine=affine,
                stats=['mean', 'count'],
                nodata=65535 # Standard fill value for VNP46, checks if using inspect_tif
            )
            
            # Combine stats with Admin info
            for idx, stat in enumerate(stats):
                region_info = admin_gdf.iloc[idx]
                
                # Try to get best available name/ID
                name = region_info.get('shapeName') or region_info.get('ADM2_NAME') or region_info.get('Name') or 'Unknown'
                region_id = region_info.get('shapeISO') or region_info.get('ADM2_CODE') or str(idx)
                
                # Check valid data
                val = stat.get('mean')
                if val is None:
                    val = np.nan
                    
                results.append({
                    'country_iso': region_info.get('shapeGroup') or country_name[:3].upper(),
                    'admin2': name,
                    'admin2_id': region_id,
                    'year': year,
                    'night_light_mean': val,
                    'pixel_count': stat.get('count', 0)
                })
                
        except Exception as e:
            logger.error(f"Failed to process year {year} for {country_name}: {e}")
            continue
            
    return results

def main():
    # Configuration
    INPUT_DIR = "data/night_lights"
    OUTPUT_FILE = "data/night_lights/night_lights_admin2.parquet"
    
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
        try:
            admin_gdf = gpd.read_file(geojson_path)
            # Ensure consistent CRS (Black Marble is usually 4326)
            if admin_gdf.crs != "EPSG:4326":
                admin_gdf = admin_gdf.to_crs("EPSG:4326")
                
            country_results = process_country_night_lights(country, admin_gdf, INPUT_DIR)
            all_results.extend(country_results)
            
        except Exception as e:
            logger.error(f"Error loading boundaries for {country}: {e}")
            continue

    if all_results:
        logger.info(f"Saving aggregated results to {OUTPUT_FILE}...")
        df = pd.DataFrame(all_results)
        
        # Ensure output directory exists (data/processed usually exists)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        df.to_parquet(OUTPUT_FILE, index=False)
        logger.info("Processing complete!")
        print(df.head())
        print(f"Total rows: {len(df)}")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()
