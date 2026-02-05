import os
import sys
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ['PROJ_LIB'] = proj_dir
    logger.info(f"Set PROJ_LIB to {proj_dir}")
except Exception as e:
    logger.warning(f"Failed to set PROJ_LIB: {e}")

try:
    import geopandas as gpd
    from shapely.geometry import box
    from blackmarble.raster import bm_raster
    from blackmarble.types import Product
    import xarray as xr
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    sys.exit(1)

def test_download():
    country_name = "Kenya"
    roi_bbox = (33.9, -4.7, 41.9, 5.5)
    year = 2012
    output_dir = "data/night_lights_debug"
    
    token = os.getenv("BEARER_TOKEN")
    if not token:
        logger.error("BEARER_TOKEN not found in .env file.")
        return

    country_dir = os.path.join(output_dir, country_name)
    os.makedirs(country_dir, exist_ok=True)
    
    logger.info(f"Testing download for {country_name} ({year})...")
    
    geometry = box(*roi_bbox)
    gdf = gpd.GeoDataFrame({'geometry': [geometry]}, index=[0], crs="EPSG:4326")
    date = datetime(year, 1, 1).date()
    
    try:
        logger.info(f"Calling bm_raster...")
        ds = bm_raster(
            gdf=gdf,
            product_id=Product.VNP46A4,
            date_range=[date],
            token=token,
            output_directory=country_dir,
            output_skip_if_exists=True
        )
        
        logger.info(f"bm_raster returned type: {type(ds)}")
        
        if ds is None:
            logger.error("bm_raster returned None!")
            return

        output_nc_path = os.path.join(country_dir, f"{country_name}_VNP46A4_{year}.nc")
        logger.info(f"Saving to {output_nc_path}...")
        ds.to_netcdf(output_nc_path)
        logger.info(f"Success! Saved NetCDF to {output_nc_path}")
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_download()
