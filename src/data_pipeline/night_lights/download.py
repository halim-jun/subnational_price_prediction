import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to fix PROJ_LIB (common issue with pyogrio/geopandas in some envs)
try:
    import pyproj
    proj_dir = pyproj.datadir.get_data_dir()
    os.environ['PROJ_LIB'] = proj_dir
    logger.info(f"Set PROJ_LIB to {proj_dir}")
except ImportError:
    logger.warning("pyproj not found. PROJ_LIB environment variable might not be set correctly.")
except Exception as e:
    logger.warning(f"Failed to set PROJ_LIB: {e}")

# Import dependencies after setting environment variables
try:
    import geopandas as gpd
    from shapely.geometry import box
    # Import the high-level raster function directly
    from blackmarble.raster import bm_raster
    from blackmarble.types import Product
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    sys.exit(1)

def download_black_marble_annual(
    country_name: str, 
    roi_bbox: tuple, 
    start_year: int, 
    end_year: int, 
    output_dir: str = "data/night_lights"
):
    """
    Downloads annual Black Marble data (VNP46A4) for a specific region.
    
    Args:
        country_name: Name of the country
        roi_bbox: Tuple of (West, South, East, North)
        start_year: Start year (e.g. 2012)
        end_year: End year (e.g. 2025)
        output_dir: Directory to save the data
    """
    
    token = os.getenv("BEARER_TOKEN")
    if not token:
        logger.error("BEARER_TOKEN not found in .env file.")
        return

    # Create output directory
    country_dir = os.path.join(output_dir, country_name)
    os.makedirs(country_dir, exist_ok=True)
    
    logger.info(f"Starting annual download for {country_name} ({start_year}-{end_year})...")
    
    # Create GeoDataFrame for ROI
    geometry = box(*roi_bbox)
    gdf = gpd.GeoDataFrame({'geometry': [geometry]}, index=[0], crs="EPSG:4326")

    # Define date range
    # VNP46A4 is Annual. We request one date per year (e.g., Jan 1st).
    dates = []
    for year in range(start_year, end_year + 1):
        if year > datetime.now().year:
            break
        dates.append(datetime(year, 1, 1).date())

    if not dates:
        logger.warning(f"No valid dates found for {country_name}")
        return

    try:
        logger.info(f"Requesting data for {len(dates)} years...")
        
        # Download data year by year to avoid full failure if one year is missing
        for date in dates:
            year = date.year
            logger.info(f"Processing {country_name} - {year}...")
            
            # Check if output file already exists
            output_nc_path = os.path.join(country_dir, f"{country_name}_VNP46A4_{year}.nc")
            if os.path.exists(output_nc_path):
                logger.info(f"File {output_nc_path} already exists. Skipping.")
                continue
            
            try:
                # Using VNP46A4 (Annual)
                ds = bm_raster(
                    gdf=gdf,
                    product_id=Product.VNP46A4,
                    date_range=[date],
                    token=token,
                    output_directory=country_dir,
                    output_skip_if_exists=True
                )
                
                # Save as NetCDF if successful
                output_nc_path = os.path.join(country_dir, f"{country_name}_VNP46A4_{year}.nc")
                ds.to_netcdf(output_nc_path)
                logger.info(f"Saved NetCDF to {output_nc_path}")
            
            except Exception as e:
                logger.warning(f"Failed to download {country_name} for {year}: {e}")
                continue

    except Exception as e:
        logger.error(f"Unexpected error during download for {country_name}: {e}")

if __name__ == "__main__":
    # Define Regions of Interest (West, South, East, North)
    ROIS = {
        "Kenya": (33.9, -4.7, 41.9, 5.5),
        "Ethiopia": (32.9, 3.3, 48.0, 14.9),
        "Somalia": (40.9, -1.7, 51.5, 12.0)
    }
    
    # Configuration
    # User requested 2007-2025, but VNP46 (Black Marble) is available from 2012.
    START_YEAR = 2012 
    END_YEAR = 2025
    
    logger.info(f"Initializing Night Time Light Downloader ({START_YEAR}-{END_YEAR})...")
    
    for country, bbox in ROIS.items():
        download_black_marble_annual(country, bbox, START_YEAR, END_YEAR)
