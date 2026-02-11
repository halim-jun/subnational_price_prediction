
import pandas as pd
import sys
import os
import glob
import logging

# Add project root to path
# This script is in src/data_pipeline/merge/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

from src.data_pipeline.merge import subnational_merger
from src.data_pipeline.merge import run_subnational_merge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_admin_coverage():
    logger.info("Initializing Admin Coverage Check...")
    data_dir = os.path.join(project_root, 'data')
    
    # 1. Load Canonical Boundaries
    logger.info("loading canonical boundaries...")
    iso3_list = ['KEN', 'SOM', 'ETH']
    boundary_dir = os.path.join(data_dir, 'geoboundaries')
    admin_gdf = subnational_merger.load_canonical_boundaries(iso3_list, boundary_dir)
    canonical_names = admin_gdf['shapeName'].unique().tolist()
    logger.info(f"Loaded {len(canonical_names)} canonical Admin2 regions.")
    
    # 2. Check Price Coverage (Optimized)
    logger.info("--- Checking Price Data ---")
    csv_path = os.path.join(data_dir, "worldbank_imputed_price_data/WLD_RTFP_mkt_2026-01-13.csv")
    if os.path.exists(csv_path):
        # Read only necessary columns
        use_cols = ['country', 'adm2_name']
        try:
            price_df = pd.read_csv(csv_path, usecols=use_cols)
            target_countries = ['Kenya', 'Ethiopia', 'Somalia']
            price_df = price_df[price_df['country'].isin(target_countries)]
            source_admins = price_df['adm2_name'].unique().tolist()
            
            mapping = subnational_merger.fuzzy_match_names(pd.Series(source_admins), canonical_names)
            unmatched = [k for k, v in mapping.items() if v is None]
            
            print(f"Price Source Regions: {len(source_admins)}")
            print(f"Matched: {len(source_admins) - len(unmatched)}")
            print(f"Unmatched: {len(unmatched)}")
            if unmatched:
                print("Unmatched Price Regions:", unmatched)
        except Exception as e:
            logger.error(f"Error reading price: {e}")
    else:
        logger.error("Price file missing!")

    # 3. Check Population Coverage
    logger.info("--- Checking Population Data ---")
    # Use run_subnational_merge for pop as it's small enough
    pop_df = run_subnational_merge.load_population_data(data_dir)
    source_admins_pop = pop_df['admin2'].unique().tolist()
    mapping_pop = subnational_merger.fuzzy_match_names(pd.Series(source_admins_pop), canonical_names)
    unmatched_pop = [k for k, v in mapping_pop.items() if v is None]
    
    print(f"Population Source Regions: {len(source_admins_pop)}")
    print(f"Matched: {len(source_admins_pop) - len(unmatched_pop)}")
    print(f"Unmatched: {len(unmatched_pop)}")
    if unmatched_pop:
        print("Unmatched Population Regions:", unmatched_pop)

    # 4. Check Crop Coverage
    logger.info("--- Checking Crop Data ---")
    # Use run_subnational_merge for crop as it's aggregated and small
    try:
        crop_df = run_subnational_merge.load_crop_data(data_dir)
        source_admins_crop = crop_df['shapeName_ADM2'].unique().tolist() if 'shapeName_ADM2' in crop_df.columns else []
        if source_admins_crop:
            mapping_crop = subnational_merger.fuzzy_match_names(pd.Series(source_admins_crop), canonical_names)
            unmatched_crop = [k for k, v in mapping_crop.items() if v is None]
            print(f"Crop Source Regions: {len(source_admins_crop)}")
            print(f"Matched: {len(source_admins_crop) - len(unmatched_crop)}")
            print(f"Unmatched: {len(unmatched_crop)}")
            if unmatched_crop:
                print("Unmatched Crop Regions:", unmatched_crop)
        else:
            print("No Crop 'shapeName_ADM2' column found or empty.")
    except Exception as e:
        logger.error(f"Error checking crop: {e}")

if __name__ == "__main__":
    check_admin_coverage()
